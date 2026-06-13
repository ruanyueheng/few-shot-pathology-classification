"""Tune non-balance-forcing methods on 200-train / 50-test (frozen_test) split.

KEY INSIGHT: The test_shuffled (61k) is class-IMBALANCED (teacher confirmed).
Methods that force balanced predictions (TIM with high λ_marg, PT-MAP Sinkhorn)
systematically distort the label distribution, hurting macro-F1 and balanced accuracy.

This script tunes ONLY methods without forced balance:
  1. Head          — cross-entropy classifier (no balance assumption)
  2. SimpleShot    — nearest prototype (no balance)
  3. Mahalanobis   — Gaussian discriminant (no balance)
  4. PT-MAP nosink — soft-EM without Sinkhorn (no balance)
  5. TIM(λ_marg=0) — CE + conditional entropy only (no marginal = no balance forcing)
  6. TIM(λ_marg=low) — very weak balance prior, may help slightly

Search spaces (designed for imbalanced generalization):
  SimpleShot:    tukey_beta × temperature
  Mahalanobis:   tukey_beta × shrink
  PT-MAP nosink: tukey_beta × n_iter × lambda_s
  TIM(λ=0):      temperature × lambda_cond × n_iter
  TIM(λ=low):    temperature × lambda_marg(0.05~0.2) × lambda_cond × n_iter

Pipeline:
  1. Train LoRA r=16 on 200 images (exclude frozen_test 50)
  2. Extract features for train + frozen_test
  3. Sweep all hyperparameters on frozen_test (with ground truth)
  4. Report best params per method with F1 + balanced accuracy
  5. Generate 61k submission CSVs with best params (train on full 250)

Usage:
  cd src
  python tune_nobalance.py --ema --lora_dropout 0.1 --verbose

Resume:
  Re-run same command. Checkpoint tracks completed folds.
  Add --force to start from scratch.
"""
from __future__ import annotations
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from datetime import timedelta
from itertools import product
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score

from data import (
    list_train_samples, list_test_samples,
    CLASS_NAMES, IDX_TO_CLASS, CLASS_TO_IDX,
)
from train_lora import (
    LoRAViTClassifier, make_eval_tf, train_one_run, seed_all,
)
from self_train import extract_features, predict_probs
from transductive import (
    simpleshot, mahalanobis, tim, pt_map,
)
from tqdm import tqdm


# ── Hyperparameter grids ──────────────────────────────────────

SIMPLESHOT_GRID = {
    "tukey_beta":    [0.3, 0.5, 0.7, 1.0],
    "temperature":   [5, 10, 15, 20, 30],
    "use_combined_mean": [True, False],
}

MAHALANOBIS_GRID = {
    "tukey_beta":    [0.3, 0.5, 0.7, 1.0],
    "shrink":        [0.1, 0.2, 0.3, 0.4, 0.5],
    "use_combined_mean": [True, False],
}

PTMAP_NOSINK_GRID = {
    "tukey_beta":    [0.3, 0.5, 0.7, 1.0],
    "n_iter":        [10, 20, 30, 40, 60],
    "lambda_s":      [5, 10, 15, 20, 30],
}

TIM_NOMARG_GRID = {
    # TIM with λ_marg = 0 (no balance forcing at all)
    "temperature":   [5, 10, 15, 20, 30],
    "lambda_cond":   [0.05, 0.1, 0.5, 1.0],
    "n_iter":        [500, 1000],
    "tukey_beta":    [0.3, 0.5, 0.7],
    "use_combined_mean": [True, False],
}

TIM_LOWMARG_GRID = {
    # TIM with very low λ_marg (mild balance prior, not aggressive)
    "temperature":   [5, 10, 15, 20],
    "lambda_marg":   [0.05, 0.1, 0.2],
    "lambda_cond":   [0.05, 0.1, 0.5],
    "n_iter":        [500, 1000],
    "tukey_beta":    [0.5],
    "use_combined_mean": [True],
}

TIM_FIXED = {"lr": 1e-4}


# ── Helpers ────────────────────────────────────────────────────

def get_frozen_test_ids(frozen_dir: str) -> set[tuple[str, str]]:
    """Parse frozen_test filenames -> set of (class_idx, image_num)."""
    frozen_ids = set()
    frozen_path = Path(frozen_dir)
    if not frozen_path.exists():
        return frozen_ids
    for f in frozen_path.glob("*.png"):
        parts = f.stem.split("_")
        # test_Class_0_Class_0_006 -> ['test', 'Class', '0', 'Class', '0', '006']
        if len(parts) >= 6:
            cls_idx = parts[2]
            img_num = parts[5]
            frozen_ids.add((cls_idx, img_num))
    return frozen_ids


def exclude_frozen_from_train(train_paths, train_labels, frozen_ids):
    """Return (filtered_paths, filtered_labels, excluded_count)."""
    keep_mask = []
    excluded = 0
    for p, l in zip(train_paths, train_labels):
        name = Path(p).stem
        parts = name.split("_")
        if len(parts) >= 3:
            cls_idx = str(l)
            img_num = parts[2]
            if (cls_idx, img_num) in frozen_ids:
                keep_mask.append(False)
                excluded += 1
                continue
        keep_mask.append(True)
    keep_mask = np.array(keep_mask)
    return (
        [p for p, k in zip(train_paths, keep_mask) if k],
        train_labels[keep_mask],
        excluded,
    )


def build_args(backbone="vits14", ema=True, lora_r=16, lora_alpha=32,
               lora_dropout=0.1, seed=42):
    """Build args namespace matching train_one_run's expectations."""
    return argparse.Namespace(
        backbone=backbone,
        image_size=224,
        epochs=30,
        batch_size=32,
        lr=5e-4,
        lr_head=1e-3,
        weight_decay=0.05,
        label_smoothing=0.1,
        grad_clip=1.0,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        head_dropout=0.1,
        ema=ema,
        ema_decay=0.95,
        hed_aug=False,
        hed_sigma=0.02,
        hed_bias=0.01,
        lora_mlp=False,
        dora=False,
        lora_init="default",
        supcon_weight=0.0,
        supcon_temp=0.07,
        mixup=0.0,
        cutmix=0.0,
        num_workers=2,
        seed=seed,
        verbose=False,
    )


def train_and_extract(train_paths, train_labels, device, args):
    """Train LoRA on given paths, extract features. Returns model + features."""
    full_idx = np.arange(len(train_paths))
    model, _, _ = train_one_run(
        train_paths, train_labels, full_idx, full_idx,
        args, device, run_id=0, refit_all=True,
    )

    n_params = model.trainable_param_count()

    sup_feats, sup_labels = extract_features(
        model, train_paths, args.image_size, args, device,
        labels=train_labels,
    )

    return model, sup_feats, sup_labels, n_params


def predict_test_feats(model, test_paths, args, device):
    """Extract features + head probs for test images."""
    test_feats, test_names = extract_features(
        model, test_paths, args.image_size, args, device,
        labels=None,
    )
    head_probs, _ = predict_probs(
        model, test_paths, args.image_size, args, device,
    )
    return test_feats, test_names, head_probs


# ── Sweep functions ────────────────────────────────────────────

def sweep_simpleshot(sup_feats, sup_labels, test_feats, y_true, num_classes=5):
    """Sweep SimpleShot hyperparameters."""
    results = []
    keys = list(SIMPLESHOT_GRID.keys())
    values = list(SIMPLESHOT_GRID.values())
    total = 1
    for v in values:
        total *= len(v)

    print(f"    Sweeping {total} SimpleShot configs ...")
    for combo in product(*values):
        params = dict(zip(keys, combo))
        try:
            pred = simpleshot(
                sup_feats, sup_labels, test_feats, num_classes=num_classes,
                tukey_beta=params["tukey_beta"],
                use_combined_mean=params["use_combined_mean"],
                temperature=params["temperature"],
            )
            f1 = float(f1_score(y_true, pred, average="macro"))
            bacc = float(balanced_accuracy_score(y_true, pred))
            results.append({"params": params, "f1": f1, "bacc": bacc})
        except Exception as e:
            results.append({"params": params, "f1": 0.0, "bacc": 0.0, "error": str(e)})

    results.sort(key=lambda x: -x["f1"])
    return results


def sweep_mahalanobis(sup_feats, sup_labels, test_feats, y_true, num_classes=5):
    """Sweep Mahalanobis hyperparameters."""
    results = []
    keys = list(MAHALANOBIS_GRID.keys())
    values = list(MAHALANOBIS_GRID.values())
    total = 1
    for v in values:
        total *= len(v)

    print(f"    Sweeping {total} Mahalanobis configs ...")
    for combo in product(*values):
        params = dict(zip(keys, combo))
        try:
            pred = mahalanobis(
                sup_feats, sup_labels, test_feats, num_classes=num_classes,
                tukey_beta=params["tukey_beta"],
                use_combined_mean=params["use_combined_mean"],
                shrink=params["shrink"],
            )
            f1 = float(f1_score(y_true, pred, average="macro"))
            bacc = float(balanced_accuracy_score(y_true, pred))
            results.append({"params": params, "f1": f1, "bacc": bacc})
        except Exception as e:
            results.append({"params": params, "f1": 0.0, "bacc": 0.0, "error": str(e)})

    results.sort(key=lambda x: -x["f1"])
    return results


def sweep_ptmap_nosink(sup_feats, sup_labels, test_feats, y_true, num_classes=5):
    """Sweep PT-MAP no-Sinkhorn hyperparameters."""
    results = []
    keys = list(PTMAP_NOSINK_GRID.keys())
    values = list(PTMAP_NOSINK_GRID.values())
    total = 1
    for v in values:
        total *= len(v)

    print(f"    Sweeping {total} PT-MAP nosink configs ...")
    for combo in product(*values):
        params = dict(zip(keys, combo))
        try:
            pred = pt_map(
                sup_feats, sup_labels, test_feats, num_classes=num_classes,
                tukey_beta=params["tukey_beta"],
                n_iter=params["n_iter"],
                lambda_s=params["lambda_s"],
                use_sinkhorn=False,
                sinkhorn_iter=10,
            )
            f1 = float(f1_score(y_true, pred, average="macro"))
            bacc = float(balanced_accuracy_score(y_true, pred))
            results.append({"params": params, "f1": f1, "bacc": bacc})
        except Exception as e:
            results.append({"params": params, "f1": 0.0, "bacc": 0.0, "error": str(e)})

    results.sort(key=lambda x: -x["f1"])
    return results


def sweep_tim_nomarg(sup_feats, sup_labels, test_feats, y_true, num_classes=5):
    """Sweep TIM with λ_marg=0 (no balance forcing)."""
    results = []
    keys = list(TIM_NOMARG_GRID.keys())
    values = list(TIM_NOMARG_GRID.values())
    total = 1
    for v in values:
        total *= len(v)

    print(f"    Sweeping {total} TIM(λ_marg=0) configs ...")
    for combo in product(*values):
        params = dict(zip(keys, combo))
        try:
            pred = tim(
                sup_feats, sup_labels, test_feats, num_classes=num_classes,
                tukey_beta=params["tukey_beta"],
                use_combined_mean=params["use_combined_mean"],
                n_iter=params["n_iter"],
                lr=TIM_FIXED["lr"],
                temperature=params["temperature"],
                lambda_marg=0.0,  # NO balance forcing!
                lambda_cond=params["lambda_cond"],
            )
            f1 = float(f1_score(y_true, pred, average="macro"))
            bacc = float(balanced_accuracy_score(y_true, pred))
            results.append({"params": params, "f1": f1, "bacc": bacc})
        except Exception as e:
            results.append({"params": params, "f1": 0.0, "bacc": 0.0, "error": str(e)})

    results.sort(key=lambda x: -x["f1"])
    return results


def sweep_tim_lowmarg(sup_feats, sup_labels, test_feats, y_true, num_classes=5):
    """Sweep TIM with very low λ_marg (mild balance prior)."""
    results = []
    keys = list(TIM_LOWMARG_GRID.keys())
    values = list(TIM_LOWMARG_GRID.values())
    total = 1
    for v in values:
        total *= len(v)

    print(f"    Sweeping {total} TIM(λ_marg=low) configs ...")
    for combo in product(*values):
        params = dict(zip(keys, combo))
        try:
            pred = tim(
                sup_feats, sup_labels, test_feats, num_classes=num_classes,
                tukey_beta=params["tukey_beta"],
                use_combined_mean=params["use_combined_mean"],
                n_iter=params["n_iter"],
                lr=TIM_FIXED["lr"],
                temperature=params["temperature"],
                lambda_marg=params["lambda_marg"],
                lambda_cond=params["lambda_cond"],
            )
            f1 = float(f1_score(y_true, pred, average="macro"))
            bacc = float(balanced_accuracy_score(y_true, pred))
            results.append({"params": params, "f1": f1, "bacc": bacc})
        except Exception as e:
            results.append({"params": params, "f1": 0.0, "bacc": 0.0, "error": str(e)})

    results.sort(key=lambda x: -x["f1"])
    return results


# ── Main ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Tune non-balance methods on 200/50 split for imbalanced 61k")
    ap.add_argument("--train_dir", default="../train_few_shot")
    ap.add_argument("--frozen_dir", default="../frozen_test",
                    help="frozen_test directory (50 held-out images)")
    ap.add_argument("--test_dir", default="../test_shuffled",
                    help="61k test images for final submission")
    ap.add_argument("--backbone", default="vits14")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no_ema", action="store_false", dest="ema")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_simpleshot", action="store_true")
    ap.add_argument("--skip_mahalanobis", action="store_true")
    ap.add_argument("--skip_ptmap", action="store_true")
    ap.add_argument("--skip_tim_nomarg", action="store_true")
    ap.add_argument("--skip_tim_lowmarg", action="store_true")
    ap.add_argument("--generate_subs", action="store_true",
                    help="Also generate 61k submission CSVs with best params")
    ap.add_argument("--out_json", default="../tune_nobalance_results.json")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Count total configs
    n_ss = 1
    for v in SIMPLESHOT_GRID.values():
        n_ss *= len(v)
    n_maha = 1
    for v in MAHALANOBIS_GRID.values():
        n_maha *= len(v)
    n_ptmap = 1
    for v in PTMAP_NOSINK_GRID.values():
        n_ptmap *= len(v)
    n_tim0 = 1
    for v in TIM_NOMARG_GRID.values():
        n_tim0 *= len(v)
    n_timl = 1
    for v in TIM_LOWMARG_GRID.values():
        n_timl *= len(v)

    print("=" * 70)
    print("  NON-BALANCE METHOD HYPERPARAMETER TUNING")
    print("  For class-IMBALANCED test_shuffled (61k)")
    print("=" * 70)
    print(f"  LoRA r={args.lora_r}, dropout={args.lora_dropout}")
    print(f"  SimpleShot:    {n_ss} configs")
    print(f"  Mahalanobis:   {n_maha} configs")
    print(f"  PT-MAP nosink: {n_ptmap} configs")
    print(f"  TIM(λ=0):      {n_tim0} configs")
    print(f"  TIM(λ=low):    {n_timl} configs")
    print(f"  Device: {device}")

    # ── Step 1: Prepare training data (200 images, excluding frozen_test) ──
    print(f"\n{'=' * 60}")
    print(f"  STEP 1: Preparing training data (200/50 split)")
    print(f"{'=' * 60}")

    all_paths, all_labels = list_train_samples(args.train_dir)
    print(f"  train_few_shot: {len(all_paths)} images")
    print(f"  Per-class: {np.bincount(all_labels).tolist()}")

    # Identify frozen_test images and exclude them
    frozen_ids = get_frozen_test_ids(args.frozen_dir)
    train_paths, train_labels, excluded = exclude_frozen_from_train(
        all_paths, all_labels, frozen_ids,
    )
    print(f"  Excluded {excluded} frozen_test images -> {len(train_paths)} train")

    # ── Step 2: Train LoRA on 200 images ──
    print(f"\n{'=' * 60}")
    print(f"  STEP 2: Training LoRA r={args.lora_r} on {len(train_paths)} images")
    print(f"{'=' * 60}")

    train_args = build_args(
        backbone=args.backbone, ema=args.ema,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, seed=args.seed,
    )

    t0 = time.time()
    model, sup_feats, sup_labels, n_params = train_and_extract(
        train_paths, train_labels, device, train_args,
    )
    train_time = time.time() - t0
    print(f"  trainable params: {n_params:,}")
    print(f"  Training time: {timedelta(seconds=int(train_time))}")

    # ── Step 3: Extract features for frozen_test ──
    print(f"\n{'=' * 60}")
    print(f"  STEP 3: Extracting features for frozen_test")
    print(f"{'=' * 60}")

    frozen_paths = list_test_samples(args.frozen_dir)
    frozen_feats, frozen_names, head_probs = predict_test_feats(
        model, frozen_paths, train_args, device,
    )
    print(f"  frozen_test features: {frozen_feats.shape}")

    # Load ground truth
    gt_path = Path(args.frozen_dir) / "_groundtruth.csv"
    gt_df = pd.read_csv(gt_path)
    gt_map = dict(zip(gt_df["filename"], gt_df["label"]))
    y_true = np.array([CLASS_TO_IDX.get(gt_map.get(n, ""), -1) for n in frozen_names])
    valid_mask = y_true >= 0
    y_true = y_true[valid_mask]
    print(f"  Ground truth: {valid_mask.sum()}/{len(frozen_names)} images")
    print(f"  Per-class GT: {np.bincount(y_true).tolist()}")

    # ── Step 4: Head baseline ──
    head_pred = head_probs.argmax(axis=1)[valid_mask]
    head_f1 = float(f1_score(y_true, head_pred, average="macro"))
    head_bacc = float(balanced_accuracy_score(y_true, head_pred))
    print(f"\n  Head baseline: F1={head_f1:.4f}, bacc={head_bacc:.4f}")

    # ── Step 5: Sweep each method ──
    all_results = {
        "head": {"f1": head_f1, "bacc": head_bacc, "params": "N/A"},
        "frozen_test_gt_dist": {str(c): int(v) for c, v in enumerate(np.bincount(y_true))},
    }

    # 5a. SimpleShot
    if not args.skip_simpleshot:
        print(f"\n{'=' * 60}")
        print(f"  STEP 5a: SimpleShot sweep ({n_ss} configs)")
        print(f"{'=' * 60}")
        t1 = time.time()
        ss_results = sweep_simpleshot(sup_feats, sup_labels, frozen_feats, y_true)
        ss_time = time.time() - t1
        best = ss_results[0]
        print(f"  Time: {ss_time:.1f}s")
        print(f"  >>> BEST SimpleShot: F1={best['f1']:.4f}, bacc={best['bacc']:.4f}")
        print(f"      params: {best['params']}")
        print(f"  Top 5:")
        for i, r in enumerate(ss_results[:5]):
            print(f"    #{i+1}: F1={r['f1']:.4f}, bacc={r['bacc']:.4f}, params={r['params']}")
        all_results["simpleshot"] = {
            "best_f1": best["f1"], "best_bacc": best["bacc"],
            "best_params": best["params"],
            "all_results": ss_results[:20],  # top 20
            "sweep_time": ss_time,
        }

    # 5b. Mahalanobis
    if not args.skip_mahalanobis:
        print(f"\n{'=' * 60}")
        print(f"  STEP 5b: Mahalanobis sweep ({n_maha} configs)")
        print(f"{'=' * 60}")
        t2 = time.time()
        maha_results = sweep_mahalanobis(sup_feats, sup_labels, frozen_feats, y_true)
        maha_time = time.time() - t2
        best = maha_results[0]
        print(f"  Time: {maha_time:.1f}s")
        print(f"  >>> BEST Mahalanobis: F1={best['f1']:.4f}, bacc={best['bacc']:.4f}")
        print(f"      params: {best['params']}")
        print(f"  Top 5:")
        for i, r in enumerate(maha_results[:5]):
            print(f"    #{i+1}: F1={r['f1']:.4f}, bacc={r['bacc']:.4f}, params={r['params']}")
        all_results["mahalanobis"] = {
            "best_f1": best["f1"], "best_bacc": best["bacc"],
            "best_params": best["params"],
            "all_results": maha_results[:20],
            "sweep_time": maha_time,
        }

    # 5c. PT-MAP nosink
    if not args.skip_ptmap:
        print(f"\n{'=' * 60}")
        print(f"  STEP 5c: PT-MAP nosink sweep ({n_ptmap} configs)")
        print(f"{'=' * 60}")
        t3 = time.time()
        ptmap_results = sweep_ptmap_nosink(sup_feats, sup_labels, frozen_feats, y_true)
        ptmap_time = time.time() - t3
        best = ptmap_results[0]
        print(f"  Time: {ptmap_time:.1f}s")
        print(f"  >>> BEST PT-MAP nosink: F1={best['f1']:.4f}, bacc={best['bacc']:.4f}")
        print(f"      params: {best['params']}")
        print(f"  Top 5:")
        for i, r in enumerate(ptmap_results[:5]):
            print(f"    #{i+1}: F1={r['f1']:.4f}, bacc={r['bacc']:.4f}, params={r['params']}")
        all_results["ptmap_nosink"] = {
            "best_f1": best["f1"], "best_bacc": best["bacc"],
            "best_params": best["params"],
            "all_results": ptmap_results[:20],
            "sweep_time": ptmap_time,
        }

    # 5d. TIM with λ_marg=0
    if not args.skip_tim_nomarg:
        print(f"\n{'=' * 60}")
        print(f"  STEP 5d: TIM(λ_marg=0) sweep ({n_tim0} configs)")
        print(f"{'=' * 60}")
        t4 = time.time()
        tim0_results = sweep_tim_nomarg(sup_feats, sup_labels, frozen_feats, y_true)
        tim0_time = time.time() - t4
        best = tim0_results[0]
        print(f"  Time: {tim0_time:.1f}s")
        print(f"  >>> BEST TIM(λ_marg=0): F1={best['f1']:.4f}, bacc={best['bacc']:.4f}")
        print(f"      params: {best['params']}")
        print(f"  Top 5:")
        for i, r in enumerate(tim0_results[:5]):
            print(f"    #{i+1}: F1={r['f1']:.4f}, bacc={r['bacc']:.4f}, params={r['params']}")
        all_results["tim_nomarg"] = {
            "best_f1": best["f1"], "best_bacc": best["bacc"],
            "best_params": best["params"],
            "all_results": tim0_results[:20],
            "sweep_time": tim0_time,
        }

    # 5e. TIM with low λ_marg
    if not args.skip_tim_lowmarg:
        print(f"\n{'=' * 60}")
        print(f"  STEP 5e: TIM(λ_marg=low) sweep ({n_timl} configs)")
        print(f"{'=' * 60}")
        t5 = time.time()
        timl_results = sweep_tim_lowmarg(sup_feats, sup_labels, frozen_feats, y_true)
        timl_time = time.time() - t5
        best = timl_results[0]
        print(f"  Time: {timl_time:.1f}s")
        print(f"  >>> BEST TIM(λ_marg=low): F1={best['f1']:.4f}, bacc={best['bacc']:.4f}")
        print(f"      params: {best['params']}")
        print(f"  Top 5:")
        for i, r in enumerate(timl_results[:5]):
            print(f"    #{i+1}: F1={r['f1']:.4f}, bacc={r['bacc']:.4f}, params={r['params']}")
        all_results["tim_lowmarg"] = {
            "best_f1": best["f1"], "best_bacc": best["bacc"],
            "best_params": best["params"],
            "all_results": timl_results[:20],
            "sweep_time": timl_time,
        }

    # ── Step 6: Summary ──
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: Best config per non-balance method")
    print(f"  (Trained on 200, evaluated on frozen_test 50)")
    print(f"{'=' * 70}")
    print(f"  {'Method':<25} {'F1':>8} {'bacc':>8}  Best params")
    print(f"  {'-' * 75}")

    ranked = []
    for method in ["head", "simpleshot", "mahalanobis", "ptmap_nosink",
                    "tim_nomarg", "tim_lowmarg"]:
        if method in all_results:
            r = all_results[method]
            f1 = r.get("best_f1", r.get("f1", 0))
            bacc = r.get("best_bacc", r.get("bacc", 0))
            params = r.get("best_params", r.get("params", "N/A"))
            params_str = str(params) if isinstance(params, dict) else str(params)
            if len(params_str) > 45:
                params_str = params_str[:42] + "..."
            print(f"  {method:<25} {f1:>8.4f} {bacc:>8.4f}  {params_str}")
            ranked.append((method, f1, bacc))

    ranked.sort(key=lambda x: -x[1])
    print(f"\n  >>> OVERALL BEST: {ranked[0][0]} "
          f"F1={ranked[0][1]:.4f}, bacc={ranked[0][2]:.4f}")

    # ── Step 7: Generate 61k submission CSVs with best params ──
    if args.generate_subs:
        print(f"\n{'=' * 70}")
        print(f"  STEP 7: Generating 61k submission CSVs with best params")
        print(f"  (Re-training on FULL 250 images for test_shuffled)")
        print(f"{'=' * 70}")

        # Re-train on ALL 250 images
        print(f"  Re-training on 250 images (no frozen exclusion)...")
        all_paths_250, all_labels_250 = list_train_samples(args.train_dir)
        train_args_250 = build_args(
            backbone=args.backbone, ema=args.ema,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, seed=args.seed,
        )
        model_250, sup_feats_250, sup_labels_250, _ = train_and_extract(
            all_paths_250, all_labels_250, device, train_args_250,
        )

        # Extract 61k test features
        test_paths = list_test_samples(args.test_dir)
        print(f"  test_shuffled: {len(test_paths)} images")
        t6 = time.time()
        test_feats_250, test_names_250, head_probs_250 = predict_test_feats(
            model_250, test_paths, train_args_250, device,
        )
        print(f"  Feature extraction: {timedelta(seconds=int(time.time() - t6))}")

        # Generate submissions for each method's best params
        out_dir = Path(args.out_json).parent

        # Head
        head_pred_61k = head_probs_250.argmax(axis=1)
        head_labels_61k = [IDX_TO_CLASS[int(i)] for i in head_pred_61k]
        df_head = pd.DataFrame({"filename": test_names_250, "label": head_labels_61k})
        df_head.to_csv(str(out_dir / "sub_tuned_head.csv"), index=False)
        print(f"  [ok] sub_tuned_head.csv")

        # SimpleShot
        if "simpleshot" in all_results:
            bp = all_results["simpleshot"]["best_params"]
            pred_ss = simpleshot(
                sup_feats_250, sup_labels_250, test_feats_250, num_classes=5,
                tukey_beta=bp["tukey_beta"],
                use_combined_mean=bp["use_combined_mean"],
                temperature=bp["temperature"],
            )
            labels_ss = [IDX_TO_CLASS[int(i)] for i in pred_ss]
            pd.DataFrame({"filename": test_names_250, "label": labels_ss}).to_csv(
                str(out_dir / "sub_tuned_simpleshot.csv"), index=False)
            print(f"  [ok] sub_tuned_simpleshot.csv (params: {bp})")

        # Mahalanobis
        if "mahalanobis" in all_results:
            bp = all_results["mahalanobis"]["best_params"]
            pred_maha = mahalanobis(
                sup_feats_250, sup_labels_250, test_feats_250, num_classes=5,
                tukey_beta=bp["tukey_beta"],
                use_combined_mean=bp["use_combined_mean"],
                shrink=bp["shrink"],
            )
            labels_maha = [IDX_TO_CLASS[int(i)] for i in pred_maha]
            pd.DataFrame({"filename": test_names_250, "label": labels_maha}).to_csv(
                str(out_dir / "sub_tuned_mahalanobis.csv"), index=False)
            print(f"  [ok] sub_tuned_mahalanobis.csv (params: {bp})")

        # PT-MAP nosink
        if "ptmap_nosink" in all_results:
            bp = all_results["ptmap_nosink"]["best_params"]
            pred_ptmap = pt_map(
                sup_feats_250, sup_labels_250, test_feats_250, num_classes=5,
                tukey_beta=bp["tukey_beta"],
                n_iter=bp["n_iter"],
                lambda_s=bp["lambda_s"],
                use_sinkhorn=False,
                sinkhorn_iter=10,
            )
            labels_ptmap = [IDX_TO_CLASS[int(i)] for i in pred_ptmap]
            pd.DataFrame({"filename": test_names_250, "label": labels_ptmap}).to_csv(
                str(out_dir / "sub_tuned_ptmap_nosink.csv"), index=False)
            print(f"  [ok] sub_tuned_ptmap_nosink.csv (params: {bp})")

        # TIM nomarg
        if "tim_nomarg" in all_results:
            bp = all_results["tim_nomarg"]["best_params"]
            pred_tim0 = tim(
                sup_feats_250, sup_labels_250, test_feats_250, num_classes=5,
                tukey_beta=bp["tukey_beta"],
                use_combined_mean=bp["use_combined_mean"],
                n_iter=bp["n_iter"],
                lr=TIM_FIXED["lr"],
                temperature=bp["temperature"],
                lambda_marg=0.0,
                lambda_cond=bp["lambda_cond"],
            )
            labels_tim0 = [IDX_TO_CLASS[int(i)] for i in pred_tim0]
            pd.DataFrame({"filename": test_names_250, "label": labels_tim0}).to_csv(
                str(out_dir / "sub_tuned_tim_nomarg.csv"), index=False)
            print(f"  [ok] sub_tuned_tim_nomarg.csv (params: {bp})")

        # TIM lowmarg
        if "tim_lowmarg" in all_results:
            bp = all_results["tim_lowmarg"]["best_params"]
            pred_timl = tim(
                sup_feats_250, sup_labels_250, test_feats_250, num_classes=5,
                tukey_beta=bp["tukey_beta"],
                use_combined_mean=bp["use_combined_mean"],
                n_iter=bp["n_iter"],
                lr=TIM_FIXED["lr"],
                temperature=bp["temperature"],
                lambda_marg=bp["lambda_marg"],
                lambda_cond=bp["lambda_cond"],
            )
            labels_timl = [IDX_TO_CLASS[int(i)] for i in pred_timl]
            pd.DataFrame({"filename": test_names_250, "label": labels_timl}).to_csv(
                str(out_dir / "sub_tuned_tim_lowmarg.csv"), index=False)
            print(f"  [ok] sub_tuned_tim_lowmarg.csv (params: {bp})")

        # Print label distributions for all submissions
        print(f"\n  Per-method label distribution on 61k:")
        print(f"  {'Method':<25} " + "  ".join(f"{'Cl'+str(c):>6}" for c in range(5))
              + "  Total")
        print(f"  {'-' * 75}")
        for csv_name in ["sub_tuned_head.csv", "sub_tuned_simpleshot.csv",
                          "sub_tuned_mahalanobis.csv", "sub_tuned_ptmap_nosink.csv",
                          "sub_tuned_tim_nomarg.csv", "sub_tuned_tim_lowmarg.csv"]:
            csv_path = out_dir / csv_name
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                counts = df["label"].value_counts()
                method = csv_name.replace("sub_tuned_", "").replace(".csv", "")
                row = [counts.get(f"Class_{c}", 0) for c in range(5)]
                total = sum(row)
                print(f"  {method:<25} " + "  ".join(f"{c:>6}" for c in row)
                      + f"  {total}")

    # ── Save results ──
    all_results["experiment"] = "tune_nobalance"
    all_results["lora_r"] = args.lora_r
    all_results["lora_dropout"] = args.lora_dropout
    all_results["train_size"] = len(train_paths)
    all_results["frozen_test_size"] = int(valid_mask.sum())

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Deep convert
    import json as _json
    with open(args.out_json, "w") as f:
        _json.dump(all_results, f, indent=2, default=convert)
    print(f"\n[ok] Results saved -> {args.out_json}")

    total_time = time.time() - t0
    print(f"  Total time: {timedelta(seconds=int(total_time))}")

    # Print next step instructions
    print(f"\n{'=' * 70}")
    print(f"  NEXT STEPS")
    print(f"{'=' * 70}")
    print(f"  1. Check results above for best method + params")
    print(f"  2. To generate 61k submissions, re-run with --generate_subs")
    print(f"  3. Or use final_submit.py with the best params")
    print(f"")
    print(f"  Example for final_submit.py:")
    if "mahalanobis" in all_results and "best_params" in all_results.get("mahalanobis", {}):
        bp = all_results["mahalanobis"]["best_params"]
        print(f"  python final_submit.py --final_method mahalanobis "
              f"--tukey_beta {bp['tukey_beta']} --maha_shrink {bp['shrink']} "
              f"--save_all_methods")


if __name__ == "__main__":
    main()
