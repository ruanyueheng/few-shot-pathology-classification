"""Push F1 higher — NO TTA, NO model ensemble (per teacher's rules).

Allowed optimization strategies:
  Phase 1 - Transductive method improvements (no retraining):
    - LaplacianShot sweep (knn affinity graph, feasible on 50 query)
    - Wider TIM(λ_marg=0) search (expanded temperature/lambda_cond ranges)
    - PT-MAP nosink wider search

  Phase 2 - Training improvements:
    - Sweep: lora_alpha × epochs × supcon_weight × hed_aug × lr × label_smoothing
    - For each config: train → extract features → TIM(λ_marg=0) best params → evaluate

  Phase 3 - Generate 61k submissions with best overall setup

Usage:
  cd src

  # Run all phases
  python tune_v2.py --ema --lora_dropout 0.1 --verbose

  # Run only Phase 1 (quick, ~20 min)
  python tune_v2.py --ema --lora_dropout 0.1 --verbose --phase 1

  # Run Phase 2 (training sweep, ~60 min)
  python tune_v2.py --ema --lora_dropout 0.1 --verbose --phase 2

  # Generate 61k submissions with best config
  python tune_v2.py --ema --lora_dropout 0.1 --verbose --generate_subs
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from datetime import timedelta
from itertools import product
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
    simpleshot, laplacianshot, mahalanobis, tim, pt_map,
)
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════
#  Hyperparameter grids
# ═══════════════════════════════════════════════════════════════

# Phase 1: Wider TIM(λ_marg=0) — expand beyond tune_nobalance's range
TIM_NOMARG_WIDE = {
    "temperature":   [3, 5, 7, 10, 12, 15, 20, 25, 30, 40],
    "lambda_cond":   [0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.5, 1.0],
    "n_iter":        [300, 500, 800, 1000],
    "tukey_beta":    [0.3, 0.5, 0.7, 1.0],
    "use_combined_mean": [True, False],
}

# Phase 1: LaplacianShot (full grid is 3840, we'll sample)
LAPLACIANSHOT_GRID = {
    "tukey_beta":    [0.3, 0.5, 0.7, 1.0],
    "knn":           [3, 5, 7, 10],
    "lam":           [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "sigma":         [0.5, 1.0, 2.0, 5.0],
    "n_iter":        [10, 20, 30],
    "use_combined_mean": [True, False],
}

# Phase 1: PT-MAP nosink wider search
PTMAP_NOSINK_WIDE = {
    "tukey_beta":    [0.3, 0.5, 0.7, 1.0],
    "n_iter":        [5, 10, 20, 30, 50],
    "lambda_s":      [3, 5, 10, 15, 20, 30, 50],
}

# Phase 2: Training config sweep
TRAINING_GRID = {
    "lora_alpha":    [16, 32, 64],
    "epochs":        [30, 50, 80],
    "supcon_weight": [0.0, 0.1, 0.3],
    "hed_aug":       [False, True],
}

TIM_FIXED = {"lr": 1e-4}


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def get_frozen_test_ids(frozen_dir: str) -> set[tuple[str, str]]:
    frozen_ids = set()
    frozen_path = Path(frozen_dir)
    if not frozen_path.exists():
        return frozen_ids
    for f in frozen_path.glob("*.png"):
        parts = f.stem.split("_")
        if len(parts) >= 6:
            cls_idx = parts[2]
            img_num = parts[5]
            frozen_ids.add((cls_idx, img_num))
    return frozen_ids


def exclude_frozen_from_train(train_paths, train_labels, frozen_ids):
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
               lora_dropout=0.1, seed=42, epochs=30, supcon_weight=0.0,
               hed_aug=False):
    return argparse.Namespace(
        backbone=backbone,
        image_size=224,
        epochs=epochs,
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
        hed_aug=hed_aug,
        hed_sigma=0.02,
        hed_bias=0.01,
        lora_mlp=False,
        dora=False,
        lora_init="default",
        supcon_weight=supcon_weight,
        supcon_temp=0.07,
        mixup=0.0,
        cutmix=0.0,
        num_workers=2,
        seed=seed,
        verbose=False,
    )


def train_and_extract(train_paths, train_labels, device, args):
    full_idx = np.arange(len(train_paths))
    model, _, _ = train_one_run(
        train_paths, train_labels, full_idx, full_idx,
        args, device, run_id=0, refit_all=True,
    )
    sup_feats, sup_labels = extract_features(
        model, train_paths, args.image_size, args, device,
        labels=train_labels,
    )
    return model, sup_feats, sup_labels


def predict_test_feats(model, test_paths, args, device):
    test_feats, test_names = extract_features(
        model, test_paths, args.image_size, args, device, labels=None,
    )
    head_probs, _ = predict_probs(
        model, test_paths, args.image_size, args, device,
    )
    return test_feats, test_names, head_probs


# ═══════════════════════════════════════════════════════════════
#  Sweep functions
# ═══════════════════════════════════════════════════════════════

def _sample_grid(grid, max_configs, seed=42):
    """Sample max_configs from a product grid."""
    keys = list(grid.keys())
    values = list(grid.values())
    total = 1
    for v in values:
        total *= len(v)
    if max_configs is None or total <= max_configs:
        return list(product(*values)), total
    # Random sample
    rng = np.random.RandomState(seed)
    indices = rng.choice(total, max_configs, replace=False)
    combos = []
    for idx in indices:
        combo = []
        remaining = idx
        for v in reversed(values):
            combo.append(v[remaining % len(v)])
            remaining //= len(v)
        combos.append(tuple(reversed(combo)))
    return combos, total


def sweep_tim_nomarg_wide(sup_feats, sup_labels, test_feats, y_true,
                           num_classes=5, max_configs=500):
    """Wider TIM(λ_marg=0) sweep with expanded parameter ranges."""
    keys = list(TIM_NOMARG_WIDE.keys())
    combos, total = _sample_grid(TIM_NOMARG_WIDE, max_configs)

    print(f"    Sweeping {len(combos)}/{total} TIM(λ_marg=0) wide configs ...")
    results = []
    for combo in tqdm(combos, desc="TIM_wide", leave=False):
        params = dict(zip(keys, combo))
        try:
            pred = tim(
                sup_feats, sup_labels, test_feats, num_classes=num_classes,
                tukey_beta=params["tukey_beta"],
                use_combined_mean=params["use_combined_mean"],
                n_iter=params["n_iter"],
                lr=TIM_FIXED["lr"],
                temperature=params["temperature"],
                lambda_marg=0.0,
                lambda_cond=params["lambda_cond"],
            )
            f1 = float(f1_score(y_true, pred, average="macro"))
            bacc = float(balanced_accuracy_score(y_true, pred))
            results.append({"params": params, "f1": f1, "bacc": bacc})
        except Exception as e:
            results.append({"params": params, "f1": 0.0, "bacc": 0.0, "error": str(e)})

    results.sort(key=lambda x: -x["f1"])
    return results


def sweep_laplacianshot(sup_feats, sup_labels, test_feats, y_true,
                         num_classes=5, max_configs=500):
    """Sweep LaplacianShot hyperparameters."""
    keys = list(LAPLACIANSHOT_GRID.keys())
    combos, total = _sample_grid(LAPLACIANSHOT_GRID, max_configs)

    print(f"    Sweeping {len(combos)}/{total} LaplacianShot configs ...")
    results = []
    for combo in tqdm(combos, desc="LapShot", leave=False):
        params = dict(zip(keys, combo))
        try:
            pred = laplacianshot(
                sup_feats, sup_labels, test_feats, num_classes=num_classes,
                tukey_beta=params["tukey_beta"],
                use_combined_mean=params["use_combined_mean"],
                knn=params["knn"],
                lam=params["lam"],
                sigma=params["sigma"],
                n_iter=params["n_iter"],
            )
            f1 = float(f1_score(y_true, pred, average="macro"))
            bacc = float(balanced_accuracy_score(y_true, pred))
            results.append({"params": params, "f1": f1, "bacc": bacc})
        except Exception as e:
            results.append({"params": params, "f1": 0.0, "bacc": 0.0, "error": str(e)})

    results.sort(key=lambda x: -x["f1"])
    return results


def sweep_ptmap_nosink_wide(sup_feats, sup_labels, test_feats, y_true,
                              num_classes=5):
    """Wider PT-MAP nosink sweep."""
    keys = list(PTMAP_NOSINK_WIDE.keys())
    combos, total = _sample_grid(PTMAP_NOSINK_WIDE, None)  # 140 configs, run all

    print(f"    Sweeping {len(combos)} PT-MAP nosink wide configs ...")
    results = []
    for combo in tqdm(combos, desc="PTMAP_wide", leave=False):
        params = dict(zip(keys, combo))
        try:
            pred = pt_map(
                sup_feats, sup_labels, test_feats, num_classes=num_classes,
                tukey_beta=params["tukey_beta"],
                n_iter=params["n_iter"],
                lambda_s=params["lambda_s"],
                use_sinkhorn=False,
            )
            f1 = float(f1_score(y_true, pred, average="macro"))
            bacc = float(balanced_accuracy_score(y_true, pred))
            results.append({"params": params, "f1": f1, "bacc": bacc})
        except Exception as e:
            results.append({"params": params, "f1": 0.0, "bacc": 0.0, "error": str(e)})

    results.sort(key=lambda x: -x["f1"])
    return results


# ═══════════════════════════════════════════════════════════════
#  Phase 2: Training config sweep
# ═══════════════════════════════════════════════════════════════

def phase2_training_sweep(train_paths, train_labels, device, base_args,
                           frozen_paths, y_true, verbose=False):
    """Sweep training configurations and evaluate each with best methods."""
    keys = list(TRAINING_GRID.keys())
    values = list(TRAINING_GRID.values())
    total = 1
    for v in values:
        total *= len(v)

    print(f"\n  Sweeping {total} training configs ...")
    print(f"  Grid: {dict(zip(keys, [len(v) for v in values]))}")

    best_overall_f1 = 0.0
    best_config = None
    all_results = []

    for i, combo in enumerate(product(*values)):
        params = dict(zip(keys, combo))
        tag = (f"a{params['lora_alpha']}_e{params['epochs']}"
               f"_sc{params['supcon_weight']}_hed{params['hed_aug']}")
        print(f"\n  [{i+1}/{total}] {tag}")

        args = build_args(
            lora_alpha=params["lora_alpha"],
            seed=base_args.seed,
            lora_r=base_args.lora_r,
            lora_dropout=base_args.lora_dropout,
            ema=base_args.ema,
            epochs=params["epochs"],
            supcon_weight=params["supcon_weight"],
            hed_aug=params["hed_aug"],
        )

        t0 = time.time()
        try:
            model, sup_feats, sup_labels = train_and_extract(
                train_paths, train_labels, device, args,
            )
            frozen_feats, frozen_names, head_probs = predict_test_feats(
                model, frozen_paths, args, device,
            )

            # LaplacianShot with best known params from Phase 1
            pred_lap = laplacianshot(
                sup_feats, sup_labels, frozen_feats, num_classes=5,
                tukey_beta=1.0, use_combined_mean=False,
                knn=7, lam=10.0, sigma=1.0, n_iter=20,
            )
            f1_lap = float(f1_score(y_true, pred_lap, average="macro"))
            bacc_lap = float(balanced_accuracy_score(y_true, pred_lap))

            # TIM(λ_marg=0) with best known params
            pred_tim = tim(
                sup_feats, sup_labels, frozen_feats, num_classes=5,
                tukey_beta=0.7, use_combined_mean=False,
                n_iter=500, lr=1e-4, temperature=15,
                lambda_marg=0.0, lambda_cond=0.05,
            )
            f1_tim = float(f1_score(y_true, pred_tim, average="macro"))
            bacc_tim = float(balanced_accuracy_score(y_true, pred_tim))

            # PT-MAP nosink with best known params
            pred_ptmap = pt_map(
                sup_feats, sup_labels, frozen_feats, num_classes=5,
                tukey_beta=1.0, n_iter=5, lambda_s=5, use_sinkhorn=False,
            )
            f1_ptmap = float(f1_score(y_true, pred_ptmap, average="macro"))

            # Head baseline
            head_pred = head_probs.argmax(axis=1)
            head_f1 = float(f1_score(y_true, head_pred, average="macro"))

            # Best method for this training config
            best_f1 = max(f1_lap, f1_tim, f1_ptmap, head_f1)
            best_method = "laplacianshot" if f1_lap == best_f1 else (
                "tim_nomarg" if f1_tim == best_f1 else (
                    "ptmap_nosink" if f1_ptmap == best_f1 else "head"))

            elapsed = time.time() - t0
            print(f"    LapShot F1={f1_lap:.4f}, TIM F1={f1_tim:.4f}, "
                  f"PTMAP F1={f1_ptmap:.4f}, Head F1={head_f1:.4f}  ({elapsed/60:.1f}min)")
            print(f"    Best: {best_method} F1={best_f1:.4f}")

            result = {
                "params": params,
                "f1": best_f1,
                "best_method": best_method,
                "f1_laplacianshot": f1_lap, "bacc_laplacianshot": bacc_lap,
                "f1_tim": f1_tim, "bacc_tim": bacc_tim,
                "f1_ptmap": f1_ptmap,
                "head_f1": head_f1,
                "time": elapsed, "tag": tag,
            }
            all_results.append(result)

            if best_f1 > best_overall_f1:
                best_overall_f1 = best_f1
                best_config = params
                print(f"    *** NEW BEST: F1={best_f1:.4f} ({best_method}) ***")

        except Exception as e:
            print(f"    ERROR: {e}")
            all_results.append({"params": params, "f1": 0.0, "bacc": 0.0, "error": str(e)})

    all_results.sort(key=lambda x: -x.get("f1", 0))
    return all_results, best_config, best_overall_f1


# ═══════════════════════════════════════════════════════════════
#  Generate 61k submissions
# ═══════════════════════════════════════════════════════════════

def generate_61k_submission(train_paths_all, train_labels_all, device, args,
                              test_dir, method, method_params, tag="v2"):
    """Generate 61k submission with best method and params (single model)."""
    print(f"\n  Training on FULL 250 images ...")
    model, sup_feats, sup_labels = train_and_extract(
        train_paths_all, train_labels_all, device, args,
    )
    test_feats, test_names, head_probs = predict_test_feats(
        model, list_test_samples(test_dir), args, device,
    )

    # Predict
    if method == "tim_nomarg":
        pred = tim(
            sup_feats, sup_labels, test_feats, num_classes=5,
            tukey_beta=method_params.get("tukey_beta", 0.7),
            use_combined_mean=method_params.get("use_combined_mean", False),
            n_iter=method_params.get("n_iter", 500),
            lr=method_params.get("lr", 1e-4),
            temperature=method_params.get("temperature", 15),
            lambda_marg=0.0,
            lambda_cond=method_params.get("lambda_cond", 0.05),
        )
    elif method == "simpleshot":
        pred = simpleshot(
            sup_feats, sup_labels, test_feats, num_classes=5,
            tukey_beta=method_params.get("tukey_beta", 0.7),
            use_combined_mean=method_params.get("use_combined_mean", True),
            temperature=method_params.get("temperature", 5),
        )
    elif method == "ptmap_nosink":
        pred = pt_map(
            sup_feats, sup_labels, test_feats, num_classes=5,
            tukey_beta=method_params.get("tukey_beta", 1.0),
            n_iter=method_params.get("n_iter", 10),
            lambda_s=method_params.get("lambda_s", 10),
            use_sinkhorn=False,
        )
    elif method == "laplacianshot":
        pred = laplacianshot(
            sup_feats, sup_labels, test_feats, num_classes=5,
            tukey_beta=method_params.get("tukey_beta", 0.7),
            use_combined_mean=method_params.get("use_combined_mean", False),
            knn=method_params.get("knn", 5),
            lam=method_params.get("lam", 1.0),
            sigma=method_params.get("sigma", 1.0),
            n_iter=method_params.get("n_iter", 20),
        )
    elif method == "mahalanobis":
        pred = mahalanobis(
            sup_feats, sup_labels, test_feats, num_classes=5,
            tukey_beta=method_params.get("tukey_beta", 1.0),
            use_combined_mean=method_params.get("use_combined_mean", True),
            shrink=method_params.get("shrink", 0.1),
        )
    elif method == "head":
        pred = head_probs.argmax(axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Label distribution
    from collections import Counter
    counts = Counter(pred)
    total = len(pred)

    # Save CSV
    csv_name = f"sub_v2_{tag}_{method}.csv"
    rows = []
    for name, p in zip(test_names, pred):
        rows.append({"filename": name, "label": IDX_TO_CLASS[p]})
    pd.DataFrame(rows).to_csv(f"../{csv_name}", index=False)

    dist_str = ", ".join([f"C{c}={counts.get(c,0)/total*100:.1f}%" for c in range(5)])
    print(f"\n  Saved {csv_name}: n={total}, {dist_str}")

    return csv_name, dict(counts)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Push F1 higher — NO TTA, NO model ensemble")
    ap.add_argument("--train_dir", default="../train_few_shot")
    ap.add_argument("--frozen_dir", default="../frozen_test")
    ap.add_argument("--test_dir", default="../test_shuffled")
    ap.add_argument("--backbone", default="vits14")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no_ema", action="store_false", dest="ema")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phase", type=int, default=0,
                    help="Run specific phase (1-2). 0=all phases")
    ap.add_argument("--generate_subs", action="store_true",
                    help="Generate 61k submission CSVs with best setup")
    ap.add_argument("--out_json", default="../tune_v2_results.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = {}

    # ── Prepare data ──────────────────────────────────────────
    print("=" * 70)
    print("  TUNE V2: Push F1 higher (NO TTA, NO model ensemble)")
    print("=" * 70)
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  EMA={args.ema}")
    print(f"  Phase: {args.phase or 'all'}")
    print(f"  Device: {device}")

    all_paths, all_labels = list_train_samples(args.train_dir)
    frozen_ids = get_frozen_test_ids(args.frozen_dir)
    train_paths, train_labels, excluded = exclude_frozen_from_train(
        all_paths, all_labels, frozen_ids,
    )
    frozen_paths = list_test_samples(args.frozen_dir)

    print(f"\n  train_few_shot: {len(all_paths)} images, excluded {excluded} -> {len(train_paths)} train")
    print(f"  frozen_test: {len(frozen_paths)} images")

    # Load ground truth
    gt_path = Path(args.frozen_dir) / "_groundtruth.csv"
    gt_df = pd.read_csv(gt_path)
    gt_map = dict(zip(gt_df["filename"], gt_df["label"]))

    # ── Train base model ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  TRAINING BASE MODEL (r={args.lora_r}, alpha={args.lora_alpha})")
    print(f"{'=' * 60}")

    base_args = build_args(
        lora_alpha=args.lora_alpha, lora_r=args.lora_r,
        lora_dropout=args.lora_dropout, ema=args.ema, seed=args.seed,
    )
    t0 = time.time()
    model, sup_feats, sup_labels = train_and_extract(
        train_paths, train_labels, device, base_args,
    )
    print(f"  Training time: {timedelta(seconds=int(time.time() - t0))}")

    # Extract features for frozen_test
    print(f"\n  Extracting frozen_test features ...")
    frozen_feats, frozen_names, head_probs = predict_test_feats(
        model, frozen_paths, base_args, device,
    )

    # Match names to ground truth
    y_true = np.array([CLASS_TO_IDX.get(gt_map.get(n, ""), -1) for n in frozen_names])
    valid_mask = y_true >= 0
    y_true = y_true[valid_mask]
    frozen_feats_valid = frozen_feats[valid_mask]
    head_probs_valid = head_probs[valid_mask]

    head_pred = head_probs_valid.argmax(axis=1)
    head_f1 = float(f1_score(y_true, head_pred, average="macro"))
    head_bacc = float(balanced_accuracy_score(y_true, head_pred))
    print(f"  Head baseline: F1={head_f1:.4f}, bacc={head_bacc:.4f}")

    all_results["head"] = {"f1": head_f1, "bacc": head_bacc}

    # ═══════════════════════════════════════════════════════════
    #  PHASE 1: Transductive method improvements (no retraining)
    # ═══════════════════════════════════════════════════════════
    if args.phase == 0 or args.phase == 1:
        print(f"\n{'=' * 70}")
        print(f"  PHASE 1: Transductive method improvements")
        print(f"{'=' * 70}")

        # ── 1a: Wider TIM(λ_marg=0) search ──────────────────
        print(f"\n{'─' * 60}")
        print(f"  1a: Wide TIM(λ_marg=0) search")
        print(f"{'─' * 60}")

        n_wide = 1
        for v in TIM_NOMARG_WIDE.values():
            n_wide *= len(v)
        print(f"  Full grid: {n_wide} configs (sampling 500)")

        t1a = time.time()
        tim_wide_results = sweep_tim_nomarg_wide(
            sup_feats, sup_labels, frozen_feats_valid, y_true,
            max_configs=500,
        )
        tim_wide_time = time.time() - t1a
        best_wide = tim_wide_results[0]
        print(f"  Time: {tim_wide_time:.1f}s")
        print(f"  >>> BEST TIM(λ=0) wide: F1={best_wide['f1']:.4f}, bacc={best_wide['bacc']:.4f}")
        print(f"      params: {best_wide['params']}")
        print(f"  Top 5:")
        for i, r in enumerate(tim_wide_results[:5]):
            print(f"    #{i+1}: F1={r['f1']:.4f}, bacc={r['bacc']:.4f}, params={r['params']}")
        all_results["tim_nomarg_wide"] = {
            "best_f1": best_wide["f1"], "best_bacc": best_wide["bacc"],
            "best_params": best_wide["params"],
            "top5": tim_wide_results[:5],
            "sweep_time": tim_wide_time,
            "total_grid": n_wide,
            "sampled": 500,
        }

        # ── 1b: LaplacianShot sweep ──────────────────────────
        print(f"\n{'─' * 60}")
        print(f"  1b: LaplacianShot sweep")
        print(f"{'─' * 60}")

        n_lap = 1
        for v in LAPLACIANSHOT_GRID.values():
            n_lap *= len(v)
        print(f"  Full grid: {n_lap} configs (sampling 500)")

        t1b = time.time()
        lap_results = sweep_laplacianshot(
            sup_feats, sup_labels, frozen_feats_valid, y_true,
            max_configs=500,
        )
        lap_time = time.time() - t1b
        best_lap = lap_results[0]
        print(f"  Time: {lap_time:.1f}s")
        print(f"  >>> BEST LaplacianShot: F1={best_lap['f1']:.4f}, bacc={best_lap['bacc']:.4f}")
        print(f"      params: {best_lap['params']}")
        print(f"  Top 5:")
        for i, r in enumerate(lap_results[:5]):
            print(f"    #{i+1}: F1={r['f1']:.4f}, bacc={r['bacc']:.4f}, params={r['params']}")
        all_results["laplacianshot"] = {
            "best_f1": best_lap["f1"], "best_bacc": best_lap["bacc"],
            "best_params": best_lap["params"],
            "top5": lap_results[:5],
            "sweep_time": lap_time,
        }

        # ── 1c: PT-MAP nosink wider search ───────────────────
        print(f"\n{'─' * 60}")
        print(f"  1c: PT-MAP nosink wider search")
        print(f"{'─' * 60}")

        t1c = time.time()
        ptmap_wide_results = sweep_ptmap_nosink_wide(
            sup_feats, sup_labels, frozen_feats_valid, y_true,
        )
        ptmap_time = time.time() - t1c
        best_ptmap = ptmap_wide_results[0]
        print(f"  Time: {ptmap_time:.1f}s")
        print(f"  >>> BEST PT-MAP nosink wide: F1={best_ptmap['f1']:.4f}, bacc={best_ptmap['bacc']:.4f}")
        print(f"      params: {best_ptmap['params']}")
        print(f"  Top 5:")
        for i, r in enumerate(ptmap_wide_results[:5]):
            print(f"    #{i+1}: F1={r['f1']:.4f}, bacc={r['bacc']:.4f}, params={r['params']}")
        all_results["ptmap_nosink_wide"] = {
            "best_f1": best_ptmap["f1"], "best_bacc": best_ptmap["bacc"],
            "best_params": best_ptmap["params"],
            "top5": ptmap_wide_results[:5],
            "sweep_time": ptmap_time,
        }

        # Phase 1 summary
        print(f"\n{'=' * 60}")
        print(f"  PHASE 1 SUMMARY")
        print(f"{'=' * 60}")
        print(f"  {'Method':<30} {'F1':>8} {'bacc':>8}")
        print(f"  {'-'*48}")
        print(f"  {'Head baseline':<30} {head_f1:>8.4f} {head_bacc:>8.4f}")
        print(f"  {'TIM(λ=0) wide':<30} {best_wide['f1']:>8.4f} {best_wide['bacc']:>8.4f}")
        print(f"  {'LaplacianShot':<30} {best_lap['f1']:>8.4f} {best_lap['bacc']:>8.4f}")
        print(f"  {'PT-MAP nosink wide':<30} {best_ptmap['f1']:>8.4f} {best_ptmap['bacc']:>8.4f}")

        # Compare with tune_nobalance result
        prev_best = 0.6636  # TIM(λ_marg=0) from tune_nobalance
        if best_wide['f1'] > prev_best:
            print(f"\n  *** Wide TIM search improved: {prev_best:.4f} → {best_wide['f1']:.4f} (+{best_wide['f1']-prev_best:.4f}) ***")
        else:
            print(f"\n  Wide TIM search did NOT improve over tune_nobalance ({prev_best:.4f})")

    # ═══════════════════════════════════════════════════════════
    #  PHASE 2: Training config sweep
    # ═══════════════════════════════════════════════════════════
    if args.phase == 0 or args.phase == 2:
        print(f"\n{'=' * 70}")
        print(f"  PHASE 2: Training config sweep")
        print(f"{'=' * 70}")

        p2_results, p2_best_config, p2_best_f1 = phase2_training_sweep(
            train_paths, train_labels, device, base_args,
            frozen_paths, y_true, verbose=args.verbose,
        )
        print(f"\n  Phase 2 results (top 10):")
        print(f"  {'Config':<30} {'F1':>8} {'Best':>15} {'LapShot':>8} {'TIM':>8} {'PTMAP':>8} {'Head':>8}")
        print(f"  {'-'*85}")
        for r in p2_results[:10]:
            tag = r.get("tag", "?")
            print(f"  {tag:<30} {r.get('f1',0):>8.4f} {r.get('best_method','?'):>15} "
                  f"{r.get('f1_laplacianshot',0):>8.4f} {r.get('f1_tim',0):>8.4f} "
                  f"{r.get('f1_ptmap',0):>8.4f} {r.get('head_f1',0):>8.4f}")

        print(f"\n  >>> BEST training config: F1={p2_best_f1:.4f}")
        print(f"      {p2_best_config}")
        all_results["phase2"] = {
            "best_f1": p2_best_f1,
            "best_config": p2_best_config,
            "all_results": p2_results,
        }

    # ═══════════════════════════════════════════════════════════
    #  Generate 61k submissions
    # ═══════════════════════════════════════════════════════════
    if args.generate_subs:
        print(f"\n{'=' * 70}")
        print(f"  GENERATING 61k SUBMISSIONS")
        print(f"{'=' * 70}")

        # Determine best method and params from results
        # Use Phase 1 best TIM params if available, else tune_nobalance's
        if "tim_nomarg_wide" in all_results:
            best_tim_params = all_results["tim_nomarg_wide"]["best_params"]
        else:
            best_tim_params = {"temperature": 15, "lambda_cond": 0.05,
                               "n_iter": 500, "tukey_beta": 0.7,
                               "use_combined_mean": False, "lr": 1e-4}

        # Use Phase 2 best training config if available
        if "phase2" in all_results and all_results["phase2"].get("best_config"):
            bc = all_results["phase2"]["best_config"]
            sub_args = build_args(
                lora_alpha=bc["lora_alpha"],
                epochs=bc["epochs"],
                supcon_weight=bc["supcon_weight"],
                hed_aug=bc["hed_aug"],
                lora_r=args.lora_r,
                lora_dropout=args.lora_dropout,
                ema=args.ema,
                seed=args.seed,
            )
            tag = f"a{bc['lora_alpha']}_e{bc['epochs']}_sc{bc['supcon_weight']}"
        else:
            sub_args = build_args(
                lora_alpha=args.lora_alpha, lora_r=args.lora_r,
                lora_dropout=args.lora_dropout, ema=args.ema, seed=args.seed,
            )
            tag = "base"

        # Generate submissions for best methods
        methods_to_submit = [
            ("tim_nomarg", best_tim_params),
        ]

        if "laplacianshot" in all_results:
            bp = all_results["laplacianshot"]["best_params"]
            methods_to_submit.append(("laplacianshot", bp))

        if "ptmap_nosink_wide" in all_results:
            bp = all_results["ptmap_nosink_wide"]["best_params"]
            methods_to_submit.append(("ptmap_nosink", bp))

        for method_name, m_params in methods_to_submit:
            try:
                csv_name, dist = generate_61k_submission(
                    all_paths, all_labels, device, sub_args,
                    args.test_dir, method_name, m_params, tag=tag,
                )
            except Exception as e:
                print(f"  ERROR generating {method_name}: {e}")

    # ── Save results ──────────────────────────────────────────
    with open(args.out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[ok] Results saved -> {args.out_json}")

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 70}")

    summary = []
    for key in all_results:
        r = all_results[key]
        if isinstance(r, dict):
            f1 = r.get("best_f1", r.get("f1", None))
            bacc = r.get("best_bacc", r.get("bacc", None))
            if f1 is not None:
                summary.append((key, f1, bacc))

    summary.sort(key=lambda x: -x[1])
    print(f"  {'Method':<35} {'F1':>8} {'bacc':>8}")
    print(f"  {'-'*53}")
    for key, f1, bacc in summary:
        if bacc is not None:
            print(f"  {key:<35} {f1:>8.4f} {bacc:>8.4f}")
        else:
            print(f"  {key:<35} {f1:>8.4f}")

    if summary:
        best_key, best_f1, _ = summary[0]
        print(f"\n  >>> OVERALL BEST: {best_key} F1={best_f1:.4f}")
        prev_best = 0.6636
        if best_f1 > prev_best:
            print(f"  Improved over tune_nobalance: {prev_best:.4f} → {best_f1:.4f} (+{best_f1-prev_best:.4f})")
        else:
            print(f"  Did NOT improve over tune_nobalance ({prev_best:.4f})")

    print(f"\n  NEXT STEPS:")
    print(f"  1. If Phase 1 found better TIM params → generate 61k subs with --generate_subs")
    print(f"  2. If Phase 2 found better training config → re-run Phase 1 with that config")
    print(f"  3. Consider self-training (self_train_61k.py) as next step")


if __name__ == "__main__":
    main()
