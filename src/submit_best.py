"""Best config → train on 250 → predict 61k with 4 methods → compare distributions.

Uses the optimal training config from Phase 2 analysis:
  - alpha=16, epochs=80, HED_aug=True, supcon=0.1, dropout=0.1, EMA=True

Generates 4 submission CSVs for comparison:
  1. sub_best_head.csv
  2. sub_best_tim_nomarg.csv
  3. sub_best_ptmap_nosink.csv
  4. sub_best_laplacianshot.csv

Usage:
  cd src
  python submit_best.py --verbose
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from datetime import timedelta
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
    simpleshot, laplacianshot, mahalanobis, tim, pt_map,
)


# ═══════════════════════════════════════════════════════════════
#  Best training config (from Phase 2 analysis)
# ═══════════════════════════════════════════════════════════════
BEST_TRAIN_CONFIG = {
    "backbone": "vits14",
    "image_size": 224,
    "epochs": 80,
    "batch_size": 32,
    "lr": 5e-4,
    "lr_head": 1e-3,
    "weight_decay": 0.05,
    "label_smoothing": 0.1,
    "grad_clip": 1.0,
    "lora_r": 16,
    "lora_alpha": 16,        # ← alpha=16 > alpha=32
    "lora_dropout": 0.1,
    "head_dropout": 0.1,
    "ema": True,
    "ema_decay": 0.95,
    "hed_aug": True,          # ← HED augmentation helps
    "hed_sigma": 0.02,
    "hed_bias": 0.01,
    "lora_mlp": False,
    "dora": False,
    "lora_init": "default",
    "supcon_weight": 0.1,     # ← mild SupCon
    "supcon_temp": 0.07,
    "mixup": 0.0,
    "cutmix": 0.0,
    "num_workers": 2,
    "seed": 42,
    "verbose": False,
}


# ═══════════════════════════════════════════════════════════════
#  Best transductive params (from tune_nobalance + tune_v2)
# ═══════════════════════════════════════════════════════════════
BEST_TIM_PARAMS = {
    "tukey_beta": 0.7,
    "use_combined_mean": False,
    "n_iter": 500,
    "lr": 1e-4,
    "temperature": 15,
    "lambda_marg": 0.0,       # ← NO marginal entropy (imbalanced test!)
    "lambda_cond": 0.05,
}

BEST_PTMAP_PARAMS = {
    "tukey_beta": 1.0,
    "n_iter": 5,
    "lambda_s": 5,
    "use_sinkhorn": False,    # ← NO Sinkhorn (imbalanced test!)
}

BEST_LAP_PARAMS = {
    "tukey_beta": 1.0,
    "use_combined_mean": False,
    "knn": 7,
    "lam": 10.0,
    "n_iter": 20,
    "sigma": 1.0,
}


def build_args():
    """Build args namespace from best config."""
    return argparse.Namespace(**BEST_TRAIN_CONFIG)


def train_and_extract(train_paths, train_labels, device, args):
    """Train LoRA, extract features. Returns model + features."""
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


def predict_test_set(model, test_paths, args, device):
    """Extract features + head probs for test images."""
    test_feats, test_names = extract_features(
        model, test_paths, args.image_size, args, device, labels=None,
    )
    head_probs, _ = predict_probs(
        model, test_paths, args.image_size, args, device,
    )
    return test_feats, test_names, head_probs


def main():
    ap = argparse.ArgumentParser(
        description="Best config: train 250 → predict 61k → 4 submission CSVs")
    ap.add_argument("--train_dir", default="../train_few_shot")
    ap.add_argument("--test_dir", default="../test_shuffled")
    ap.add_argument("--frozen_dir", default="../frozen_test")
    ap.add_argument("--out_dir", default="..", help="directory for submission CSVs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  BEST CONFIG SUBMISSION")
    print("  Train: 250 images (full, no leakage on test_shuffled)")
    print("  Config: alpha=16, epochs=80, HED=True, SupCon=0.1, dropout=0.1, EMA")
    print("  Methods: Head + TIM(λ=0) + PT-MAP nosink + LaplacianShot")
    print(f"  Device: {device}")
    print("=" * 70)

    # ── Step 1: Load training data ──
    train_paths, train_labels = list_train_samples(args.train_dir)
    print(f"\n  Training on {len(train_paths)} images")
    print(f"  Per-class: {np.bincount(train_labels).tolist()}")

    # ── Step 2: Train ──
    train_args = build_args()
    print(f"\n  Training config:")
    print(f"    lora_alpha={train_args.lora_alpha}, epochs={train_args.epochs}, "
          f"dropout={train_args.lora_dropout}")
    print(f"    HED_aug={train_args.hed_aug}, SupCon={train_args.supcon_weight}, "
          f"EMA={train_args.ema}")

    t0 = time.time()
    model, sup_feats, sup_labels, n_params = train_and_extract(
        train_paths, train_labels, device, train_args,
    )
    train_time = time.time() - t0
    print(f"  Trainable params: {n_params:,}")
    print(f"  Training time: {timedelta(seconds=int(train_time))}")

    # ── Step 3: Evaluate on frozen_test (leaky but useful reference) ──
    gt_path = Path(args.frozen_dir) / "_groundtruth.csv"
    if gt_path.exists():
        print(f"\n{'=' * 70}")
        print(f"  EVALUATION on frozen_test (LEAKY — trained on all 250)")
        print(f"{'=' * 70}")

        frozen_paths = list_test_samples(args.frozen_dir)
        frozen_feats, frozen_names = extract_features(
            model, frozen_paths, train_args.image_size, train_args, device,
            labels=None,
        )
        head_probs_frozen, _ = predict_probs(
            model, frozen_paths, train_args.image_size, train_args, device,
        )

        # Run 4 methods on frozen_test
        pred_tim = tim(sup_feats, sup_labels, frozen_feats, num_classes=5,
                       **BEST_TIM_PARAMS)
        pred_ptmap = pt_map(sup_feats, sup_labels, frozen_feats, num_classes=5,
                            tukey_beta=BEST_PTMAP_PARAMS["tukey_beta"],
                            n_iter=BEST_PTMAP_PARAMS["n_iter"],
                            lambda_s=BEST_PTMAP_PARAMS["lambda_s"],
                            use_sinkhorn=False)
        pred_lap = laplacianshot(sup_feats, sup_labels, frozen_feats, num_classes=5,
                                 **BEST_LAP_PARAMS)
        pred_head = head_probs_frozen.argmax(axis=1)

        # Load ground truth
        gt_df = pd.read_csv(gt_path)
        gt_map = dict(zip(gt_df["filename"], gt_df["label"]))
        y_true = np.array([CLASS_TO_IDX.get(gt_map.get(n, ""), -1)
                           for n in frozen_names])
        valid_mask = y_true >= 0
        if valid_mask.sum() > 0:
            y_true = y_true[valid_mask]
            print(f"  Ground truth: {valid_mask.sum()}/{len(frozen_names)} images")
            print(f"\n  {'Method':<20} {'F1':>8} {'BalAcc':>8}")
            print(f"  {'-'*38}")
            for name, pred in [("head", pred_head), ("tim_nomarg", pred_tim),
                               ("ptmap_nosink", pred_ptmap), ("laplacianshot", pred_lap)]:
                pv = pred[valid_mask] if len(pred) == valid_mask.sum() or len(pred) == len(frozen_names) else pred
                if len(pv) != len(y_true):
                    pv = pred[valid_mask]
                f1 = float(f1_score(y_true, pv, average="macro"))
                bacc = float(balanced_accuracy_score(y_true, pv))
                print(f"  {name:<20} {f1:>8.4f} {bacc:>8.4f}")

    # ── Step 4: Predict on test_shuffled (61k) ──
    print(f"\n{'=' * 70}")
    print(f"  PREDICTING on test_shuffled (61,881 images)")
    print(f"{'=' * 70}")

    test_paths = list_test_samples(args.test_dir)
    print(f"  test_shuffled: {len(test_paths)} images")

    t1 = time.time()
    test_feats, test_names, head_probs = predict_test_set(
        model, test_paths, train_args, device,
    )
    extract_time = time.time() - t1
    print(f"  Feature extraction: {timedelta(seconds=int(extract_time))}")

    # ── Run 4 methods ──
    out_dir = Path(args.out_dir)
    methods = {}

    # 1. Head
    print(f"\n  [1/4] Head ...")
    pred_head_61k = head_probs.argmax(axis=1)
    methods["head"] = pred_head_61k

    # 2. TIM(λ_marg=0)
    print(f"  [2/4] TIM(λ_marg=0) ...")
    t2 = time.time()
    pred_tim_61k = tim(sup_feats, sup_labels, test_feats, num_classes=5,
                        **BEST_TIM_PARAMS)
    print(f"    TIM: {time.time()-t2:.1f}s")
    methods["tim_nomarg"] = pred_tim_61k

    # 3. PT-MAP nosink
    print(f"  [3/4] PT-MAP nosink ...")
    t3 = time.time()
    pred_ptmap_61k = pt_map(sup_feats, sup_labels, test_feats, num_classes=5,
                             tukey_beta=BEST_PTMAP_PARAMS["tukey_beta"],
                             n_iter=BEST_PTMAP_PARAMS["n_iter"],
                             lambda_s=BEST_PTMAP_PARAMS["lambda_s"],
                             use_sinkhorn=False)
    print(f"    PT-MAP: {time.time()-t3:.1f}s")
    methods["ptmap_nosink"] = pred_ptmap_61k

    # 4. LaplacianShot (sparse for 61k)
    print(f"  [4/4] LaplacianShot (sparse knn for 61k) ...")
    t4 = time.time()
    pred_lap_61k = laplacianshot(sup_feats, sup_labels, test_feats, num_classes=5,
                                  **BEST_LAP_PARAMS)
    print(f"    LaplacianShot: {time.time()-t4:.1f}s")
    methods["laplacianshot"] = pred_lap_61k

    # ── Print distribution comparison ──
    print(f"\n{'=' * 70}")
    print(f"  LABEL DISTRIBUTION COMPARISON (61k test_shuffled)")
    print(f"{'=' * 70}")
    total = len(test_names)
    header = f"  {'Method':<20} "
    for c in range(5):
        header += f"{'Cl_'+str(c):>8}"
    header += f"  {'Total':>8}"
    print(header)
    print(f"  {'-'*72}")

    for name, pred in methods.items():
        counts = np.bincount(pred, minlength=5)
        row = f"  {name:<20} "
        for c in range(5):
            row += f"{counts[c]:>8}"
        row += f"  {total:>8}"
        print(row)
        # Also print percentages
        row_pct = f"  {'':<20} "
        for c in range(5):
            row_pct += f"{counts[c]/total*100:>7.1f}%"
        row_pct += f"  {'100%':>8}"
        print(row_pct)

    # ── Pairwise agreement ──
    print(f"\n  PAIRWISE AGREEMENT between methods:")
    method_names = list(methods.keys())
    for i in range(len(method_names)):
        for j in range(i+1, len(method_names)):
            n1, n2 = method_names[i], method_names[j]
            agree = (methods[n1] == methods[n2]).sum()
            print(f"    {n1} vs {n2}: {agree}/{total} = {agree/total*100:.1f}%")

    # ── Save 4 submission CSVs ──
    print(f"\n{'=' * 70}")
    print(f"  SAVING SUBMISSION CSVs")
    print(f"{'=' * 70}")

    for name, pred in methods.items():
        labels = [IDX_TO_CLASS[int(i)] for i in pred]
        df = pd.DataFrame({"filename": test_names, "label": labels})
        path = out_dir / f"sub_best_{name}.csv"
        df.to_csv(str(path), index=False)
        print(f"  [ok] {name} -> {path}")

    # ── Save checkpoint ──
    ckpt_dir = Path("../artifacts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_a16_e80_hed_sc01.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "args": vars(train_args),
        "train_size": len(train_paths),
        "config": "alpha=16, epochs=80, HED=True, SupCon=0.1, dropout=0.1, EMA",
    }, ckpt_path)
    print(f"\n  [ok] checkpoint -> {ckpt_path}")

    # ── Save log ──
    log = {
        "config": "alpha=16, epochs=80, HED=True, SupCon=0.1, dropout=0.1, EMA",
        "train_size": len(train_paths),
        "train_time_sec": round(train_time, 1),
        "test_size": len(test_paths),
        "methods": {},
    }
    for name, pred in methods.items():
        counts = np.bincount(pred, minlength=5)
        log["methods"][name] = {
            "distribution": counts.tolist(),
            "distribution_pct": [round(c/total*100, 1) for c in counts],
        }
    if gt_path.exists() and valid_mask.sum() > 0:
        for name, pred in [("head", pred_head), ("tim_nomarg", pred_tim),
                           ("ptmap_nosink", pred_ptmap), ("laplacianshot", pred_lap)]:
            pv = pred[valid_mask] if len(pred) == len(frozen_names) else pred
            f1 = float(f1_score(y_true, pv, average="macro"))
            bacc = float(balanced_accuracy_score(y_true, pv))
            log["methods"][name]["frozen_f1"] = round(f1, 4)
            log["methods"][name]["frozen_bacc"] = round(bacc, 4)

    log_path = out_dir / "submit_best_results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  [ok] log -> {log_path}")

    total_time = time.time() - t0
    print(f"\n  Total time: {timedelta(seconds=int(total_time))}")
    print(f"\n  ✅ DONE — 4 submission CSVs ready for comparison")


if __name__ == "__main__":
    main()
