"""TIM hyperparameter search on cv5 (r=16, lora_dropout=0.2).

CRITICAL INSIGHT: TIM hyperparameters have NEVER been tuned — they're defaults
from the original paper. With 61k test images, TIM's marginal entropy estimation
becomes much more reliable, so the optimal hyperparameters may differ.

This script:
  1. Trains LoRA r=16 once per fold (5 folds)
  2. Extracts features once per fold
  3. Sweeps TIM hyperparameters on those features (FAST — no re-training)

Search space:
  temperature    ∈ {5, 10, 15, 20, 30}
  lambda_marg    ∈ {0.5, 1.0, 2.0, 5.0}
  lambda_cond    ∈ {0.05, 0.1, 0.5}
  n_iter         ∈ {500, 1000}

Total: 5 × 4 × 3 × 2 = 120 TIM configs per fold
BUT: feature extraction is the bottleneck (1-2 min/fold), TIM runs are seconds.

Also tests PT-MAP with/without Sinkhorn (to verify it works on large sets).

Usage:
  cd src
  python tim_search.py --ema --tukey_beta 0.5 --lora_dropout 0.2 --verbose

Resume:
  Re-run same command. Uses checkpoint to skip completed folds.
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
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score

from data import list_train_samples, list_test_samples, CLASS_NAMES
from train_lora import train_one_run, seed_all
from self_train import extract_features, predict_probs
from transductive import (
    simpleshot, mahalanobis, tim, pt_map, alpha_tim,
)


# ── TIM search grid ────────────────────────────────────────────

TIM_GRID = {
    "temperature": [5, 10, 15, 20, 30],
    "lambda_marg": [0.5, 1.0, 2.0, 5.0],
    "lambda_cond": [0.05, 0.1, 0.5],
    "n_iter":      [500, 1000],
}

# Fixed params
TIM_FIXED = {"lr": 1e-4, "tukey_beta": 0.5, "use_combined_mean": True}


# ── Also test alpha-TIM and PT-MAP variants ────────────────────

ALPHA_TIM_GRID = {
    "alpha":        [1.5, 2.0, 3.0],
    "temperature":  [10, 15, 20],
    "lambda_marg":  [0.5, 1.0, 2.0],
    "lambda_cond":  [0.05, 0.1],
    "n_iter":       [1000],
}

PTMAP_VARIANTS = [
    {"use_sinkhorn": True,  "lambda_s": 10.0, "n_iter": 20, "label": "ptmap_sink"},
    {"use_sinkhorn": True,  "lambda_s": 20.0, "n_iter": 20, "label": "ptmap_sink_l20"},
    {"use_sinkhorn": False, "lambda_s": 10.0, "n_iter": 20, "label": "ptmap_nosink"},
    {"use_sinkhorn": False, "lambda_s": 20.0, "n_iter": 20, "label": "ptmap_nosink_l20"},
    {"use_sinkhorn": False, "lambda_s": 10.0, "n_iter": 40, "label": "ptmap_nosink_40iter"},
]


def build_args(backbone="vits14", ema=True, lora_r=16, lora_alpha=32,
               lora_dropout=0.2, seed=42):
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


def train_and_extract(train_paths, train_labels, holdout_paths, device, args):
    """Train LoRA, extract features for both splits. Returns features + head probs."""
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
    test_feats, test_names = extract_features(
        model, holdout_paths, args.image_size, args, device,
        labels=None,
    )
    head_probs, _ = predict_probs(
        model, holdout_paths, args.image_size, args, device,
    )

    del model
    torch.cuda.empty_cache()

    return sup_feats, sup_labels, test_feats, head_probs, test_names, n_params


def sweep_tim(sup_feats, sup_labels, test_feats, y_true, num_classes=5,
              tukey_beta=0.5, use_combined_mean=True):
    """Sweep TIM hyperparameters. Returns list of (params, f1, bacc)."""
    results = []

    # Generate all combos
    keys = list(TIM_GRID.keys())
    values = list(TIM_GRID.values())
    total = 1
    for v in values:
        total *= len(v)
    print(f"    Sweeping {total} TIM configurations ...")

    for combo in product(*values):
        params = dict(zip(keys, combo))
        try:
            pred = tim(
                sup_feats, sup_labels, test_feats, num_classes=num_classes,
                tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
                n_iter=params["n_iter"], lr=TIM_FIXED["lr"],
                temperature=params["temperature"],
                lambda_marg=params["lambda_marg"],
                lambda_cond=params["lambda_cond"],
            )
            f1 = float(f1_score(y_true, pred, average="macro"))
            bacc = float(balanced_accuracy_score(y_true, pred))
            results.append({"params": params, "f1": f1, "bacc": bacc})
        except Exception as e:
            results.append({"params": params, "f1": 0.0, "bacc": 0.0, "error": str(e)})

    # Sort by F1
    results.sort(key=lambda x: -x["f1"])
    return results


def sweep_alpha_tim(sup_feats, sup_labels, test_feats, y_true, num_classes=5,
                    tukey_beta=0.5, use_combined_mean=True):
    """Sweep alpha-TIM hyperparameters. Returns list of (params, f1, bacc)."""
    results = []

    keys = list(ALPHA_TIM_GRID.keys())
    values = list(ALPHA_TIM_GRID.values())
    total = 1
    for v in values:
        total *= len(v)
    print(f"    Sweeping {total} alpha-TIM configurations ...")

    for combo in product(*values):
        params = dict(zip(keys, combo))
        try:
            pred = alpha_tim(
                sup_feats, sup_labels, test_feats, num_classes=num_classes,
                tukey_beta=tukey_beta, use_combined_mean=True,
                n_iter=params["n_iter"], lr=1e-4,
                temperature=params["temperature"],
                alpha=params["alpha"],
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


def sweep_ptmap(sup_feats, sup_labels, test_feats, y_true, num_classes=5,
                tukey_beta=0.5):
    """Test PT-MAP variants. Returns list of (params, f1, bacc)."""
    results = []

    for variant in PTMAP_VARIANTS:
        label = variant["label"]
        try:
            pred = pt_map(
                sup_feats, sup_labels, test_feats, num_classes=num_classes,
                tukey_beta=tukey_beta,
                n_iter=variant["n_iter"],
                lambda_s=variant["lambda_s"],
                use_sinkhorn=variant["use_sinkhorn"],
                sinkhorn_iter=10,
            )
            f1 = float(f1_score(y_true, pred, average="macro"))
            bacc = float(balanced_accuracy_score(y_true, pred))
            results.append({"label": label, "params": variant, "f1": f1, "bacc": bacc})
        except Exception as e:
            results.append({"label": label, "params": variant, "f1": 0.0,
                            "bacc": 0.0, "error": str(e)})

    results.sort(key=lambda x: -x["f1"])
    return results


# ── Main ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="TIM/alpha-TIM/PT-MAP hyperparameter search on cv5")
    ap.add_argument("--train_dir", default="../train_few_shot")
    ap.add_argument("--backbone", default="vits14")
    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no_ema", action="store_false", dest="ema")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.2,
                    help="LoRA dropout (0.2 was best from reg_search)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tukey_beta", type=float, default=0.5)
    ap.add_argument("--maha_shrink", type=float, default=0.3)
    ap.add_argument("--use_combined_mean", action="store_true", default=True)
    ap.add_argument("--skip_tim", action="store_true",
                    help="skip TIM sweep (faster, only alpha-TIM + PT-MAP)")
    ap.add_argument("--skip_alpha_tim", action="store_true",
                    help="skip alpha-TIM sweep")
    ap.add_argument("--skip_ptmap", action="store_true",
                    help="skip PT-MAP variants")
    ap.add_argument("--out_json", default="../tim_search_results.json")
    ap.add_argument("--force", action="store_true",
                    help="ignore checkpoint, start from scratch")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Count TIM configs
    n_tim = 1
    for v in TIM_GRID.values():
        n_tim *= len(v)
    n_atim = 1
    for v in ALPHA_TIM_GRID.values():
        n_atim *= len(v)

    print(f"[info] TIM grid: {n_tim} configs")
    print(f"[info] alpha-TIM grid: {n_atim} configs")
    print(f"[info] PT-MAP variants: {len(PTMAP_VARIANTS)}")
    print(f"[info] LoRA r={args.lora_r}, dropout={args.lora_dropout}")
    print(f"[info] {args.folds}-fold CV")

    # Load all 250 images
    all_paths, all_labels = list_train_samples(args.train_dir)
    all_labels = np.array(all_labels)
    N = len(all_paths)
    print(f"[info] {N} samples, device={device}")
    print(f"[info] per-class: {np.bincount(all_labels).tolist()}")

    # ── Checkpoint ──
    ckpt_path = Path(args.out_json).with_suffix(".checkpoint.json")
    if ckpt_path.exists() and not args.force:
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        completed_folds = ckpt.get("completed_folds", [])
        all_fold_results = ckpt.get("fold_results", [])
        print(f"[resume] {len(completed_folds)} folds already done, continuing")
    else:
        completed_folds = []
        all_fold_results = []
        ckpt = {}

    # ── Run 5-fold CV ──
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    for fold_idx, (train_idx, holdout_idx) in enumerate(
            skf.split(np.arange(N), all_labels)):

        fold_key = f"fold_{fold_idx}"
        if fold_key in completed_folds:
            print(f"\n[resume] skipping fold {fold_idx+1} (already done)")
            continue

        t0 = time.time()
        print(f"\n{'='*70}")
        print(f"  FOLD {fold_idx+1}/{args.folds}: "
              f"{len(train_idx)} train / {len(holdout_idx)} holdout")
        print(f"{'='*70}")

        train_args = build_args(
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, ema=args.ema, seed=args.seed,
        )

        with tempfile.TemporaryDirectory(prefix=f"tim_search_f{fold_idx}_") as tmpdir:
            tmpdir = Path(tmpdir)

            # Build train dir
            fold_train_dir = tmpdir / "train"
            for c in CLASS_NAMES:
                (fold_train_dir / c).mkdir(parents=True, exist_ok=True)
            train_paths_fold = [all_paths[i] for i in train_idx]
            train_labels_fold = all_labels[train_idx]
            for p, l in zip(train_paths_fold, train_labels_fold):
                cls_dir = fold_train_dir / CLASS_NAMES[l]
                shutil.copy2(p, cls_dir / Path(p).name)

            # Build holdout dir
            fold_holdout_dir = tmpdir / "holdout"
            fold_holdout_dir.mkdir(parents=True, exist_ok=True)
            holdout_paths_fold = [all_paths[i] for i in holdout_idx]
            for p in holdout_paths_fold:
                shutil.copy2(p, fold_holdout_dir / Path(p).name)

            # Ground truth
            gt_map = {}
            for i in holdout_idx:
                gt_map[Path(all_paths[i]).name] = int(all_labels[i])

            # Load split data
            st_paths, st_labels = list_train_samples(str(fold_train_dir))
            holdout_paths = list_test_samples(str(fold_holdout_dir))

            # ── Step 1: Train LoRA + Extract features (only once per fold) ──
            print(f"  Training LoRA r={args.lora_r}, dropout={args.lora_dropout} ...")
            sup_feats, sup_labels_arr, test_feats, head_probs, test_names, n_params = \
                train_and_extract(st_paths, st_labels, holdout_paths, device, train_args)

            print(f"  Features: support={sup_feats.shape}, query={test_feats.shape}")

            # True labels for holdout
            y_true = np.array([gt_map[n] for n in test_names])

            # Head baseline
            head_pred = head_probs.argmax(axis=1)
            head_f1 = float(f1_score(y_true, head_pred, average="macro"))
            head_bacc = float(balanced_accuracy_score(y_true, head_pred))
            print(f"  Head baseline: F1={head_f1:.4f}, bacc={head_bacc:.4f}")

            # SimpleShot + Mahalanobis baselines
            pred_ss = simpleshot(
                sup_feats, sup_labels_arr, test_feats, num_classes=5,
                tukey_beta=args.tukey_beta, use_combined_mean=args.use_combined_mean,
            )
            ss_f1 = float(f1_score(y_true, pred_ss, average="macro"))

            pred_maha = mahalanobis(
                sup_feats, sup_labels_arr, test_feats, num_classes=5,
                tukey_beta=args.tukey_beta, use_combined_mean=args.use_combined_mean,
                shrink=args.maha_shrink,
            )
            maha_f1 = float(f1_score(y_true, pred_maha, average="macro"))

            print(f"  SimpleShot: F1={ss_f1:.4f}, Mahalanobis: F1={maha_f1:.4f}")

            # ── Step 2: Sweep TIM ──
            fold_result = {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "holdout_size": len(holdout_idx),
                "head_f1": head_f1,
                "ss_f1": ss_f1,
                "maha_f1": maha_f1,
            }

            if not args.skip_tim:
                print(f"\n  [TIM Sweep] {n_tim} configs ...")
                t1 = time.time()
                tim_results = sweep_tim(
                    sup_feats, sup_labels_arr, test_feats, y_true,
                    num_classes=5, tukey_beta=args.tukey_beta,
                    use_combined_mean=args.use_combined_mean,
                )
                tim_time = time.time() - t1
                print(f"  TIM sweep: {tim_time:.1f}s")
                # Top 5
                print(f"  Top 5 TIM configs:")
                for i, r in enumerate(tim_results[:5]):
                    p = r["params"]
                    print(f"    #{i+1}: temp={p['temperature']}, "
                          f"λ_marg={p['lambda_marg']}, λ_cond={p['lambda_cond']}, "
                          f"n_iter={p['n_iter']} → F1={r['f1']:.4f}")

                fold_result["tim_sweep"] = tim_results
                fold_result["tim_best"] = tim_results[0]
                fold_result["tim_sweep_time"] = tim_time

            # ── Step 3: Sweep alpha-TIM ──
            if not args.skip_alpha_tim:
                print(f"\n  [alpha-TIM Sweep] {n_atim} configs ...")
                t2 = time.time()
                atim_results = sweep_alpha_tim(
                    sup_feats, sup_labels_arr, test_feats, y_true,
                    num_classes=5, tukey_beta=args.tukey_beta,
                    use_combined_mean=args.use_combined_mean,
                )
                atim_time = time.time() - t2
                print(f"  alpha-TIM sweep: {atim_time:.1f}s")
                print(f"  Top 5 alpha-TIM configs:")
                for i, r in enumerate(atim_results[:5]):
                    p = r["params"]
                    print(f"    #{i+1}: α={p['alpha']}, temp={p['temperature']}, "
                          f"λ_marg={p['lambda_marg']}, λ_cond={p['lambda_cond']} "
                          f"→ F1={r['f1']:.4f}")

                fold_result["atim_sweep"] = atim_results
                fold_result["atim_best"] = atim_results[0]
                fold_result["atim_sweep_time"] = atim_time

            # ── Step 4: PT-MAP variants ──
            if not args.skip_ptmap:
                print(f"\n  [PT-MAP Variants] {len(PTMAP_VARIANTS)} configs ...")
                ptmap_results = sweep_ptmap(
                    sup_feats, sup_labels_arr, test_feats, y_true,
                    num_classes=5, tukey_beta=args.tukey_beta,
                )
                print(f"  PT-MAP results:")
                for r in ptmap_results:
                    marker = " ← best" if r == ptmap_results[0] else ""
                    print(f"    {r['label']:<30}: F1={r['f1']:.4f}{marker}")

                fold_result["ptmap_variants"] = ptmap_results
                fold_result["ptmap_best"] = ptmap_results[0]

        elapsed = time.time() - t0
        fold_result["elapsed_sec"] = round(elapsed, 1)

        # Find overall best for this fold
        best_f1 = head_f1
        best_method = "head"
        best_params = {}

        if "tim_best" in fold_result and fold_result["tim_best"]["f1"] > best_f1:
            best_f1 = fold_result["tim_best"]["f1"]
            best_method = "tim"
            best_params = fold_result["tim_best"]["params"]
        if "atim_best" in fold_result and fold_result["atim_best"]["f1"] > best_f1:
            best_f1 = fold_result["atim_best"]["f1"]
            best_method = "alpha_tim"
            best_params = fold_result["atim_best"]["params"]
        if "ptmap_best" in fold_result and fold_result["ptmap_best"]["f1"] > best_f1:
            best_f1 = fold_result["ptmap_best"]["f1"]
            best_method = fold_result["ptmap_best"]["label"]
            best_params = fold_result["ptmap_best"]["params"]
        if ss_f1 > best_f1:
            best_f1 = ss_f1
            best_method = "simpleshot"
            best_params = {}
        if maha_f1 > best_f1:
            best_f1 = maha_f1
            best_method = "mahalanobis"
            best_params = {}

        fold_result["best_method"] = best_method
        fold_result["best_f1"] = best_f1
        fold_result["best_params"] = best_params

        print(f"\n  >>> Fold {fold_idx+1} best: {best_method} "
              f"F1={best_f1:.4f} ({timedelta(seconds=int(elapsed))})")

        # Save checkpoint
        all_fold_results.append(fold_result)
        completed_folds.append(fold_key)
        ckpt["completed_folds"] = completed_folds
        ckpt["fold_results"] = all_fold_results
        with open(ckpt_path, "w") as f:
            json.dump(ckpt, f, indent=2)

    # ── Aggregate across folds ──
    print(f"\n{'='*75}")
    print(f"  TIM HYPERPARAMETER SEARCH RESULTS")
    print(f"  {args.folds}-fold CV, LoRA r={args.lora_r}, dropout={args.lora_dropout}")
    print(f"{'='*75}")

    # ── Find best TIM params across folds ──
    if not args.skip_tim:
        # Collect all TIM results per fold
        # For each TIM config, compute average F1 across folds
        tim_avg = {}
        for fold_res in all_fold_results:
            if "tim_sweep" not in fold_res:
                continue
            for r in fold_res["tim_sweep"]:
                key = str(sorted(r["params"].items()))
                if key not in tim_avg:
                    tim_avg[key] = {"params": r["params"], "f1s": [], "baccs": []}
                tim_avg[key]["f1s"].append(r["f1"])
                tim_avg[key]["baccs"].append(r["bacc"])

        # Sort by mean F1
        tim_ranked = []
        for key, v in tim_avg.items():
            if len(v["f1s"]) == args.folds:
                tim_ranked.append({
                    "params": v["params"],
                    "mean_f1": float(np.mean(v["f1s"])),
                    "std_f1": float(np.std(v["f1s"])),
                    "mean_bacc": float(np.mean(v["baccs"])),
                })
        tim_ranked.sort(key=lambda x: -x["mean_f1"])

        print(f"\n  TOP 20 TIM configurations (averaged across {args.folds} folds):")
        print(f"  {'#':>3}  {'temp':>4}  {'λ_marg':>6}  {'λ_cond':>6}  "
              f"{'n_iter':>6}  {'mean F1':>8}  {'±std':>6}  {'bacc':>8}")
        print(f"  {'-'*60}")
        for i, r in enumerate(tim_ranked[:20]):
            p = r["params"]
            print(f"  {i+1:>3}  {p['temperature']:>4}  {p['lambda_marg']:>6}  "
                  f"{p['lambda_cond']:>6}  {p['n_iter']:>6}  "
                  f"{r['mean_f1']:>8.4f}  {r['std_f1']:>6.4f}  "
                  f"{r['mean_bacc']:>8.4f}")

        # Compare with default TIM (temp=15, λ_marg=1.0, λ_cond=0.1, n_iter=1000)
        default_key = str(sorted({"temperature": 15, "lambda_marg": 1.0,
                                  "lambda_cond": 0.1, "n_iter": 1000}.items()))
        default_f1 = tim_avg.get(default_key, {}).get("f1s", [])
        if default_f1:
            default_mean = np.mean(default_f1)
            best_tim_mean = tim_ranked[0]["mean_f1"]
            diff = best_tim_mean - default_mean
            print(f"\n  Default TIM F1 = {default_mean:.4f}")
            print(f"  Best TIM F1    = {best_tim_mean:.4f}  ({diff:+.4f})")

    # ── Find best alpha-TIM params across folds ──
    if not args.skip_alpha_tim:
        atim_avg = {}
        for fold_res in all_fold_results:
            if "atim_sweep" not in fold_res:
                continue
            for r in fold_res["atim_sweep"]:
                key = str(sorted(r["params"].items()))
                if key not in atim_avg:
                    atim_avg[key] = {"params": r["params"], "f1s": [], "baccs": []}
                atim_avg[key]["f1s"].append(r["f1"])
                atim_avg[key]["baccs"].append(r["bacc"])

        atim_ranked = []
        for key, v in atim_avg.items():
            if len(v["f1s"]) == args.folds:
                atim_ranked.append({
                    "params": v["params"],
                    "mean_f1": float(np.mean(v["f1s"])),
                    "std_f1": float(np.std(v["f1s"])),
                    "mean_bacc": float(np.mean(v["baccs"])),
                })
        atim_ranked.sort(key=lambda x: -x["mean_f1"])

        print(f"\n  TOP 10 alpha-TIM configurations:")
        print(f"  {'#':>3}  {'α':>4}  {'temp':>4}  {'λ_marg':>6}  {'λ_cond':>6}  "
              f"{'mean F1':>8}  {'±std':>6}")
        print(f"  {'-'*55}")
        for i, r in enumerate(atim_ranked[:10]):
            p = r["params"]
            print(f"  {i+1:>3}  {p['alpha']:>4}  {p['temperature']:>4}  "
                  f"{p['lambda_marg']:>6}  {p['lambda_cond']:>6}  "
                  f"{r['mean_f1']:>8.4f}  {r['std_f1']:>6.4f}")

    # ── PT-MAP comparison ──
    if not args.skip_ptmap:
        ptmap_avg = {}
        for fold_res in all_fold_results:
            if "ptmap_variants" not in fold_res:
                continue
            for r in fold_res["ptmap_variants"]:
                label = r["label"]
                if label not in ptmap_avg:
                    ptmap_avg[label] = {"f1s": [], "baccs": []}
                ptmap_avg[label]["f1s"].append(r["f1"])
                ptmap_avg[label]["baccs"].append(r["bacc"])

        print(f"\n  PT-MAP variants (averaged):")
        print(f"  {'Variant':<30}  {'mean F1':>8}  {'±std':>6}  {'bacc':>8}")
        print(f"  {'-'*58}")
        for label in ["ptmap_sink", "ptmap_sink_l20",
                       "ptmap_nosink", "ptmap_nosink_l20",
                       "ptmap_nosink_40iter"]:
            if label in ptmap_avg:
                f1s = ptmap_avg[label]["f1s"]
                baccs = ptmap_avg[label]["baccs"]
                print(f"  {label:<30}  {np.mean(f1s):>8.4f}  "
                      f"{np.std(f1s):>6.4f}  {np.mean(baccs):>8.4f}")

    # ── Overall best ──
    print(f"\n  Per-fold best method summary:")
    for fold_res in all_fold_results:
        print(f"    Fold {fold_res['fold']+1}: {fold_res['best_method']} "
              f"F1={fold_res['best_f1']:.4f} "
              f"(head={fold_res['head_f1']:.4f}, "
              f"SS={fold_res['ss_f1']:.4f}, "
              f"Maha={fold_res['maha_f1']:.4f})")

    # Save final results
    final = {
        "experiment": "tim_search",
        "lora_r": args.lora_r,
        "lora_dropout": args.lora_dropout,
        "folds": args.folds,
        "seed": args.seed,
        "tukey_beta": args.tukey_beta,
        "tim_grid": TIM_GRID,
        "alpha_tim_grid": ALPHA_TIM_GRID,
        "ptmap_variants_tested": [v["label"] for v in PTMAP_VARIANTS],
        "fold_results": all_fold_results,
    }
    if not args.skip_tim and tim_ranked:
        final["tim_best_params"] = tim_ranked[0]["params"]
        final["tim_best_f1"] = tim_ranked[0]["mean_f1"]
    if not args.skip_alpha_tim and atim_ranked:
        final["atim_best_params"] = atim_ranked[0]["params"]
        final["atim_best_f1"] = atim_ranked[0]["mean_f1"]

    with open(args.out_json, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n[ok] saved results -> {args.out_json}")

    # Clean up checkpoint
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"[ok] checkpoint cleaned up")


if __name__ == "__main__":
    main()
