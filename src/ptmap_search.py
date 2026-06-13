"""PT-MAP hyperparameter search — NO DATA LEAKAGE.

For each random split:
  1. Split 250 images into 200 train + 50 holdout
  2. Train LoRA on 200 train images ONLY
  3. Extract features for train + holdout
  4. Search 320 PT-MAP parameter combos on features
  5. Evaluate on 50 holdout images

Search grid:
  lambda_s:      [5, 10, 15, 20, 30]
  n_iter:        [10, 20, 30, 50]
  sinkhorn_iter: [5, 10, 20, 30]
  tukey_beta:    [0.3, 0.5, 0.7, 1.0]

Usage:
  cd src
  python ptmap_search.py --ema --verbose
  python ptmap_search.py --ema --n_repeats 3 --verbose

Resume:
  If interrupted, just re-run the same command — it will skip
  already-completed splits and continue from where it left off.
  Use --force to start from scratch.
"""
from __future__ import annotations
import argparse
import json
import itertools
import shutil
import tempfile
import time
from pathlib import Path
from datetime import timedelta
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score

from data import list_train_samples, list_test_samples, CLASS_NAMES
from train_lora import train_one_run, seed_all
from self_train import extract_features, predict_probs
from transductive import pt_map


# ── Search grid ──────────────────────────────────────────────────

SEARCH_GRID = {
    "lambda_s":      [5, 10, 15, 20, 30],
    "n_iter":        [10, 20, 30, 50],
    "sinkhorn_iter": [5, 10, 20, 30],
    "tukey_beta":    [0.3, 0.5, 0.7, 1.0],
}

# Total combos: 5 x 4 x 4 x 4 = 320


def build_args(backbone="vits14", ema=True, seed=42):
    """Build args namespace matching train_one_run's expectations."""
    batch_size = 24 if backbone == "vitb14" else 32
    return argparse.Namespace(
        backbone=backbone,
        image_size=224,
        epochs=30,
        batch_size=batch_size,
        lr=5e-4,
        lr_head=1e-3,
        weight_decay=0.05,
        label_smoothing=0.1,
        grad_clip=1.0,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.1,
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


def train_on_split(train_paths, train_labels, holdout_paths, device, args):
    """Train LoRA on train split only, extract features for both splits.

    This avoids data leakage: the model NEVER sees holdout images during
    training.

    Returns:
      sup_feats:   np.ndarray (N_train, D)
      sup_labels:  np.ndarray (N_train,)
      test_feats:  np.ndarray (N_holdout, D)
      head_probs:  np.ndarray (N_holdout, 5)
      test_names:  list of filenames
    """
    # Train LoRA on train split
    full_idx = np.arange(len(train_paths))
    model, _, _ = train_one_run(
        train_paths, train_labels, full_idx, full_idx,
        args, device, run_id=0, refit_all=True,
    )

    # Extract features
    sup_feats, sup_labels = extract_features(
        model, train_paths, args.image_size, args, device,
        labels=train_labels,
    )
    test_feats, test_names = extract_features(
        model, holdout_paths, args.image_size, args, device,
        labels=None,
    )

    # Head predictions on holdout
    head_probs, _ = predict_probs(
        model, holdout_paths, args.image_size, args, device,
    )

    del model
    torch.cuda.empty_cache()

    return sup_feats, sup_labels, test_feats, head_probs, test_names


def search_ptmap_on_split(sup_feats, sup_labels, test_feats, y_true,
                          param_combos, num_classes=5):
    """Run PT-MAP with all parameter combos on one split.

    Returns:
      results: list of (params_dict, f1, bacc) sorted by f1 desc
    """
    results = []
    for combo in param_combos:
        params = dict(combo)
        try:
            pred = pt_map(
                sup_feats, sup_labels, test_feats,
                num_classes=num_classes,
                tukey_beta=params["tukey_beta"],
                n_iter=params["n_iter"],
                lambda_s=params["lambda_s"],
                use_sinkhorn=True,
                sinkhorn_iter=params["sinkhorn_iter"],
            )
            f1 = float(f1_score(y_true, pred, average="macro"))
            bacc = float(balanced_accuracy_score(y_true, pred))
            results.append((params, f1, bacc))
        except Exception:
            results.append((params, -1.0, -1.0))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def main():
    ap = argparse.ArgumentParser(
        description="PT-MAP hyperparameter search (proper train/holdout split)")
    ap.add_argument("--train_dir", default="../train_few_shot")
    ap.add_argument("--backbone", default="vits14", choices=["vits14", "vitb14"])
    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no_ema", action="store_false", dest="ema")
    ap.add_argument("--n_repeats", type=int, default=3,
                    help="random splits for evaluation (default 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", default="../ptmap_search_results.json")
    ap.add_argument("--force", action="store_true",
                    help="ignore checkpoint and start from scratch")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load all 250 images
    all_paths, all_labels = list_train_samples(args.train_dir)
    N = len(all_paths)
    print(f"[info] {N} samples, device={device}")
    print(f"[info] per-class: {np.bincount(all_labels).tolist()}")

    # Build parameter grid
    keys = list(SEARCH_GRID.keys())
    values = [SEARCH_GRID[k] for k in keys]
    param_combos = list(itertools.product(*values))
    param_combos = [dict(zip(keys, combo)) for combo in param_combos]
    print(f"[info] search grid: {len(param_combos)} parameter combos")

    # ── Checkpoint ──
    ckpt_path = Path(args.out_json).with_suffix(".checkpoint.json")
    if ckpt_path.exists() and not args.force:
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        completed_repeats = ckpt.get("completed_repeats", 0)
        all_split_results = ckpt.get("split_results", [])
        if completed_repeats >= args.n_repeats:
            print(f"[resume] all {args.n_repeats} repeats already done!")
        else:
            print(f"[resume] {completed_repeats} repeats already done, "
                  f"continuing from repeat {completed_repeats + 1}")
    else:
        all_split_results = []
        completed_repeats = 0
        ckpt = {}

    # Build args
    train_args = build_args(backbone=args.backbone, ema=args.ema, seed=args.seed)

    # ── Search on multiple splits ──
    splitter = StratifiedShuffleSplit(
        n_splits=args.n_repeats, test_size=0.2, random_state=args.seed,
    )

    for rep, (train_idx, holdout_idx) in enumerate(
            splitter.split(np.arange(N), all_labels)):
        if rep < completed_repeats:
            continue

        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  REPEAT {rep+1}/{args.n_repeats}: "
              f"{len(train_idx)} train / {len(holdout_idx)} holdout")
        print(f"{'='*60}")

        # ── Create temp dirs for this split ──
        with tempfile.TemporaryDirectory(prefix=f"ptmap_split_{rep}_") as tmpdir:
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

            # Ground truth for evaluation
            gt_map = {}
            for i in holdout_idx:
                gt_map[Path(all_paths[i]).name] = int(all_labels[i])

            # Load split data properly
            st_paths, st_labels = list_train_samples(str(fold_train_dir))
            holdout_paths = list_test_samples(str(fold_holdout_dir))

            # Train LoRA on train split ONLY (no leakage)
            print(f"  Training LoRA on {len(st_paths)} images ...")
            sup_feats, sup_labels_arr, test_feats, head_probs, test_names = \
                train_on_split(st_paths, st_labels, holdout_paths, device, train_args)

            # True labels for holdout
            y_true = np.array([gt_map[n] for n in test_names])

            # Head baseline
            head_pred = head_probs.argmax(axis=1)
            head_f1 = float(f1_score(y_true, head_pred, average="macro"))
            head_bacc = float(balanced_accuracy_score(y_true, head_pred))

            # Search PT-MAP params on features
            print(f"  Searching {len(param_combos)} PT-MAP combos ...")
            split_results = search_ptmap_on_split(
                sup_feats, sup_labels_arr, test_feats, y_true,
                param_combos, num_classes=5,
            )

        elapsed = time.time() - t0
        best = split_results[0]
        print(f"  >>> Best combo: lambda_s={best[0]['lambda_s']}, "
              f"n_iter={best[0]['n_iter']}, "
              f"sinkhorn={best[0]['sinkhorn_iter']}, "
              f"tukey={best[0]['tukey_beta']}  "
              f"F1={best[1]:.4f}  bacc={best[2]:.4f}")
        print(f"  >>> Head baseline: F1={head_f1:.4f}  bacc={head_bacc:.4f}")
        print(f"  >>> Time: {timedelta(seconds=int(elapsed))}")

        if args.verbose:
            # Show top-5 combos
            print(f"\n  Top-5 combos:")
            for rank, (p, f1, bacc) in enumerate(split_results[:5]):
                print(f"    {rank+1}. λ={p['lambda_s']:>2d}  "
                      f"n_iter={p['n_iter']:>2d}  "
                      f"sink={p['sinkhorn_iter']:>2d}  "
                      f"tukey={p['tukey_beta']:.1f}  "
                      f"F1={f1:.4f}")

        # Save checkpoint
        split_data = {
            "repeat": rep + 1,
            "train_size": len(train_idx),
            "holdout_size": len(holdout_idx),
            "head_f1": head_f1,
            "head_bacc": head_bacc,
            "best_params": split_results[0][0],
            "best_f1": split_results[0][1],
            "best_bacc": split_results[0][2],
            "top5": [(r[0], round(r[1], 4), round(r[2], 4))
                     for r in split_results[:5]],
            "all_results": [
                (r[0], round(r[1], 4), round(r[2], 4))
                for r in split_results if r[1] >= 0
            ],
        }
        all_split_results.append(split_data)
        ckpt["completed_repeats"] = rep + 1
        ckpt["split_results"] = all_split_results
        with open(ckpt_path, "w") as f:
            json.dump(ckpt, f, indent=2)

    # ── Aggregate results ──
    print(f"\n{'='*70}")
    print(f"  PT-MAP HYPERPARAMETER SEARCH RESULTS")
    print(f"  {args.n_repeats} repeats, {len(param_combos)} combos each")
    print(f"{'='*70}")

    # Aggregate by parameter combo across all repeats
    combo_scores = {}
    for split_data in all_split_results:
        for item in split_data["all_results"]:
            params = item[0]
            f1 = item[1]
            bacc = item[2]
            key = (params["lambda_s"], params["n_iter"],
                   params["sinkhorn_iter"], params["tukey_beta"])
            if key not in combo_scores:
                combo_scores[key] = {"f1": [], "bacc": []}
            combo_scores[key]["f1"].append(f1)
            combo_scores[key]["bacc"].append(bacc)

    # Compute mean and sort
    combo_list = []
    for key, scores in combo_scores.items():
        mean_f1 = np.mean(scores["f1"])
        std_f1 = np.std(scores["f1"]) if len(scores["f1"]) > 1 else 0.0
        mean_bacc = np.mean(scores["bacc"])
        combo_list.append({
            "lambda_s": key[0], "n_iter": key[1],
            "sinkhorn_iter": key[2], "tukey_beta": key[3],
            "mean_f1": mean_f1, "std_f1": std_f1,
            "mean_bacc": mean_bacc,
        })
    combo_list.sort(key=lambda x: x["mean_f1"], reverse=True)

    # Head baseline
    head_f1s = [s["head_f1"] for s in all_split_results]
    head_mean = np.mean(head_f1s)
    head_std = np.std(head_f1s) if len(head_f1s) > 1 else 0.0

    print(f"\n  Head baseline: F1 = {head_mean:.4f} ± {head_std:.4f}\n")
    print(f"  {'Rank':>4}  {'λ_s':>4}  {'n_iter':>6}  {'sink':>4}  "
          f"{'tukey':>5}  {'mean F1':>8}  {'±std':>6}  {'vs head':>8}")
    print(f"  {'-'*60}")

    for rank, c in enumerate(combo_list[:20]):
        vs_head = c["mean_f1"] - head_mean
        print(f"  {rank+1:>4}  {c['lambda_s']:>4}  {c['n_iter']:>6}  "
              f"{c['sinkhorn_iter']:>4}  {c['tukey_beta']:>5.1f}  "
              f"{c['mean_f1']:>8.4f}  {c['std_f1']:>6.4f}  "
              f"{vs_head:>+8.4f}")

    # Per-param sensitivity (marginal effect)
    print(f"\n  --- Parameter Sensitivity (marginal mean F1) ---")
    for param_name in ["lambda_s", "n_iter", "sinkhorn_iter", "tukey_beta"]:
        param_vals = sorted(set(c[param_name] for c in combo_list))
        print(f"\n  {param_name}:")
        for v in param_vals:
            subset = [c for c in combo_list if c[param_name] == v]
            mean = np.mean([c["mean_f1"] for c in subset])
            print(f"    {param_name}={v:>5}  →  mean F1 = {mean:.4f}  "
                  f"(n={len(subset)})")

    # Best overall
    best = combo_list[0]
    print(f"\n  >>> BEST COMBO: lambda_s={best['lambda_s']}, "
          f"n_iter={best['n_iter']}, "
          f"sinkhorn_iter={best['sinkhorn_iter']}, "
          f"tukey_beta={best['tukey_beta']}")
    print(f"      mean F1 = {best['mean_f1']:.4f} ± {best['std_f1']:.4f}  "
          f"(head baseline = {head_mean:.4f})")

    # Save final results
    final = {
        "search_grid": SEARCH_GRID,
        "n_repeats": args.n_repeats,
        "backbone": args.backbone,
        "ema": args.ema,
        "head_baseline": {"mean_f1": float(head_mean),
                          "std_f1": float(head_std)},
        "best_combo": {k: v for k, v in best.items()
                       if k != "mean_f1" and k != "std_f1"
                       and k != "mean_bacc"},
        "best_mean_f1": float(best["mean_f1"]),
        "best_std_f1": float(best["std_f1"]),
        "top20": combo_list[:20],
        "per_split": all_split_results,
    }
    with open(args.out_json, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n[ok] saved results -> {args.out_json}")

    # Clean up checkpoint
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"[ok] checkpoint cleaned up")


if __name__ == "__main__":
    main()
