"""Distribution Calibration (DC) for few-shot feature augmentation.

Implements the ICLR 2021 Oral paper:
  "Distribution Calibration for Few-Shot Learning" — Yang et al.

Core idea:
  1. Estimate each class's feature distribution (mean + covariance)
  2. Find the k nearest classes and borrow their covariance to calibrate
  3. Sample synthetic features from the calibrated distribution
  4. Use original + synthetic features for downstream classification

This is a "free lunch" — no retraining needed, operates purely on
extracted features.

Usage:
  cd src
  python dc_experiment.py --ema --verbose

Resume:
  If interrupted, re-run the same command. Uses checkpoint to skip
  completed folds. Add --force to start from scratch.
"""
from __future__ import annotations
import argparse
import itertools
import json
import shutil
import tempfile
import time
from pathlib import Path
from datetime import timedelta
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.covariance import MinCovDet

from data import list_train_samples, list_test_samples, CLASS_NAMES
from train_lora import train_one_run, seed_all
from self_train import extract_features, predict_probs
from transductive import pt_map


# ── Distribution Calibration ────────────────────────────────────

def distribution_calibration(
    sup_feats: np.ndarray,
    sup_labels: np.ndarray,
    num_classes: int = 5,
    k_neighbors: int = 2,
    n_samples: int = 200,
    alpha: float = 0.21,
    use_robust_cov: bool = False,
    cov_reg: float = 1e-3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate per-class distributions and generate synthetic features.

    For each class c:
      1. Compute mean μ_c and covariance Σ_c from support features
      2. Find k nearest classes (by Euclidean distance between class means)
      3. Calibrate: Σ_cal = Σ_c + α * (Σ_neighbor1 + Σ_neighbor2 + ...)
      4. Add regularization: Σ_cal += cov_reg * I
      5. Sample n_samples synthetic features from N(μ_c, Σ_cal)

    Args:
      sup_feats:    (N, D) support features
      sup_labels:   (N,)  support labels
      num_classes:  number of classes
      k_neighbors:  number of nearest classes to borrow covariance from
      n_samples:    number of synthetic features per class
      alpha:        weight for borrowed covariance (default 0.21 from paper)
      use_robust_cov: use MinCovDet for robust covariance estimation
      cov_reg:      regularization added to covariance diagonal
      seed:         random seed for reproducibility

    Returns:
      aug_feats:  (N + num_classes*n_samples, D) augmented features
      aug_labels: (N + num_classes*n_samples,) augmented labels
    """
    rng = np.random.RandomState(seed)
    D = sup_feats.shape[1]

    # Per-class statistics
    class_means = np.zeros((num_classes, D))
    class_covs = np.zeros((num_classes, D, D))

    for c in range(num_classes):
        mask = sup_labels == c
        feats_c = sup_feats[mask]
        class_means[c] = feats_c.mean(axis=0)
        if use_robust_cov and len(feats_c) > D + 1:
            try:
                mcd = MinCovDet().fit(feats_c)
                class_covs[c] = mcd.covariance_
            except Exception:
                class_covs[c] = np.cov(feats_c.T) + cov_reg * np.eye(D)
        else:
            if len(feats_c) > 1:
                class_covs[c] = np.cov(feats_c.T) + cov_reg * np.eye(D)
            else:
                class_covs[c] = np.eye(D) * cov_reg

    # Find k nearest classes for each class
    # Distance matrix between class means
    dist_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            dist_matrix[i, j] = np.linalg.norm(class_means[i] - class_means[j])

    # Generate synthetic features
    all_syn_feats = []
    all_syn_labels = []

    for c in range(num_classes):
        # Find k nearest classes (excluding self)
        distances = dist_matrix[c].copy()
        distances[c] = np.inf  # exclude self
        neighbor_idx = np.argsort(distances)[:k_neighbors]

        # Calibrated covariance: Σ_c + α * Σ_neighbors
        calibrated_cov = class_covs[c].copy()
        for ni in neighbor_idx:
            calibrated_cov += alpha * class_covs[ni]

        # Regularize
        calibrated_cov += cov_reg * np.eye(D)

        # Ensure positive definite
        try:
            L = np.linalg.cholesky(calibrated_cov)
        except np.linalg.LinAlgError:
            # Fallback: use eigendecomposition to fix
            eigvals, eigvecs = np.linalg.eigh(calibrated_cov)
            eigvals = np.maximum(eigvals, cov_reg)
            calibrated_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = np.linalg.cholesky(calibrated_cov)

        # Sample synthetic features
        z = rng.randn(n_samples, D)
        syn_feats = class_means[c][np.newaxis, :] + z @ L.T
        all_syn_feats.append(syn_feats)
        all_syn_labels.append(np.full(n_samples, c, dtype=np.int64))

    syn_feats = np.concatenate(all_syn_feats, axis=0)
    syn_labels = np.concatenate(all_syn_labels, axis=0)

    # Combine original + synthetic
    aug_feats = np.concatenate([sup_feats, syn_feats], axis=0)
    aug_labels = np.concatenate([sup_labels, syn_labels], axis=0)

    return aug_feats, aug_labels


# ── Training helper ─────────────────────────────────────────────

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


def train_and_extract(train_paths, train_labels, holdout_paths, device, args):
    """Train LoRA on train split, extract features for both splits.

    Returns:
      sup_feats, sup_labels, test_feats, head_probs, test_names
    """
    full_idx = np.arange(len(train_paths))
    model, _, _ = train_one_run(
        train_paths, train_labels, full_idx, full_idx,
        args, device, run_id=0, refit_all=True,
    )

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

    return sup_feats, sup_labels, test_feats, head_probs, test_names


# ── DC parameter search ─────────────────────────────────────────

DC_SEARCH_GRID = {
    "k_neighbors": [1, 2, 3],
    "n_samples":   [50, 100, 200, 400],
    "alpha":       [0.1, 0.21, 0.5, 1.0],
    "cov_reg":     [1e-4, 1e-3, 1e-2],
}


def search_dc_params(sup_feats, sup_labels, test_feats, y_true,
                     num_classes=5, tukey_beta=1.0, maha_shrink=0.3,
                     use_combined_mean=True, head_probs=None):
    """Search DC parameters using PT-MAP as downstream classifier.

    For each DC config, generate synthetic features, then run PT-MAP
    (with best params from previous search) and record F1.

    Returns list of (dc_params, method, f1, bacc) sorted by f1 desc.
    """
    results = []
    keys = list(DC_SEARCH_GRID.keys())
    values = [DC_SEARCH_GRID[k] for k in keys]

    total = 1
    for v in values:
        total *= len(v)
    print(f"  Searching {total} DC parameter combos ...")

    for i, combo in enumerate(itertools.product(*values)):
        dc_params = dict(zip(keys, combo))
        try:
            # Generate augmented features
            aug_feats, aug_labels = distribution_calibration(
                sup_feats, sup_labels,
                num_classes=num_classes,
                k_neighbors=dc_params["k_neighbors"],
                n_samples=dc_params["n_samples"],
                alpha=dc_params["alpha"],
                cov_reg=dc_params["cov_reg"],
            )

            # Run PT-MAP on augmented features (best params from search)
            pred = pt_map(
                aug_feats, aug_labels, test_feats,
                num_classes=num_classes,
                tukey_beta=tukey_beta, n_iter=10, lambda_s=20.0,
                use_sinkhorn=True, sinkhorn_iter=10,
            )
            f1 = float(f1_score(y_true, pred, average="macro"))
            bacc = float(balanced_accuracy_score(y_true, pred))
            results.append((dc_params, "ptmap", f1, bacc))

        except Exception as e:
            results.append((dc_params, "ptmap", -1.0, -1.0))

    # Also test without DC (baseline)
    # PT-MAP without DC
    pred = pt_map(
        sup_feats, sup_labels, test_feats,
        num_classes=num_classes,
        tukey_beta=tukey_beta, n_iter=10, lambda_s=20.0,
        use_sinkhorn=True, sinkhorn_iter=10,
    )
    f1 = float(f1_score(y_true, pred, average="macro"))
    bacc = float(balanced_accuracy_score(y_true, pred))
    results.append(({"k_neighbors": 0, "n_samples": 0, "alpha": 0, "cov_reg": 0},
                    "ptmap_nodc", f1, bacc))

    # Head baseline
    if head_probs is not None:
        head_pred = head_probs.argmax(axis=1)
        head_f1 = float(f1_score(y_true, head_pred, average="macro"))
        head_bacc = float(balanced_accuracy_score(y_true, head_pred))
        results.append(({}, "head", head_f1, head_bacc))

    results.sort(key=lambda x: x[2], reverse=True)
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Distribution Calibration experiment (ICLR 2021)")
    ap.add_argument("--train_dir", default="../train_few_shot")
    ap.add_argument("--backbone", default="vits14", choices=["vits14", "vitb14"])
    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no_ema", action="store_false", dest="ema")
    ap.add_argument("--n_repeats", type=int, default=3,
                    help="random splits (default 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tukey_beta", type=float, default=1.0,
                    help="Tukey beta for PT-MAP (1.0=off, from search)")
    ap.add_argument("--maha_shrink", type=float, default=0.3)
    ap.add_argument("--use_combined_mean", action="store_true", default=True)
    ap.add_argument("--out_json", default="../dc_experiment_results.json")
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

    # Build args
    train_args = build_args(backbone=args.backbone, ema=args.ema, seed=args.seed)

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

    # ── Run experiment ──
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

        with tempfile.TemporaryDirectory(prefix=f"dc_split_{rep}_") as tmpdir:
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

            # Train LoRA on train split ONLY
            print(f"  Training LoRA on {len(st_paths)} images ...")
            sup_feats, sup_labels_arr, test_feats, head_probs, test_names = \
                train_and_extract(st_paths, st_labels, holdout_paths,
                                  device, train_args)

            # True labels for holdout
            y_true = np.array([gt_map[n] for n in test_names])

            # Search DC parameters
            dc_results = search_dc_params(
                sup_feats, sup_labels_arr, test_feats, y_true,
                num_classes=5,
                tukey_beta=args.tukey_beta,
                maha_shrink=args.maha_shrink,
                use_combined_mean=args.use_combined_mean,
                head_probs=head_probs,
            )

        elapsed = time.time() - t0

        # Report
        best = dc_results[0]
        # Find no-DC baseline
        nodc = [r for r in dc_results if r[1] == "ptmap_nodc"][0]
        head = [r for r in dc_results if r[1] == "head"][0]

        print(f"  >>> Best DC: k={best[0].get('k_neighbors','?')}, "
              f"n_samples={best[0].get('n_samples','?')}, "
              f"alpha={best[0].get('alpha','?')}, "
              f"reg={best[0].get('cov_reg','?')}  "
              f"F1={best[2]:.4f}")
        print(f"  >>> PT-MAP (no DC): F1={nodc[2]:.4f}")
        print(f"  >>> Head baseline: F1={head[2]:.4f}")
        print(f"  >>> Time: {timedelta(seconds=int(elapsed))}")

        if args.verbose and best[1] == "ptmap":
            print(f"\n  Top-5 DC combos:")
            dc_only = [r for r in dc_results if r[1] == "ptmap" and r[2] >= 0]
            for rank, (p, m, f1, bacc) in enumerate(dc_only[:5]):
                print(f"    {rank+1}. k={p['k_neighbors']}, "
                      f"n={p['n_samples']}, "
                      f"α={p['alpha']:.2f}, "
                      f"reg={p['cov_reg']:.0e}  "
                      f"F1={f1:.4f}")

        # Save checkpoint
        split_data = {
            "repeat": rep + 1,
            "train_size": len(train_idx),
            "holdout_size": len(holdout_idx),
            "best_dc_params": best[0],
            "best_dc_f1": best[2],
            "best_dc_bacc": best[3],
            "nodc_f1": nodc[2],
            "nodc_bacc": nodc[3],
            "head_f1": head[2],
            "head_bacc": head[3],
            "all_results": [
                {
                    "dc_params": r[0],
                    "method": r[1],
                    "f1": round(r[2], 4),
                    "bacc": round(r[3], 4),
                }
                for r in dc_results if r[2] >= 0
            ],
        }
        all_split_results.append(split_data)
        ckpt["completed_repeats"] = rep + 1
        ckpt["split_results"] = all_split_results
        with open(ckpt_path, "w") as f:
            json.dump(ckpt, f, indent=2)

    # ── Aggregate ──
    print(f"\n{'='*70}")
    print(f"  DISTRIBUTION CALIBRATION EXPERIMENT RESULTS")
    print(f"  {args.n_repeats} repeats")
    print(f"{'='*70}")

    # Aggregate DC params across repeats
    dc_combo_scores = {}
    for split_data in all_split_results:
        for item in split_data["all_results"]:
            if item["method"] != "ptmap":
                continue
            p = item["dc_params"]
            if p.get("k_neighbors", 0) == 0:
                continue  # skip no-DC baseline
            key = (p["k_neighbors"], p["n_samples"], p["alpha"], p["cov_reg"])
            if key not in dc_combo_scores:
                dc_combo_scores[key] = {"f1": [], "bacc": []}
            dc_combo_scores[key]["f1"].append(item["f1"])
            dc_combo_scores[key]["bacc"].append(item["bacc"])

    # Compute mean and sort
    dc_list = []
    for key, scores in dc_combo_scores.items():
        mean_f1 = np.mean(scores["f1"])
        std_f1 = np.std(scores["f1"]) if len(scores["f1"]) > 1 else 0.0
        dc_list.append({
            "k_neighbors": key[0], "n_samples": key[1],
            "alpha": key[2], "cov_reg": key[3],
            "mean_f1": mean_f1, "std_f1": std_f1,
        })
    dc_list.sort(key=lambda x: x["mean_f1"], reverse=True)

    # No-DC and head baselines
    nodc_f1s = [s["nodc_f1"] for s in all_split_results]
    head_f1s = [s["head_f1"] for s in all_split_results]
    nodc_mean = np.mean(nodc_f1s)
    nodc_std = np.std(nodc_f1s) if len(nodc_f1s) > 1 else 0.0
    head_mean = np.mean(head_f1s)
    head_std = np.std(head_f1s) if len(head_f1s) > 1 else 0.0

    print(f"\n  PT-MAP (no DC): F1 = {nodc_mean:.4f} ± {nodc_std:.4f}")
    print(f"  Head baseline:  F1 = {head_mean:.4f} ± {head_std:.4f}\n")

    print(f"  {'Rank':>4}  {'k':>2}  {'n_samp':>6}  {'alpha':>5}  "
          f"{'reg':>6}  {'mean F1':>8}  {'±std':>6}  {'vs noDC':>8}")
    print(f"  {'-'*65}")

    for rank, c in enumerate(dc_list[:20]):
        vs_nodc = c["mean_f1"] - nodc_mean
        print(f"  {rank+1:>4}  {c['k_neighbors']:>2}  {c['n_samples']:>6}  "
              f"{c['alpha']:>5.2f}  {c['cov_reg']:>6.0e}  "
              f"{c['mean_f1']:>8.4f}  {c['std_f1']:>6.4f}  "
              f"{vs_nodc:>+8.4f}")

    # Per-param sensitivity
    print(f"\n  --- DC Parameter Sensitivity (marginal mean F1) ---")
    for param_name in ["k_neighbors", "n_samples", "alpha", "cov_reg"]:
        param_vals = sorted(set(c[param_name] for c in dc_list))
        print(f"\n  {param_name}:")
        for v in param_vals:
            subset = [c for c in dc_list if c[param_name] == v]
            mean = np.mean([c["mean_f1"] for c in subset])
            print(f"    {param_name}={v}  →  mean F1 = {mean:.4f}  "
                  f"(n={len(subset)})")

    # Best overall
    if dc_list:
        best = dc_list[0]
        print(f"\n  >>> BEST DC COMBO: k={best['k_neighbors']}, "
              f"n_samples={best['n_samples']}, "
              f"alpha={best['alpha']}, "
              f"cov_reg={best['cov_reg']}")
        print(f"      mean F1 = {best['mean_f1']:.4f} ± {best['std_f1']:.4f}  "
              f"(no DC = {nodc_mean:.4f}, head = {head_mean:.4f})")

        improvement = best["mean_f1"] - nodc_mean
        if improvement > 0.005:
            print(f"      ✅ DC improves by +{improvement:.4f}")
        elif improvement > -0.005:
            print(f"      ➖ DC has negligible effect ({improvement:+.4f})")
        else:
            print(f"      ❌ DC hurts by {improvement:.4f}")

    # Save final results
    final = {
        "search_grid": DC_SEARCH_GRID,
        "n_repeats": args.n_repeats,
        "backbone": args.backbone,
        "ema": args.ema,
        "baselines": {
            "ptmap_nodc": {"mean_f1": float(nodc_mean), "std_f1": float(nodc_std)},
            "head": {"mean_f1": float(head_mean), "std_f1": float(head_std)},
        },
        "best_dc": {k: v for k, v in dc_list[0].items()} if dc_list else None,
        "top20": dc_list[:20],
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
