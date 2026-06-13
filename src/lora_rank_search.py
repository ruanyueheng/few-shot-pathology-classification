"""LoRA rank search: find the best LoRA rank to balance capacity vs overfitting.

Current default: r=32 → 1,084,549 trainable params, 200 train images.
Params/sample = 5,423:1 — heavily overfitting.

This script searches r ∈ {2, 4, 8, 16, 32} with matching lora_alpha = 2*r,
keeping all other hyperparameters identical, then evaluates with ALL 8
transductive methods + head on holdout splits, taking the best per split.

This matches the evaluation logic of cv5_improve.py exactly, so results
are directly comparable with the baseline 0.6529.

Key hypothesis: smaller LoRA rank → less overfitting → better feature quality
→ higher transductive inference F1.

Usage:
  cd src
  python lora_rank_search.py --ema --tukey_beta 0.5 --verbose

Resume:
  If interrupted, re-run the same command. Uses checkpoint to skip
  completed configurations. Add --force to start from scratch.
"""
from __future__ import annotations
import argparse
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

from data import list_train_samples, list_test_samples, CLASS_NAMES
from train_lora import train_one_run, seed_all
from self_train import extract_features, predict_probs
from transductive import (
    simpleshot, laplacianshot, mahalanobis, tim,
    label_propagation, pt_map, alpha_tim,
)


# ── Rank configurations ────────────────────────────────────────

RANK_CONFIGS = {
    # rank: (lora_r, lora_alpha)
    2:  (2,  4),
    4:  (4,  8),
    8:  (8,  16),
    16: (16, 32),
    32: (32, 64),   # current default
}


def build_args(backbone="vits14", ema=True, lora_r=32, lora_alpha=64, seed=42):
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
        lora_r=lora_r,
        lora_alpha=lora_alpha,
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
      sup_feats, sup_labels, test_feats, head_probs, test_names, n_params
    """
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


def run_all_methods(sup_feats, sup_labels, test_feats, num_classes=5,
                    tukey_beta=0.5, maha_shrink=0.3, use_combined_mean=True):
    """Run ALL 8 transductive methods (same as cv5_improve / holdout_compare).

    Returns dict of {method_name: {"pred": ..., "probs": ...}}.
    """
    results = {}

    # 1. SimpleShot
    pred_ss, probs_ss = simpleshot(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        return_probs=True,
    )
    results["simpleshot"] = {"pred": pred_ss, "probs": probs_ss}

    # 2. Mahalanobis
    pred_maha, probs_maha = mahalanobis(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        shrink=maha_shrink, return_probs=True,
    )
    results["mahalanobis"] = {"pred": pred_maha, "probs": probs_maha}

    # 3. Label Propagation
    pred_lp = label_propagation(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        knn=10, alpha=0.7, sigma=1.0,
    )
    results["lp"] = {"pred": pred_lp}

    # 4. LaplacianShot
    pred_lap, probs_lap = laplacianshot(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        knn=5, lam=1.0, n_iter=20, sigma=1.0, return_probs=True,
    )
    results["laplacianshot"] = {"pred": pred_lap, "probs": probs_lap}

    # 5. PT-MAP (original params from cv5_improve)
    pred_ptmap, probs_ptmap = pt_map(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta,
        n_iter=20, lambda_s=10.0, use_sinkhorn=True, sinkhorn_iter=10,
        return_probs=True,
    )
    results["ptmap"] = {"pred": pred_ptmap, "probs": probs_ptmap}

    # 6. TIM
    pred_tim = tim(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        n_iter=1000, lr=1e-4, temperature=15.0,
        lambda_marg=1.0, lambda_cond=0.1,
    )
    results["tim"] = {"pred": pred_tim}

    # 7. alpha-TIM (α=2.0)
    pred_atim = alpha_tim(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=True,
        n_iter=1000, lr=1e-4, temperature=15.0,
        alpha=2.0, lambda_marg=1.0, lambda_cond=0.1,
    )
    results["alpha_tim"] = {"pred": pred_atim}

    return results


def evaluate_all(sup_feats, sup_labels, test_feats, y_true,
                 num_classes=5, tukey_beta=0.5, maha_shrink=0.3,
                 use_combined_mean=True, head_probs=None):
    """Run ALL methods + head, return dict of {method: {f1, bacc}}."""
    # Transductive methods
    trans_results = run_all_methods(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, maha_shrink=maha_shrink,
        use_combined_mean=use_combined_mean,
    )

    method_scores = {}
    for name, res in trans_results.items():
        pred = res["pred"]
        method_scores[name] = {
            "f1": float(f1_score(y_true, pred, average="macro")),
            "bacc": float(balanced_accuracy_score(y_true, pred)),
        }

    # Head baseline
    if head_probs is not None:
        head_pred = head_probs.argmax(axis=1)
        method_scores["head"] = {
            "f1": float(f1_score(y_true, head_pred, average="macro")),
            "bacc": float(balanced_accuracy_score(y_true, head_pred)),
        }

    return method_scores


def main():
    ap = argparse.ArgumentParser(
        description="LoRA rank search — find optimal rank to reduce overfitting")
    ap.add_argument("--train_dir", default="../train_few_shot")
    ap.add_argument("--backbone", default="vits14", choices=["vits14", "vitb14"])
    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no_ema", action="store_false", dest="ema")
    ap.add_argument("--n_repeats", type=int, default=3,
                    help="random splits (default 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tukey_beta", type=float, default=0.5,
                    help="Tukey beta for transductive methods (0.5=default, same as cv5)")
    ap.add_argument("--maha_shrink", type=float, default=0.3)
    ap.add_argument("--use_combined_mean", action="store_true", default=True)
    ap.add_argument("--ranks", type=str, default="2,4,8,16,32",
                    help="comma-separated LoRA ranks to search")
    ap.add_argument("--out_json", default="../lora_rank_results.json")
    ap.add_argument("--force", action="store_true",
                    help="ignore checkpoint and start from scratch")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse rank list
    rank_list = [int(x) for x in args.ranks.split(",")]
    print(f"[info] Searching LoRA ranks: {rank_list}")
    print(f"[info] PT-MAP params: lambda_s=10, n_iter=20, "
          f"sinkhorn_iter=10, tukey_beta={args.tukey_beta}")
    print(f"[info] (Same params as cv5_improve baseline for direct comparison)")

    # Load all 250 images
    all_paths, all_labels = list_train_samples(args.train_dir)
    N = len(all_paths)
    print(f"[info] {N} samples, device={device}")
    print(f"[info] per-class: {np.bincount(all_labels).tolist()}")

    # ── Checkpoint ──
    ckpt_path = Path(args.out_json).with_suffix(".checkpoint.json")
    if ckpt_path.exists() and not args.force:
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        completed = ckpt.get("completed", [])  # list of "rep0_r32"
        all_results = ckpt.get("results", [])
        print(f"[resume] {len(completed)} configs already done, continuing")
    else:
        completed = []
        all_results = []
        ckpt = {}

    # ── Run experiment ──
    splitter = StratifiedShuffleSplit(
        n_splits=args.n_repeats, test_size=0.2, random_state=args.seed,
    )

    total_configs = args.n_repeats * len(rank_list)
    done_count = len(completed)

    for rep, (train_idx, holdout_idx) in enumerate(
            splitter.split(np.arange(N), all_labels)):
        for rank in rank_list:
            task_key = f"rep{rep}_r{rank}"
            if task_key in completed:
                continue

            lora_r, lora_alpha = RANK_CONFIGS[rank]
            train_args = build_args(
                backbone=args.backbone, ema=args.ema,
                lora_r=lora_r, lora_alpha=lora_alpha, seed=args.seed,
            )

            done_count += 1
            t0 = time.time()
            print(f"\n{'='*60}")
            print(f"  [{done_count}/{total_configs}] REPEAT {rep+1}/{args.n_repeats}, "
                  f"rank={rank} (alpha={lora_alpha})")
            print(f"{'='*60}")

            with tempfile.TemporaryDirectory(prefix=f"rank_r{rank}_") as tmpdir:
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
                print(f"  Training LoRA r={rank}, alpha={lora_alpha} "
                      f"on {len(st_paths)} images ...")
                sup_feats, sup_labels_arr, test_feats, head_probs, test_names, n_params = \
                    train_and_extract(st_paths, st_labels, holdout_paths,
                                      device, train_args)

                print(f"  trainable params: {n_params:,}  "
                      f"(params/sample = {n_params/len(st_paths):.0f}:1)")

                # True labels for holdout
                y_true = np.array([gt_map[n] for n in test_names])

                # Evaluate ALL methods (same as cv5_improve)
                method_scores = evaluate_all(
                    sup_feats, sup_labels_arr, test_feats, y_true,
                    num_classes=5,
                    tukey_beta=args.tukey_beta,
                    maha_shrink=args.maha_shrink,
                    use_combined_mean=args.use_combined_mean,
                    head_probs=head_probs,
                )

            elapsed = time.time() - t0

            # Find best method for this split
            best_method = max(method_scores, key=lambda k: method_scores[k]["f1"])
            best_f1 = method_scores[best_method]["f1"]

            # Report
            if args.verbose:
                print(f"  All methods:")
                for m, s in sorted(method_scores.items(),
                                   key=lambda x: -x[1]["f1"]):
                    marker = " ← best" if m == best_method else ""
                    print(f"    {m:<20}: F1={s['f1']:.4f}{marker}")

            print(f"  >>> r={rank}: best={best_method} F1={best_f1:.4f}")
            print(f"  >>> Time: {timedelta(seconds=int(elapsed))}")

            # Save checkpoint
            result_entry = {
                "repeat": rep + 1,
                "lora_r": rank,
                "lora_alpha": lora_alpha,
                "n_params": n_params,
                "params_per_sample": round(n_params / len(st_paths), 1),
                "train_size": len(train_idx),
                "holdout_size": len(holdout_idx),
                "best_method": best_method,
                "best_f1": best_f1,
                "all_methods": method_scores,
                "elapsed_sec": round(elapsed, 1),
            }
            all_results.append(result_entry)
            completed.append(task_key)
            ckpt["completed"] = completed
            ckpt["results"] = all_results
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, indent=2)

    # ── Aggregate ──
    print(f"\n{'='*70}")
    print(f"  LORA RANK SEARCH RESULTS")
    print(f"  {args.n_repeats} repeats × {len(rank_list)} ranks")
    print(f"  (same evaluation as cv5_improve: ALL methods, pick best per split)")
    print(f"{'='*70}")

    # Aggregate by rank
    rank_stats = {}
    for rank in rank_list:
        rank_entries = [r for r in all_results if r["lora_r"] == rank]
        if not rank_entries:
            continue

        # Best-F1 per split (same metric as cv5_improve's mean_best_f1)
        best_f1s = [r["best_f1"] for r in rank_entries]

        # Per-method average across repeats
        method_avg = {}
        for method in ["simpleshot", "mahalanobis", "lp", "laplacianshot",
                        "ptmap", "tim", "alpha_tim", "head"]:
            f1s = [r["all_methods"][method]["f1"] for r in rank_entries
                   if method in r.get("all_methods", {})]
            baccs = [r["all_methods"][method]["bacc"] for r in rank_entries
                     if method in r.get("all_methods", {})]
            if f1s:
                method_avg[method] = {
                    "mean_f1": float(np.mean(f1s)),
                    "std_f1": float(np.std(f1s)) if len(f1s) > 1 else 0.0,
                    "mean_bacc": float(np.mean(baccs)),
                }

        rank_stats[rank] = {
            "n_params": rank_entries[0]["n_params"],
            "params_per_sample": rank_entries[0]["params_per_sample"],
            "n_repeats": len(rank_entries),
            "mean_best_f1": float(np.mean(best_f1s)),
            "std_best_f1": float(np.std(best_f1s)) if len(best_f1s) > 1 else 0.0,
            "method_avg": method_avg,
        }

    # Print summary table
    print(f"\n  {'rank':>4}  {'alpha':>5}  {'params':>10}  {'p/s':>6}  "
          f"{'Best F1':>8}  {'±std':>6}  "
          f"{'PT-MAP':>8}  {'SS':>8}  {'Head':>8}")
    print(f"  {'-'*75}")

    for rank in rank_list:
        if rank not in rank_stats:
            continue
        s = rank_stats[rank]
        ptmap = s["method_avg"].get("ptmap", {})
        ss = s["method_avg"].get("simpleshot", {})
        head = s["method_avg"].get("head", {})

        lora_r, lora_alpha = RANK_CONFIGS[rank]
        print(f"  {rank:>4}  {lora_alpha:>5}  {s['n_params']:>10,}  "
              f"{s['params_per_sample']:>6.0f}  "
              f"{s['mean_best_f1']:>8.4f}  {s['std_best_f1']:>6.4f}  "
              f"{ptmap.get('mean_f1', 0):>8.4f}  "
              f"{ss.get('mean_f1', 0):>8.4f}  "
              f"{head.get('mean_f1', 0):>8.4f}")

    # Detailed per-method breakdown for each rank
    for rank in rank_list:
        if rank not in rank_stats:
            continue
        s = rank_stats[rank]
        _, lora_alpha = RANK_CONFIGS[rank]
        print(f"\n  r={rank} (alpha={lora_alpha}) — per-method average macro-F1:")
        method_sorted = sorted(s["method_avg"].items(),
                               key=lambda x: -x[1]["mean_f1"])
        print(f"    {'Method':<20} {'mean F1':>8} {'±std':>8} {'mean bacc':>10}")
        print(f"    {'-'*48}")
        for m, stats in method_sorted:
            print(f"    {m:<20} {stats['mean_f1']:>8.4f} "
                  f"{stats['std_f1']:>8.4f} "
                  f"{stats['mean_bacc']:>10.4f}")

    # Find best rank
    best_rank = max(rank_stats, key=lambda k: rank_stats[k]["mean_best_f1"])
    best_s = rank_stats[best_rank]
    print(f"\n  >>> BEST RANK: r={best_rank}")
    print(f"      mean best-F1 = {best_s['mean_best_f1']:.4f} "
          f"± {best_s['std_best_f1']:.4f}")

    # Comparison with r=32 baseline
    if 32 in rank_stats:
        baseline_f1 = rank_stats[32]["mean_best_f1"]
        print(f"\n  --- vs r=32 baseline (best-F1={baseline_f1:.4f}) ---")
        for rank in rank_list:
            if rank == 32 or rank not in rank_stats:
                continue
            s = rank_stats[rank]
            diff = s["mean_best_f1"] - baseline_f1
            marker = "✅" if diff > 0.005 else ("➖" if diff > -0.005 else "❌")
            print(f"    r={rank:>2}: best-F1={s['mean_best_f1']:.4f}  "
                  f"({diff:+.4f}) {marker}")

    # Save final results
    final = {
        "ranks_searched": rank_list,
        "rank_configs": {str(k): {"r": v[0], "alpha": v[1]}
                        for k, v in RANK_CONFIGS.items() if k in rank_list},
        "n_repeats": args.n_repeats,
        "backbone": args.backbone,
        "ema": args.ema,
        "tukey_beta": args.tukey_beta,
        "maha_shrink": args.maha_shrink,
        "use_combined_mean": args.use_combined_mean,
        "rank_stats": {str(k): v for k, v in rank_stats.items()},
        "per_config_results": all_results,
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
