"""Regularization search: Mixup + lora_dropout on r=16 (best rank from lora_rank_search).

r=16 + TIM achieved 0.7000 mean best-F1 (vs r=32 baseline 0.6445).
This script searches additional regularization to further reduce overfitting:

  - mixup_alpha ∈ {0.0, 0.2, 0.5, 0.8}    (Mixup interpolation strength)
  - lora_dropout ∈ {0.0, 0.1, 0.2, 0.3}    (Dropout on LoRA weights)

Total: 4 × 4 = 16 configs × 3 repeats = 48 training runs (~6.4 hours).
Checkpoint/resume supported.

Evaluation: ALL 8 transductive methods + head, pick best per split
(same as lora_rank_search and cv5_improve).

Usage:
  cd src
  python reg_search.py --ema --tukey_beta 0.5 --verbose

Resume:
  Re-run same command. Uses checkpoint to skip completed configs.
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


# ── Fixed base config (from lora_rank_search best: r=16) ────────

BASE_LORA_R = 16
BASE_LORA_ALPHA = 32


def build_args(backbone="vits14", ema=True, lora_r=16, lora_alpha=32,
               mixup=0.0, lora_dropout=0.1, seed=42):
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
        mixup=mixup,
        cutmix=0.0,
        num_workers=2,
        seed=seed,
        verbose=False,
    )


def train_and_extract(train_paths, train_labels, holdout_paths, device, args):
    """Train LoRA on train split, extract features for both splits."""
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
    """Run ALL 8 transductive methods (same as cv5_improve / lora_rank_search)."""
    results = {}

    results["simpleshot"] = {"pred": simpleshot(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
    )}

    results["mahalanobis"] = {"pred": mahalanobis(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        shrink=maha_shrink,
    )}

    results["lp"] = {"pred": label_propagation(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        knn=10, alpha=0.7, sigma=1.0,
    )}

    results["laplacianshot"] = {"pred": laplacianshot(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        knn=5, lam=1.0, n_iter=20, sigma=1.0,
    )}

    results["ptmap"] = {"pred": pt_map(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta,
        n_iter=20, lambda_s=10.0, use_sinkhorn=True, sinkhorn_iter=10,
    )}

    results["tim"] = {"pred": tim(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        n_iter=1000, lr=1e-4, temperature=15.0,
        lambda_marg=1.0, lambda_cond=0.1,
    )}

    results["alpha_tim"] = {"pred": alpha_tim(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=True,
        n_iter=1000, lr=1e-4, temperature=15.0,
        alpha=2.0, lambda_marg=1.0, lambda_cond=0.1,
    )}

    return results


def evaluate_all(sup_feats, sup_labels, test_feats, y_true,
                 num_classes=5, tukey_beta=0.5, maha_shrink=0.3,
                 use_combined_mean=True, head_probs=None):
    """Run ALL methods + head, return dict of {method: {f1, bacc}}."""
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

    if head_probs is not None:
        head_pred = head_probs.argmax(axis=1)
        method_scores["head"] = {
            "f1": float(f1_score(y_true, head_pred, average="macro")),
            "bacc": float(balanced_accuracy_score(y_true, head_pred)),
        }

    return method_scores


def main():
    ap = argparse.ArgumentParser(
        description="Regularization search: Mixup + lora_dropout on r=16")
    ap.add_argument("--train_dir", default="../train_few_shot")
    ap.add_argument("--backbone", default="vits14", choices=["vits14", "vitb14"])
    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no_ema", action="store_false", dest="ema")
    ap.add_argument("--n_repeats", type=int, default=3,
                    help="random splits (default 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tukey_beta", type=float, default=0.5,
                    help="Tukey beta for transductive methods")
    ap.add_argument("--maha_shrink", type=float, default=0.3)
    ap.add_argument("--use_combined_mean", action="store_true", default=True)
    # Mixup values to search
    ap.add_argument("--mixup_values", type=str, default="0.0,0.2,0.5,0.8",
                    help="comma-separated mixup alpha values")
    # LoRA dropout values to search
    ap.add_argument("--dropout_values", type=str, default="0.0,0.1,0.2,0.3",
                    help="comma-separated lora_dropout values")
    ap.add_argument("--out_json", default="../reg_search_results.json")
    ap.add_argument("--force", action="store_true",
                    help="ignore checkpoint and start from scratch")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mixup_list = [float(x) for x in args.mixup_values.split(",")]
    dropout_list = [float(x) for x in args.dropout_values.split(",")]

    print(f"[info] Regularization search on r={BASE_LORA_R}")
    print(f"[info] Mixup values: {mixup_list}")
    print(f"[info] LoRA dropout values: {dropout_list}")
    print(f"[info] Total configs: {len(mixup_list)} × {len(dropout_list)} = "
          f"{len(mixup_list) * len(dropout_list)}")
    print(f"[info] PT-MAP params: lambda_s=10, n_iter=20, "
          f"sinkhorn_iter=10, tukey_beta={args.tukey_beta}")

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
        completed = ckpt.get("completed", [])
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

    total_configs = args.n_repeats * len(mixup_list) * len(dropout_list)
    done_count = len(completed)

    for rep, (train_idx, holdout_idx) in enumerate(
            splitter.split(np.arange(N), all_labels)):
        for mixup_alpha in mixup_list:
            for lora_dropout in dropout_list:
                task_key = f"rep{rep}_m{mixup_alpha}_d{lora_dropout}"
                if task_key in completed:
                    continue

                train_args = build_args(
                    backbone=args.backbone, ema=args.ema,
                    lora_r=BASE_LORA_R, lora_alpha=BASE_LORA_ALPHA,
                    mixup=mixup_alpha, lora_dropout=lora_dropout,
                    seed=args.seed,
                )

                done_count += 1
                t0 = time.time()
                print(f"\n{'='*60}")
                print(f"  [{done_count}/{total_configs}] REPEAT {rep+1}/{args.n_repeats}, "
                      f"mixup={mixup_alpha}, lora_dropout={lora_dropout}")
                print(f"{'='*60}")

                with tempfile.TemporaryDirectory(prefix=f"reg_m{mixup_alpha}_d{lora_dropout}_") as tmpdir:
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
                    print(f"  Training LoRA r={BASE_LORA_R}, mixup={mixup_alpha}, "
                          f"lora_dropout={lora_dropout} on {len(st_paths)} images ...")
                    sup_feats, sup_labels_arr, test_feats, head_probs, test_names, n_params = \
                        train_and_extract(st_paths, st_labels, holdout_paths,
                                          device, train_args)

                    print(f"  trainable params: {n_params:,}  "
                          f"(params/sample = {n_params/len(st_paths):.0f}:1)")

                    # True labels for holdout
                    y_true = np.array([gt_map[n] for n in test_names])

                    # Evaluate ALL methods (same as cv5_improve / lora_rank_search)
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

                print(f"  >>> mixup={mixup_alpha}, dropout={lora_dropout}: "
                      f"best={best_method} F1={best_f1:.4f}")
                print(f"  >>> Time: {timedelta(seconds=int(elapsed))}")

                # Save checkpoint
                result_entry = {
                    "repeat": rep + 1,
                    "mixup": mixup_alpha,
                    "lora_dropout": lora_dropout,
                    "lora_r": BASE_LORA_R,
                    "lora_alpha": BASE_LORA_ALPHA,
                    "n_params": n_params,
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
    print(f"  REGULARIZATION SEARCH RESULTS")
    print(f"  {args.n_repeats} repeats × {len(mixup_list)} mixup × {len(dropout_list)} dropout")
    print(f"  Base: r={BASE_LORA_R}, alpha={BASE_LORA_ALPHA}")
    print(f"  (same evaluation as cv5_improve: ALL methods, pick best per split)")
    print(f"{'='*70}")

    # Aggregate by (mixup, dropout) combo
    combo_stats = {}
    for mixup_alpha in mixup_list:
        for lora_dropout in dropout_list:
            entries = [r for r in all_results
                       if r["mixup"] == mixup_alpha and r["lora_dropout"] == lora_dropout]
            if not entries:
                continue

            best_f1s = [r["best_f1"] for r in entries]

            # Per-method average across repeats
            method_avg = {}
            for method in ["simpleshot", "mahalanobis", "lp", "laplacianshot",
                           "ptmap", "tim", "alpha_tim", "head"]:
                f1s = [r["all_methods"][method]["f1"] for r in entries
                       if method in r.get("all_methods", {})]
                baccs = [r["all_methods"][method]["bacc"] for r in entries
                         if method in r.get("all_methods", {})]
                if f1s:
                    method_avg[method] = {
                        "mean_f1": float(np.mean(f1s)),
                        "std_f1": float(np.std(f1s)) if len(f1s) > 1 else 0.0,
                        "mean_bacc": float(np.mean(baccs)),
                    }

            key = (mixup_alpha, lora_dropout)
            combo_stats[key] = {
                "n_repeats": len(entries),
                "mean_best_f1": float(np.mean(best_f1s)),
                "std_best_f1": float(np.std(best_f1s)) if len(best_f1s) > 1 else 0.0,
                "method_avg": method_avg,
            }

    # Print summary table
    # Reference: r=16, mixup=0, dropout=0.1 from lora_rank_search
    ref_key = (0.0, 0.1)
    ref_f1 = combo_stats.get(ref_key, {}).get("mean_best_f1", None)

    print(f"\n  {'mixup':>5}  {'dropout':>7}  {'Best F1':>8}  {'±std':>6}  "
          f"{'vs ref':>7}  {'Best method':>12}  {'TIM':>8}  {'SS':>8}  {'Head':>8}")
    print(f"  {'-'*80}")

    # Sort by mean_best_f1 descending
    sorted_combos = sorted(combo_stats.items(), key=lambda x: -x[1]["mean_best_f1"])

    for (mixup_alpha, lora_dropout), s in sorted_combos:
        tim_f1 = s["method_avg"].get("tim", {}).get("mean_f1", 0)
        ss_f1 = s["method_avg"].get("simpleshot", {}).get("mean_f1", 0)
        head_f1 = s["method_avg"].get("head", {}).get("mean_f1", 0)

        # Find best method name
        best_m = max(s["method_avg"].items(), key=lambda x: x[1]["mean_f1"])[0]

        vs_ref = ""
        if ref_f1 is not None:
            diff = s["mean_best_f1"] - ref_f1
            vs_ref = f"{diff:+.4f}"

        print(f"  {mixup_alpha:>5.1f}  {lora_dropout:>7.1f}  "
              f"{s['mean_best_f1']:>8.4f}  {s['std_best_f1']:>6.4f}  "
              f"{vs_ref:>7}  {best_m:>12}  "
              f"{tim_f1:>8.4f}  {ss_f1:>8.4f}  {head_f1:>8.4f}")

    # Find best combo
    best_key = max(combo_stats, key=lambda k: combo_stats[k]["mean_best_f1"])
    best_s = combo_stats[best_key]
    print(f"\n  >>> BEST COMBO: mixup={best_key[0]}, lora_dropout={best_key[1]}")
    print(f"      mean best-F1 = {best_s['mean_best_f1']:.4f} "
          f"± {best_s['std_best_f1']:.4f}")

    # ── Sensitivity analysis ──
    print(f"\n  --- Mixup sensitivity (marginal mean F1) ---")
    for mixup_alpha in mixup_list:
        f1s = [combo_stats[(mixup_alpha, d)]["mean_best_f1"]
               for d in dropout_list if (mixup_alpha, d) in combo_stats]
        if f1s:
            print(f"    mixup={mixup_alpha:.1f} → mean F1 = {np.mean(f1s):.4f}")

    print(f"\n  --- LoRA dropout sensitivity (marginal mean F1) ---")
    for lora_dropout in dropout_list:
        f1s = [combo_stats[(m, lora_dropout)]["mean_best_f1"]
               for m in mixup_list if (m, lora_dropout) in combo_stats]
        if f1s:
            print(f"    dropout={lora_dropout:.1f} → mean F1 = {np.mean(f1s):.4f}")

    # Save final results
    final = {
        "base_lora_r": BASE_LORA_R,
        "base_lora_alpha": BASE_LORA_ALPHA,
        "mixup_values": mixup_list,
        "dropout_values": dropout_list,
        "n_repeats": args.n_repeats,
        "backbone": args.backbone,
        "ema": args.ema,
        "tukey_beta": args.tukey_beta,
        "maha_shrink": args.maha_shrink,
        "use_combined_mean": args.use_combined_mean,
        "combo_stats": {f"mixup{k[0]}_drop{k[1]}": v for k, v in combo_stats.items()},
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
