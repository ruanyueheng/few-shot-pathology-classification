"""5-Fold Cross-Validation improvement experiments with checkpoint/resume.

Tests these configurations (each with 5-fold CV):
  A. baseline    : vits14 + EMA + tukey_beta=0.5        (current best)
  B. +vitb14     : vitb14 + EMA + tukey_beta=0.5        (bigger backbone)
  C. +supcon     : vits14 + EMA + SupCon + tukey_beta=0.5
  D. +hed_aug    : vits14 + EMA + HED stain aug + tukey_beta=0.5
  E. best_combo  : vitb14 + EMA + SupCon + HED + tukey_beta=0.5

For each config, runs 5-fold stratified CV (200 train / 50 holdout per fold),
trains LoRA, runs ALL 8 inference methods, and records macro-F1.
At the end prints a comparison table so you can see which upgrade matters most.

CHECKPOINT / RESUME:
  After each fold completes, saves progress to cv5_resume_state.json.
  If the script is interrupted, just re-run the same command -- it will
  skip completed folds and resume from where it stopped.

Usage:
  cd src
  python cv5_improve.py                           # all 5 configs, 5-fold CV
  python cv5_improve.py --configs baseline vitb14 # only 2 configs
  python cv5_improve.py --folds 3                 # 3-fold instead of 5
  python cv5_improve.py --force                   # ignore checkpoint, start fresh
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from datetime import timedelta
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report

from data import list_train_samples, list_test_samples, CLASS_NAMES
from train_lora import train_one_run, seed_all
from self_train import extract_features, predict_probs
from transductive import (
    simpleshot, laplacianshot, mahalanobis, tim,
    label_propagation, pt_map, alpha_tim,
)


# ── Configuration definitions ──────────────────────────────────────

CONFIGS = {
    "baseline": {
        "backbone": "vits14", "ema": True, "tukey_beta": 0.5,
        "supcon_weight": 0.0, "hed_aug": False,
        "lora_r": 32, "lora_alpha": 64, "batch_size": 32,
        "label": "A. baseline (vits14+EMA)",
    },
    "vitb14": {
        "backbone": "vitb14", "ema": True, "tukey_beta": 0.5,
        "supcon_weight": 0.0, "hed_aug": False,
        "lora_r": 32, "lora_alpha": 64, "batch_size": 24,
        "label": "B. +ViT-B/14 backbone",
    },
    "supcon": {
        "backbone": "vits14", "ema": True, "tukey_beta": 0.5,
        "supcon_weight": 0.5, "hed_aug": False,
        "lora_r": 32, "lora_alpha": 64, "batch_size": 32,
        "label": "C. +SupCon loss (0.5)",
    },
    "hed_aug": {
        "backbone": "vits14", "ema": True, "tukey_beta": 0.5,
        "supcon_weight": 0.0, "hed_aug": True,
        "lora_r": 32, "lora_alpha": 64, "batch_size": 32,
        "label": "D. +HED stain aug",
    },
    "best_combo": {
        "backbone": "vitb14", "ema": True, "tukey_beta": 0.5,
        "supcon_weight": 0.5, "hed_aug": True,
        "lora_r": 32, "lora_alpha": 64, "batch_size": 24,
        "label": "E. best combo (B+C+D)",
    },
}

RESUME_FILE = Path(__file__).parent.parent / "cv5_resume_state.json"


# ── Transductive methods ──────────────────────────────────────────

def run_all_methods(sup_feats, sup_labels, test_feats, num_classes=5,
                    tukey_beta=0.5, maha_shrink=0.3,
                    use_combined_mean=False):
    """Run ALL transductive methods, return dict of {name: {pred, probs?}}."""
    results = {}

    print("    [1/8] SimpleShot ...")
    pred_ss, probs_ss = simpleshot(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        return_probs=True,
    )
    results["simpleshot"] = {"pred": pred_ss, "probs": probs_ss}

    print("    [2/8] Mahalanobis ...")
    pred_maha, probs_maha = mahalanobis(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        shrink=maha_shrink, return_probs=True,
    )
    results["mahalanobis"] = {"pred": pred_maha, "probs": probs_maha}

    print("    [3/8] Label Propagation ...")
    pred_lp = label_propagation(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        knn=10, alpha=0.7, sigma=1.0,
    )
    results["lp"] = {"pred": pred_lp}

    print("    [4/8] LaplacianShot ...")
    pred_lap, probs_lap = laplacianshot(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        knn=5, lam=1.0, n_iter=20, sigma=1.0, return_probs=True,
    )
    results["laplacianshot"] = {"pred": pred_lap, "probs": probs_lap}

    print("    [5/8] PT-MAP ...")
    # pt_map() does NOT accept use_combined_mean (hardcoded internally)
    pred_ptmap, probs_ptmap = pt_map(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta,
        n_iter=20, lambda_s=10.0, use_sinkhorn=True, sinkhorn_iter=10,
        return_probs=True,
    )
    results["ptmap"] = {"pred": pred_ptmap, "probs": probs_ptmap}

    print("    [6/8] TIM ...")
    pred_tim = tim(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        n_iter=1000, lr=1e-4, temperature=15.0,
        lambda_marg=1.0, lambda_cond=0.1,
    )
    results["tim"] = {"pred": pred_tim}

    print("    [7/8] alpha-TIM (a=2.0) ...")
    pred_atim = alpha_tim(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=True,
        n_iter=1000, lr=1e-4, temperature=15.0,
        alpha=2.0, lambda_marg=1.0, lambda_cond=0.1,
    )
    results["alpha_tim"] = {"pred": pred_atim}

    return results


# ── Build args namespace for train_one_run ────────────────────────

def build_args_from_config(cfg: dict, seed: int = 42):
    """Build a namespace object from a config dict, matching train_one_run's expectations."""
    args = argparse.Namespace(
        backbone=cfg["backbone"],
        image_size=224,
        epochs=30,
        batch_size=cfg["batch_size"],
        lr=5e-4,
        lr_head=1e-3,
        weight_decay=0.05,
        label_smoothing=0.1,
        grad_clip=1.0,
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=0.1,
        head_dropout=0.1,
        ema=cfg["ema"],
        ema_decay=0.95,
        hed_aug=cfg["hed_aug"],
        hed_sigma=0.02,
        hed_bias=0.01,
        lora_mlp=False,
        dora=False,
        lora_init="default",
        supcon_weight=cfg["supcon_weight"],
        supcon_temp=0.07,
        mixup=0.0,
        cutmix=0.0,
        num_workers=2,
        seed=seed,
        verbose=False,
    )
    return args


# ── Checkpoint / Resume helpers ───────────────────────────────────

def load_resume_state() -> dict:
    """Load resume state from disk. Returns {} if not found."""
    if RESUME_FILE.exists():
        with open(RESUME_FILE, "r") as f:
            return json.load(f)
    return {}


def save_resume_state(state: dict):
    """Save resume state to disk."""
    with open(RESUME_FILE, "w") as f:
        json.dump(state, f, indent=2)


def make_fold_key(cfg_name: str, fold_idx: int) -> str:
    """Unique key for a (config, fold) pair in resume state."""
    return f"{cfg_name}__fold_{fold_idx}"


# ── Single fold runner ─────────────────────────────────────────────

def run_one_fold(cfg_name, cfg, fold_idx, train_idx, holdout_idx,
                 all_paths, all_labels, seed, device,
                 tukey_beta, maha_shrink, use_combined_mean):
    """Train + evaluate one fold. Returns dict with all method scores."""
    t0 = time.time()
    print(f"\n  --- fold {fold_idx+1}: "
          f"{len(train_idx)} train / {len(holdout_idx)} holdout ---")

    with tempfile.TemporaryDirectory(prefix=f"cv5_{cfg_name}_f{fold_idx}_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Build train dir (with class subdirs)
        fold_train_dir = tmpdir / "train"
        for c in CLASS_NAMES:
            (fold_train_dir / c).mkdir(parents=True, exist_ok=True)
        for i in train_idx:
            p = all_paths[i]
            l = all_labels[i]
            shutil.copy2(p, fold_train_dir / CLASS_NAMES[l] / Path(p).name)

        # Build holdout dir (flat)
        fold_holdout_dir = tmpdir / "holdout"
        fold_holdout_dir.mkdir(parents=True, exist_ok=True)
        for i in holdout_idx:
            shutil.copy2(all_paths[i], fold_holdout_dir / Path(all_paths[i]).name)

        # Ground truth for evaluation
        gt_map = {}
        for i in holdout_idx:
            gt_map[Path(all_paths[i]).name] = int(all_labels[i])

        # Build args from config
        args = build_args_from_config(cfg, seed=seed)

        # Train LoRA on this fold
        st_paths, st_labels = list_train_samples(str(fold_train_dir))
        full_idx = np.arange(len(st_paths))
        model, _, _ = train_one_run(
            st_paths, st_labels, full_idx, full_idx,
            args, device, run_id=fold_idx, refit_all=True,
        )

        # Extract features
        sup_feats, sup_lab = extract_features(
            model, st_paths, args.image_size, args, device,
            labels=st_labels,
        )
        holdout_paths_list = list_test_samples(str(fold_holdout_dir))
        test_feats, test_names = extract_features(
            model, holdout_paths_list, args.image_size, args, device,
            labels=None,
        )

        # Head prediction
        head_probs, _ = predict_probs(
            model, holdout_paths_list, args.image_size, args, device,
        )
        head_pred = head_probs.argmax(axis=1)

        # All transductive methods
        trans_results = run_all_methods(
            sup_feats, sup_lab, test_feats, num_classes=5,
            tukey_beta=tukey_beta, maha_shrink=maha_shrink,
            use_combined_mean=use_combined_mean,
        )

        # Evaluate all methods
        y_true = np.array([gt_map[n] for n in test_names])

        method_scores = {}
        for name, res in trans_results.items():
            pred = res["pred"]
            method_scores[name] = {
                "f1": float(f1_score(y_true, pred, average="macro")),
                "bacc": float(balanced_accuracy_score(y_true, pred)),
            }
        method_scores["head"] = {
            "f1": float(f1_score(y_true, head_pred, average="macro")),
            "bacc": float(balanced_accuracy_score(y_true, head_pred)),
        }

        # Find best method
        best_method = max(method_scores, key=lambda k: method_scores[k]["f1"])
        best_f1 = method_scores[best_method]["f1"]

        elapsed = time.time() - t0
        print(f"  >>> fold {fold_idx+1} best: {best_method} "
              f"F1={best_f1:.4f}  ({timedelta(seconds=int(elapsed))})")

        del model
        torch.cuda.empty_cache()

    return {
        "fold": fold_idx,
        "best_method": best_method,
        "best_f1": best_f1,
        "all_methods": method_scores,
        "elapsed_sec": elapsed,
    }


# ── Per-config runner with resume ─────────────────────────────────

def run_config_with_resume(cfg_name, cfg, all_paths, all_labels,
                           n_folds, seed, device,
                           tukey_beta, maha_shrink, use_combined_mean,
                           resume_state, verbose):
    """Run one config across all folds, skipping completed ones via resume."""
    print(f"\n{'#'*70}")
    print(f"  CONFIG: {cfg['label']}")
    print(f"  backbone={cfg['backbone']}  ema={cfg['ema']}  "
          f"supcon={cfg['supcon_weight']}  hed_aug={cfg['hed_aug']}")
    print(f"  {n_folds}-fold CV x {int(len(all_paths) * (n_folds-1)/n_folds)} train / "
          f"{int(len(all_paths) / n_folds)} holdout")
    print(f"{'#'*70}")

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (train_idx, holdout_idx) in enumerate(
            skf.split(np.arange(len(all_paths)), all_labels)):

        key = make_fold_key(cfg_name, fold_idx)

        # Check resume: skip if already done
        if key in resume_state.get("completed_folds", {}):
            saved = resume_state["completed_folds"][key]
            print(f"\n  [resume] skipping fold {fold_idx+1} (already done, "
                  f"best={saved['best_method']} F1={saved['best_f1']:.4f})")
            fold_results.append(saved)
            continue

        # Run this fold
        result = run_one_fold(
            cfg_name, cfg, fold_idx, train_idx, holdout_idx,
            all_paths, all_labels, seed, device,
            tukey_beta, maha_shrink, use_combined_mean,
        )

        fold_results.append(result)

        # Save checkpoint after each fold
        if "completed_folds" not in resume_state:
            resume_state["completed_folds"] = {}
        resume_state["completed_folds"][key] = result
        save_resume_state(resume_state)
        print(f"  [checkpoint] saved fold {fold_idx+1} progress")

    return fold_results


# ── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="5-Fold CV improvement experiments with checkpoint/resume")
    ap.add_argument("--train_dir", default="../train_few_shot")
    ap.add_argument("--folds", type=int, default=5,
                    help="number of CV folds (default 5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", nargs="+", default=None,
                    choices=list(CONFIGS.keys()),
                    help="which configs to run (default: all)")
    ap.add_argument("--tukey_beta", type=float, default=0.5)
    ap.add_argument("--maha_shrink", type=float, default=0.3)
    ap.add_argument("--use_combined_mean", action="store_true")
    ap.add_argument("--out_json", default="../cv5_improve_results.json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="ignore checkpoint, start from scratch")
    args = ap.parse_args()

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load all 250 images
    all_paths, all_labels = list_train_samples(args.train_dir)
    all_labels = np.array(all_labels)
    print(f"[info] {len(all_paths)} samples, device={device}")
    print(f"[info] per-class: {np.bincount(all_labels).tolist()}")

    # Which configs to run
    config_names = args.configs or list(CONFIGS.keys())
    print(f"[info] running {len(config_names)} configs: {config_names}")
    print(f"[info] {args.folds}-fold CV each\n")

    # Load or init resume state
    if args.force:
        resume_state = {}
        print("[info] --force: ignoring checkpoint, starting fresh\n")
    else:
        resume_state = load_resume_state()
        if resume_state:
            n_done = len(resume_state.get("completed_folds", {}))
            print(f"[info] checkpoint found: {n_done} folds already completed")
            print(f"[info] will skip completed folds and resume from there\n")
        else:
            print("[info] no checkpoint found, starting fresh\n")

    # Record metadata in resume state
    resume_state["meta"] = {
        "folds": args.folds,
        "seed": args.seed,
        "tukey_beta": args.tukey_beta,
        "maha_shrink": args.maha_shrink,
        "use_combined_mean": args.use_combined_mean,
        "configs": config_names,
    }

    # Run each config
    all_results = {}
    total_t0 = time.time()

    for cfg_name in config_names:
        cfg = CONFIGS[cfg_name]
        t0 = time.time()

        fold_results = run_config_with_resume(
            cfg_name, cfg, all_paths, all_labels,
            n_folds=args.folds, seed=args.seed, device=device,
            tukey_beta=args.tukey_beta, maha_shrink=args.maha_shrink,
            use_combined_mean=args.use_combined_mean,
            resume_state=resume_state, verbose=args.verbose,
        )

        elapsed = time.time() - t0

        # Aggregate across folds
        f1s = [r["best_f1"] for r in fold_results]

        # Per-method average across folds
        method_f1s = {}
        method_baccs = {}
        for r in fold_results:
            for m, scores in r["all_methods"].items():
                method_f1s.setdefault(m, []).append(scores["f1"])
                method_baccs.setdefault(m, []).append(scores["bacc"])

        all_results[cfg_name] = {
            "label": cfg["label"],
            "config": {k: v for k, v in cfg.items() if k != "label"},
            "n_folds": args.folds,
            "mean_best_f1": float(np.mean(f1s)),
            "std_best_f1": float(np.std(f1s)),
            "per_fold": fold_results,
            "method_avg_f1": {
                m: {"mean": float(np.mean(fs)), "std": float(np.std(fs))}
                for m, fs in method_f1s.items()
            },
            "method_avg_bacc": {
                m: {"mean": float(np.mean(bs)), "std": float(np.std(bs))}
                for m, bs in method_baccs.items()
            },
            "elapsed_sec": elapsed,
        }

        print(f"\n  >>> {cfg['label']}: mean F1={np.mean(f1s):.4f} "
              f"+/- {np.std(f1s):.4f}  ({timedelta(seconds=int(elapsed))})")

    # ── Final comparison table ─────────────────────────────────────
    total_elapsed = time.time() - total_t0
    print(f"\n{'='*75}")
    print(f"  5-FOLD CV IMPROVEMENT RESULTS")
    print(f"  {args.folds}-fold CV per config, "
          f"total time: {timedelta(seconds=int(total_elapsed))}")
    print(f"{'='*75}")

    # Sort by mean_best_f1 descending
    sorted_configs = sorted(all_results.items(),
                            key=lambda x: -x[1]["mean_best_f1"])

    baseline_f1 = all_results.get("baseline", {}).get("mean_best_f1", None)

    print(f"\n  {'Config':<35} {'mean F1':>8} {'+/-':>8} {'vs base':>10}")
    print(f"  {'-'*65}")
    for name, res in sorted_configs:
        vs_base = ""
        if baseline_f1 is not None and name != "baseline":
            diff = res["mean_best_f1"] - baseline_f1
            sign = "+" if diff >= 0 else ""
            vs_base = f"{sign}{diff:.4f}"
        print(f"  {res['label']:<35} {res['mean_best_f1']:>8.4f} "
              f"{res['std_best_f1']:>8.4f} {vs_base:>10}")

    # Per-method breakdown for each config
    for name, res in sorted_configs:
        print(f"\n  {res['label']} — per-method average macro-F1:")
        method_sorted = sorted(res["method_avg_f1"].items(),
                               key=lambda x: -x[1]["mean"])
        print(f"    {'Method':<20} {'mean F1':>8} {'+/-':>8} {'mean bacc':>10}")
        print(f"    {'-'*48}")
        for m, stats in method_sorted:
            bacc = res["method_avg_bacc"][m]["mean"]
            print(f"    {m:<20} {stats['mean']:>8.4f} {stats['std']:>8.4f} "
                  f"{bacc:>10.4f}")

    # Best overall config + method
    best_cfg_name = sorted_configs[0][0]
    best_res = sorted_configs[0][1]
    best_method_name = max(best_res["method_avg_f1"],
                           key=lambda k: best_res["method_avg_f1"][k]["mean"])
    best_method_f1 = best_res["method_avg_f1"][best_method_name]

    print(f"\n  >>> BEST OVERALL: {best_res['label']} + {best_method_name}")
    print(f"      mean macro-F1 = {best_method_f1['mean']:.4f} "
          f"+/- {best_method_f1['std']:.4f}")

    # Save results
    output = {
        "experiment": "cv5_improve",
        "n_folds": args.folds,
        "seed": args.seed,
        "tukey_beta": args.tukey_beta,
        "maha_shrink": args.maha_shrink,
        "use_combined_mean": args.use_combined_mean,
        "configs": all_results,
        "total_elapsed_sec": total_elapsed,
    }
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[ok] saved results -> {args.out_json}")

    # Clean up checkpoint on successful completion
    if RESUME_FILE.exists():
        RESUME_FILE.unlink()
        print(f"[ok] checkpoint cleaned up ({RESUME_FILE.name})")


if __name__ == "__main__":
    main()
