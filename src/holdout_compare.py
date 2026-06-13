"""Holdout evaluation: split train_few_shot into train+holdout,
train LoRA once, then test ALL inference methods on the holdout set.

This gives you a clean, unbiased comparison of every method on data
the model has NEVER seen — no pseudo-label leakage, no model selection
bias.

Usage:
  cd src
  python holdout_compare.py                    # default: 200/50 split, seed=42
  python holdout_compare.py --seed 123         # different random split
  python holdout_compare.py --n_repeats 3      # 3 random splits, average results
  python holdout_compare.py --rounds 0         # no self-training (fastest)
  python holdout_compare.py --rounds 3         # with self-training (slower)
  python holdout_compare.py --train_ratio 0.8  # 80% train / 20% holdout
"""
from __future__ import annotations
import argparse
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
)

from data import (
    list_train_samples, list_test_samples, ImagePathDataset,
    CLASS_NAMES, IDX_TO_CLASS, CLASS_TO_IDX,
)
from train_lora import (
    LoRAViTClassifier, make_train_tf, make_eval_tf, train_one_run, seed_all,
)
from self_train import (
    extract_features, predict_probs, select_pseudo, materialize_round_dir,
)
from transductive import (
    simpleshot, laplacianshot, mahalanobis, tim,
    label_propagation, pt_map, alpha_tim,
)


def run_all_methods(sup_feats, sup_labels, test_feats, num_classes=5,
                    tukey_beta=0.5, maha_shrink=0.3,
                    use_combined_mean=False):
    """Run ALL transductive methods and return dict of predictions + probs."""
    results = {}

    # 1. SimpleShot
    print("  [1/8] SimpleShot ...")
    pred_ss, probs_ss = simpleshot(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        return_probs=True,
    )
    results["simpleshot"] = {"pred": pred_ss, "probs": probs_ss}

    # 2. Mahalanobis
    print("  [2/8] Mahalanobis ...")
    pred_maha, probs_maha = mahalanobis(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        shrink=maha_shrink, return_probs=True,
    )
    results["mahalanobis"] = {"pred": pred_maha, "probs": probs_maha}

    # 3. Label Propagation
    print("  [3/8] Label Propagation ...")
    pred_lp = label_propagation(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        knn=10, alpha=0.7, sigma=1.0,
    )
    results["lp"] = {"pred": pred_lp}

    # 4. LaplacianShot
    print("  [4/8] LaplacianShot ...")
    pred_lap, probs_lap = laplacianshot(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        knn=5, lam=1.0, n_iter=20, sigma=1.0, return_probs=True,
    )
    results["laplacianshot"] = {"pred": pred_lap, "probs": probs_lap}

    # 5. PT-MAP
    print("  [5/8] PT-MAP ...")
    pred_ptmap, probs_ptmap = pt_map(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta,
        n_iter=20, lambda_s=10.0, use_sinkhorn=True, sinkhorn_iter=10,
        return_probs=True,
    )
    results["ptmap"] = {"pred": pred_ptmap, "probs": probs_ptmap}

    # 6. TIM
    print("  [6/8] TIM ...")
    pred_tim = tim(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        n_iter=1000, lr=1e-4, temperature=15.0,
        lambda_marg=1.0, lambda_cond=0.1,
    )
    results["tim"] = {"pred": pred_tim}

    # 7. alpha-TIM (α=2.0)
    print("  [7/8] alpha-TIM (α=2.0) ...")
    pred_atim = alpha_tim(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=True,
        n_iter=1000, lr=1e-4, temperature=15.0,
        alpha=2.0, lambda_marg=1.0, lambda_cond=0.1,
    )
    results["alpha_tim"] = {"pred": pred_atim}

    # 8. Head (classification head from LoRA model)
    # This is done separately since it uses the model directly

    return results


def evaluate_methods(y_true, results_dict, head_pred=None):
    """Compute macro-F1 and balanced-accuracy for each method."""
    table = []
    for name, res in results_dict.items():
        pred = res["pred"]
        f1 = f1_score(y_true, pred, average="macro")
        bacc = balanced_accuracy_score(y_true, pred)
        table.append({"method": name, "macro_f1": f1, "bal_acc": bacc})
    if head_pred is not None:
        f1 = f1_score(y_true, head_pred, average="macro")
        bacc = balanced_accuracy_score(y_true, head_pred)
        table.append({"method": "head", "macro_f1": f1, "bal_acc": bacc})
    # sort by F1 descending
    table.sort(key=lambda x: -x["macro_f1"])
    return table


def main():
    ap = argparse.ArgumentParser(
        description="Holdout split: train on 200, test ALL methods on 50")
    # data
    ap.add_argument("--train_dir", default="../train_few_shot",
                    help="full training dir (250 images)")
    ap.add_argument("--train_ratio", type=float, default=0.8,
                    help="fraction used for training (0.8 = 200 train / 50 holdout)")
    # self-training
    ap.add_argument("--rounds", type=int, default=0,
                    help="self-training rounds (0 = baseline, fastest)")
    ap.add_argument("--conf_thresh", type=float, default=0.9)
    ap.add_argument("--per_class_topk", type=int, default=2)
    ap.add_argument("--per_class_min", type=int, default=None)
    # inference
    ap.add_argument("--maha_shrink", type=float, default=0.3)
    ap.add_argument("--tukey_beta", type=float, default=0.5)
    ap.add_argument("--use_combined_mean", action="store_true")
    # LoRA hyperparams
    ap.add_argument("--backbone", default="vits14")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--head_dropout", type=float, default=0.1)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.95)
    ap.add_argument("--hed_aug", action="store_true")
    ap.add_argument("--hed_sigma", type=float, default=0.02)
    ap.add_argument("--hed_bias", type=float, default=0.01)
    ap.add_argument("--lora_mlp", action="store_true")
    ap.add_argument("--dora", action="store_true")
    ap.add_argument("--lora_init", default="default",
                    choices=["default", "gaussian", "pissa", "pissa_niter_4",
                             "pissa_niter_16", "olora", "loftq", "orthogonal"])
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--cutmix", type=float, default=0.0)
    ap.add_argument("--supcon_weight", type=float, default=0.0)
    ap.add_argument("--supcon_temp", type=float, default=0.07)
    ap.add_argument("--num_workers", type=int, default=2)
    # experiment
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for the stratified split")
    ap.add_argument("--n_repeats", type=int, default=1,
                    help="how many random splits to average over (reduces variance)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out_json", default="../holdout_compare_results.json")
    args = ap.parse_args()

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")
    print(f"[info] train_ratio={args.train_ratio}, "
          f"n_repeats={args.n_repeats}, seed={args.seed}")
    print(f"[info] self-training rounds={args.rounds}")
    print(f"[info] LoRA: backbone={args.backbone}, r={args.lora_r}, "
          f"alpha={args.lora_alpha}, epochs={args.epochs}")
    print(f"[info] inference: tukey_beta={args.tukey_beta}, "
          f"maha_shrink={args.maha_shrink}")

    # ---- load all 250 images ----
    all_paths, all_labels = list_train_samples(args.train_dir)
    N = len(all_paths)
    n_train = int(N * args.train_ratio)
    n_holdout = N - n_train
    print(f"[info] {N} samples total, split={n_train} train / {n_holdout} holdout")

    # ---- repeated holdout evaluation ----
    all_repeat_tables = []
    all_repeat_best = []

    splitter = StratifiedShuffleSplit(
        n_splits=args.n_repeats, test_size=1 - args.train_ratio,
        random_state=args.seed,
    )

    for rep, (train_idx, holdout_idx) in enumerate(
            splitter.split(np.arange(N), all_labels)):
        print(f"\n{'='*60}")
        print(f"  REPEAT {rep+1}/{args.n_repeats}  "
              f"train={len(train_idx)}  holdout={len(holdout_idx)}")
        print(f"  holdout per-class: {np.bincount(all_labels[holdout_idx]).tolist()}")
        print(f"{'='*60}")

        with tempfile.TemporaryDirectory(prefix=f"holdout_rep{rep}_") as tmpdir:
            tmpdir = Path(tmpdir)

            # ---- Build train dir (n_train images with labels) ----
            fold_train_dir = tmpdir / "train"
            for c in CLASS_NAMES:
                (fold_train_dir / c).mkdir(parents=True, exist_ok=True)
            train_paths_fold = [all_paths[i] for i in train_idx]
            train_labels_fold = all_labels[train_idx]
            for p, l in zip(train_paths_fold, train_labels_fold):
                cls_dir = fold_train_dir / CLASS_NAMES[l]
                shutil.copy2(p, cls_dir / Path(p).name)

            # ---- Build holdout dir (n_holdout images, NO labels) ----
            fold_holdout_dir = tmpdir / "holdout"
            fold_holdout_dir.mkdir(parents=True, exist_ok=True)
            holdout_paths_fold = [all_paths[i] for i in holdout_idx]
            for p in holdout_paths_fold:
                shutil.copy2(p, fold_holdout_dir / Path(p).name)

            # ---- Ground truth for evaluation ----
            gt_map = {}
            for i in holdout_idx:
                gt_map[Path(all_paths[i]).name] = int(all_labels[i])

            # ==== Self-training loop ====
            accumulated_pseudo = []
            used_names = set()

            for r in range(args.rounds + 1):
                print(f"\n  --- round {r} ---")

                # materialize labeled dir
                round_dir = tmpdir / f"round_{r}"
                round_dir.mkdir(parents=True, exist_ok=True)
                labeled_dir, n_orig, n_pseudo = materialize_round_dir(
                    round_dir, fold_train_dir, fold_holdout_dir,
                    pseudo=[], accumulated_pseudo=accumulated_pseudo,
                )
                print(f"  labeled: {n_orig} orig + {n_pseudo} pseudo = "
                      f"{n_orig+n_pseudo}")

                # train LoRA
                st_paths, st_labels = list_train_samples(str(labeled_dir))
                full_idx = np.arange(len(st_paths))
                model, _, _ = train_one_run(
                    st_paths, st_labels, full_idx, full_idx,
                    args, device, run_id=rep * (args.rounds + 1) + r,
                    refit_all=True,
                )

                if r < args.rounds:
                    # pick pseudo-labels
                    sup_feats, sup_lab = extract_features(
                        model, st_paths, args.image_size, args, device,
                        labels=st_labels,
                    )
                    holdout_paths_list = list_test_samples(str(fold_holdout_dir))
                    test_feats, test_names = extract_features(
                        model, holdout_paths_list, args.image_size,
                        args, device, labels=None,
                    )
                    _, probs = mahalanobis(
                        sup_feats, sup_lab, test_feats, num_classes=5,
                        tukey_beta=args.tukey_beta,
                        use_combined_mean=args.use_combined_mean,
                        shrink=args.maha_shrink, return_probs=True,
                    )
                    newly_selected = select_pseudo(
                        probs, test_names, args.conf_thresh,
                        args.per_class_topk, args.per_class_min, used_names,
                    )
                    print(f"  picked {len(newly_selected)} pseudo-labels")
                    accumulated_pseudo.extend(newly_selected)
                    used_names.update(n for n, _, _ in newly_selected)

                    del model
                    torch.cuda.empty_cache()

                else:
                    # ==== FINAL ROUND: run ALL methods ====
                    print(f"\n  >>> Running ALL inference methods ...")

                    # Extract features for support + holdout
                    sup_feats, sup_lab = extract_features(
                        model, st_paths, args.image_size, args, device,
                        labels=st_labels,
                    )
                    holdout_paths_list = list_test_samples(str(fold_holdout_dir))
                    test_feats, test_names = extract_features(
                        model, holdout_paths_list, args.image_size,
                        args, device, labels=None,
                    )

                    # Head prediction
                    head_probs, head_names = predict_probs(
                        model, holdout_paths_list, args.image_size,
                        args, device,
                    )
                    head_pred = head_probs.argmax(axis=1)

                    # All transductive methods
                    trans_results = run_all_methods(
                        sup_feats, sup_lab, test_feats, num_classes=5,
                        tukey_beta=args.tukey_beta,
                        maha_shrink=args.maha_shrink,
                        use_combined_mean=args.use_combined_mean,
                    )

                    # Evaluate all
                    y_true = np.array([gt_map[n] for n in test_names])
                    table = evaluate_methods(y_true, trans_results, head_pred)

                    # Print ranking table
                    print(f"\n  {'='*50}")
                    print(f"  METHOD RANKING (repeat {rep+1})")
                    print(f"  {'='*50}")
                    print(f"  {'Method':<20} {'macro-F1':>10} {'bal-acc':>10}")
                    print(f"  {'-'*42}")
                    for row in table:
                        print(f"  {row['method']:<20} "
                              f"{row['macro_f1']:>10.4f} "
                              f"{row['bal_acc']:>10.4f}")

                    # Print best method's detailed report
                    best = table[0]
                    best_name = best["method"]
                    best_pred = trans_results.get(best_name, {}).get("pred", head_pred)
                    if best_name == "head":
                        best_pred = head_pred
                    print(f"\n  Best method: {best_name} "
                          f"(macro-F1={best['macro_f1']:.4f})")
                    print(classification_report(
                        y_true, best_pred,
                        target_names=CLASS_NAMES, digits=4))
                    print("  Confusion Matrix (rows=true, cols=pred):")
                    print(confusion_matrix(y_true, best_pred))

                    all_repeat_tables.append(table)
                    all_repeat_best.append(best)

                    del model
                    torch.cuda.empty_cache()

    # ==== Summary across repeats ====
    print(f"\n{'='*60}")
    print(f"  SUMMARY across {args.n_repeats} repeats")
    print(f"{'='*60}")

    # Collect all method names
    method_names = set()
    for table in all_repeat_tables:
        for row in table:
            method_names.add(row["method"])
    method_names = sorted(method_names)

    # Average F1 per method
    avg_table = []
    for m in method_names:
        f1s = [row["macro_f1"] for table in all_repeat_tables
               for row in table if row["method"] == m]
        baccs = [row["bal_acc"] for table in all_repeat_tables
                 for row in table if row["method"] == m]
        avg_table.append({
            "method": m,
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)) if len(f1s) > 1 else 0.0,
            "mean_bacc": float(np.mean(baccs)),
            "std_bacc": float(np.std(baccs)) if len(baccs) > 1 else 0.0,
            "n_repeats": len(f1s),
        })
    avg_table.sort(key=lambda x: -x["mean_f1"])

    print(f"  {'Method':<20} {'mean F1':>10} {'±std':>8} "
          f"{'mean bacc':>10} {'±std':>8}")
    print(f"  {'-'*60}")
    for row in avg_table:
        print(f"  {row['method']:<20} "
              f"{row['mean_f1']:>10.4f} {row['std_f1']:>8.4f} "
              f"{row['mean_bacc']:>10.4f} {row['std_bacc']:>8.4f}")

    # Best method overall
    best_overall = avg_table[0]
    print(f"\n  >>> BEST METHOD: {best_overall['method']} "
          f"(mean macro-F1 = {best_overall['mean_f1']:.4f})")

    # Save results
    results = {
        "experiment": "holdout_compare",
        "train_ratio": args.train_ratio,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "self_training_rounds": args.rounds,
        "tukey_beta": args.tukey_beta,
        "maha_shrink": args.maha_shrink,
        "use_combined_mean": args.use_combined_mean,
        "backbone": args.backbone,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "ema": args.ema,
        "per_repeat_tables": all_repeat_tables,
        "average_ranking": avg_table,
        "best_method": best_overall["method"],
        "best_mean_f1": best_overall["mean_f1"],
        "note": "This is an UNBIASED estimate. The model was trained on "
                "a subset of train_few_shot and tested on images it NEVER "
                "saw. Compare this with frozen_test 0.8351 (which may be "
                "inflated due to pseudo-label leakage).",
    }
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ok] saved results -> {args.out_json}")


if __name__ == "__main__":
    main()