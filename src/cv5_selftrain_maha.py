"""5-Fold Cross-Validation for the self-training + Mahalanobis pipeline.

This script answers the question: "Is 0.835 inflated?"

It runs the FULL pipeline (LoRA training → self-training → Mahalanobis inference)
inside a stratified 5-fold CV loop on train_few_shot/ (250 images).

Key design: each fold treats its val split as the "unlabeled test set" for
pseudo-labeling — exactly mirroring how self_train.py works, but with a proper
CV guarantee that every sample is evaluated exactly once out-of-fold.

Output:
  - Per-fold macro-F1 and balanced-accuracy
  - OOF (out-of-fold) macro-F1 across all 250 samples
  - Per-class breakdown
  - Comparison with the frozen_test 0.8351 score

Usage:
  cd src
  python cv5_selftrain_maha.py

  # Or with different settings:
  python cv5_selftrain_maha.py --rounds 3 --conf_thresh 0.9 --maha_shrink 0.3
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
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
from transductive import mahalanobis


def main():
    ap = argparse.ArgumentParser(
        description="5-Fold CV for self-training + Mahalanobis pipeline")
    # data
    ap.add_argument("--train_dir", default="../train_few_shot",
                    help="full training dir (250 images)")
    ap.add_argument("--n_splits", type=int, default=5)
    # self-training
    ap.add_argument("--rounds", type=int, default=3,
                    help="number of self-training rounds (0 = baseline, no pseudo)")
    ap.add_argument("--conf_thresh", type=float, default=0.9)
    ap.add_argument("--per_class_topk", type=int, default=2)
    ap.add_argument("--per_class_min", type=int, default=None)
    # final inference method
    ap.add_argument("--maha_shrink", type=float, default=0.3)
    ap.add_argument("--tukey_beta", type=float, default=1.0)
    ap.add_argument("--use_combined_mean", action="store_true")
    # LoRA training hyperparams (same as best_v02)
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
    ap.add_argument("--ema", action="store_true", help="enable EMA")
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true",
                    help="print per-epoch loss during training")
    ap.add_argument("--out_json", default="../cv5_selftrain_maha_results.json")
    args = ap.parse_args()

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")
    print(f"[info] self-training rounds={args.rounds}, "
          f"conf_thresh={args.conf_thresh}, "
          f"maha_shrink={args.maha_shrink}")
    print(f"[info] LoRA: backbone={args.backbone}, r={args.lora_r}, "
          f"alpha={args.lora_alpha}, epochs={args.epochs}")

    # ---- load all 250 images ----
    all_paths, all_labels = list_train_samples(args.train_dir)
    N = len(all_paths)
    print(f"[info] {N} samples, per-class {np.bincount(all_labels).tolist()}")

    # ---- CV loop ----
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True,
                          random_state=args.seed)
    oof_pred = np.zeros(N, dtype=int)  # out-of-fold predictions
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.arange(N), all_labels)):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold+1}/{args.n_splits}  "
              f"train={len(train_idx)}  val={len(val_idx)}")
        print(f"  val per-class: {np.bincount(all_labels[val_idx]).tolist()}")
        print(f"{'='*60}")

        # Build temporary directories for this fold
        with tempfile.TemporaryDirectory(prefix=f"cv5_fold{fold}_") as tmpdir:
            tmpdir = Path(tmpdir)
            # -- train split as "labeled" dir (for self-training base) --
            fold_train_dir = tmpdir / "train"
            for c in CLASS_NAMES:
                (fold_train_dir / c).mkdir(parents=True, exist_ok=True)
            train_paths_fold = [all_paths[i] for i in train_idx]
            train_labels_fold = all_labels[train_idx]
            for p, l in zip(train_paths_fold, train_labels_fold):
                cls_dir = fold_train_dir / CLASS_NAMES[l]
                shutil.copy2(p, cls_dir / Path(p).name)

            # -- val split as "unlabeled test" dir --
            fold_test_dir = tmpdir / "test"
            fold_test_dir.mkdir(parents=True, exist_ok=True)
            val_paths_fold = [all_paths[i] for i in val_idx]
            for p in val_paths_fold:
                shutil.copy2(p, fold_test_dir / Path(p).name)

            # -- ground-truth map for eval --
            gt_map = {}
            for i in val_idx:
                gt_map[Path(all_paths[i]).name] = int(all_labels[i])

            # ==== Self-training loop within this fold ====
            accumulated_pseudo = []
            used_names = set()

            for r in range(args.rounds + 1):
                print(f"\n  --- fold{fold+1} round {r} ---")

                # materialize labeled dir (base train + accumulated pseudos)
                round_dir = tmpdir / f"round_{r}"
                round_dir.mkdir(parents=True, exist_ok=True)
                labeled_dir, n_orig, n_pseudo = materialize_round_dir(
                    round_dir, fold_train_dir, fold_test_dir,
                    pseudo=[], accumulated_pseudo=accumulated_pseudo,
                )
                print(f"  labeled: {n_orig} orig + {n_pseudo} pseudo = "
                      f"{n_orig+n_pseudo}")

                # train LoRA on labeled set
                st_paths, st_labels = list_train_samples(str(labeled_dir))
                full_idx = np.arange(len(st_paths))
                model, _, _ = train_one_run(
                    st_paths, st_labels, full_idx, full_idx,
                    args, device, run_id=fold * (args.rounds + 1) + r,
                    refit_all=True,
                )

                # On the last round, do Mahalanobis inference
                if r == args.rounds:
                    # Extract features
                    sup_feats, sup_lab = extract_features(
                        model, st_paths, args.image_size, args, device,
                        labels=st_labels,
                    )
                    test_paths_list = list_test_samples(str(fold_test_dir))
                    test_feats, test_names = extract_features(
                        model, test_paths_list, args.image_size, args, device,
                        labels=None,
                    )

                    # Mahalanobis prediction
                    final_pred = mahalanobis(
                        sup_feats, sup_lab, test_feats, num_classes=5,
                        tukey_beta=args.tukey_beta,
                        use_combined_mean=args.use_combined_mean,
                        shrink=args.maha_shrink,
                    )

                    # Evaluate
                    y_true = np.array([gt_map[n] for n in test_names])
                    fold_f1 = float(f1_score(y_true, final_pred, average="macro"))
                    fold_bacc = float(balanced_accuracy_score(y_true, final_pred))

                    print(f"  >>> fold{fold+1} round{r} "
                          f"macro-F1={fold_f1:.4f}  bal-acc={fold_bacc:.4f}")
                    print(classification_report(
                        y_true, final_pred,
                        target_names=CLASS_NAMES, digits=4))

                    # Map predictions back to oof array
                    for name, pred in zip(test_names, final_pred):
                        # find the index in all_paths
                        for j in val_idx:
                            if Path(all_paths[j]).name == name:
                                oof_pred[j] = int(pred)
                                break

                else:
                    # Pick pseudo-labels for next round
                    # Use Mahalanobis confidence for pseudo selection
                    sup_feats, sup_lab = extract_features(
                        model, st_paths, args.image_size, args, device,
                        labels=st_labels,
                    )
                    test_paths_list = list_test_samples(str(fold_test_dir))
                    test_feats, test_names = extract_features(
                        model, test_paths_list, args.image_size, args, device,
                        labels=None,
                    )
                    _, probs = mahalanobis(
                        sup_feats, sup_lab, test_feats, num_classes=5,
                        tukey_beta=args.tukey_beta,
                        use_combined_mean=args.use_combined_mean,
                        shrink=args.maha_shrink,
                        return_probs=True,
                    )

                    newly_selected = select_pseudo(
                        probs, test_names, args.conf_thresh,
                        args.per_class_topk, args.per_class_min, used_names,
                    )
                    print(f"  picked {len(newly_selected)} pseudo-labels")
                    if newly_selected:
                        cls_count = np.bincount(
                            [c for _, c, _ in newly_selected],
                            minlength=5).tolist()
                        print(f"  per-class: {cls_count}")
                    accumulated_pseudo.extend(newly_selected)
                    used_names.update(n for n, _, _ in newly_selected)

                del model
                torch.cuda.empty_cache()

        fold_results.append({
            "fold": fold + 1,
            "macro_f1": fold_f1,
            "bal_acc": fold_bacc,
            "n_val": len(val_idx),
        })

    # ==== Final summary ====
    oof_f1 = float(f1_score(all_labels, oof_pred, average="macro"))
    oof_bacc = float(balanced_accuracy_score(all_labels, oof_pred))
    f1s = [r["macro_f1"] for r in fold_results]
    baccs = [r["bal_acc"] for r in fold_results]

    print(f"\n{'='*60}")
    print(f"  5-FOLD CV RESULTS  (self-training + Mahalanobis)")
    print(f"{'='*60}")
    print(f"per-fold macro-F1:  {[round(x, 4) for x in f1s]}")
    print(f"per-fold bal-acc :  {[round(x, 4) for x in baccs]}")
    print(f"mean macro-F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"mean bal-acc  = {np.mean(baccs):.4f} ± {np.std(baccs):.4f}")
    print(f"OOF  macro-F1 = {oof_f1:.4f}")
    print(f"OOF  bal-acc  = {oof_bacc:.4f}")
    print()
    print("OOF Classification Report:")
    print(classification_report(all_labels, oof_pred,
                               target_names=CLASS_NAMES, digits=4))
    print("OOF Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_labels, oof_pred))

    # Save results
    results = {
        "method": "self-training + Mahalanobis",
        "rounds": args.rounds,
        "conf_thresh": args.conf_thresh,
        "maha_shrink": args.maha_shrink,
        "tukey_beta": args.tukey_beta,
        "backbone": args.backbone,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "per_fold": fold_results,
        "mean_macro_f1": float(np.mean(f1s)),
        "std_macro_f1": float(np.std(f1s)),
        "mean_bal_acc": float(np.mean(baccs)),
        "std_bal_acc": float(np.std(baccs)),
        "oof_macro_f1": oof_f1,
        "oof_bal_acc": oof_bacc,
        "note": "This is the TRUE generalization estimate. "
                "Compare with frozen_test 0.8351 (which may be inflated "
                "due to self-training pseudo-label leakage and model selection "
                "on test set).",
    }
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ok] saved results -> {args.out_json}")


if __name__ == "__main__":
    main()
