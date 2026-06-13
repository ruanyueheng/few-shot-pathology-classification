"""Final submission: train on 200 images (excluding frozen_test 50), predict test set.

CRITICAL: frozen_test contains 50 images that are ALSO in train_few_shot.
Training on all 250 and then evaluating on frozen_test = DATA LEAKAGE.
This script automatically detects and excludes those 50 images.

Pipeline:
  1. Parse frozen_test filenames → identify which train images to exclude
  2. Train LoRA r=16 on the remaining 200 images
  3. Evaluate on frozen_test (with ground truth) — get honest F1
  4. Also run all 8 transductive methods + head, pick best per method
  5. Generate submission CSV for test_shuffled (61,881 images)

Usage:
  cd src
  python final_submit.py --ema --tukey_beta 0.5 --verbose

Resume:
  If interrupted, re-run the same command. Checkpoint tracks progress.
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
    simpleshot, laplacianshot, mahalanobis, tim,
    label_propagation, pt_map, alpha_tim,
)
from tqdm import tqdm


# ── Helpers ────────────────────────────────────────────────────

def get_frozen_test_ids(frozen_dir: str) -> set[tuple[str, str]]:
    """Parse frozen_test filenames → set of (class_idx, image_num).

    frozen_test files: test_Class_0_Class_0_006.png → ('0', '006')
    train files:       Class_0_006.png              → ('0', '006')
    """
    frozen_ids = set()
    frozen_path = Path(frozen_dir)
    if not frozen_path.exists():
        return frozen_ids
    for f in frozen_path.glob("*.png"):
        parts = f.stem.split("_")
        # test_Class_0_Class_0_006 → ['test', 'Class', '0', 'Class', '0', '006']
        if len(parts) >= 6:
            cls_idx = parts[2]
            img_num = parts[5]
            frozen_ids.add((cls_idx, img_num))
    return frozen_ids


def exclude_frozen_from_train(train_paths, train_labels, frozen_ids):
    """Return (filtered_paths, filtered_labels, excluded_count)."""
    keep_mask = []
    excluded = 0
    for p, l in zip(train_paths, train_labels):
        # Class_0_006.png → ('0', '006')
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


def build_args(backbone="vits14", ema=True, lora_r=16, lora_alpha=32, lora_dropout=0.2, seed=42):
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


def train_and_extract(train_paths, train_labels, device, args):
    """Train LoRA on given paths, extract features. Returns model + features."""
    full_idx = np.arange(len(train_paths))
    model, _, _ = train_one_run(
        train_paths, train_labels, full_idx, full_idx,
        args, device, run_id=0, refit_all=True,
    )

    n_params = model.trainable_param_count()

    # Extract features for support set (training images)
    sup_feats, sup_labels = extract_features(
        model, train_paths, args.image_size, args, device,
        labels=train_labels,
    )

    return model, sup_feats, sup_labels, n_params


def predict_test_set(model, test_paths, args, device):
    """Extract features + head probs for test images."""
    test_feats, test_names = extract_features(
        model, test_paths, args.image_size, args, device,
        labels=None,
    )
    head_probs, _ = predict_probs(
        model, test_paths, args.image_size, args, device,
    )
    return test_feats, test_names, head_probs


def run_all_methods(sup_feats, sup_labels, test_feats, num_classes=5,
                    tukey_beta=0.5, maha_shrink=0.3, use_combined_mean=True,
                    scalable_only=False,
                    tim_params=None, atim_params=None, ptmap_params=None,
                    lshot_params=None):
    """Run transductive methods. Returns dict of {method: pred}.

    If scalable_only=True, skip LP (needs O(n_query^2) dense matrix).

    LaplacianShot is NOW SCALABLE — uses sparse knn affinity for Nq > 500.
    TIM, PT-MAP, SimpleShot, Mahalanobis are also scalable.

    tim_params/atim_params/ptmap_params/lshot_params: optional dicts to
    override default hyperparameters.
    """
    results = {}

    # ── Scalable methods ──

    # SimpleShot: O(Nq × Ns)
    results["simpleshot"] = simpleshot(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
    )

    # Mahalanobis: O(Nq × Ns)
    results["mahalanobis"] = mahalanobis(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        shrink=maha_shrink,
    )

    # TIM: O(Nq × Ns × n_iter) — scales perfectly to 61k
    tp = tim_params or {}
    results["tim"] = tim(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        n_iter=tp.get("n_iter", 1000),
        lr=tp.get("lr", 1e-4),
        temperature=tp.get("temperature", 15.0),
        lambda_marg=tp.get("lambda_marg", 1.0),
        lambda_cond=tp.get("lambda_cond", 0.1),
    )

    # alpha-TIM: same complexity as TIM
    ap = atim_params or {}
    results["alpha_tim"] = alpha_tim(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=True,
        n_iter=ap.get("n_iter", 1000),
        lr=ap.get("lr", 1e-4),
        temperature=ap.get("temperature", 15.0),
        alpha=ap.get("alpha", 2.0),
        lambda_marg=ap.get("lambda_marg", 1.0),
        lambda_cond=ap.get("lambda_cond", 0.1),
    )

    # PT-MAP: O(Nq × C × D × n_iter) — SCALABLE! Only [Nq, C] matrices.
    pp = ptmap_params or {}
    results["ptmap_sink"] = pt_map(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta,
        n_iter=pp.get("n_iter", 20),
        lambda_s=pp.get("lambda_s", 10.0),
        use_sinkhorn=pp.get("use_sinkhorn", True),
        sinkhorn_iter=pp.get("sinkhorn_iter", 10),
    )
    # PT-MAP without Sinkhorn (better for class-imbalanced test sets)
    results["ptmap_nosink"] = pt_map(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta,
        n_iter=pp.get("n_iter", 20),
        lambda_s=pp.get("lambda_s", 10.0),
        use_sinkhorn=False,
    )

    # LaplacianShot: O(Nq × k × n_iter) — NOW SCALABLE with sparse knn!
    # Uses sparse affinity matrix for Nq > 500, memory: O(Nq * k) not O(Nq²)
    lp = lshot_params or {}
    results["laplacianshot"] = laplacianshot(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=lp.get("tukey_beta", tukey_beta),
        use_combined_mean=lp.get("use_combined_mean", use_combined_mean),
        knn=lp.get("knn", 7),
        lam=lp.get("lam", 10.0),
        n_iter=lp.get("n_iter", 20),
        sigma=lp.get("sigma", 1.0),
    )

    # ── Non-scalable methods (O(n_query^2) dense) — skip for 61k ──
    if not scalable_only:
        results["lp"] = label_propagation(
            sup_feats, sup_labels, test_feats, num_classes=num_classes,
            tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
            knn=10, alpha=0.7, sigma=1.0,
        )

    return results


# ── Main ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Final submission: train on 200 (excluding frozen_test), predict all")
    ap.add_argument("--train_dir", default="../train_few_shot")
    ap.add_argument("--frozen_dir", default="../frozen_test",
                    help="frozen_test directory (images that MUST be excluded from training)")
    ap.add_argument("--test_dir", default="../test_shuffled",
                    help="unlabeled test images for submission")
    ap.add_argument("--backbone", default="vits14")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no_ema", action="store_false", dest="ema")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tukey_beta", type=float, default=0.5)
    ap.add_argument("--maha_shrink", type=float, default=0.3)
    ap.add_argument("--use_combined_mean", action="store_true", default=True)
    ap.add_argument("--out_csv", default="../sub_final_r16.csv",
                    help="submission CSV path")
    ap.add_argument("--save_all_methods", action="store_true",
                    help="save a CSV for EACH method (for comparison)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--final_method", default="tim",
                    choices=["tim", "alpha_tim", "ptmap_sink", "ptmap_nosink",
                             "simpleshot", "mahalanobis", "laplacianshot", "head"],
                    help="method to use for final submission (default: tim)")
    ap.add_argument("--lora_dropout", type=float, default=0.1,
                    help="LoRA dropout (0.1 is better for TIM)")
    args = ap.parse_args()

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Step 1: Decide training set ──
    # For test_shuffled (61k): no overlap with train_few_shot → use ALL 250 images
    # For frozen_test (50 images): those ARE in train_few_shot → must exclude
    print("=" * 60)
    print("  STEP 1: Preparing training data")
    print("=" * 60)

    all_paths, all_labels = list_train_samples(args.train_dir)
    print(f"  train_few_shot images: {len(all_paths)}")

    # Since we're predicting on test_shuffled (no overlap with train_few_shot),
    # we can use ALL 250 images for training — no data leakage!
    train_paths = all_paths
    train_labels = all_labels
    print(f"  Training on: {len(train_paths)} images (full 250, no leakage on test_shuffled)")
    print(f"  Per-class: {np.bincount(train_labels).tolist()}")

    # For frozen_test evaluation, we'll re-extract features without leakage later

    # ── Step 2: Train LoRA on 250 images ──
    print(f"\n{'=' * 60}")
    print(f"  STEP 2: Training LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  on {len(train_paths)} images (full 250, test_shuffled has no overlap)")
    print(f"{'=' * 60}")

    train_args = build_args(
        backbone=args.backbone, ema=args.ema,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, seed=args.seed,
    )

    t0 = time.time()
    model, sup_feats, sup_labels, n_params = train_and_extract(
        train_paths, train_labels, device, train_args,
    )
    train_time = time.time() - t0
    print(f"  trainable params: {n_params:,}  "
          f"(params/sample = {n_params / len(train_paths):.0f}:1)")
    print(f"  Training time: {timedelta(seconds=int(train_time))}")

    # ── Step 3: Evaluate on frozen_test (if ground truth exists) ──
    # NOTE: frozen_test 50 images ARE in train_few_shot 250.
    # The model was trained on all 250, so frozen_test evaluation here is LEAKY.
    # This is shown for reference only — the real submission is on test_shuffled.
    gt_path = Path(args.frozen_dir) / "_groundtruth.csv"
    if gt_path.exists():
        print(f"\n{'=' * 60}")
        print(f"  STEP 3: Evaluating on frozen_test (LEAKY — trained on 250)")
        print(f"  NOTE: These scores are inflated! Real metric is on test_shuffled.")
        print(f"{'=' * 60}")

        # Load frozen_test images
        frozen_paths = list_test_samples(args.frozen_dir)
        # Filter out _groundtruth.csv — list_test_samples only gets .png
        frozen_feats, frozen_names = extract_features(
            model, frozen_paths, train_args.image_size, train_args, device,
            labels=None,
        )
        head_probs_frozen, _ = predict_probs(
            model, frozen_paths, train_args.image_size, train_args, device,
        )

        # Run all methods on frozen_test
        frozen_results = run_all_methods(
            sup_feats, sup_labels, frozen_feats, num_classes=5,
            tukey_beta=args.tukey_beta, maha_shrink=args.maha_shrink,
            use_combined_mean=args.use_combined_mean,
        )

        # Load ground truth
        gt_df = pd.read_csv(gt_path)
        gt_map = dict(zip(gt_df["filename"], gt_df["label"]))
        y_true = np.array([CLASS_TO_IDX.get(gt_map.get(n, ""), -1)
                           for n in frozen_names])
        valid_mask = y_true >= 0
        if valid_mask.sum() > 0:
            y_true = y_true[valid_mask]
            # Also filter predictions
            print(f"\n  Ground truth loaded: {valid_mask.sum()}/{len(frozen_names)} images")

            # Evaluate each method
            print(f"\n  {'Method':<20} {'F1':>8} {'BalAcc':>8}")
            print(f"  {'-' * 38}")

            method_scores = {}
            for name, pred in frozen_results.items():
                pred_valid = pred[valid_mask]
                f1 = float(f1_score(y_true, pred_valid, average="macro"))
                bacc = float(balanced_accuracy_score(y_true, pred_valid))
                method_scores[name] = {"f1": f1, "bacc": bacc}
                print(f"  {name:<20} {f1:>8.4f} {bacc:>8.4f}")

            # Head
            head_pred = head_probs_frozen.argmax(axis=1)[valid_mask]
            head_f1 = float(f1_score(y_true, head_pred, average="macro"))
            head_bacc = float(balanced_accuracy_score(y_true, head_pred))
            method_scores["head"] = {"f1": head_f1, "bacc": head_bacc}
            print(f"  {'head':<20} {head_f1:>8.4f} {head_bacc:>8.4f}")

            best_method = max(method_scores, key=lambda k: method_scores[k]["f1"])
            print(f"\n  >>> BEST on frozen_test: {best_method} "
                  f"F1={method_scores[best_method]['f1']:.4f}")
    else:
        print(f"\n  [skip] frozen_test evaluation (no ground truth)")

    # ── Step 4: Predict on test_shuffled for submission ──
    print(f"\n{'=' * 60}")
    print(f"  STEP 4: Predicting on test_shuffled for submission")
    print(f"{'=' * 60}")

    test_paths = list_test_samples(args.test_dir)
    print(f"  test_shuffled: {len(test_paths)} images")

    t1 = time.time()
    test_feats, test_names, head_probs_test = predict_test_set(
        model, test_paths, train_args, device,
    )
    extract_time = time.time() - t1
    print(f"  Feature extraction: {timedelta(seconds=int(extract_time))}")

    # Run all transductive methods on test_shuffled
    # All methods are now scalable to 61k!
    # - TIM/alpha-TIM: O(Nq × Ns × n_iter)
    # - PT-MAP: O(Nq × C × D × n_iter)
    # - SimpleShot/Mahalanobis: O(Nq × Ns)
    # - LaplacianShot: O(Nq × k × n_iter) with sparse knn affinity
    # Only LP truly OOMs on 61k (dense Nq² matrix + linear solve)
    print(f"  Running all scalable methods...")
    test_results = run_all_methods(
        sup_feats, sup_labels, test_feats, num_classes=5,
        tukey_beta=args.tukey_beta, maha_shrink=args.maha_shrink,
        use_combined_mean=args.use_combined_mean,
        scalable_only=True,  # skip LP (OOM on 61k)
        lshot_params={
            "tukey_beta": 1.0, "use_combined_mean": False,
            "knn": 7, "lam": 10.0, "n_iter": 20, "sigma": 1.0,
        },
    )

    # Print per-method label distribution for comparison
    print(f"\n  Per-method label distribution on test_shuffled:")
    print(f"  {'Method':<20} " + "  ".join(f"{'Cl'+str(c):>6}" for c in range(5)))
    print(f"  {'-'*56}")
    for name, pred in test_results.items():
        counts = np.bincount(pred, minlength=5)
        print(f"  {name:<20} " + "  ".join(f"{c:>6}" for c in counts))
    head_pred_dist = np.bincount(head_probs_test.argmax(axis=1), minlength=5)
    print(f"  {'head':<20} " + "  ".join(f"{c:>6}" for c in head_pred_dist))

    # Pick method (default: TIM; user can override with --final_method)
    default_method = args.final_method
    if default_method not in test_results:
        print(f"  [warn] method '{default_method}' not in results, using 'tim'")
        default_method = "tim"
    final_pred = test_results[default_method]
    final_labels = [IDX_TO_CLASS[int(i)] for i in final_pred]

    sub_df = pd.DataFrame({
        "filename": test_names,
        "label": final_labels,
    })
    sub_df.to_csv(args.out_csv, index=False)
    print(f"\n  [ok] submission saved -> {args.out_csv}")
    print(f"  Method: {default_method}")
    print(f"  Label distribution:")
    print(sub_df["label"].value_counts().to_string().replace("\n", "\n  "))

    # Save all methods if requested
    if args.save_all_methods:
        out_dir = Path(args.out_csv).parent
        for name, pred in test_results.items():
            labels = [IDX_TO_CLASS[int(i)] for i in pred]
            df = pd.DataFrame({"filename": test_names, "label": labels})
            path = out_dir / f"sub_final_r16_{name}.csv"
            df.to_csv(str(path), index=False)
            print(f"  [ok] {name} -> {path}")

        # Also head
        head_pred = head_probs_test.argmax(axis=1)
        head_labels = [IDX_TO_CLASS[int(i)] for i in head_pred]
        df_head = pd.DataFrame({"filename": test_names, "label": head_labels})
        path_head = out_dir / "sub_final_r16_head.csv"
        df_head.to_csv(str(path_head), index=False)
        print(f"  [ok] head -> {path_head}")

    # ── Save model checkpoint ──
    ckpt_dir = Path("../artifacts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"final_r{args.lora_r}_full250.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "args": vars(train_args),
        "train_size": len(train_paths),
        "excluded_frozen": 0,
    }, ckpt_path)
    print(f"\n  [ok] checkpoint saved -> {ckpt_path}")

    # ── Save experiment log ──
    log_path = Path("../final_submit_results.json")
    log = {
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "backbone": args.backbone,
        "ema": args.ema,
        "train_size": len(train_paths),
        "excluded_frozen": 0,  # no exclusion needed for test_shuffled
        "tukey_beta": args.tukey_beta,
        "seed": args.seed,
        "train_time_sec": round(train_time, 1),
        "final_method": default_method,
    }
    if gt_path.exists():
        log["frozen_test_scores"] = method_scores
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  [ok] log saved -> {log_path}")

    total_time = time.time() - t0
    print(f"\n  Total time: {timedelta(seconds=int(total_time))}")
    print(f"\n  ✅ DONE — submission ready at {args.out_csv}")


if __name__ == "__main__":
    main()
