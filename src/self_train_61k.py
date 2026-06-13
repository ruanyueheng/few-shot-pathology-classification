"""Self-training with TIM on test_shuffled (61k images).

Pipeline per round:
  1. Train LoRA r=16 on 250 labeled images (+ accumulated pseudo-labels)
  2. Extract features for all 61k test images
  3. Run TIM to get transductive predictions + soft probs
  4. Select high-confidence pseudo-labels (conf >= threshold)
  5. Add pseudo-labels to training set for next round
  6. Repeat for N rounds

Key design choices:
  - TIM for pseudo-label selection (not raw head probs) — more reliable
  - Per-class top-k cap to prevent class imbalance drift
  - Confidence threshold increases each round (conservative start)
  - Resume support: interrupt and continue later

Usage:
  cd src
  python self_train_61k.py --ema --tukey_beta 0.5 --lora_dropout 0.1 --verbose

Resume:
  python self_train_61k.py --ema --tukey_beta 0.5 --lora_dropout 0.1 --resume --verbose
"""
from __future__ import annotations
import argparse
import json
import time
import shutil
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
    simpleshot, mahalanobis, tim, pt_map,
)
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────

DEFAULT_TIM_PARAMS = {
    "temperature": 10.0,
    "lambda_marg": 2.0,
    "lambda_cond": 0.05,
    "n_iter": 1000,
    "lr": 1e-4,
}

# Confidence thresholds per round (increasingly conservative)
# NOTE: We use HEAD probs (trained with cross-entropy) for selection,
# NOT TIM probs — TIM's softmax with temp=10 is too soft (max ~0.25-0.45),
# making high thresholds impossible to reach.
# Head probs can reach 0.9+, so 0.7/0.8/0.9 are reasonable.
CONF_SCHEDULE = [0.70, 0.80, 0.90]

# Per-class max pseudo-labels added per round
# Round 0: 500/class, Round 1: 300/class, Round 2: 200/class
TOPK_SCHEDULE = [500, 300, 200]


# ── Helpers ────────────────────────────────────────────────────

def build_args(backbone="vits14", ema=True, lora_r=16, lora_alpha=32,
               lora_dropout=0.1, seed=42):
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
    model.eval()

    # Extract support features
    sup_feats, _ = extract_features(
        model, train_paths, args.image_size, args, device,
        labels=train_labels,
    )
    return model, sup_feats, train_labels, sum(p.numel() for p in model.parameters() if p.requires_grad)


def select_pseudo_labels(head_probs, test_names, conf_thresh,
                         per_class_topk, already_used):
    """Select pseudo-labels from HEAD soft predictions.

    We use HEAD probs (not TIM) because TIM's soft probs with temp=10
    max out at ~0.25-0.45, making confidence thresholds >= 0.5 unreachable.
    HEAD probs from cross-entropy training can reach 0.9+, making them
    suitable for confident pseudo-label selection.

    The FINAL submission still uses TIM predictions — we only use HEAD
    for deciding WHICH test images are confident enough to pseudo-label.
    """
    pred = head_probs.argmax(axis=1)
    conf = head_probs.max(axis=1)

    selected = []
    per_class_counts = np.zeros(5, dtype=int)

    for c in range(5):
        # All test images predicted as class c, not already used
        candidates = []
        for i in np.where(pred == c)[0]:
            name = test_names[i]
            if name not in already_used and conf[i] >= conf_thresh:
                candidates.append((i, name, float(conf[i])))

        # Sort by confidence descending
        candidates.sort(key=lambda x: -x[2])

        # Cap per class
        if per_class_topk is not None:
            candidates = candidates[:per_class_topk]

        for i, name, c_conf in candidates:
            selected.append((name, int(c), c_conf))
            per_class_counts[c] += 1

    return selected, per_class_counts


def materialize_pseudo_dir(out_dir, base_train_dir, test_dir,
                           accumulated_pseudo):
    """Create a training directory with original 250 + pseudo-labeled images."""
    labeled_dir = out_dir / "labeled"
    if labeled_dir.exists():
        shutil.rmtree(labeled_dir)
    for c in CLASS_NAMES:
        (labeled_dir / c).mkdir(parents=True, exist_ok=True)

    # Copy original train data
    n_orig = 0
    for c in CLASS_NAMES:
        for f in sorted((base_train_dir / c).glob("*.png")):
            shutil.copy2(f, labeled_dir / c / f.name)
            n_orig += 1

    # Copy pseudo-labeled test images
    n_pseudo = 0
    for name, c_idx, _conf in accumulated_pseudo:
        src = test_dir / name
        dst = labeled_dir / CLASS_NAMES[c_idx] / f"pseudo_{name}"
        if src.exists():
            shutil.copy2(src, dst)
            n_pseudo += 1

    return labeled_dir, n_orig, n_pseudo


def run_all_scalable(sup_feats, sup_labels, test_feats, num_classes=5,
                     tukey_beta=0.5, maha_shrink=0.3, use_combined_mean=True,
                     tim_params=None, ptmap_params=None):
    """Run all scalable transductive methods. Returns dict of {method: pred}."""
    results = {}
    tp = tim_params or DEFAULT_TIM_PARAMS

    results["simpleshot"] = simpleshot(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
    )
    results["mahalanobis"] = mahalanobis(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        shrink=maha_shrink,
    )
    results["tim"] = tim(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta, use_combined_mean=use_combined_mean,
        n_iter=tp.get("n_iter", 1000),
        lr=tp.get("lr", 1e-4),
        temperature=tp.get("temperature", 10.0),
        lambda_marg=tp.get("lambda_marg", 2.0),
        lambda_cond=tp.get("lambda_cond", 0.05),
    )
    pp = ptmap_params or {}
    results["ptmap_sink"] = pt_map(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta,
        n_iter=pp.get("n_iter", 20),
        lambda_s=pp.get("lambda_s", 20.0),
        use_sinkhorn=True,
    )
    results["ptmap_nosink"] = pt_map(
        sup_feats, sup_labels, test_feats, num_classes=num_classes,
        tukey_beta=tukey_beta,
        n_iter=pp.get("n_iter", 20),
        lambda_s=pp.get("lambda_s", 10.0),
        use_sinkhorn=False,
    )
    return results


def get_tim_probs(sup_feats, sup_labels, test_feats, num_classes=5,
                  tukey_beta=0.5, use_combined_mean=True, tim_params=None):
    """Run TIM and return soft probability matrix [Nq, C]."""
    import torch.nn.functional as F
    import torch

    tp = tim_params or DEFAULT_TIM_PARAMS
    temperature = tp.get("temperature", 10.0)
    lambda_marg = tp.get("lambda_marg", 2.0)
    lambda_cond = tp.get("lambda_cond", 0.05)
    n_iter = tp.get("n_iter", 1000)
    lr = tp.get("lr", 1e-4)

    from transductive import tukey_transform, center_and_normalize

    if tukey_beta != 1.0:
        sup_feats = tukey_transform(sup_feats, tukey_beta)
        test_feats = tukey_transform(test_feats, tukey_beta)

    q_s, q_q = center_and_normalize(sup_feats, test_feats,
                                     use_combined_mean=use_combined_mean)

    q_s = torch.from_numpy(q_s).float()
    q_q = torch.from_numpy(q_q).float()

    ns, d = q_s.shape
    nq = q_q.shape[0]
    C = num_classes

    # One-hot support labels
    y_s = torch.zeros(ns, C)
    y_s[torch.arange(ns), torch.from_numpy(sup_labels).long()] = 1.0

    # Initialize soft query labels from nearest prototype
    with torch.no_grad():
        proto = (y_s.T @ q_s) / (y_s.sum(0, keepdim=True).T)  # [C, d]
        sim = q_q @ proto.T  # [nq, C]
        soft = torch.softmax(sim / temperature, dim=1)

    soft = soft.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([soft], lr=lr)

    for it in range(n_iter):
        optimizer.zero_grad()

        # Marginal entropy: H(p_bar)
        p_bar = soft.mean(dim=0)  # [C]
        h_marg = -(p_bar * (p_bar + 1e-8).log()).sum()

        # Conditional entropy: E[H(p_i)]
        h_cond = -(soft * (soft + 1e-8).log()).sum(dim=1).mean()

        loss = lambda_marg * h_marg - lambda_cond * h_cond
        loss.backward()
        optimizer.step()

        # Project to simplex per row
        with torch.no_grad():
            soft.copy_(torch.softmax(soft / temperature, dim=1))

    probs = soft.detach().numpy()
    return probs


# ── Resume ────────────────────────────────────────────────────

RESUME_FILE = "selftrain_61k_state.json"


def save_resume_state(out_root, completed_round, accumulated_pseudo,
                      used_names, round_log):
    state = {
        "completed_round": completed_round,
        "accumulated_pseudo": [
            {"name": n, "class_idx": int(c), "conf": float(cf)}
            for n, c, cf in accumulated_pseudo
        ],
        "used_names": sorted(used_names),
        "round_log": round_log,
    }
    with open(out_root / RESUME_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"  [resume] state saved after round {completed_round}")


def load_resume_state(out_root):
    path = out_root / RESUME_FILE
    if not path.exists():
        return None
    with open(path) as f:
        state = json.load(f)
    accumulated_pseudo = [
        (item["name"], int(item["class_idx"]), float(item["conf"]))
        for item in state["accumulated_pseudo"]
    ]
    used_names = set(state["used_names"])
    round_log = state.get("round_log", [])
    completed = state["completed_round"]
    print(f"  [resume] found state: round {completed} done, "
          f"{len(accumulated_pseudo)} pseudo-labels accumulated")
    return completed + 1, accumulated_pseudo, used_names, round_log


# ── Main ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Self-training on test_shuffled (61k)")
    ap.add_argument("--train_dir", default="../dev_few_shot",
                    help="250 labeled images")
    ap.add_argument("--test_dir", default="../test_shuffled",
                    help="61,881 unlabeled test images")
    ap.add_argument("--out_dir", default="../artifacts/selftrain_61k")
    ap.add_argument("--rounds", type=int, default=3,
                    help="number of self-training rounds")
    ap.add_argument("--resume", action="store_true",
                    help="resume from last completed round")

    # Model
    ap.add_argument("--backbone", default="vits14")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no_ema", action="store_false", dest="ema")
    ap.add_argument("--seed", type=int, default=42)

    # Transductive
    ap.add_argument("--tukey_beta", type=float, default=0.5)
    ap.add_argument("--maha_shrink", type=float, default=0.3)
    ap.add_argument("--use_combined_mean", action="store_true", default=True)

    # TIM params (tuned from tim_search)
    ap.add_argument("--tim_temp", type=float, default=10.0)
    ap.add_argument("--tim_lambda_marg", type=float, default=2.0)
    ap.add_argument("--tim_lambda_cond", type=float, default=0.05)
    ap.add_argument("--tim_n_iter", type=int, default=1000)
    ap.add_argument("--tim_lr", type=float, default=1e-4)

    # Pseudo-label selection
    ap.add_argument("--conf_thresh", type=float, default=None,
                    help="confidence threshold (default: auto schedule 0.70/0.80/0.90)")
    ap.add_argument("--per_class_topk", type=int, default=None,
                    help="max pseudo per class per round (default: auto 500/300/200)")

    # Final submission method
    ap.add_argument("--final_method", default="tim",
                    choices=["tim", "ptmap_sink", "ptmap_nosink",
                             "simpleshot", "mahalanobis", "head"])

    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    base_train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    tim_params = {
        "temperature": args.tim_temp,
        "lambda_marg": args.tim_lambda_marg,
        "lambda_cond": args.tim_lambda_cond,
        "n_iter": args.tim_n_iter,
        "lr": args.tim_lr,
    }

    print("=" * 70)
    print("  SELF-TRAINING ON test_shuffled (61k)")
    print("=" * 70)
    print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  TIM: temp={tim_params['temperature']}, "
          f"lambda_marg={tim_params['lambda_marg']}, "
          f"lambda_cond={tim_params['lambda_cond']}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Device: {device}")

    # Load test paths
    test_paths = list_test_samples(test_dir)
    print(f"  test_shuffled: {len(test_paths)} images")

    # Load original train — use ALL 250 images (test_shuffled has no overlap)
    orig_paths, orig_labels = list_train_samples(base_train_dir)
    print(f"  train_few_shot: {len(orig_paths)} images (using ALL 250)")

    # Resume
    start_round = 0
    accumulated_pseudo = []
    used_names = set()
    round_log = []

    if args.resume:
        state = load_resume_state(out_root)
        if state is not None:
            start_round, accumulated_pseudo, used_names, round_log = state
        else:
            print("  [resume] no state file, starting from scratch")

    # ── Self-training loop ──
    for r in range(start_round, args.rounds + 1):
        t_round_start = time.time()
        print(f"\n{'=' * 70}")
        print(f"  ROUND {r}/{args.rounds}")
        print(f"{'=' * 70}")

        # Build training set
        if r == 0:
            train_paths = orig_paths
            train_labels = orig_labels
            n_pseudo = 0
        else:
            labeled_dir, n_orig, n_pseudo = materialize_pseudo_dir(
                out_root / f"round_{r}", base_train_dir, test_dir,
                accumulated_pseudo,
            )
            train_paths, train_labels = list_train_samples(labeled_dir)

        print(f"  Training set: {len(orig_paths)} orig + {n_pseudo} pseudo = "
              f"{len(train_paths)} total")
        print(f"  Per-class: {np.bincount(train_labels, minlength=5).tolist()}")

        # Train LoRA
        print(f"  Training LoRA r={args.lora_r}...")
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
        print(f"  trainable params: {n_params:,}")
        print(f"  Training time: {timedelta(seconds=int(train_time))}")

        # Extract test features
        print(f"  Extracting features for {len(test_paths)} test images...")
        t1 = time.time()
        test_feats, test_names = extract_features(
            model, test_paths, train_args.image_size, train_args, device,
            labels=None,
        )
        extract_time = time.time() - t1
        print(f"  Feature extraction: {timedelta(seconds=int(extract_time))}")
        print(f"  sup_feats: {sup_feats.shape}, test_feats: {test_feats.shape}")

        # Head predictions
        head_probs, _ = predict_probs(
            model, test_paths, train_args.image_size, train_args, device,
        )

        # Run all scalable methods
        print(f"  Running transductive methods...")
        test_results = run_all_scalable(
            sup_feats, sup_labels, test_feats, num_classes=5,
            tukey_beta=args.tukey_beta, maha_shrink=args.maha_shrink,
            use_combined_mean=args.use_combined_mean,
            tim_params=tim_params,
        )

        # Print label distribution per method
        print(f"\n  Per-method label distribution on test_shuffled:")
        print(f"  {'Method':<20} " + "  ".join(
            f"{'Cl'+str(c):>6}" for c in range(5)))
        print(f"  {'-'*56}")
        for name, pred in test_results.items():
            counts = np.bincount(pred, minlength=5)
            print(f"  {name:<20} " + "  ".join(f"{c:>6}" for c in counts))
        head_pred_dist = np.bincount(head_probs.argmax(axis=1), minlength=5)
        print(f"  {'head':<20} " + "  ".join(f"{c:>6}" for c in head_pred_dist))

        # Save per-round predictions for all methods
        round_dir = out_root / f"round_{r}"
        round_dir.mkdir(parents=True, exist_ok=True)
        for name, pred in test_results.items():
            labels = [IDX_TO_CLASS[int(i)] for i in pred]
            df = pd.DataFrame({"filename": test_names, "label": labels})
            df.to_csv(round_dir / f"sub_{name}.csv", index=False)
        # Also head
        head_labels = [IDX_TO_CLASS[int(i)] for i in head_probs.argmax(axis=1)]
        df_head = pd.DataFrame({"filename": test_names, "label": head_labels})
        df_head.to_csv(round_dir / "sub_head.csv", index=False)
        print(f"  [ok] saved all method CSVs -> {round_dir}/")

        # Save checkpoint
        torch.save({
            "state_dict": model.state_dict(),
            "args": vars(train_args),
            "round": r,
            "train_size": len(train_paths),
            "n_pseudo": n_pseudo,
        }, round_dir / "ckpt.pt")

        # ── Pseudo-label selection (not on last round) ──
        if r < args.rounds:
            print(f"\n  Selecting pseudo-labels for round {r+1}...")

            # Use HEAD probs for confidence-based selection
            # TIM probs are too soft (max ~0.25-0.45) for thresholding
            print(f"  Using HEAD probs for pseudo-label selection...")
            print(f"  HEAD confidence stats: "
                  f"min={head_probs.max(axis=1).min():.4f}, "
                  f"mean={head_probs.max(axis=1).mean():.4f}, "
                  f"max={head_probs.max(axis=1).max():.4f}")

            # Determine threshold and topk for this round
            conf_thresh = args.conf_thresh if args.conf_thresh is not None \
                else CONF_SCHEDULE[min(r, len(CONF_SCHEDULE) - 1)]
            topk = args.per_class_topk if args.per_class_topk is not None \
                else TOPK_SCHEDULE[min(r, len(TOPK_SCHEDULE) - 1)]

            print(f"  Confidence threshold: {conf_thresh}")
            print(f"  Per-class top-k: {topk}")

            newly_selected, per_class = select_pseudo_labels(
                head_probs, test_names, conf_thresh, topk, used_names,
            )

            print(f"  Selected {len(newly_selected)} new pseudo-labels")
            print(f"  Per-class: {per_class.tolist()}")

            if newly_selected:
                # Confidence stats
                confs = [c for _, _, c in newly_selected]
                print(f"  Confidence: min={min(confs):.4f}, "
                      f"mean={np.mean(confs):.4f}, max={max(confs):.4f}")

                # Save pseudo-labels for this round
                pd.DataFrame(newly_selected,
                             columns=["filename", "pseudo_class", "confidence"]
                             ).to_csv(round_dir / "pseudo_added.csv", index=False)

            accumulated_pseudo.extend(newly_selected)
            used_names.update(n for n, _, _ in newly_selected)

            total_pseudo = len(accumulated_pseudo)
            print(f"  Total accumulated pseudo-labels: {total_pseudo}")

            round_elapsed = time.time() - t_round_start
            round_log.append({
                "round": r,
                "train_size": len(train_paths),
                "n_pseudo_used": n_pseudo,
                "n_new_pseudo": len(newly_selected),
                "n_total_pseudo": total_pseudo,
                "conf_thresh": conf_thresh,
                "per_class_topk": topk,
                "train_time_sec": round(train_time, 1),
                "extract_time_sec": round(extract_time, 1),
                "total_time_sec": round(round_elapsed, 1),
            })

            # Save resume state
            save_resume_state(out_root, r, accumulated_pseudo,
                              used_names, round_log)

        else:
            # Last round — generate final submission
            print(f"\n  Generating final submission with method={args.final_method}...")
            if args.final_method == "head":
                final_pred = head_probs.argmax(axis=1)
            else:
                final_pred = test_results[args.final_method]

            final_labels = [IDX_TO_CLASS[int(i)] for i in final_pred]
            sub_df = pd.DataFrame({
                "filename": test_names,
                "label": final_labels,
            })
            sub_path = out_root / "final_submission.csv"
            sub_df.to_csv(sub_path, index=False)
            print(f"\n  [ok] final submission -> {sub_path}")
            print(f"  Method: {args.final_method}")
            print(f"  Label distribution:")
            print(sub_df["label"].value_counts().to_string().replace("\n", "\n  "))

            # Also save all method submissions
            for name, pred in test_results.items():
                labels = [IDX_TO_CLASS[int(i)] for i in pred]
                df = pd.DataFrame({"filename": test_names, "label": labels})
                df.to_csv(out_root / f"final_sub_{name}.csv", index=False)
            head_labels = [IDX_TO_CLASS[int(i)] for i in head_probs.argmax(axis=1)]
            pd.DataFrame({"filename": test_names, "label": head_labels}
                         ).to_csv(out_root / "final_sub_head.csv", index=False)
            print(f"  [ok] saved all method submissions -> {out_root}/")

            round_elapsed = time.time() - t_round_start
            round_log.append({
                "round": r,
                "train_size": len(train_paths),
                "n_pseudo_used": n_pseudo,
                "n_new_pseudo": 0,
                "n_total_pseudo": len(accumulated_pseudo),
                "train_time_sec": round(train_time, 1),
                "extract_time_sec": round(extract_time, 1),
                "total_time_sec": round(round_elapsed, 1),
            })

        # Clean up
        del model
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  SELF-TRAINING SUMMARY")
    print(f"{'=' * 70}")
    for entry in round_log:
        print(f"  Round {entry['round']}: "
              f"train={entry['train_size']} "
              f"(+{entry['n_pseudo_used']} pseudo used), "
              f"new_pseudo={entry['n_new_pseudo']}, "
              f"total_pseudo={entry['n_total_pseudo']}, "
              f"time={timedelta(seconds=int(entry['total_time_sec']))}")

    with open(out_root / "summary.json", "w") as f:
        json.dump(round_log, f, indent=2)
    print(f"  [ok] summary -> {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
