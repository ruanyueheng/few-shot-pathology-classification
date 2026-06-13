"""Self-training (pseudo-labeling) on unlabeled test set — with resume support.

Pipeline (one round):
  1. Train LoRA model on current labeled set (=train_dir + accumulated pseudo-labels)
  2. Run inference on --test_dir, get softmax probs per image
  3. Select pseudo-labels:
       - confidence >= --conf_thresh  AND
       - per-class quota: top-K most-confident per class (to handle test imbalance)
  4. Append pseudo-labeled images to the labeled pool
  5. Repeat for --rounds rounds

Resume support:
  After each round completes, a `resume_state.json` is written to --out_dir.
  On restart, if --resume is given and `resume_state.json` exists, the script
  skips all already-completed rounds and continues from where it left off.

Iteration writes:
  artifacts/selftrain/round_K/
    - ckpt.pt              (LoRA checkpoint of that round)
    - pseudo_added.csv     (which test imgs got which pseudo-label this round)
    - eval.json            (if gt_csv: macro-F1, bal-acc on frozen test)
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
from transductive import simpleshot, laplacianshot, mahalanobis, tim
from tqdm import tqdm


@torch.no_grad()
def extract_features(model, paths, image_size: int, args, device,
                     labels: np.ndarray | None = None):
    """Extract CLS features from model.backbone for either labeled or unlabeled images."""
    eval_tf = make_eval_tf(image_size)
    if labels is not None:
        ds = ImagePathDataset(paths, labels, eval_tf)
    else:
        ds = ImagePathDataset(paths, None, eval_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    model.eval()
    feats, second = [], []
    for x, y_or_name in tqdm(loader, desc="extract_feats", leave=False):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
            f = model.backbone(x)
        feats.append(f.float().cpu().numpy())
        if labels is not None:
            second.append(y_or_name.numpy())
        else:
            second.extend(list(y_or_name))
    feats = np.concatenate(feats, axis=0)
    if labels is not None:
        second = np.concatenate(second, axis=0)
    return feats, second


@torch.no_grad()
def predict_probs(model, paths, image_size: int, args, device):
    eval_tf = make_eval_tf(image_size)
    ds = ImagePathDataset(paths, None, eval_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    model.eval()
    all_probs, all_names = [], []
    for x, names in tqdm(loader, desc="predict_probs", leave=False):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
            logits = model(x)
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_names.extend(list(names))
    return np.concatenate(all_probs, axis=0), all_names


def select_pseudo(probs: np.ndarray, names: list[str],
                  conf_thresh: float, per_class_topk: int | None,
                  per_class_min: int | None,
                  already_used: set[str]) -> list[tuple[str, int, float]]:
    """Return list of (filename, predicted_class, confidence) to be added.

    Selection per class c:
      1. all images predicted as c with conf >= conf_thresh
      2. if fewer than per_class_min, top up by taking highest-confidence
         predictions-as-c regardless of threshold (protects minority classes)
      3. then cap to per_class_topk
    """
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    selected: list[tuple[str, int, float]] = []
    for c in range(probs.shape[1]):
        # all images predicted as class c, not already used
        all_c = [i for i in np.where(pred == c)[0] if names[i] not in already_used]
        all_c.sort(key=lambda i: -conf[i])
        thresh_c = [i for i in all_c if conf[i] >= conf_thresh]
        # if too few above threshold, force per_class_min from highest-conf in this class
        if per_class_min is not None and len(thresh_c) < per_class_min:
            picked = all_c[:per_class_min]
        else:
            picked = thresh_c
        if per_class_topk is not None:
            picked = picked[:per_class_topk]
        for i in picked:
            selected.append((names[i], int(c), float(conf[i])))
    return selected


def materialize_round_dir(round_dir: Path, base_train_dir: Path,
                          test_dir: Path,
                          pseudo: list[tuple[str, int, float]],
                          accumulated_pseudo: list[tuple[str, int, float]]):
    """Build a directory with the labeled training data for this round:
       base train images + accumulated pseudo-labeled test images."""
    new_train = round_dir / "labeled"
    if new_train.exists():
        shutil.rmtree(new_train)
    for c in CLASS_NAMES:
        (new_train / c).mkdir(parents=True, exist_ok=True)

    # copy original train data
    n_orig = 0
    for c in CLASS_NAMES:
        for f in sorted((base_train_dir / c).glob("*.png")):
            shutil.copy2(f, new_train / c / f.name)
            n_orig += 1

    # write pseudo-labeled test images (accumulated)
    n_pseudo = 0
    for name, c_idx, _conf in accumulated_pseudo:
        src = test_dir / name
        dst = new_train / CLASS_NAMES[c_idx] / f"pseudo_{name}"
        if src.exists():
            shutil.copy2(src, dst)
            n_pseudo += 1
    return new_train, n_orig, n_pseudo


def evaluate_on_test(model, paths, gt_map: dict[str, int], args, device):
    """If gt_map is given (filename -> class idx), compute test metrics."""
    probs, names = predict_probs(model, paths, args.image_size, args, device)
    pred = probs.argmax(axis=1)
    y_true = np.array([gt_map[n] for n in names])
    macro_f1 = f1_score(y_true, pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, pred)
    report = classification_report(y_true, pred, target_names=CLASS_NAMES,
                                   digits=4, output_dict=True)
    cm = confusion_matrix(y_true, pred).tolist()
    return {"macro_f1": float(macro_f1), "bal_acc": float(bal_acc),
            "per_class": report, "confusion": cm,
            "preds": list(zip(names, pred.tolist(),
                              probs.max(axis=1).tolist()))}


# ---------------------------------------------------------------------------
#  Resume helpers
# ---------------------------------------------------------------------------

RESUME_FILE = "resume_state.json"


def save_resume_state(out_root: Path, completed_round: int,
                      accumulated_pseudo: list[tuple[str, int, float]],
                      used_names: set[str],
                      round_metrics: list[dict]):
    """Save resume state after a round completes."""
    state = {
        "completed_round": completed_round,
        "accumulated_pseudo": [
            {"name": n, "class_idx": int(c), "conf": float(cf)}
            for n, c, cf in accumulated_pseudo
        ],
        "used_names": sorted(used_names),
        "round_metrics": round_metrics,
    }
    with open(out_root / RESUME_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"[resume] state saved after round {completed_round} -> "
          f"{out_root / RESUME_FILE}")


def load_resume_state(out_root: Path):
    """Load resume state. Returns (start_round, accumulated_pseudo, used_names,
    round_metrics) or None if no state file exists."""
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
    round_metrics = state.get("round_metrics", [])
    completed_round = state["completed_round"]
    print(f"[resume] found state file, completed_round={completed_round}, "
          f"accumulated_pseudo={len(accumulated_pseudo)}, "
          f"used_names={len(used_names)}")
    return completed_round + 1, accumulated_pseudo, used_names, round_metrics


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True,
                    help="labeled training dir (e.g. ../dev_few_shot)")
    ap.add_argument("--test_dir", required=True,
                    help="unlabeled test image dir (e.g. ../frozen_test)")
    ap.add_argument("--gt_csv", default=None,
                    help="optional ground-truth csv (filename,label) for internal eval")
    ap.add_argument("--out_dir", default="../artifacts/selftrain")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--resume", action="store_true",
                    help="resume from last completed round (reads resume_state.json)")
    ap.add_argument("--conf_thresh", type=float, default=0.9)
    ap.add_argument("--pseudo_method", default="head",
                    choices=["head", "simpleshot", "mahalanobis"],
                    help="how to compute confidence for pseudo-label selection")
    ap.add_argument("--final_method", default="head",
                    choices=["head", "simpleshot", "laplacianshot", "mahalanobis", "tim"],
                    help="how to predict after all rounds finish (writes final_submission.csv)")
    ap.add_argument("--tim_iter", type=int, default=1000)
    ap.add_argument("--tim_lr", type=float, default=1e-4)
    ap.add_argument("--tim_temp", type=float, default=15.0)
    ap.add_argument("--tim_lambda_marg", type=float, default=1.0)
    ap.add_argument("--tim_lambda_cond", type=float, default=0.1)
    ap.add_argument("--use_combined_mean", action="store_true",
                    help="transductive: use mean(support ∪ test) for feature centering")
    ap.add_argument("--maha_shrink", type=float, default=0.3)
    ap.add_argument("--lap_knn", type=int, default=3)
    ap.add_argument("--lap_lam", type=float, default=0.1)
    ap.add_argument("--lap_n_iter", type=int, default=20)
    ap.add_argument("--lap_sigma", type=float, default=1.0)
    ap.add_argument("--tukey_beta", type=float, default=1.0)
    ap.add_argument("--per_class_topk", type=int, default=None,
                    help="if given, max pseudo-labels per class per round")
    ap.add_argument("--per_class_min", type=int, default=None,
                    help="if given, force at least this many pseudo-labels per class "
                         "(ignores conf_thresh for under-represented classes)")

    # training hyperparameters (forwarded to train_one_run)
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
    ap.add_argument("--lora_mlp", action="store_true")
    ap.add_argument("--dora", action="store_true")
    ap.add_argument("--lora_init", default="default",
                    choices=["default", "gaussian", "pissa", "pissa_niter_4",
                             "pissa_niter_16", "olora", "loftq", "orthogonal"])
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.95)
    ap.add_argument("--hed_aug", action="store_true")
    ap.add_argument("--hed_sigma", type=float, default=0.02)
    ap.add_argument("--hed_bias", type=float, default=0.01)
    ap.add_argument("--head_dropout", type=float, default=0.1)
    ap.add_argument("--supcon_weight", type=float, default=0.0,
                    help="weight for Supervised Contrastive loss (0 = off, "
                         "0.3-1.0 typical; shapes feature geometry vs drift)")
    ap.add_argument("--supcon_temp", type=float, default=0.07,
                    help="SupCon temperature")
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--cutmix", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    base_train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    # load ground truth map if provided (internal eval only)
    gt_map = None
    if args.gt_csv:
        gt_df = pd.read_csv(args.gt_csv)
        gt_map = {row.filename: CLASS_TO_IDX[row.label] for row in gt_df.itertuples()}
        print(f"[info] loaded {len(gt_map)} ground-truth labels for internal eval")

    test_paths = list_test_samples(test_dir)
    print(f"[info] {len(test_paths)} unlabeled test images at {test_dir}")

    # ----- Resume logic -----
    start_round = 0
    accumulated_pseudo: list[tuple[str, int, float]] = []
    used_names: set[str] = set()
    round_metrics = []

    if args.resume:
        state = load_resume_state(out_root)
        if state is not None:
            start_round, accumulated_pseudo, used_names, round_metrics = state
            print(f"[resume] skipping rounds 0..{start_round - 1}, "
                  f"starting from round {start_round}")
            # rebuild round_metrics summary if resuming
            if round_metrics:
                print("[resume] previous round metrics:")
                for m in round_metrics:
                    print(f"  round {m['round']}: macro-F1={m['macro_f1']:.4f}  "
                          f"bal-acc={m['bal_acc']:.4f}  (+{m['n_pseudo']} pseudo)")
        else:
            print("[resume] no resume_state.json found, starting from scratch")
    # ----- End resume logic -----

    for r in tqdm(range(start_round, args.rounds + 1), desc="self-train rounds",
                  initial=start_round, total=args.rounds + 1):
        round_dir = out_root / f"round_{r}"
        round_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n========== round {r} ==========")

        # build labeled set for this round
        labeled_dir, n_orig, n_pseudo = materialize_round_dir(
            round_dir, base_train_dir, test_dir,
            pseudo=[], accumulated_pseudo=accumulated_pseudo,
        )
        print(f"[info] labeled pool: {n_orig} orig + {n_pseudo} pseudo = "
              f"{n_orig+n_pseudo} images")

        # train on this round's labeled set (using full set, no held-out val
        # since we already split frozen_test out)
        paths, labels = list_train_samples(labeled_dir)
        full_idx = np.arange(len(paths))
        model, _, _ = train_one_run(
            paths, labels, full_idx, full_idx, args, device,
            run_id=r, refit_all=True,
        )

        # evaluate on internal frozen test if gt provided
        metrics = None
        if gt_map is not None:
            metrics = evaluate_on_test(model, test_paths, gt_map, args, device)
            print(f"[eval] round{r} macro-F1 = {metrics['macro_f1']:.4f}  "
                  f"bal-acc = {metrics['bal_acc']:.4f}")
            with open(round_dir / "eval.json", "w") as f:
                json.dump({k: v for k, v in metrics.items() if k != "preds"},
                          f, indent=2)
            # save predictions
            preds_df = pd.DataFrame(metrics["preds"],
                                    columns=["filename", "pred_idx", "confidence"])
            preds_df["pred_label"] = preds_df["pred_idx"].map(IDX_TO_CLASS)
            preds_df.to_csv(round_dir / "predictions.csv", index=False)
            round_metrics.append({"round": r, **{k: v for k, v in metrics.items()
                                                  if k in ("macro_f1", "bal_acc")},
                                  "n_pseudo": n_pseudo})

        # save model
        torch.save({"state_dict": model.state_dict(), "args": vars(args)},
                   round_dir / "ckpt.pt")

        # if not the last round, pick pseudo-labels for next round
        if r < args.rounds:
            if args.pseudo_method == "head":
                probs, names = predict_probs(model, test_paths, args.image_size,
                                              args, device)
            else:
                # use transductive method (simpleshot/mahalanobis) on extracted features
                sup_paths_p, sup_labels_p = list_train_samples(labeled_dir)
                sup_feats_p, sup_lab_p = extract_features(
                    model, sup_paths_p, args.image_size, args, device,
                    labels=sup_labels_p,
                )
                test_feats_p, names = extract_features(
                    model, test_paths, args.image_size, args, device, labels=None,
                )
                if args.pseudo_method == "simpleshot":
                    _, probs = simpleshot(
                        sup_feats_p, sup_lab_p, test_feats_p, num_classes=5,
                        tukey_beta=args.tukey_beta,
                        use_combined_mean=args.use_combined_mean,
                        return_probs=True,
                    )
                else:  # mahalanobis
                    _, probs = mahalanobis(
                        sup_feats_p, sup_lab_p, test_feats_p, num_classes=5,
                        tukey_beta=args.tukey_beta,
                        use_combined_mean=args.use_combined_mean,
                        shrink=args.maha_shrink,
                        return_probs=True,
                    )
                print(f"[info] pseudo confidences from method={args.pseudo_method}")
            newly_selected = select_pseudo(
                probs, names, args.conf_thresh,
                args.per_class_topk, args.per_class_min, used_names,
            )
            print(f"[info] round{r} picked {len(newly_selected)} new pseudo-labels "
                  f"(conf>={args.conf_thresh}"
                  f"{f', topk={args.per_class_topk}/class' if args.per_class_topk else ''})")
            # distribution
            if newly_selected:
                cls_count = np.bincount([c for _, c, _ in newly_selected],
                                         minlength=5).tolist()
                print(f"        per-class added: {cls_count}")
            accumulated_pseudo.extend(newly_selected)
            used_names.update(n for n, _, _ in newly_selected)
            pd.DataFrame(newly_selected,
                         columns=["filename", "pseudo_class", "confidence"]).to_csv(
                round_dir / "pseudo_added.csv", index=False,
            )
            # ---- save resume state after each non-final round ----
            save_resume_state(out_root, r, accumulated_pseudo, used_names,
                              round_metrics)

        # On the LAST round, do the final prediction with chosen method, save submission.
        if r == args.rounds:
            print(f"\n[info] final prediction with method={args.final_method}")
            # use the FINAL (last-round) labeled pool as support — original train + accepted pseudos.
            sup_paths, sup_labels = list_train_samples(labeled_dir)
            if args.final_method == "head":
                # reuse existing head-based predictions written above
                if metrics is not None and "preds" in metrics:
                    final_pred = np.array([p for _, p, _ in metrics["preds"]])
                    final_names = [n for n, _, _ in metrics["preds"]]
                else:
                    # no gt_map, need to predict
                    probs_head, names_head = predict_probs(
                        model, test_paths, args.image_size, args, device)
                    final_pred = probs_head.argmax(axis=1)
                    final_names = names_head
            else:
                sup_feats, sup_lab = extract_features(
                    model, sup_paths, args.image_size, args, device,
                    labels=sup_labels,
                )
                test_feats, final_names = extract_features(
                    model, test_paths, args.image_size, args, device, labels=None,
                )
                print(f"[info] support feats {sup_feats.shape}, "
                      f"query feats {test_feats.shape}")
                if args.final_method == "simpleshot":
                    final_pred = simpleshot(
                        sup_feats, sup_lab, test_feats, num_classes=5,
                        tukey_beta=args.tukey_beta,
                        use_combined_mean=args.use_combined_mean,
                    )
                elif args.final_method == "mahalanobis":
                    final_pred = mahalanobis(
                        sup_feats, sup_lab, test_feats, num_classes=5,
                        tukey_beta=args.tukey_beta,
                        use_combined_mean=args.use_combined_mean,
                        shrink=args.maha_shrink,
                    )
                elif args.final_method == "tim":
                    final_pred = tim(
                        sup_feats, sup_lab, test_feats, num_classes=5,
                        tukey_beta=args.tukey_beta,
                        use_combined_mean=args.use_combined_mean,
                        n_iter=args.tim_iter, lr=args.tim_lr,
                        temperature=args.tim_temp,
                        lambda_marg=args.tim_lambda_marg,
                        lambda_cond=args.tim_lambda_cond,
                    )
                else:  # laplacianshot
                    final_pred = laplacianshot(
                        sup_feats, sup_lab, test_feats, num_classes=5,
                        knn=args.lap_knn, lam=args.lap_lam,
                        n_iter=args.lap_n_iter, sigma=args.lap_sigma,
                        tukey_beta=args.tukey_beta,
                        use_combined_mean=args.use_combined_mean,
                    )
            # write final submission
            final_labels = [IDX_TO_CLASS[int(i)] for i in final_pred]
            sub_df = pd.DataFrame({"filename": final_names, "label": final_labels})
            sub_path = out_root / "final_submission.csv"
            sub_df.to_csv(sub_path, index=False)
            print(f"[ok] final submission -> {sub_path}")
            print(sub_df["label"].value_counts())

            # if gt available, eval the final method
            if gt_map is not None:
                y_true = np.array([gt_map[n] for n in final_names])
                f1 = float(f1_score(y_true, final_pred, average="macro"))
                ba = float(balanced_accuracy_score(y_true, final_pred))
                print(f"[eval] FINAL ({args.final_method}) "
                      f"macro-F1 = {f1:.4f}  bal-acc = {ba:.4f}")
                with open(out_root / "final_eval.json", "w") as f:
                    json.dump({"final_method": args.final_method,
                               "macro_f1": f1, "bal_acc": ba}, f, indent=2)

            # ---- save final resume state (all rounds done) ----
            save_resume_state(out_root, r, accumulated_pseudo, used_names,
                              round_metrics)

        del model
        torch.cuda.empty_cache()

    # summary
    if round_metrics:
        print("\n========== self-training summary ==========")
        for m in round_metrics:
            print(f"  round {m['round']}: macro-F1={m['macro_f1']:.4f}  "
                  f"bal-acc={m['bal_acc']:.4f}  (+{m['n_pseudo']} pseudo)")
        with open(out_root / "summary.json", "w") as f:
            json.dump(round_metrics, f, indent=2)


if __name__ == "__main__":
    main()
