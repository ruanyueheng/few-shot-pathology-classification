"""Quick inference on test_shuffled using an existing trained checkpoint.

Usage:
    python predict_existing.py \
        --ckpt ../artifacts/best_v02_f0.8351_mahalanobis/round_3/ckpt.pt \
        --train_dir ../train_few_shot \
        --test_dir ../test_shuffled \
        --final_method mahalanobis \
        --tukey_beta 0.5 \
        --out_csv ../sub_existing_maha.csv

This loads the saved LoRA checkpoint, extracts features, and runs
Mahalanobis (or other transductive method) to produce predictions.
No training involved — takes ~1-2 minutes.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from self_train import (
    extract_features, list_train_samples, list_test_samples, CLASS_NAMES,
    CLASS_TO_IDX, IDX_TO_CLASS,
)
from train_lora import LoRAViTClassifier, make_eval_tf
from transductive import simpleshot, laplacianshot, mahalanobis, tim
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to ckpt.pt")
    ap.add_argument("--train_dir", required=True,
                    help="labeled training dir (support set for transductive)")
    ap.add_argument("--test_dir", required=True, help="unlabeled test image dir")
    ap.add_argument("--final_method", default="mahalanobis",
                    choices=["head", "simpleshot", "laplacianshot", "mahalanobis", "tim"])
    ap.add_argument("--maha_shrink", type=float, default=0.3)
    ap.add_argument("--tukey_beta", type=float, default=1.0)
    ap.add_argument("--use_combined_mean", action="store_true")
    ap.add_argument("--tim_iter", type=int, default=1000)
    ap.add_argument("--tim_lr", type=float, default=1e-4)
    ap.add_argument("--tim_temp", type=float, default=15.0)
    ap.add_argument("--tim_lambda_marg", type=float, default=1.0)
    ap.add_argument("--tim_lambda_cond", type=float, default=0.1)
    ap.add_argument("--lap_knn", type=int, default=3)
    ap.add_argument("--lap_lam", type=float, default=0.1)
    ap.add_argument("--lap_n_iter", type=int, default=20)
    ap.add_argument("--lap_sigma", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out_csv", default="../sub_existing_maha.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    saved = ckpt["args"]

    # reconstruct model from saved config
    model = LoRAViTClassifier(
        backbone=saved.get("backbone", "vits14"),
        num_classes=5,
        lora_r=saved.get("lora_r", 32),
        lora_alpha=saved.get("lora_alpha", 64),
        lora_dropout=saved.get("lora_dropout", 0.1),
        lora_mlp=saved.get("lora_mlp", False),
        dora=saved.get("dora", False),
        lora_init=saved.get("lora_init", "default"),
        head_dropout=saved.get("head_dropout", 0.1),
        image_size=saved.get("image_size", 224),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[info] loaded checkpoint from {args.ckpt}")
    print(f"       backbone={saved.get('backbone')}, lora_r={saved.get('lora_r')}, "
          f"lora_alpha={saved.get('lora_alpha')}")

    # list samples
    sup_paths, sup_labels = list_train_samples(Path(args.train_dir))
    test_paths = list_test_samples(Path(args.test_dir))
    print(f"[info] support: {len(sup_paths)} images, test: {len(test_paths)} images")

    # extract features
    print("[1/2] extracting support features...")
    sup_feats, _ = extract_features(
        model, sup_paths, saved.get("image_size", 224), args, device,
        labels=sup_labels,
    )
    print(f"       support feats: {sup_feats.shape}")

    print("[2/2] extracting test features...")
    test_feats, test_names = extract_features(
        model, test_paths, saved.get("image_size", 224), args, device,
        labels=None,
    )
    print(f"       test feats: {test_feats.shape}")

    # predict with chosen method
    print(f"[info] predicting with method={args.final_method}")
    if args.final_method == "head":
        # use model's classification head directly
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from self_train import ImagePathDataset
        eval_tf = make_eval_tf(saved.get("image_size", 224))
        ds = ImagePathDataset(test_paths, None, eval_tf)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        all_preds = []
        model.eval()
        with torch.no_grad():
            for x, _ in tqdm(loader, desc="head predict"):
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=(device == "cuda")):
                    logits = model(x)
                all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        final_pred = np.concatenate(all_preds)
    elif args.final_method == "simpleshot":
        final_pred = simpleshot(
            sup_feats, sup_labels, test_feats, num_classes=5,
            tukey_beta=args.tukey_beta,
            use_combined_mean=args.use_combined_mean,
        )
    elif args.final_method == "mahalanobis":
        final_pred = mahalanobis(
            sup_feats, sup_labels, test_feats, num_classes=5,
            tukey_beta=args.tukey_beta,
            use_combined_mean=args.use_combined_mean,
            shrink=args.maha_shrink,
        )
    elif args.final_method == "tim":
        final_pred = tim(
            sup_feats, sup_labels, test_feats, num_classes=5,
            tukey_beta=args.tukey_beta,
            use_combined_mean=args.use_combined_mean,
            n_iter=args.tim_iter, lr=args.tim_lr,
            temperature=args.tim_temp,
            lambda_marg=args.tim_lambda_marg,
            lambda_cond=args.tim_lambda_cond,
        )
    else:  # laplacianshot
        final_pred = laplacianshot(
            sup_feats, sup_labels, test_feats, num_classes=5,
            knn=args.lap_knn, lam=args.lap_lam,
            n_iter=args.lap_n_iter, sigma=args.lap_sigma,
            tukey_beta=args.tukey_beta,
            use_combined_mean=args.use_combined_mean,
        )

    # write submission
    final_labels = [IDX_TO_CLASS[int(i)] for i in final_pred]
    sub_df = pd.DataFrame({"filename": test_names, "label": final_labels})
    sub_df.to_csv(args.out_csv, index=False)
    print(f"\n[ok] submission saved -> {args.out_csv}")
    print(sub_df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
