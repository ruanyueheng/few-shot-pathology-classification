"""Inference using a LoRA-finetuned ViT checkpoint -> submission.csv.

Supports three inference methods:
  - head:           use the trained classification head (default, inductive)
  - simpleshot:     center+L2 features, nearest cosine prototype (training-free, transductive ish)
  - laplacianshot:  prototypes + graph-Laplacian smoothing over query embeddings (transductive)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import (
    list_train_samples, list_test_samples, ImagePathDataset,
    IDX_TO_CLASS, CLASS_NAMES,
)
from train_lora import LoRAViTClassifier, make_eval_tf
from transductive import (simpleshot, laplacianshot, mahalanobis, tim,
                          label_propagation, pt_map, alpha_tim)


@torch.no_grad()
def extract_feats(model, paths, image_size: int, args, device, with_labels: bool):
    eval_tf = make_eval_tf(image_size)
    if with_labels:
        # paths is (paths_list, labels_array)
        plist, labels = paths
        ds = ImagePathDataset(plist, labels, eval_tf)
    else:
        ds = ImagePathDataset(paths, None, eval_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    model.eval()
    feats, ys_or_names = [], []
    for batch in loader:
        x, second = batch
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
            f = model.backbone(x)        # CLS features, [B, D]
        feats.append(f.float().cpu().numpy())
        if with_labels:
            ys_or_names.append(second.numpy())
        else:
            ys_or_names.extend(list(second))
    feats = np.concatenate(feats, axis=0)
    if with_labels:
        ys_or_names = np.concatenate(ys_or_names, axis=0)
    return feats, ys_or_names


@torch.no_grad()
def predict_with_head(model, paths, args, device, image_size: int):
    ds = ImagePathDataset(paths, None, make_eval_tf(image_size))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    model.eval()
    preds, names = [], []
    for x, n in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
            logits = model(x)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        names.extend(list(n))
    return np.concatenate(preds), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--ckpt", required=True, help="LoRA checkpoint .pt path")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--method", default="head",
                    choices=["head", "simpleshot", "laplacianshot", "mahalanobis",
                             "tim", "lp", "ptmap", "alpha_tim"])
    ap.add_argument("--train_dir", default=None,
                    help="needed for simpleshot / laplacianshot (support set)")
    # LaplacianShot hyperparameters
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--n_iter", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--tukey_beta", type=float, default=1.0,
                    help="Tukey power (β=0.5 often helps few-shot; 1.0 = no transform)")
    ap.add_argument("--use_combined_mean", action="store_true",
                    help="transductive centering: use mean(support ∪ test) for L2 norm")
    ap.add_argument("--maha_shrink", type=float, default=0.3,
                    help="shrinkage for Mahalanobis pooled covariance (0 = no shrinkage)")
    # TIM hyperparameters
    ap.add_argument("--tim_iter", type=int, default=1000)
    ap.add_argument("--tim_lr", type=float, default=1e-4)
    ap.add_argument("--tim_temp", type=float, default=15.0)
    ap.add_argument("--tim_lambda_marg", type=float, default=1.0,
                    help="weight on marginal class entropy (cover all classes)")
    ap.add_argument("--tim_lambda_cond", type=float, default=0.1,
                    help="weight on conditional entropy (per-sample confidence)")
    # PT-MAP hyperparameters
    ap.add_argument("--ptmap_lambda", type=float, default=10.0,
                    help="PT-MAP temperature λ=1/(2σ²) (default 10.0)")
    ap.add_argument("--ptmap_iter", type=int, default=20,
                    help="PT-MAP soft-EM iterations (default 20)")
    ap.add_argument("--ptmap_sinkhorn", type=int, default=10,
                    help="Sinkhorn iterations per EM step (0 = disable Sinkhorn)")
    # alpha-TIM hyperparameters
    ap.add_argument("--atim_alpha", type=float, default=2.0,
                    help="alpha-TIM divergence order (α>1 → push toward balance; α→1 = TIM)")
    ap.add_argument("--atim_iter", type=int, default=1000)
    ap.add_argument("--atim_lr", type=float, default=1e-4)
    ap.add_argument("--atim_temp", type=float, default=15.0)
    ap.add_argument("--atim_lambda_marg", type=float, default=1.0)
    ap.add_argument("--atim_lambda_cond", type=float, default=0.1)
    # Label Propagation hyperparameters
    ap.add_argument("--lp_knn", type=int, default=10)
    ap.add_argument("--lp_alpha", type=float, default=0.7)
    ap.add_argument("--lp_sigma", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    print(f"[info] device={device}, backbone={cfg['backbone']}, "
          f"image_size={cfg['image_size']}, method={args.method}")

    model = LoRAViTClassifier(
        backbone_key=cfg["backbone"], image_size=cfg["image_size"],
        lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"], head_dropout=cfg["head_dropout"],
        lora_mlp=cfg.get("lora_mlp", False),
        use_dora=cfg.get("dora", False),
        lora_init=cfg.get("lora_init", "default"),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    test_paths = list_test_samples(args.test_dir)
    if not test_paths:
        raise SystemExit(f"No .png files in {args.test_dir}")
    print(f"[info] test images: {len(test_paths)}")

    if args.method == "head":
        preds, names = predict_with_head(model, test_paths, args, device,
                                         image_size=cfg["image_size"])
    else:
        if args.train_dir is None:
            raise SystemExit(f"--train_dir required for method={args.method}")
        sup_paths, sup_labels = list_train_samples(args.train_dir)
        print(f"[info] support set: {len(sup_paths)} from {args.train_dir}")
        sup_feats, _ = extract_feats(
            model, (sup_paths, sup_labels),
            image_size=cfg["image_size"], args=args, device=device, with_labels=True,
        )
        test_feats, names = extract_feats(
            model, test_paths,
            image_size=cfg["image_size"], args=args, device=device, with_labels=False,
        )
        print(f"[info] features: support={sup_feats.shape}, test={test_feats.shape}")
        if args.method == "simpleshot":
            preds = simpleshot(sup_feats, sup_labels, test_feats, num_classes=5,
                               tukey_beta=args.tukey_beta,
                               use_combined_mean=args.use_combined_mean)
        elif args.method == "mahalanobis":
            preds = mahalanobis(sup_feats, sup_labels, test_feats, num_classes=5,
                                tukey_beta=args.tukey_beta,
                                use_combined_mean=args.use_combined_mean,
                                shrink=args.maha_shrink)
        elif args.method == "tim":
            preds = tim(sup_feats, sup_labels, test_feats, num_classes=5,
                        tukey_beta=args.tukey_beta,
                        use_combined_mean=args.use_combined_mean,
                        n_iter=args.tim_iter, lr=args.tim_lr,
                        temperature=args.tim_temp,
                        lambda_marg=args.tim_lambda_marg,
                        lambda_cond=args.tim_lambda_cond)
        elif args.method == "lp":
            preds = label_propagation(sup_feats, sup_labels, test_feats, num_classes=5,
                                      knn=args.lp_knn, alpha=args.lp_alpha,
                                      sigma=args.lp_sigma,
                                      tukey_beta=args.tukey_beta,
                                      use_combined_mean=args.use_combined_mean)
        elif args.method == "ptmap":
            preds = pt_map(sup_feats, sup_labels, test_feats, num_classes=5,
                           tukey_beta=args.tukey_beta,
                           n_iter=args.ptmap_iter,
                           lambda_s=args.ptmap_lambda,
                           use_sinkhorn=(args.ptmap_sinkhorn > 0),
                           sinkhorn_iter=args.ptmap_sinkhorn)
        elif args.method == "alpha_tim":
            preds = alpha_tim(sup_feats, sup_labels, test_feats, num_classes=5,
                              tukey_beta=args.tukey_beta,
                              use_combined_mean=args.use_combined_mean,
                              n_iter=args.atim_iter, lr=args.atim_lr,
                              temperature=args.atim_temp,
                              alpha=args.atim_alpha,
                              lambda_marg=args.atim_lambda_marg,
                              lambda_cond=args.atim_lambda_cond)
        else:
            preds = laplacianshot(
                sup_feats, sup_labels, test_feats, num_classes=5,
                knn=args.knn, lam=args.lam, n_iter=args.n_iter, sigma=args.sigma,
                tukey_beta=args.tukey_beta,
                use_combined_mean=args.use_combined_mean,
            )

    labels = [IDX_TO_CLASS[int(i)] for i in preds]
    df = pd.DataFrame({"filename": names, "label": labels})
    df.to_csv(args.out, index=False)
    print(f"[ok] wrote {args.out}  ({len(df)} rows)")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
