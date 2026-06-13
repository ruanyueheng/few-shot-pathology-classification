"""Inference using pathology-specific foundation models (no LoRA needed).

Supported backbones (all already cached):
  --backbone phikon      owkin/phikon            (0.3 GB, TCGA DINO ViT-B/16)
  --backbone plip        vinid/plip              (0.6 GB, pathology CLIP vision)
  --backbone conch       MahmoodLab/conch        (0.8 GB, pathology CLIP vision)
  --backbone lunit       1aurent/vit_small_patch16_224.lunit_dino  (0.1 GB)

Usage:
    cd src
    python infer_phikon.py --backbone phikon --train_dir ../dev_few_shot \
        --test_dir ../frozen_test --out ../sub_phikon.csv

    # Ensemble with existing DINOv2-LoRA features:
    python infer_phikon.py --backbone phikon --train_dir ../dev_few_shot \
        --test_dir ../frozen_test --out ../sub_phikon_ensemble.csv \
        --lora_ckpt ../artifacts/best_v02_f0.8351_mahalanobis/round_3/ckpt.pt \
        --ensemble_weight 0.5
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# reuse project utilities
sys.path.insert(0, str(Path(__file__).parent))
from data import (
    list_train_samples, list_test_samples,
    IDX_TO_CLASS, CLASS_NAMES,
)
from transductive import mahalanobis, simpleshot


# ---------------------------------------------------------------------------
# Backbone loaders
# ---------------------------------------------------------------------------

BACKBONE_REGISTRY = {
    "phikon":  "owkin/phikon",
    "plip":    "vinid/plip",
    "conch":   "MahmoodLab/conch",
    "lunit":   "1aurent/vit_small_patch16_224.lunit_dino",
    "clip_l":  "openai/clip-vit-large-patch14",   # general CLIP (transformers, cached)
}

# Normalisation stats: all these models were trained with ImageNet stats
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


def load_backbone(backbone_key: str, device: str):
    """Return (model, feature_dim, transform) for the requested backbone."""
    repo = BACKBONE_REGISTRY[backbone_key]
    print(f"[info] loading backbone: {repo}")

    if backbone_key == "phikon":
        from transformers import ViTModel
        model = ViTModel.from_pretrained(repo, add_pooling_layer=False)
        model = model.to(device).eval()
        feat_dim = model.config.hidden_size          # 768 for ViT-B
        tf = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
        def extract_fn(batch_tensor):
            # batch_tensor: [B, 3, 224, 224] already on device
            with torch.no_grad():
                out = model(pixel_values=batch_tensor)
            return out.last_hidden_state[:, 0, :]    # CLS token

    elif backbone_key == "lunit":
        import timm
        model = timm.create_model(repo, pretrained=True, num_classes=0)
        model = model.to(device).eval()
        data_cfg = timm.data.resolve_model_data_config(model)
        feat_dim = model.num_features
        tf = timm.data.create_transform(**data_cfg, is_training=False)
        def extract_fn(batch_tensor):
            with torch.no_grad():
                return model(batch_tensor)

    elif backbone_key in ("plip", "conch", "clip_l"):
        # CLIP-style models; use the vision encoder's pooled CLS embedding
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained(repo)
        model = model.to(device).eval()
        feat_dim = model.config.vision_config.hidden_size
        clip_mean = (0.4815, 0.4578, 0.4082)   # CLIP's own normalization
        clip_std = (0.2686, 0.2613, 0.2758)
        tf = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(clip_mean, clip_std),
        ])
        def extract_fn(batch_tensor):
            with torch.no_grad():
                out = model.vision_model(pixel_values=batch_tensor)
            return out.pooler_output

    else:
        raise ValueError(f"Unknown backbone: {backbone_key}")

    return model, feat_dim, tf, extract_fn


# ---------------------------------------------------------------------------
# Dataset / feature extraction
# ---------------------------------------------------------------------------

class SimpleDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths   = paths
        self.labels  = labels   # None for test set
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        return img, str(self.paths[idx].name)


@torch.no_grad()
def extract_feats(extract_fn, paths, labels, transform, batch_size, num_workers, device):
    ds = SimpleDataset(paths, labels, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    feats, second = [], []
    for batch in loader:
        x, y_or_n = batch
        x = x.to(device, non_blocking=True)
        f = extract_fn(x).float().cpu().numpy()
        feats.append(f)
        if labels is not None:
            second.append(y_or_n.numpy())
        else:
            second.extend(list(y_or_n))
    feats = np.concatenate(feats, axis=0)
    if labels is not None:
        second = np.concatenate(second, axis=0)
    return feats, second


# ---------------------------------------------------------------------------
# Optional: load DINOv2-LoRA features for ensemble
# ---------------------------------------------------------------------------

def load_lora_feats(ckpt_path: str, train_dir: str, test_dir: str,
                    batch_size: int, num_workers: int, device: str):
    """Extract features from an existing LoRA checkpoint."""
    import torch
    from train_lora import LoRAViTClassifier, make_eval_tf
    from data import ImagePathDataset

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
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

    eval_tf = make_eval_tf(cfg["image_size"])

    sup_paths, sup_labels = list_train_samples(train_dir)
    test_paths = list_test_samples(test_dir)

    sup_feats_list, test_feats_list, test_names_out = [], [], []
    for paths, labels, is_train in [
        (sup_paths, sup_labels, True),
        (test_paths, None, False),
    ]:
        ds = ImagePathDataset(paths, labels, eval_tf)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        feats_l, second_l = [], []
        for x, y_or_n in loader:
            x = x.to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                      enabled=(device == "cuda")):
                f = model.backbone(x)
            feats_l.append(f.float().cpu().numpy())
            if is_train:
                second_l.append(y_or_n.numpy())
            else:
                second_l.extend(list(y_or_n))
        feats = np.concatenate(feats_l, axis=0)
        if is_train:
            sup_feats_list = feats
            sup_lab_out = np.concatenate(second_l, axis=0)
        else:
            test_feats_list = feats
            test_names_out = second_l

    return sup_feats_list, sup_lab_out, test_feats_list, test_names_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="phikon",
                    choices=list(BACKBONE_REGISTRY.keys()))
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir",  required=True)
    ap.add_argument("--out",       default="submission.csv")
    ap.add_argument("--method",    default="mahalanobis",
                    choices=["mahalanobis", "simpleshot"])
    ap.add_argument("--tukey_beta",    type=float, default=0.5)
    ap.add_argument("--maha_shrink",   type=float, default=0.3)
    # Optional: ensemble with existing LoRA checkpoint
    ap.add_argument("--lora_ckpt",      default=None,
                    help="path to LoRA .pt checkpoint for ensemble")
    ap.add_argument("--ensemble_weight", type=float, default=0.5,
                    help="weight for phikon features (0=LoRA only, 1=phikon only)")
    ap.add_argument("--batch_size",  type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}, backbone={args.backbone}, method={args.method}")

    # 1. Load pathology backbone
    _, feat_dim, transform, extract_fn = load_backbone(args.backbone, device)
    print(f"[info] feature dim: {feat_dim}")

    # 2. Extract features
    sup_paths, sup_labels = list_train_samples(args.train_dir)
    test_paths = list_test_samples(args.test_dir)
    print(f"[info] support={len(sup_paths)}, test={len(test_paths)}")

    sup_feats, _ = extract_feats(extract_fn, sup_paths, sup_labels,
                                  transform, args.batch_size, args.num_workers, device)
    test_feats, test_names = extract_feats(extract_fn, test_paths, None,
                                            transform, args.batch_size, args.num_workers, device)
    print(f"[info] phikon feats: sup={sup_feats.shape}, test={test_feats.shape}")

    # 3. Optional: concat LoRA features
    if args.lora_ckpt:
        print(f"[info] loading LoRA features from {args.lora_ckpt}")
        lora_sup, lora_sup_lab, lora_test, lora_names = load_lora_feats(
            args.lora_ckpt, args.train_dir, args.test_dir,
            args.batch_size, args.num_workers, device,
        )
        # Verify alignment (same order)
        assert list(lora_names) == list(test_names), "name mismatch between phikon and LoRA"
        sup_labels = lora_sup_lab   # use LoRA's labels (same, just being explicit)

        # L2-normalise each block before concat, then re-weight
        def l2(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        w = args.ensemble_weight
        sup_feats  = np.concatenate([w * l2(sup_feats),  (1-w) * l2(lora_sup)],  axis=1)
        test_feats = np.concatenate([w * l2(test_feats), (1-w) * l2(lora_test)], axis=1)
        print(f"[info] ensemble feats dim={sup_feats.shape[1]}")

    # 4. Transductive inference
    if args.method == "mahalanobis":
        preds = mahalanobis(sup_feats, sup_labels, test_feats, num_classes=5,
                            tukey_beta=args.tukey_beta, shrink=args.maha_shrink)
    else:
        preds = simpleshot(sup_feats, sup_labels, test_feats, num_classes=5,
                           tukey_beta=args.tukey_beta)

    labels = [IDX_TO_CLASS[int(i)] for i in preds]
    df = pd.DataFrame({"filename": test_names, "label": labels})
    df.to_csv(args.out, index=False)
    print(f"[ok] wrote {args.out}  ({len(df)} rows)")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
