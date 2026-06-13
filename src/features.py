"""Backbone loaders + feature extraction.

Supports two families:
  - DINOv2 (timm): vits14 / vitb14 / vitl14   — natural-image SSL
  - Pathology SSL (HuggingFace transformers): phikon / phikon_v2 / hibou_b
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ImagePathDataset, build_transform

# timm-loadable backbones (natural-image SSL / supervised)
TIMM_MODEL_NAMES = {
    "vits14": "vit_small_patch14_dinov2.lvd142m",
    "vitb14": "vit_base_patch14_dinov2.lvd142m",
    "vitl14": "vit_large_patch14_dinov2.lvd142m",
    # EVA-02 — strong supervised + MIM pretraining
    "eva02_s": "eva02_small_patch14_336.mim_in22k_ft_in1k",
    "eva02_b": "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
    # ConvNeXt-V2 — CNN inductive bias, good on textures
    "cnxt_n": "convnextv2_nano.fcmae_ft_in22k_in1k",
    "cnxt_t": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    # Lunit-DINO — DINO self-supervised on TCGA pathology slides, ViT-S/16 open weights
    "lunit_dino": "hf-hub:1aurent/vit_small_patch16_224.lunit_dino",
}

# HuggingFace transformers AutoModel (pathology / domain SSL)
HF_MODEL_NAMES = {
    "phikon": "owkin/phikon",          # ViT-B/16, iBOT on TCGA, ~86M
    "phikon_v2": "owkin/phikon-v2",    # ViT-L/16, DINOv2 on PANCAN-XL, ~307M
    "hibou_b": "histai/hibou-b",       # ViT-B/14, DINOv2 on histopath, ~86M
}

# HuggingFace transformers CLIPVisionModel (vision-language pretraining)
CLIP_MODEL_NAMES = {
    "plip":   "vinid/plip",       # CLIP ViT-B/32 fine-tuned on TwitterPath (pathology)
}

# Mahmood Lab pathology models (custom loader, open_clip-based)
CONCH_MODEL_NAMES = {
    "conch":  "MahmoodLab/CONCH",  # ViT-B/16 vision encoder, CoCa-style VL pretraining
}

ALL_BACKBONES = (list(TIMM_MODEL_NAMES) + list(HF_MODEL_NAMES)
                 + list(CLIP_MODEL_NAMES) + list(CONCH_MODEL_NAMES))


class HFCLSWrapper(nn.Module):
    """Wraps a HF transformers vision model so model(x) returns the CLS token [B, D]."""
    def __init__(self, hf_model: nn.Module, use_pooler: bool = False):
        super().__init__()
        self.m = hf_model
        self.use_pooler = use_pooler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.m(pixel_values=x)
        if self.use_pooler and getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0]


class CONCHVisionWrapper(nn.Module):
    """Wraps the CONCH (MahmoodLab) CoCa-style model to return CLS embedding [B, D]."""
    def __init__(self, conch_model: nn.Module):
        super().__init__()
        # CONCH model has .visual which is the vision encoder
        # encode_image returns the image embedding (post-projection)
        self.m = conch_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encode_image with proj_contrast=False returns the unprojected CLS embedding
        # so that we can fine-tune on top of raw vision features.
        emb = self.m.encode_image(x, proj_contrast=False, normalize=False)
        return emb


def load_backbone(name: str, device: str = "cuda",
                  image_size: int = 224) -> nn.Module:
    if name in TIMM_MODEL_NAMES:
        # ConvNeXt doesn't accept img_size; only ViT-style does
        kwargs = dict(pretrained=True, num_classes=0)
        if not name.startswith("cnxt"):
            kwargs["img_size"] = image_size
        model = timm.create_model(TIMM_MODEL_NAMES[name], **kwargs)
    elif name in HF_MODEL_NAMES:
        from transformers import AutoModel
        hf = AutoModel.from_pretrained(HF_MODEL_NAMES[name], trust_remote_code=True)
        model = HFCLSWrapper(hf)
    elif name in CLIP_MODEL_NAMES:
        from transformers import CLIPVisionModel
        hf = CLIPVisionModel.from_pretrained(CLIP_MODEL_NAMES[name])
        model = HFCLSWrapper(hf, use_pooler=True)
    elif name in CONCH_MODEL_NAMES:
        # CONCH requires the conch package (pip install git+https://github.com/Mahmoodlab/CONCH.git)
        from conch.open_clip_custom import create_model_from_pretrained
        conch, _ = create_model_from_pretrained(
            "conch_ViT-B-16", f"hf_hub:{CONCH_MODEL_NAMES[name]}",
        )
        model = CONCHVisionWrapper(conch)
    else:
        raise ValueError(f"Unknown backbone {name}. Options: {ALL_BACKBONES}")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


# Backward-compat alias
load_dinov2 = load_backbone


@torch.no_grad()
def extract(
    paths: List[Path],
    labels: np.ndarray | None,
    model: torch.nn.Module,
    image_size: int = 224,
    batch_size: int = 64,
    device: str = "cuda",
    num_workers: int = 2,
    n_views: int = 1,
    train_aug: bool = False,
) -> Tuple[np.ndarray, np.ndarray | None, List[str]]:
    transform = build_transform(image_size=image_size, train=train_aug)
    ds = ImagePathDataset(paths, labels, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    feats_acc: list[np.ndarray] = []
    lbls_acc: list[int] = []
    names_acc: list[str] = []

    for view in range(n_views):
        view_feats: list[np.ndarray] = []
        for batch in tqdm(loader, desc=f"feat view {view+1}/{n_views}"):
            x, y = batch
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                f = model(x)
            f = torch.nn.functional.normalize(f.float(), dim=-1)
            view_feats.append(f.cpu().numpy())
            if view == 0:
                if labels is not None:
                    lbls_acc.extend(y.tolist())
                else:
                    names_acc.extend(list(y))
        feats_acc.append(np.concatenate(view_feats, axis=0))

    feats = np.mean(np.stack(feats_acc, axis=0), axis=0)
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    lbls = np.asarray(lbls_acc, dtype=np.int64) if labels is not None else None
    return feats, lbls, names_acc


def save_npz(path: str | Path, feats: np.ndarray, labels: np.ndarray | None,
             filenames: List[str] | None) -> None:
    payload = {"feats": feats}
    if labels is not None:
        payload["labels"] = labels
    if filenames:
        payload["filenames"] = np.array(filenames)
    np.savez(path, **payload)
