"""Few-shot dataset utilities: load 5-class PNG folder, stratified split, transforms."""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

CLASS_NAMES = [f"Class_{i}" for i in range(5)]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# ImageNet stats — DINOv2 was trained with these
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int = 224, train: bool = False) -> T.Compose:
    if train:
        return T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def list_train_samples(root: str | Path) -> Tuple[List[Path], np.ndarray]:
    root = Path(root)
    paths: List[Path] = []
    labels: List[int] = []
    for cls in CLASS_NAMES:
        cls_dir = root / cls
        for p in sorted(cls_dir.glob("*.png")):
            paths.append(p)
            labels.append(CLASS_TO_IDX[cls])
    return paths, np.asarray(labels, dtype=np.int64)


def list_test_samples(root: str | Path) -> List[Path]:
    root = Path(root)
    return sorted(root.glob("*.png"))


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[Path], labels: np.ndarray | None, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.transform(img)
        if self.labels is None:
            return x, str(self.paths[idx].name)
        return x, int(self.labels[idx])
