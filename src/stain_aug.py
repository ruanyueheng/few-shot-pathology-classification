"""HED color augmentation for H&E stained histopathology images.

Reference: Tellez et al. (2018) — "Quantifying the effects of data augmentation
and stain color normalization in convolutional neural networks for computational
pathology." Standard hematoxylin/eosin/DAB stain decomposition (Ruifrok-Johnston).

Pipeline per image:
  RGB --(-log10(I/255))--> OD --(M^-1)--> stain concentrations [H, E, D]
  jitter each channel: c_i *= (1 + alpha_i),  alpha_i ~ U(-sigma, sigma)
  optional shift:      c_i += beta_i,         beta_i ~ U(-bias, bias)
  back: OD = M @ c   ->  I = 255 * 10^(-OD)
"""
from __future__ import annotations
import numpy as np
from PIL import Image

# Ruifrok-Johnston H&E stain matrix (rows = stain RGB optical density)
HE_STAIN_MATRIX = np.array([
    [0.65, 0.70, 0.29],   # Hematoxylin
    [0.07, 0.99, 0.11],   # Eosin
    [0.27, 0.57, 0.78],   # DAB / residual
], dtype=np.float32)

HE_STAIN_INV = np.linalg.inv(HE_STAIN_MATRIX).astype(np.float32)


class HEDColorJitter:
    """Apply HED-space color jitter as a PIL-in / PIL-out transform.

    Args:
      sigma: multiplicative jitter scale per stain channel. 0.05-0.10 is typical for
             pathology — strong enough to vary stain darkness, gentle enough not to
             destroy class signal.
      bias:  additive shift per stain channel.
      p:     probability of applying (otherwise return unchanged).
    """
    def __init__(self, sigma: float = 0.02, bias: float = 0.01, p: float = 0.5):
        self.sigma = sigma
        self.bias = bias
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return img
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        H, W, _ = arr.shape
        eps = 1.0   # 1/255 floor on intensity
        # OD = -log10((I + eps) / 256)
        od = -np.log10((arr + eps) / 256.0)            # [H, W, 3]
        od_flat = od.reshape(-1, 3)
        # In row-vector convention: OD_row = c_row @ M_stain  (each row of M = stain OD signature)
        # ⇒  c_row = OD_row @ M_stain^{-1}.
        stains = od_flat @ HE_STAIN_INV                # [N, 3]
        # jitter
        alpha = np.random.uniform(1.0 - self.sigma, 1.0 + self.sigma, size=(1, 3)).astype(np.float32)
        beta = np.random.uniform(-self.bias, self.bias, size=(1, 3)).astype(np.float32)
        stains = stains * alpha + beta
        # back to OD and intensity
        od_new = stains @ HE_STAIN_MATRIX               # [N, 3]
        od_new = od_new.reshape(H, W, 3)
        intensity = np.power(10.0, -od_new) * 256.0 - eps
        intensity = np.clip(intensity, 0, 255).astype(np.uint8)
        return Image.fromarray(intensity)
