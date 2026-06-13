"""方向2: better covariance estimation for Mahalanobis. Baseline = fixed shrink 0.3 (0.835)."""
import sys
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf, OAS
from sklearn.metrics import f1_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from data import list_train_samples, list_test_samples, CLASS_NAMES, CLASS_TO_IDX
from train_lora import LoRAViTClassifier
from vlm_rag_qualitative import extract_lora_feats
from transductive import center_and_normalize

base = Path(__file__).parent.parent
ckpt_path = base / "artifacts" / "best_v02_f0.8351_mahalanobis" / "round_3" / "ckpt.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = ck["args"]
model = LoRAViTClassifier(
    backbone_key=cfg["backbone"], image_size=cfg["image_size"],
    lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
    lora_dropout=cfg["lora_dropout"], head_dropout=cfg["head_dropout"],
    lora_mlp=cfg.get("lora_mlp", False), use_dora=cfg.get("dora", False),
    lora_init=cfg.get("lora_init", "default"),
).to(device)
model.load_state_dict(ck["state_dict"], strict=False)
model.eval()

sup_paths, sup_lab = list_train_samples(base / "dev_few_shot")
test_paths = list_test_samples(base / "frozen_test")
args = Namespace(batch_size=32, num_workers=0)
sup_feats, sup_lab = extract_lora_feats(model, sup_paths, cfg["image_size"], args, device, labels=sup_lab)
test_feats, test_names = extract_lora_feats(model, test_paths, cfg["image_size"], args, device, labels=None)

gt = {r.filename: CLASS_TO_IDX[r.label] for r in pd.read_csv(base / "frozen_test" / "_groundtruth.csv").itertuples()}
y_true = np.array([gt[n] for n in test_names])

# Tukey + center/L2 (same as v02)
s, qq = center_and_normalize(sup_feats, test_feats, tukey_beta=0.5, use_combined_mean=False)
D = s.shape[1]
NC = 5
means = np.zeros((NC, D), dtype=s.dtype)
res = []
for c in range(NC):
    means[c] = s[sup_lab == c].mean(0)
    res.append(s[sup_lab == c] - means[c])
R = np.concatenate(res, 0)

def maha_eval(cov_inv, name, s_=s, qq_=qq, means_=means):
    W = means_ @ cov_inv
    b = (means_ * W).sum(1) * 0.5
    preds = (qq_ @ W.T - b[None, :]).argmax(1)
    f1 = f1_score(y_true, preds, average="macro")
    ba = balanced_accuracy_score(y_true, preds)
    print(f"{name:28s} macro-F1={f1:.4f}  bal-acc={ba:.4f}")
    return f1

print("=" * 56)
print("方向2: Mahalanobis 协方差估计对比 (frozen, baseline shrink=0.3=0.835)")
print("=" * 56)

# 1. fixed shrink 0.3 (baseline)
cov = (R.T @ R) / max(1, R.shape[0] - NC)
tr = np.trace(cov) / D
cov_s = (1 - 0.3) * cov + 0.3 * tr * np.eye(D) + 1e-6 * np.eye(D)
maha_eval(np.linalg.inv(cov_s), "fixed shrink=0.3 (baseline)")

# 2. Ledoit-Wolf (analytic optimal shrinkage)
lw = LedoitWolf().fit(R)
print(f"   [LedoitWolf chose shrinkage = {lw.shrinkage_:.4f}]")
maha_eval(lw.precision_, "Ledoit-Wolf (auto)")

# 3. OAS
oas = OAS().fit(R)
print(f"   [OAS chose shrinkage = {oas.shrinkage_:.4f}]")
maha_eval(oas.precision_, "OAS (auto)")

# 4. PCA-whitening + cosine prototype (decorrelate dominant directions)
from numpy.linalg import svd
mu_all = s.mean(0)
U, Sv, Vt = svd(s - mu_all, full_matrices=False)
k = min(100, len(Sv))
W_pca = Vt[:k] / (Sv[:k][:, None] / np.sqrt(len(s)) + 1e-6)  # whiten
s_w = (s - mu_all) @ W_pca.T
qq_w = (qq - mu_all) @ W_pca.T
s_w /= (np.linalg.norm(s_w, axis=1, keepdims=True) + 1e-9)
qq_w /= (np.linalg.norm(qq_w, axis=1, keepdims=True) + 1e-9)
means_w = np.stack([s_w[sup_lab == c].mean(0) for c in range(NC)])
preds = (qq_w @ means_w.T).argmax(1)
f1 = f1_score(y_true, preds, average="macro")
ba = balanced_accuracy_score(y_true, preds)
print(f"{'PCA-whiten(100) + cosine':28s} macro-F1={f1:.4f}  bal-acc={ba:.4f}")
print("=" * 56)
