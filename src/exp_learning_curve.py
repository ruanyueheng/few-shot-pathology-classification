"""Learning curve: does macro-F1 still rise with more training data?
Train LoRA on N/class (10/20/30/40) subsets of dev, eval Mahalanobis on frozen.
If the curve is still rising at N=40, training-data size is the bottleneck."""
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from data import list_train_samples, list_test_samples, ImagePathDataset, CLASS_TO_IDX
from train_lora import train_one_run, make_eval_tf
from transductive import mahalanobis

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
base = Path(__file__).parent.parent
device = "cuda" if torch.cuda.is_available() else "cpu"

def make_args(seed=42):
    return Namespace(
        backbone="vits14", image_size=224, epochs=30, batch_size=32,
        lr=5e-4, lr_head=1e-3, weight_decay=0.05, label_smoothing=0.1,
        grad_clip=1.0, lora_r=32, lora_alpha=64, lora_dropout=0.1,
        head_dropout=0.1, lora_mlp=False, dora=False, lora_init="default",
        ema=True, ema_decay=0.95, hed_aug=False, hed_sigma=0.02, hed_bias=0.01,
        mixup=0.0, cutmix=0.0, num_workers=0, seed=seed, verbose=False,
        supcon_weight=0.0, supcon_temp=0.07,
    )

dev_paths, dev_labels = list_train_samples(base / "dev_few_shot")
by_cls = defaultdict(list)
for p, l in zip(dev_paths, dev_labels):
    by_cls[int(l)].append(p)
test_paths = list_test_samples(base / "frozen_test")
gt = {r.filename: CLASS_TO_IDX[r.label] for r in pd.read_csv(base / "frozen_test" / "_groundtruth.csv").itertuples()}

@torch.no_grad()
def extract(model, paths, labels=None):
    ds = ImagePathDataset(paths, labels, make_eval_tf(224))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    feats, second = [], []
    model.eval()
    for x, s in loader:
        x = x.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            f = model.backbone(x)
        feats.append(f.float().cpu().numpy())
        if labels is not None: second.append(s.numpy())
        else: second.extend(list(s))
    feats = np.concatenate(feats, 0)
    return (feats, np.concatenate(second)) if labels is not None else (feats, second)

results = []
print("=" * 56)
print("学习曲线：纯 LoRA (无自训练)，frozen Mahalanobis")
print("=" * 56)
for N in [10, 20, 30, 40]:
    rng = np.random.default_rng(42)
    sub_paths, sub_labels = [], []
    for c in range(5):
        ps = by_cls[c]
        idx = rng.choice(len(ps), size=N, replace=False)
        for i in idx:
            sub_paths.append(ps[i]); sub_labels.append(c)
    sub_labels = np.array(sub_labels)
    args = make_args()
    fi = np.arange(len(sub_paths))
    model = train_one_run(sub_paths, sub_labels, fi, fi, args, device, run_id=99, refit_all=True)[0]
    sup_f, _ = extract(model, sub_paths, sub_labels)
    test_f, names = extract(model, test_paths, None)
    y_true = np.array([gt[n] for n in names])
    preds = mahalanobis(sup_f, sub_labels, test_f, num_classes=5, tukey_beta=0.5, shrink=0.3)
    f1 = f1_score(y_true, preds, average="macro")
    ba = balanced_accuracy_score(y_true, preds)
    print(f"N={N}/class (total {N*5:3d}):  macro-F1={f1:.4f}  bal-acc={ba:.4f}")
    results.append((N * 5, f1, ba))
    del model; torch.cuda.empty_cache()

xs = [r[0] for r in results]; f1s = [r[1] for r in results]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(xs, f1s, "-o", color="#2b6cb0", lw=2, ms=9)
for x, f in zip(xs, f1s):
    ax.annotate(f"{f:.3f}", (x, f), textcoords="offset points", xytext=(0, 11), ha="center", fontweight="bold")
ax.axhline(0.85, ls="--", color="#e53e3e", alpha=0.7, label="题目目标 0.85")
ax.set_xlabel("训练集大小（总样本数，每类均分）", fontsize=11)
ax.set_ylabel("frozen_test macro-F1", fontsize=11)
ax.set_title("学习曲线：指标随训练数据量的变化（纯 LoRA）", fontsize=12)
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(base / "report" / "figures" / "fig_learning_curve.png", dpi=150, bbox_inches="tight")
print("[ok] saved fig_learning_curve.png")
print("=" * 56)
