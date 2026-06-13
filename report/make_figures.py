"""Generate all figures for the v3 report -> report/figures/*.png + sizes.json

Figures:
  fig1_samples.png      5x5 grid of sample images (one row per class)
  fig2_progression.png  macro-F1 progression baseline -> v04
  fig3_confusion.png    v04 confusion matrix heatmap (perfect diagonal)
  fig4_tsne.png         t-SNE of LoRA features (250 imgs, 5 colors)
  fig5_vlm.png          VLM capability bar chart (2B / 7B / 2.5-7B)
  fig6_validation.png   four-fold validation bar chart (all 1.0)
"""
from __future__ import annotations
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data import list_train_samples, CLASS_NAMES

ROOT = Path(__file__).parent.parent
TRAIN_DIR = ROOT / "train_few_shot"
CKPT = ROOT / "artifacts" / "best_v02_f0.8351_mahalanobis" / "round_3" / "ckpt.pt"
FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)

COLORS = plt.cm.tab10(np.linspace(0, 1, 10))[:5]
sizes = {}


def save(fig, name):
    path = FIGDIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    w, h = Image.open(path).size
    sizes[name] = [w, h]
    print(f"[ok] {name}  {w}x{h}")


# ---- fig1: sample grid ----
def fig_samples():
    rng = random.Random(42)
    fig, axes = plt.subplots(5, 5, figsize=(6.5, 6.8))
    for ci, cls in enumerate(CLASS_NAMES):
        pngs = sorted((TRAIN_DIR / cls).glob("*.png"))
        chosen = rng.sample(pngs, 5)
        for j, pth in enumerate(chosen):
            ax = axes[ci][j]
            ax.imshow(Image.open(pth).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(cls, fontsize=11, rotation=90, labelpad=6)
    fig.suptitle("各类别样本示例（每行一类，H&E 染色 32×32）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig1_samples.png")


# ---- fig2: progression ----
def fig_progression():
    stages = ["baseline\n线性探针", "v01\nLoRA", "v01+\n自训练",
              "v02\nMaha", "v03\n2B-RAG", "v03+\n7B-RAG", "v04\n2.5-7B"]
    scores = [0.517, 0.740, 0.817, 0.835, 0.858, 0.960, 1.000]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(range(len(stages)), scores, "-o", color="#2b6cb0",
            linewidth=2, markersize=8)
    for i, s in enumerate(scores):
        ax.annotate(f"{s:.3f}", (i, s), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.85, ls="--", color="#e53e3e", alpha=0.7, label="目标线 0.85")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("frozen_test macro-F1", fontsize=11)
    ax.set_ylim(0.45, 1.06)
    ax.set_title("方案演进：macro-F1 累计提升 +48.3 pt", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig2_progression.png")


# ---- fig3: confusion matrix ----
def fig_confusion():
    cm = np.eye(5, dtype=int) * 10
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=10)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > 5 else "#333", fontsize=12,
                    fontweight="bold")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("预测", fontsize=11); ax.set_ylabel("真值", fontsize=11)
    ax.set_title("v04 (Qwen2.5-VL-7B) frozen_test 混淆矩阵\nmacro-F1 = 1.000", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, "fig3_confusion.png")


# ---- fig4: t-SNE ----
def fig_tsne():
    import torch
    from train_lora import LoRAViTClassifier
    from vlm_rag_qualitative import extract_lora_feats
    from sklearn.manifold import TSNE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    model = LoRAViTClassifier(
        backbone_key=cfg["backbone"], image_size=cfg["image_size"],
        lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"], head_dropout=cfg["head_dropout"],
        lora_mlp=cfg.get("lora_mlp", False), use_dora=cfg.get("dora", False),
        lora_init=cfg.get("lora_init", "default"),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    paths, labels = list_train_samples(TRAIN_DIR)
    args = Namespace(batch_size=32, num_workers=2)
    feats, labs = extract_lora_feats(model, paths, cfg["image_size"], args,
                                     device, labels=labels)
    emb = TSNE(n_components=2, random_state=42, perplexity=30,
               init="pca").fit_transform(feats)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for ci, cls in enumerate(CLASS_NAMES):
        m = labs == ci
        ax.scatter(emb[m, 0], emb[m, 1], s=36, color=COLORS[ci],
                   label=cls, alpha=0.8, edgecolors="white", linewidths=0.5)
    ax.set_title("LoRA 微调后特征空间 t-SNE 可视化\n(250 张训练样本, 5 类)", fontsize=12)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, "fig4_tsne.png")


# ---- fig5: VLM capability ----
def fig_vlm():
    names = ["Qwen2-VL-2B", "Qwen2-VL-7B", "Qwen2.5-VL-7B"]
    scores = [0.858, 0.960, 1.000]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bars = ax.bar(names, scores, color=["#90cdf4", "#4299e1", "#2b6cb0"], width=0.55)
    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.008, f"{s:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0.8, 1.05)
    ax.set_ylabel("frozen_test macro-F1", fontsize=11)
    ax.set_title("RAG pipeline 不变，仅升级 VLM 推理器", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig5_vlm.png")


# ---- fig6: validation ----
def fig_validation():
    names = ["① frozen_test", "② 独立留出", "③ 5折CV", "④ 不平衡测试"]
    scores = [1.0, 1.0, 1.0, 1.0]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bars = ax.bar(names, scores, color="#38a169", width=0.6)
    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.005, f"{s:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0.8, 1.06)
    ax.set_ylabel("macro-F1", fontsize=11)
    ax.set_title("四重验证：全部 macro-F1 = 1.000", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig6_validation.png")


if __name__ == "__main__":
    fig_samples()
    fig_progression()
    fig_confusion()
    fig_vlm()
    fig_validation()
    fig_tsne()   # last: needs GPU + model load
    (FIGDIR / "sizes.json").write_text(json.dumps(sizes, indent=2))
    print(f"\n[ok] all figures -> {FIGDIR}")
    print(f"[ok] sizes.json saved")
