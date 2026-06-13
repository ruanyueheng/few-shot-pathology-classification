"""Generate TRUE-data figures for report v4 (post-leakage-discovery)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import json
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)
sizes = json.loads((FIGDIR / "sizes.json").read_text()) if (FIGDIR / "sizes.json").exists() else {}

def save(fig, name):
    p = FIGDIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    sizes[name] = list(Image.open(p).size)
    print("[ok]", name, sizes[name])

# ---- fig2 v4: TRUE progression (ends at 0.835, VLM-RAG flat) ----
def fig_prog():
    stages = ["baseline\n线性探针", "LoRA r=32\n+ 分类头", "+ 自训练\n3轮伪标签",
              "+ Mahalanobis\n(v02 主线)", "VLM-RAG\n(修复泄漏后)"]
    scores = [0.517, 0.740, 0.817, 0.835, 0.835]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(range(4), scores[:4], "-o", color="#2b6cb0", lw=2, ms=8)
    ax.plot([3, 4], [0.835, 0.835], "--o", color="#888", lw=2, ms=8)
    for i, s in enumerate(scores):
        ax.annotate(f"{s:.3f}", (i, s), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")
    # leaked 1.0 marker
    ax.annotate("泄漏前虚高\n1.000 (作废)", (4, 1.0), fontsize=9, color="#c00",
                ha="center", va="bottom")
    ax.scatter([4], [1.0], marker="x", s=80, color="#c00", zorder=5)
    ax.axhline(0.85, ls=":", color="#e53e3e", alpha=0.7, label="题目目标 0.85")
    ax.set_xticks(range(5)); ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("frozen_test macro-F1", fontsize=11)
    ax.set_ylim(0.45, 1.05)
    ax.set_title("真实方案演进：到 v02 = 0.835；VLM-RAG 修复泄漏后持平(退化成KNN)", fontsize=11.5)
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); save(fig, "fig2_progression_v4.png")

# ---- fig3 v4: TRUE confusion matrix (0.835) ----
def fig_conf():
    cm = np.array([[9,0,0,1,0],[0,9,0,0,1],[0,1,8,0,1],[1,0,2,6,1],[0,0,0,0,10]])
    names = [f"Class_{i}" for i in range(5)]
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=10)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > 5 else "#333", fontsize=12, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("预测", fontsize=11); ax.set_ylabel("真值", fontsize=11)
    ax.set_title("真实最佳方法混淆矩阵 (v02 / 修复后VLM)\nmacro-F1 = 0.835，Class_3 召回 0.60", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); save(fig, "fig3_confusion_v4.png")

# ---- fig_leakage: leaked vs true (frozen) ----
def fig_leak():
    labels = ["文件名泄漏版\n(VLM读文件名)", "修复后真实\n(VLM真看图)"]
    vals = [1.000, 0.835]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    bars = ax.bar(labels, vals, color=["#e57373", "#388e3c"], width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center",
                fontsize=12, fontweight="bold")
    ax.set_ylim(0.7, 1.05); ax.set_ylabel("frozen_test macro-F1", fontsize=11)
    ax.set_title("文件名泄漏的影响：去掉文件名后真实分数下降 0.165", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); save(fig, "fig_leakage.png")

# ---- fig_ablation: 4 directions ----
def fig_abl():
    labels = ["方向1\nLoRA微调\n(主线)", "方向2\nCLIP\nzero-shot", "方向3\nVLM-RAG\n(=KNN)", "方向4\nSupCon\n(best)"]
    vals = [0.835, 0.454, 0.835, 0.676]
    colors = ["#388e3c", "#e57373", "#fbc02d", "#e57373"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.012, f"{v:.3f}", ha="center",
                fontsize=11, fontweight="bold")
    ax.axhline(0.835, ls="--", color="#388e3c", alpha=0.6)
    ax.set_ylim(0.3, 0.95); ax.set_ylabel("frozen_test macro-F1", fontsize=11)
    ax.set_title("老师 4 个方向完整消融：均 ≤ 0.835(收敛到数据级天花板)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); save(fig, "fig_ablation.png")

# ---- fig_supcon: weight ablation ----
def fig_sc():
    w = ["0\n(纯CE)", "0.3", "0.5"]
    vals = [0.835, 0.676, 0.627]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    bars = ax.bar(w, vals, color=["#388e3c", "#e57373", "#c62828"], width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center",
                fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 0.92); ax.set_ylabel("frozen_test macro-F1", fontsize=11)
    ax.set_xlabel("SupCon weight λ", fontsize=11)
    ax.set_title("SupCon 失败消融：weight 越大伤害越大\n(对比学习在200张小数据上水土不服)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); save(fig, "fig_supcon.png")

if __name__ == "__main__":
    fig_prog(); fig_conf(); fig_leak(); fig_abl(); fig_sc()
    (FIGDIR / "sizes.json").write_text(json.dumps(sizes, indent=2))
    print("[ok] all v4 figures + sizes.json updated")
