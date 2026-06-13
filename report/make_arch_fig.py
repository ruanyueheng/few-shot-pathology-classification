"""Architecture diagram of the best real method (v02): LoRA + self-training + Mahalanobis."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(8.6, 11))
ax.set_xlim(0, 10)
ax.set_ylim(0, 15.5)
ax.axis("off")

# color scheme
C_DATA = "#D5E8F0"
C_PREP = "#BBDEFB"
C_TRAIN = "#C8E6C9"
C_FEAT = "#FFF9C4"
C_CLS = "#FFE0B2"
C_OUT = "#F8BBD0"

def box(x, y, w, h, text, fill, fs=11, bold=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                       linewidth=1.5, edgecolor="#555", facecolor=fill)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)

def arrow(x, y1, y2, text=""):
    a = FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=18,
                        linewidth=1.8, color="#444")
    ax.add_patch(a)
    if text:
        ax.text(x + 0.25, (y1 + y2) / 2, text, ha="left", va="center",
                fontsize=9, color="#444", style="italic")

CX = 3.6   # main column center-x
W = 5.6
X = CX - W / 2

# title
ax.text(5, 15.0, "最佳方法框架图 (v02)", ha="center", fontsize=16, fontweight="bold")
ax.text(5, 14.55, "LoRA 微调 DINOv2 + 自训练 + Mahalanobis  (frozen_test macro-F1 = 0.835)",
        ha="center", fontsize=10, color="#555", style="italic")

# boxes (top -> bottom)
box(X, 13.2, W, 0.95, "输入：250 张 32×32 RGB H&E 图\n(Class_0~4，每类 50 张)", C_DATA, bold=True)
arrow(CX, 13.2, 12.7)

box(X, 11.7, W, 0.95, "预处理\nbicubic 上采样 → 224×224 + ImageNet 归一化\n训练时温和增广 (翻转/±15°旋转/弱 ColorJitter)", C_PREP, fs=10)
arrow(CX, 11.7, 11.2)

box(X, 9.9, W, 1.2, "Backbone：DINOv2 ViT-S/14 (冻结)\n+ LoRA 适配器 r=32, α=64\n(注入 attn.qkv + attn.proj，仅 ~92 万可训练参数)", C_TRAIN, fs=10, bold=True)
arrow(CX, 9.9, 9.4)

box(X, 8.4, W, 0.95, "训练：交叉熵 + 自训练 3 轮\n(伪标签 conf≥0.8，每类配额防坍塌)", C_TRAIN, fs=10)
arrow(CX, 8.4, 7.9, "得到 LoRA ckpt")

box(X, 6.6, W, 0.95, "特征提取：[CLS] token 384-d\n(support 标注集 + test 集 都提特征)", C_FEAT, fs=10)
arrow(CX, 6.6, 6.1)

box(X, 5.1, W, 0.95, "特征后处理\nTukey 幂变换 (β=0.5) + 中心化 + L2 归一化", C_FEAT, fs=10)
arrow(CX, 5.1, 4.6)

box(X, 3.3, W, 1.0, "分类器：Mahalanobis\n类共享池化协方差 (shrink=0.3)\n按马氏距离指派最近类原型", C_CLS, fs=10, bold=True)
arrow(CX, 3.3, 2.8)

box(X, 1.5, W, 0.95, "输出：submission.csv\nmacro-F1 = 0.835 / balanced-acc = 0.840", C_OUT, fs=11, bold=True)

# side annotations: training vs inference
ax.plot([6.7, 6.7], [8.4, 11.1], color="#2e7d32", lw=2)
ax.text(6.85, 9.75, "训练阶段\n(dev 200 张)", color="#2e7d32", fontsize=9,
        va="center", fontweight="bold")
ax.plot([6.7, 6.7], [1.5, 7.55], color="#e65100", lw=2)
ax.text(6.85, 4.5, "推理阶段\n(support 检索\n+ 距离分类)", color="#e65100", fontsize=9,
        va="center", fontweight="bold")

# footnote
ax.text(5, 0.6, "注：单模型，无 TTA、无集成。VLM+RAG 经泄漏修正后退化为 KNN(≈本方法)，故主线采用更简单可靠的 v02。",
        ha="center", fontsize=8, color="#777", style="italic", wrap=True)

fig.tight_layout()
out = "E:/University/Homework/Machine Learning2/final_project/report/figures/fig_architecture.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("[ok] saved", out)
