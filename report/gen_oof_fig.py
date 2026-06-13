import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
FIGDIR = Path(__file__).parent / "figures"

# 5-fold CV OOF confusion matrix (real data from cv5_selftrain_maha)
cm = np.array([[34, 1, 6, 8, 1],
               [0, 33, 6, 5, 6],
               [0, 1, 23, 7, 19],
               [5, 3, 6, 29, 7],
               [0, 6, 12, 4, 28]])
names = [f"Class_{i}" for i in range(5)]

fig, ax = plt.subplots(figsize=(5.8, 5.2))
im = ax.imshow(cm, cmap="Oranges", vmin=0, vmax=34)
for i in range(5):
    for j in range(5):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > 17 else "#333", fontsize=12, fontweight="bold")
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("预测", fontsize=11); ax.set_ylabel("真值", fontsize=11)
ax.set_title("5 折 CV 全样本(OOF)混淆矩阵\nClass_2↔Class_4 互相混淆最严重(红框区)", fontsize=11)
# highlight the 2<->4 confusion
import matplotlib.patches as mp
ax.add_patch(mp.Rectangle((3.5, 1.5), 1, 1, fill=False, edgecolor="red", lw=2.5))  # C2->C4 =19
ax.add_patch(mp.Rectangle((1.5, 3.5), 1, 1, fill=False, edgecolor="red", lw=2.5))  # C4->C2 =12
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
out = FIGDIR / "fig_oof_confusion.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)

sizes = json.loads((FIGDIR / "sizes.json").read_text())
sizes["fig_oof_confusion.png"] = list(Image.open(out).size)
(FIGDIR / "sizes.json").write_text(json.dumps(sizes, indent=2))
print("ok", sizes["fig_oof_confusion.png"])
