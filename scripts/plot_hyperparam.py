"""
Plot hyperparameter sensitivity on Rest14 (Qwen3-8B):
  Top row: (a) GATv2 layer depth, (b) LoRA rank
  Bottom row (full width): (c) Stage 2 epochs (P/R/F1 tradeoff)
Output: figures/hyperparam_sensitivity.png and .pdf
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.gridspec import GridSpec

# ---- Data ----
layers = [1, 2, 3, 4]
f1_layers = [79.68, 80.22, 79.85, 79.41]

ranks = [16, 32, 64, 128]
f1_ranks = [78.53, 79.41, 80.22, 80.08]

epochs = [1, 2, 3, 5, 7, 10]
p_epochs  = [79.29, 79.14, 78.94, 78.59, 78.11, 77.60]
r_epochs  = [78.34, 79.96, 80.92, 81.92, 82.32, 81.95]
f1_epochs = [78.81, 79.55, 79.92, 80.22, 80.16, 79.74]

# ---- Style ----
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 11.5,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#444444",
})

fig = plt.figure(figsize=(8.6, 6.4), dpi=200)
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.05], hspace=0.42, wspace=0.28)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, :])

# ---- Panel (a): GATv2 layer depth ----
ax = ax_a
colors_a = ["#9cc4dc" if v != max(f1_layers) else "#2a9d8f" for v in f1_layers]
bars = ax.bar(range(len(layers)), f1_layers, width=0.55,
              color=colors_a, edgecolor="#333333", linewidth=0.6)
for b, v in zip(bars, f1_layers):
    fw = "bold" if v == max(f1_layers) else "normal"
    ax.text(b.get_x()+b.get_width()/2, v+0.04, f"{v:.2f}",
            ha="center", va="bottom", fontsize=9.5,
            fontweight=fw, color="#222222")
    if v == max(f1_layers):
        b.set_edgecolor("#145c55"); b.set_linewidth(1.3)
ax.set_xticks(range(len(layers)))
ax.set_xticklabels(layers)
ax.set_xlabel("Number of GATv2 layers")
ax.set_ylabel("F1 (%)")
ax.set_ylim(79.0, 80.6)
ax.set_title("(a) GATv2 Layer Depth")
ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ---- Panel (b): LoRA rank ----
ax = ax_b
colors_b = ["#9cc4dc" if v != max(f1_ranks) else "#2a9d8f" for v in f1_ranks]
bars = ax.bar(range(len(ranks)), f1_ranks, width=0.55,
              color=colors_b, edgecolor="#333333", linewidth=0.6)
for b, v in zip(bars, f1_ranks):
    fw = "bold" if v == max(f1_ranks) else "normal"
    ax.text(b.get_x()+b.get_width()/2, v+0.06, f"{v:.2f}",
            ha="center", va="bottom", fontsize=9.5,
            fontweight=fw, color="#222222")
    if v == max(f1_ranks):
        b.set_edgecolor("#145c55"); b.set_linewidth(1.3)
ax.set_xticks(range(len(ranks)))
ax.set_xticklabels(ranks)
ax.set_xlabel("LoRA rank $r$")
ax.set_ylabel("F1 (%)")
ax.set_ylim(78.0, 80.8)
ax.set_title("(b) LoRA Rank")
ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ---- Panel (c): Stage 2 epochs (P/R/F1 lines) — full width ----
ax = ax_c
ax.plot(epochs, p_epochs, marker="s", markersize=7, linewidth=1.9,
        color="#56a3dc", linestyle="--", label="Precision")
ax.plot(epochs, r_epochs, marker="^", markersize=7.5, linewidth=1.9,
        color="#78c9a8", linestyle="--", label="Recall")
ax.plot(epochs, f1_epochs, marker="o", markersize=8, linewidth=2.4,
        color="#e07a2b", label="F1")
# Annotate best F1
best_i = int(np.argmax(f1_epochs))
ax.annotate(f"Best F1: {f1_epochs[best_i]:.2f}",
            xy=(epochs[best_i], f1_epochs[best_i]),
            xytext=(epochs[best_i]+0.6, f1_epochs[best_i]+0.55),
            fontsize=10, color="#e07a2b", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#e07a2b", lw=0.9))
ax.set_xticks(epochs)
ax.set_xlabel("Stage 2 training epochs")
ax.set_ylabel("Score (%)")
ax.set_ylim(76.8, 83.2)
ax.set_title("(c) Stage 2 Training Duration")
ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(loc="lower right", frameon=True, framealpha=0.95,
          edgecolor="#cccccc", fontsize=10).get_frame().set_linewidth(0.6)

plt.tight_layout()

out_png = "./figures/hyperparam_sensitivity.png"
out_pdf = "./figures/hyperparam_sensitivity.pdf"
plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
