"""
Plot 5-way F1 comparison across 4 ASTE benchmarks.
Methods: MvP / DESS-xxL / Syn-Chain / STaR* (re-impl) / Ours (Qwen3)
Output: figures/main_results.png and figures/main_results.pdf
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ---- Data (F1 %) ----
datasets = ["Rest14", "Lap14", "Rest15", "Rest16"]

methods = ["MvP", "DESS-xxL", "Syn-Chain", "STaR$^*$ (re-impl.)", "Ours (Qwen3-8B)"]
# rows aligned with methods, columns with datasets
f1 = np.array([
    [74.05, 63.33, 65.89, 73.48],  # MvP
    [79.98, 69.08, 74.22, 77.16],  # DESS-xxL
    [73.51, 62.78, 66.43, 74.19],  # Syn-Chain
    [79.12, 68.74, 75.48, 79.83],  # STaR
    [80.22, 70.15, 77.19, 80.86],  # Ours
])

# ---- Colors: grays for baselines, teal for ours ----
colors = [
    "#c7c9cc",  # MvP        - light gray
    "#8c919a",  # DESS-xxL   - medium gray
    "#9cc4dc",  # Syn-Chain  - light blue
    "#f4a261",  # STaR       - orange
    "#2a9d8f",  # Ours       - teal (emphasis)
]

# ---- Style ----
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#444444",
})

fig, ax = plt.subplots(figsize=(7.0, 4.1), dpi=200)

n_groups = len(datasets)
n_methods = len(methods)
group_width = 0.80
bar_width = group_width / n_methods
x = np.arange(n_groups)

for i, (m, c) in enumerate(zip(methods, colors)):
    offset = (i - (n_methods - 1) / 2) * bar_width
    bars = ax.bar(
        x + offset, f1[i], bar_width,
        label=m, color=c,
        edgecolor="#333333", linewidth=0.5,
    )
    # Highlight "Ours": thicker edge
    if i == n_methods - 1:
        for b in bars:
            b.set_edgecolor("#145c55")
            b.set_linewidth(1.2)
    # Value labels on top
    for j, b in enumerate(bars):
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 0.2,
            f"{h:.1f}",
            ha="center", va="bottom",
            fontsize=7.5,
            color="#222222",
            fontweight="bold" if i == n_methods - 1 else "normal",
        )

# ---- Axes & grid ----
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11.5)
ax.set_ylabel("F1 (%)")
ax.set_ylim(58, 86)
ax.set_yticks(np.arange(60, 86, 5))
ax.yaxis.grid(True, linestyle="--", alpha=0.35, linewidth=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ---- Legend ----
leg = ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=5,
    frameon=False,
    fontsize=9.3,
    columnspacing=1.2,
    handletextpad=0.5,
)

plt.tight_layout()

# ---- Save ----
out_png = "./figures/main_results.png"
out_pdf = "./figures/main_results.pdf"
plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
