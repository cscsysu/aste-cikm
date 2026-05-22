"""
Plot STaR convergence curve vs. single-pass gold-guided CoT distillation.
Dual y-axis: left = F1, right = Cumulative GPU-hours.
Output: figures/star_curve.png and figures/star_curve.pdf
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- Data ----
iters      = [1, 2, 3, 4, 5]
star_f1    = [77.92, 78.54, 78.89, 79.05, 79.12]
star_gpuhr = [2.3, 4.5, 6.7, 8.8, 10.9]
ours_f1    = 80.22
ours_gpuhr = 4.1

# ---- Style ----
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})

fig, ax1 = plt.subplots(figsize=(6.4, 4.0), dpi=200)

# ---- Left axis: F1 ----
color_star_f1 = "#1f77b4"   # blue
color_ours_f1 = "#2ca02c"   # green (teal-ish)

l1 = ax1.plot(
    iters, star_f1,
    marker="o", markersize=7, linewidth=2.0,
    color=color_star_f1, label="STaR F1 (per iter.)"
)
l2 = ax1.axhline(
    ours_f1, linestyle="--", linewidth=2.0,
    color=color_ours_f1, label=f"Ours F1 (single pass) = {ours_f1}"
)
# annotate final STaR F1
ax1.annotate(
    f"{star_f1[-1]}",
    xy=(iters[-1], star_f1[-1]),
    xytext=(iters[-1] - 0.15, star_f1[-1] - 0.35),
    fontsize=9, color=color_star_f1,
)
# annotate Ours
ax1.annotate(
    "Ours = 80.22",
    xy=(3.0, ours_f1),
    xytext=(2.4, ours_f1 + 0.12),
    fontsize=9.5, color=color_ours_f1, fontweight="bold",
)

ax1.set_xlabel("STaR iteration")
ax1.set_ylabel("F1 (%)", color="#222222")
ax1.set_xticks(iters)
ax1.set_ylim(77, 81)
ax1.set_xlim(0.7, 5.3)
ax1.tick_params(axis="y", labelcolor="#222222")

# ---- Right axis: GPU-hours ----
ax2 = ax1.twinx()
ax2.grid(False)
color_star_gpu = "#d62728"  # red
color_ours_gpu = "#ff7f0e"  # orange

l3 = ax2.plot(
    iters, star_gpuhr,
    marker="s", markersize=7, linewidth=2.0,
    color=color_star_gpu, label="STaR cumulative GPU-hr"
)
l4 = ax2.axhline(
    ours_gpuhr, linestyle="--", linewidth=2.0,
    color=color_ours_gpu, label=f"Ours GPU-hr = {ours_gpuhr}"
)
# annotate last STaR point
ax2.annotate(
    f"{star_gpuhr[-1]}h",
    xy=(iters[-1], star_gpuhr[-1]),
    xytext=(iters[-1] - 0.45, star_gpuhr[-1] + 0.25),
    fontsize=9, color=color_star_gpu,
)
# annotate Ours GPU-hr
ax2.annotate(
    f"Ours = {ours_gpuhr}h",
    xy=(1.5, ours_gpuhr),
    xytext=(1.1, ours_gpuhr - 1.2),
    fontsize=9.5, color=color_ours_gpu, fontweight="bold",
)

ax2.set_ylabel("Cumulative GPU-hours", color="#222222")
ax2.set_ylim(0, 12)
ax2.tick_params(axis="y", labelcolor="#222222")

# ---- Combined legend ----
lines = l1 + [l2] + l3 + [l4]
labels = [ln.get_label() for ln in lines]
leg = ax1.legend(
    lines, labels,
    loc="lower right", bbox_to_anchor=(0.99, 0.02),
    frameon=True, framealpha=0.95, edgecolor="#cccccc",
    fancybox=False,
)
leg.get_frame().set_linewidth(0.6)

# ---- Title ----
ax1.set_title(
    "STaR cost-quality convergence (Rest14, Qwen3-8B student, GLM-5 teacher)",
    pad=10, fontsize=11.5,
)

plt.tight_layout()

# ---- Save ----
out_png = "./figures/star_curve.png"
out_pdf = "./figures/star_curve.pdf"
plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
