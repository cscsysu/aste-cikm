"""
Plot False Positive breakdown pie charts for ALL FOUR ASTE datasets.

Rest14 uses the original hard-coded numbers (115/51/32, total FP = 198) as
in the previously published version of the figure; the other three datasets
are computed from prediction files using RELAXED matching (aspect/opinion
substring tolerance, sentiment exact) so that the resulting P/R/F1 align
with the numbers reported in the main table (Qwen3-8B Stage 2).

For the computed datasets, each unmatched prediction is assigned to exactly
one category, in this order:
  1. Sentiment error  -- some gold has relaxed-matching aspect AND opinion,
                         but different sentiment.
  2. Span boundary    -- some gold has the same sentiment AND at least one
                         span (aspect or opinion) overlaps (substring),
                         but the triplet is not a strict-equal match.
  3. Over-extraction  -- everything else (hallucinated / out-of-scope).

Outputs (separate figures, one per dataset):
  pie_pic/error_pie_rest14.{png,pdf}
  pie_pic/error_pie_lap14.{png,pdf}
  pie_pic/error_pie_rest15.{png,pdf}
  pie_pic/error_pie_rest16.{png,pdf}
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

BASE     = os.environ.get(
    "REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
OUT_DIR  = os.path.join(BASE, "pie_pic")
os.makedirs(OUT_DIR, exist_ok=True)

# Qwen3-8B Stage 2 prediction files for the three datasets we recompute.
COMPUTED_DATASETS = [
    ("Lap14",  os.path.join(BASE, "lap14_predictions.jsonl")),
    ("Rest15", os.path.join(BASE, "rest15_predictions.jsonl")),
    ("Rest16", os.path.join(BASE, "rest16_predictions.jsonl")),
]

# Rest14 keeps its previously published breakdown (Over, Sent, Span).
REST14_FIXED = {"fp_over": 115, "fp_sent": 51, "fp_span": 32}

def norm(s):
    return (s or "").lower().strip()

def relaxed(a, b):
    if not a or not b:
        return a == b
    return a == b or a in b or b in a

def is_match(p, g):
    return (norm(p["sentiment"]) == norm(g["sentiment"])
            and relaxed(norm(p["aspect"]),  norm(g["aspect"]))
            and relaxed(norm(p["opinion"]), norm(g["opinion"])))

def categorize(path):
    """Return (n_samples, total_pred, total_gold, tp, fp_over, fp_sent, fp_span)."""
    data = [json.loads(l) for l in open(path)]
    tp = 0
    total_p = 0
    total_g = 0
    fp_over = fp_sent = fp_span = 0

    for d in data:
        preds = d["pred_triplets"]
        golds = d["gold_triplets"]
        total_p += len(preds)
        total_g += len(golds)

        # Greedy bipartite TP matching (relaxed)
        matched_g = [False] * len(golds)
        matched_p = [False] * len(preds)
        for i, p in enumerate(preds):
            for j, g in enumerate(golds):
                if matched_g[j]:
                    continue
                if is_match(p, g):
                    matched_g[j] = True
                    matched_p[i] = True
                    tp += 1
                    break

        # Classify unmatched predictions
        for i, p in enumerate(preds):
            if matched_p[i]:
                continue
            ap, op, sp = norm(p["aspect"]), norm(p["opinion"]), norm(p["sentiment"])

            # 1. Sentiment error
            is_sent_err = False
            for g in golds:
                ag, og, sg = norm(g["aspect"]), norm(g["opinion"]), norm(g["sentiment"])
                if relaxed(ap, ag) and relaxed(op, og) and sp != sg:
                    is_sent_err = True
                    break
            if is_sent_err:
                fp_sent += 1
                continue

            # 2. Span boundary
            is_span_err = False
            for g in golds:
                ag, og, sg = norm(g["aspect"]), norm(g["opinion"]), norm(g["sentiment"])
                if sp != sg:
                    continue
                a_overlap = (ap == ag) or (ap and ag and (ap in ag or ag in ap))
                o_overlap = (op == og) or (op and og and (op in og or og in op))
                strict_eq = (ap == ag and op == og)
                if (a_overlap or o_overlap) and not strict_eq:
                    is_span_err = True
                    break
            if is_span_err:
                fp_span += 1
                continue

            # 3. Over-extraction
            fp_over += 1

    return {
        "n":           len(data),
        "total_pred":  total_p,
        "total_gold":  total_g,
        "tp":          tp,
        "fp_total":    total_p - tp,
        "fp_over":     fp_over,
        "fp_sent":     fp_sent,
        "fp_span":     fp_span,
    }

# Style
mpl.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
})

COLORS  = ["#e74c3c", "#f39c12", "#3498db"]   # Over, Sent, Span
LABELS  = ["Over-extraction\n(Hallucination)",
           "Sentiment\nError",
           "Span Boundary\nMismatch"]
EXPLODE = (0.05, 0.03, 0.03)

print(f"{'Dataset':<8} {'N':>5} {'Gold':>5} {'Pred':>5} {'TP':>5} {'FP':>5}  "
      f"{'Over':>5} {'Sent':>5} {'Span':>5}   F1")
all_stats = {}

# Rest14: fixed values from the previously published figure
r14 = dict(REST14_FIXED)
r14["fp_total"] = r14["fp_over"] + r14["fp_sent"] + r14["fp_span"]
all_stats["Rest14"] = r14
print(f"{'Rest14':<8} {'-':>5} {'-':>5} {'-':>5} {'-':>5} "
      f"{r14['fp_total']:>5}  "
      f"{r14['fp_over']:>5} {r14['fp_sent']:>5} {r14['fp_span']:>5}   "
      f"(fixed)")

# Other datasets: recompute from predictions
for name, path in COMPUTED_DATASETS:
    s = categorize(path)
    all_stats[name] = s
    P = s["tp"] / s["total_pred"] * 100 if s["total_pred"] else 0
    R = s["tp"] / s["total_gold"] * 100 if s["total_gold"] else 0
    F = 2*P*R/(P+R) if (P+R) else 0
    print(f"{name:<8} {s['n']:>5} {s['total_gold']:>5} {s['total_pred']:>5} "
          f"{s['tp']:>5} {s['fp_total']:>5}  "
          f"{s['fp_over']:>5} {s['fp_sent']:>5} {s['fp_span']:>5}   {F:.2f}")

# Draw one figure per dataset
for name, s in all_stats.items():
    sizes = [s["fp_over"], s["fp_sent"], s["fp_span"]]
    total = sum(sizes)
    assert total == s["fp_total"], f"{name}: pie total {total} != FP {s['fp_total']}"

    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=EXPLODE,
        labels=LABELS,
        colors=COLORS,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * total / 100))})",
        startangle=140,
        textprops={"fontsize": 16},
        pctdistance=0.68,
        labeldistance=1.12,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(15)
        t.set_fontweight("bold")
        t.set_color("white")

    # Force identical canvas across all four datasets so subfigures render at
    # the same visual size in the paper. Tighten the axis range so the pie
    # itself fills more of the canvas (less surrounding whitespace), then the
    # entire \linewidth in LaTeX is mostly pie rather than padding.
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.20, 1.20)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    stem = name.lower()
    out_png = os.path.join(OUT_DIR, f"error_pie_{stem}.png")
    out_pdf = os.path.join(OUT_DIR, f"error_pie_{stem}.pdf")
    plt.savefig(out_png, dpi=300, facecolor="white")
    plt.savefig(out_pdf,            facecolor="white")
    plt.close(fig)
    print(f"  saved {out_png}")
    print(f"  saved {out_pdf}")
