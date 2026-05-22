"""
LLM+Graph complementarity analysis script
Compares Full model (with graph) vs w/o Graph on Rest14 sample-by-sample.
"""

import json
import re
from collections import Counter

def load_predictions(path):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def triplet_to_tuple(t):
    """Normalize a triplet to (aspect, opinion, sentiment) tuple, lowercased"""
    return (t["aspect"].lower().strip(), t["opinion"].lower().strip(), t["sentiment"].upper().strip())

def get_correct_triplets(pred_triplets, gold_triplets):
    """Return the set of correctly predicted triplets"""
    pred_set = set(triplet_to_tuple(t) for t in pred_triplets)
    gold_set = set(triplet_to_tuple(t) for t in gold_triplets)
    return pred_set & gold_set

def sample_f1(pred_triplets, gold_triplets):
    """Per-sample P/R/F1"""
    pred_set = set(triplet_to_tuple(t) for t in pred_triplets)
    gold_set = set(triplet_to_tuple(t) for t in gold_triplets)
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else 0
    r = tp / len(gold_set) if gold_set else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1

def has_negation(text):
    """Detect whether the sentence contains negation words"""
    neg_words = [r"\bnot\b", r"\bn't\b", r"\bnever\b", r"\bno\b", r"\bnor\b", r"\bneither\b", r"\bhardly\b", r"\bbarely\b"]
    for pattern in neg_words:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def has_contrast(text):
    """Detect whether the sentence contains contrast words"""
    contrast_words = [r"\bbut\b", r"\balthough\b", r"\bhowever\b", r"\byet\b", r"\bdespite\b", r"\bthough\b", r"\bwhile\b"]
    for pattern in contrast_words:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def count_gold_triplets(gold_triplets):
    return len(gold_triplets)

def avg_span_distance(gold_triplets):
    """Average token distance between aspect and opinion across gold triplets"""
    distances = []
    for t in gold_triplets:
        if "aspect_indices" in t and "opinion_indices" in t:
            a_center = sum(t["aspect_indices"]) / len(t["aspect_indices"])
            o_center = sum(t["opinion_indices"]) / len(t["opinion_indices"])
            distances.append(abs(a_center - o_center))
    return sum(distances) / len(distances) if distances else 0

def main():
    full = load_predictions("./full_rest14.jsonl")
    no_graph = load_predictions("./no_graph_rest14.jsonl")

    assert len(full) == len(no_graph), f"Sample count mismatch: {len(full)} vs {len(no_graph)}"
    n = len(full)

    # ========== Sample-level classification ==========
    # treat F1=1.0 as "fully correct"
    both_correct = []    # A: both fully correct
    full_only = []       # B: Full correct, w/o Graph wrong -> Graph contribution
    no_graph_only = []   # C: w/o Graph correct, Full wrong -> Graph hurt
    both_wrong = []      # D: both wrong

    # finer granularity: per-triplet
    full_recovered = []  # triplets that Full got right but no_graph didn't
    graph_hurt = []      # triplets that no_graph got right but Full didn't

    for i in range(n):
        f_sample = full[i]
        ng_sample = no_graph[i]

        f_p, f_r, f_f1 = sample_f1(f_sample["pred_triplets"], f_sample["gold_triplets"])
        ng_p, ng_r, ng_f1 = sample_f1(ng_sample["pred_triplets"], ng_sample["gold_triplets"])

        f_correct = get_correct_triplets(f_sample["pred_triplets"], f_sample["gold_triplets"])
        ng_correct = get_correct_triplets(ng_sample["pred_triplets"], ng_sample["gold_triplets"])

        # triplet level
        recovered = f_correct - ng_correct  # ones the graph helped recover
        hurt = ng_correct - f_correct       # ones the graph caused us to lose

        # we use id to label samples (we can't recover the original sentence from generated_text)
        info = {
            "id": f_sample["id"],
            "full_f1": f_f1,
            "no_graph_f1": ng_f1,
            "delta_f1": f_f1 - ng_f1,
            "gold_triplets": f_sample["gold_triplets"],
            "full_pred": f_sample["pred_triplets"],
            "no_graph_pred": ng_sample["pred_triplets"],
            "recovered_triplets": list(recovered),
            "hurt_triplets": list(hurt),
        }

        if f_f1 == 1.0 and ng_f1 == 1.0:
            both_correct.append(info)
        elif f_f1 == 1.0 and ng_f1 < 1.0:
            full_only.append(info)
        elif f_f1 < 1.0 and ng_f1 == 1.0:
            no_graph_only.append(info)
        else:
            both_wrong.append(info)

        full_recovered.extend([(r, info) for r in recovered])
        graph_hurt.extend([(h, info) for h in hurt])

    # ========== Output statistics ==========
    print("=" * 70)
    print("LLM + Graph complementarity analysis (Rest14, Qwen3-8B, Stage 2)")
    print("=" * 70)
    print(f"\nTotal samples: {n}")
    print(f"\n=== Sample-level statistics (F1=1.0 == fully correct) ===")
    print(f"  A: both fully correct:        {len(both_correct):>4} ({len(both_correct)/n*100:.1f}%)")
    print(f"  B: Full correct, w/o Graph wrong:   {len(full_only):>4} ({len(full_only)/n*100:.1f}%)  <- Graph contribution")
    print(f"  C: w/o Graph correct, Full wrong:   {len(no_graph_only):>4} ({len(no_graph_only)/n*100:.1f}%)  <- Graph interference")
    print(f"  D: both not fully correct:    {len(both_wrong):>4} ({len(both_wrong)/n*100:.1f}%)")

    print(f"\n=== Triplet-level statistics ===")
    print(f"  Triplets recovered by Graph: {len(full_recovered)}")
    print(f"  Triplets lost due to Graph:  {len(graph_hurt)}")
    print(f"  Net gain: +{len(full_recovered) - len(graph_hurt)} triplets")

    # ========== Analyze characteristics of B-class samples (Graph contribution) ==========
    print(f"\n=== Graph-contribution samples (class B, {len(full_only)}) feature analysis ===")

    # We'd need the original sentence to analyze negation/contrast. We can't recover it
    # from generated_text, but we can infer some signal from gold_triplets aspect/opinion.
    # We use samples in B+D with delta_f1 > 0 for finer analysis.
    improved = [info for info in (full_only + both_wrong) if info["delta_f1"] > 0]
    degraded = [info for info in (no_graph_only + both_wrong) if info["delta_f1"] < 0]
    neutral = [info for info in both_wrong if info["delta_f1"] == 0]

    print(f"\n=== Sample classification by F1 change ===")
    print(f"  Samples where Graph improves F1: {len(improved):>4} ({len(improved)/n*100:.1f}%)")
    print(f"  Samples where Graph reduces F1:  {len(degraded):>4} ({len(degraded)/n*100:.1f}%)")
    print(f"  Samples where Graph has no effect: {n - len(improved) - len(degraded):>4} ({(n - len(improved) - len(degraded))/n*100:.1f}%)")

    # gold-triplet count distribution within improved samples
    improved_triplet_counts = [count_gold_triplets(s["gold_triplets"]) for s in improved]
    degraded_triplet_counts = [count_gold_triplets(s["gold_triplets"]) for s in degraded]
    all_triplet_counts = [count_gold_triplets(full[i]["gold_triplets"]) for i in range(n)]

    print(f"\n=== Gold triplet count distribution ===")
    print(f"  All samples mean:               {sum(all_triplet_counts)/len(all_triplet_counts):.2f}")
    if improved_triplet_counts:
        print(f"  Graph-improved samples mean:    {sum(improved_triplet_counts)/len(improved_triplet_counts):.2f}")
    if degraded_triplet_counts:
        print(f"  Graph-degraded samples mean:    {sum(degraded_triplet_counts)/len(degraded_triplet_counts):.2f}")

    # aspect-opinion distance statistics
    improved_dists = [avg_span_distance(s["gold_triplets"]) for s in improved]
    degraded_dists = [avg_span_distance(s["gold_triplets"]) for s in degraded]
    all_dists = [avg_span_distance(full[i]["gold_triplets"]) for i in range(n)]

    print(f"\n=== Aspect-Opinion average token distance ===")
    print(f"  All samples mean:               {sum(all_dists)/len(all_dists):.2f}")
    if improved_dists:
        print(f"  Graph-improved samples mean:    {sum(improved_dists)/len(improved_dists):.2f}")
    if degraded_dists:
        print(f"  Graph-degraded samples mean:    {sum(degraded_dists)/len(degraded_dists):.2f}")

    # ========== Overall F1 verification ==========
    print(f"\n=== Overall F1 verification ===")
    total_tp_full, total_pred_full, total_gold_full = 0, 0, 0
    total_tp_ng, total_pred_ng, total_gold_ng = 0, 0, 0
    for i in range(n):
        f_pred = set(triplet_to_tuple(t) for t in full[i]["pred_triplets"])
        f_gold = set(triplet_to_tuple(t) for t in full[i]["gold_triplets"])
        ng_pred = set(triplet_to_tuple(t) for t in no_graph[i]["pred_triplets"])
        ng_gold = set(triplet_to_tuple(t) for t in no_graph[i]["gold_triplets"])

        total_tp_full += len(f_pred & f_gold)
        total_pred_full += len(f_pred)
        total_gold_full += len(f_gold)

        total_tp_ng += len(ng_pred & ng_gold)
        total_pred_ng += len(ng_pred)
        total_gold_ng += len(ng_gold)

    p_full = total_tp_full / total_pred_full if total_pred_full else 0
    r_full = total_tp_full / total_gold_full if total_gold_full else 0
    f1_full = 2 * p_full * r_full / (p_full + r_full) if (p_full + r_full) else 0

    p_ng = total_tp_ng / total_pred_ng if total_pred_ng else 0
    r_ng = total_tp_ng / total_gold_ng if total_gold_ng else 0
    f1_ng = 2 * p_ng * r_ng / (p_ng + r_ng) if (p_ng + r_ng) else 0

    print(f"  Full model:  P={p_full*100:.2f}, R={r_full*100:.2f}, F1={f1_full*100:.2f}")
    print(f"  w/o Graph:   P={p_ng*100:.2f}, R={r_ng*100:.2f}, F1={f1_ng*100:.2f}")
    print(f"  Δ F1: {(f1_full - f1_ng)*100:+.2f}")

    # ========== Print typical cases ==========
    print(f"\n=== Top 5 samples with biggest Graph contribution (sorted by delta F1) ===")
    all_info = []
    for i in range(n):
        f_p, f_r, f_f1 = sample_f1(full[i]["pred_triplets"], full[i]["gold_triplets"])
        ng_p, ng_r, ng_f1 = sample_f1(no_graph[i]["pred_triplets"], no_graph[i]["gold_triplets"])
        all_info.append({
            "id": full[i]["id"],
            "delta_f1": f_f1 - ng_f1,
            "full_f1": f_f1,
            "no_graph_f1": ng_f1,
            "gold": [triplet_to_tuple(t) for t in full[i]["gold_triplets"]],
            "full_pred": [triplet_to_tuple(t) for t in full[i]["pred_triplets"]],
            "no_graph_pred": [triplet_to_tuple(t) for t in no_graph[i]["pred_triplets"]],
        })

    all_info.sort(key=lambda x: x["delta_f1"], reverse=True)
    for info in all_info[:5]:
        print(f"\n  [{info['id']}] ΔF1={info['delta_f1']:+.2f} (Full={info['full_f1']:.2f}, w/o Graph={info['no_graph_f1']:.2f})")
        print(f"    Gold:     {info['gold']}")
        print(f"    Full:     {info['full_pred']}")
        print(f"    w/o Graph:{info['no_graph_pred']}")

    print(f"\n=== Top 5 samples where Graph hurts the most ===")
    for info in all_info[-5:]:
        print(f"\n  [{info['id']}] ΔF1={info['delta_f1']:+.2f} (Full={info['full_f1']:.2f}, w/o Graph={info['no_graph_f1']:.2f})")
        print(f"    Gold:     {info['gold']}")
        print(f"    Full:     {info['full_pred']}")
        print(f"    w/o Graph:{info['no_graph_pred']}")

if __name__ == "__main__":
    main()
