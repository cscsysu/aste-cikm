"""
Analyze why ReasonGraph is much lower than ABSA-R1 on Rest15 and Rest16.
- ABSA-R1: Rest15=84.81, Rest16=84.52
- ReasonGraph (Qwen3): Rest15=77.19, Rest16=80.86
- Gap: Rest15 -7.62, Rest16 -3.66
"""
import json
import re
from collections import Counter

def load(path):
    return [json.loads(line) for line in open(path)]

def ttup(t):
    return (t["aspect"].lower().strip(), t["opinion"].lower().strip(), t["sentiment"].upper().strip())

def sample_stats(sample):
    p = set(ttup(t) for t in sample["pred_triplets"])
    g = set(ttup(t) for t in sample["gold_triplets"])
    return len(p & g), len(p), len(g), p, g

def analyze(path, dataset_name):
    data = load(path)
    n = len(data)

    # overall F1
    tp = sum(len(set(ttup(t) for t in d["pred_triplets"]) & set(ttup(t) for t in d["gold_triplets"])) for d in data)
    total_pred = sum(len(d["pred_triplets"]) for d in data)
    total_gold = sum(len(d["gold_triplets"]) for d in data)
    p = tp / total_pred if total_pred else 0
    r = tp / total_gold if total_gold else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0

    # error type classification
    fully_correct = 0
    no_pred = 0  # prediction empty but gold non-empty
    over_predict = 0  # over-predicted
    under_predict = 0  # under-predicted
    wrong_predict = 0  # right count but wrong content
    empty_gold = 0  # gold empty

    # triplet-level error classification
    fp_over_extract = 0  # predicted something gold doesn't have
    fp_sentiment = 0  # aspect-opinion pair correct but sentiment wrong
    fp_span = 0  # sentiment correct but span boundary differs
    fn = 0  # gold has but not predicted

    # sentiment distribution
    gold_sentiments = Counter()
    pred_sentiments = Counter()

    # gold triplet count distribution
    gold_count_dist = Counter()

    for d in data:
        gold = d["gold_triplets"]
        pred = d["pred_triplets"]
        gold_count_dist[len(gold)] += 1

        for g in gold:
            gold_sentiments[g["sentiment"].upper()] += 1
        for pr in pred:
            pred_sentiments[pr["sentiment"].upper()] += 1

        gset = set(ttup(t) for t in gold)
        pset = set(ttup(t) for t in pred)

        if not gold and not pred:
            continue
        if not gold:
            empty_gold += 1
            continue

        if pset == gset:
            fully_correct += 1
        elif not pset:
            no_pred += 1
        elif len(pset) > len(gset):
            over_predict += 1
        elif len(pset) < len(gset):
            under_predict += 1
        else:
            wrong_predict += 1

        # triplet-level FP classification
        tp_set = gset & pset
        fp_set = pset - gset
        fn_set = gset - pset
        fn += len(fn_set)

        gold_ao = {(g[0], g[1]): g[2] for g in gset}
        gold_as_span = {(g[0], g[2]): g[1] for g in gset}  # aspect+sent -> opinion
        gold_os_span = {(g[1], g[2]): g[0] for g in gset}  # opinion+sent -> aspect

        for fp_trip in fp_set:
            a, o, s = fp_trip
            if (a, o) in gold_ao and gold_ao[(a, o)] != s:
                fp_sentiment += 1
            elif any(g[0] == a and g[2] == s for g in gset) or any(g[1] == o and g[2] == s for g in gset):
                fp_span += 1
            else:
                fp_over_extract += 1

    print(f"\n{'='*70}")
    print(f"  {dataset_name}")
    print(f"{'='*70}")
    print(f"Total samples: {n}")
    print(f"F1: {f1*100:.2f} (P={p*100:.2f}, R={r*100:.2f})")
    print(f"Total pred triplets: {total_pred}, total gold triplets: {total_gold}, TP: {tp}")
    print()
    print(f"--- Sample-level classification ---")
    print(f"  Fully correct:           {fully_correct:>4} ({fully_correct/n*100:.1f}%)")
    print(f"  Empty prediction:        {no_pred:>4} ({no_pred/n*100:.1f}%)")
    print(f"  Over-predicted (>gold):  {over_predict:>4} ({over_predict/n*100:.1f}%)")
    print(f"  Under-predicted (<gold): {under_predict:>4} ({under_predict/n*100:.1f}%)")
    print(f"  Right count wrong content: {wrong_predict:>4} ({wrong_predict/n*100:.1f}%)")
    print(f"  Empty gold:              {empty_gold:>4}")
    print()
    print(f"--- Triplet-level FP analysis (total FP = {total_pred - tp}) ---")
    print(f"  Over-extraction:       {fp_over_extract:>4} ({fp_over_extract/max(total_pred-tp,1)*100:.1f}%)")
    print(f"  Sentiment error:       {fp_sentiment:>4} ({fp_sentiment/max(total_pred-tp,1)*100:.1f}%)")
    print(f"  Span boundary error:   {fp_span:>4} ({fp_span/max(total_pred-tp,1)*100:.1f}%)")
    print(f"  Missing (FN):          {fn:>4}")
    print()
    print(f"--- Gold sentiment distribution ---")
    for s, c in gold_sentiments.most_common():
        print(f"  {s}: {c} ({c/sum(gold_sentiments.values())*100:.1f}%)")
    print(f"--- Pred sentiment distribution ---")
    for s, c in pred_sentiments.most_common():
        print(f"  {s}: {c} ({c/max(sum(pred_sentiments.values()),1)*100:.1f}%)")
    print()
    print(f"--- Gold triplet count distribution ---")
    for k in sorted(gold_count_dist):
        print(f"  {k} triplets: {gold_count_dist[k]} samples")

    return {
        "f1": f1*100, "p": p*100, "r": r*100,
        "fully_correct": fully_correct, "n": n,
        "fp_over": fp_over_extract, "fp_sent": fp_sentiment, "fp_span": fp_span,
        "fn": fn,
    }

# Analyze Qwen3 on Rest15 and Rest16
print("\n" + "#"*70)
print("# Qwen3 analysis")
print("#"*70)
r15_q = analyze("./rest15_predictions.jsonl", "Rest15 (Qwen3)")
r16_q = analyze("./rest16_predictions.jsonl", "Rest16 (Qwen3)")
r14_q = analyze("./full_rest14.jsonl", "Rest14 (Qwen3)")
lap_q = analyze("./lap14_predictions.jsonl", "Lap14 (Qwen3)")

# summary comparison
print("\n" + "#"*70)
print("# Summary comparison")
print("#"*70)
print(f"\n{'Dataset':<12} {'F1':>7} {'P':>7} {'R':>7} {'Over%':>7} {'Sent%':>7} {'Span%':>7} {'FN':>6}")
for name, d in [("Rest14", r14_q), ("Lap14", lap_q), ("Rest15", r15_q), ("Rest16", r16_q)]:
    total_fp = d["fp_over"] + d["fp_sent"] + d["fp_span"]
    print(f"{name:<12} {d['f1']:>7.2f} {d['p']:>7.2f} {d['r']:>7.2f} "
          f"{d['fp_over']/max(total_fp,1)*100:>6.1f}% "
          f"{d['fp_sent']/max(total_fp,1)*100:>6.1f}% "
          f"{d['fp_span']/max(total_fp,1)*100:>6.1f}% "
          f"{d['fn']:>6}")
