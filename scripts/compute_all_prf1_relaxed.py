"""
Compute P/R/F1 from all local jsonl prediction files using RELAXED matching.

Relaxed matching rule:
  A predicted triplet (a_p, o_p, s_p) matches a gold triplet (a_g, o_g, s_g) iff
    - sentiment match exactly: s_p == s_g
    - aspect relaxed match:  a_p == a_g  OR  a_p in a_g  OR  a_g in a_p
    - opinion relaxed match: o_p == o_g  OR  o_p in o_g  OR  o_g in o_p
  (all comparisons after .lower().strip())

Each gold can be matched by at most one pred (greedy bipartite matching).
"""
import json
import os
import re

# Paths are resolved relative to the repository root.
# Override via env vars if your layout differs.
REPO_ROOT = os.environ.get(
    "REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
ASTE_DATA = os.path.join(REPO_ROOT, "data", "aste")
GLM5_DIR  = os.path.join(REPO_ROOT, "results", "llm_baselines", "glm5")
BASE      = REPO_ROOT

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

def norm(s):
    return (s or "").lower().strip()

def relaxed_span_match(a, b):
    """True if a == b or a is substring of b or b is substring of a."""
    if not a or not b:
        return a == b
    return a == b or a in b or b in a

def count_matches(preds, golds):
    """Greedy bipartite matching: each gold matched by at most one pred."""
    matched_gold = [False] * len(golds)
    tp = 0
    for p in preds:
        ap, op, sp = norm(p.get("aspect", "")), norm(p.get("opinion", "")), norm(p.get("sentiment", ""))
        for i, g in enumerate(golds):
            if matched_gold[i]:
                continue
            ag, og, sg = norm(g.get("aspect", "")), norm(g.get("opinion", "")), norm(g.get("sentiment", ""))
            if sp == sg and relaxed_span_match(ap, ag) and relaxed_span_match(op, og):
                matched_gold[i] = True
                tp += 1
                break
    return tp

def prf1_relaxed(preds_list, golds_list):
    tp = 0
    total_p = 0
    total_g = 0
    for p, g in zip(preds_list, golds_list):
        tp += count_matches(p, g)
        total_p += len(p)
        total_g += len(g)
    P = tp / total_p if total_p else 0
    R = tp / total_g if total_g else 0
    F = 2 * P * R / (P + R) if (P + R) else 0
    return P * 100, R * 100, F * 100, tp, total_p, total_g

def parse_glm5_content(content):
    if not content:
        return []
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        arr = json.loads(content)
        if isinstance(arr, list):
            out = []
            for t in arr:
                if isinstance(t, dict):
                    s = (t.get("sentiment", "") or "").upper()
                    s = s.replace("POSITIVE","POS").replace("NEGATIVE","NEG").replace("NEUTRAL","NEU")
                    out.append({
                        "aspect":    t.get("aspect", ""),
                        "opinion":   t.get("opinion", ""),
                        "sentiment": s,
                    })
            return out
    except Exception:
        pass
    return []

def load_gold_data(dataset):
    path = f"{ASTE_DATA}/{dataset}_test.jsonl"
    return load_jsonl(path)

def eval_glm5(dataset, shot):
    pred_file = f"{GLM5_DIR}/{dataset}_{shot}.jsonl"
    if not os.path.exists(pred_file):
        return None
    pred_data = load_jsonl(pred_file)
    gold_data = load_gold_data(dataset)
    preds_list, golds_list = [], []
    for i, p in enumerate(pred_data):
        preds_list.append(parse_glm5_content(p.get("content", "")))
        if i < len(gold_data):
            golds_list.append(gold_data[i].get("triplets") or gold_data[i].get("gold_triplets") or [])
    return prf1_relaxed(preds_list, golds_list)

def eval_ours(pred_file):
    if not os.path.exists(pred_file):
        return None
    data = load_jsonl(pred_file)
    preds = [d["pred_triplets"] for d in data]
    golds = [d["gold_triplets"] for d in data]
    return prf1_relaxed(preds, golds)

def fmt(res, name):
    if res is None:
        print(f"{name:40} NOT FOUND")
        return
    P, R, F, tp, tp_, tg = res
    print(f"{name:40} P={P:6.2f}  R={R:6.2f}  F1={F:6.2f}  (TP={tp}, Pred={tp_}, Gold={tg})")

print("=" * 95)
print("RELAXED MATCHING — GLM-5 (API baseline)")
print("=" * 95)
for d in ["rest14", "lap14", "rest15", "rest16"]:
    for s in ["zero-shot", "few-shot"]:
        fmt(eval_glm5(d, s), f"GLM-5 {d} {s}")

print("\n" + "=" * 95)
print("RELAXED MATCHING — Ours (Qwen3-8B, Stage 2)")
print("=" * 95)
fmt(eval_ours(f"{BASE}/full_rest14.jsonl"),        "Ours-Qwen3 rest14")
fmt(eval_ours(f"{BASE}/lap14_predictions.jsonl"),  "Ours-Qwen3 lap14")
fmt(eval_ours(f"{BASE}/rest15_predictions.jsonl"), "Ours-Qwen3 rest15")
fmt(eval_ours(f"{BASE}/rest16_predictions.jsonl"), "Ours-Qwen3 rest16")

print("\n" + "=" * 95)
print("RELAXED MATCHING — Ours (Llama3.1-8B, Stage 2)")
print("=" * 95)
fmt(eval_ours(f"{BASE}/llama_rest14_predictions.jsonl"), "Ours-Llama rest14")
fmt(eval_ours(f"{BASE}/llama_lap14_predictions.jsonl"),  "Ours-Llama lap14")
fmt(eval_ours(f"{BASE}/llama_rest15_predictions.jsonl"), "Ours-Llama rest15")
fmt(eval_ours(f"{BASE}/llama_rest16_predictions.jsonl"), "Ours-Llama rest16")

print("\n" + "=" * 95)
print("RELAXED MATCHING — Ablation: no-graph rest14")
print("=" * 95)
fmt(eval_ours(f"{BASE}/no_graph_rest14.jsonl"), "w/o Graph rest14")
