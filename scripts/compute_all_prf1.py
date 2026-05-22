"""
Compute P/R/F1 from all local jsonl prediction files:
  (1) Ours (Qwen3) — full_rest14.jsonl, *_predictions.jsonl (rest15/16, lap14)
  (2) Ours (Llama) — llama_rest14/15/16/lap14_predictions.jsonl
  (3) GLM-5 zero/few-shot — results/llm_baselines/glm5/*.jsonl
  (4) no-graph ablation — no_graph_rest14.jsonl

For GLM-5 the format is {id, content (JSON str), reasoning}; we need to parse content
and align with gold from data/aste/*_test.jsonl by id.
"""
import json
import os
import re

# Paths are resolved relative to the repository root.
REPO_ROOT = os.environ.get(
    "REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
ASTE_DATA = os.path.join(REPO_ROOT, "data", "aste")
GLM5_DIR  = os.path.join(REPO_ROOT, "results", "llm_baselines", "glm5")
BASE      = REPO_ROOT

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

def ttup(t):
    a = t.get("aspect", "") or ""
    o = t.get("opinion", "") or ""
    s = t.get("sentiment", "") or ""
    return (a.lower().strip(), o.lower().strip(), s.upper().strip())

def prf1(preds_list, golds_list):
    tp = 0
    total_p = 0
    total_g = 0
    for p, g in zip(preds_list, golds_list):
        ps = set(ttup(t) for t in p)
        gs = set(ttup(t) for t in g)
        tp += len(ps & gs)
        total_p += len(ps)
        total_g += len(gs)
    P = tp / total_p if total_p else 0
    R = tp / total_g if total_g else 0
    F = 2 * P * R / (P + R) if (P + R) else 0
    return P * 100, R * 100, F * 100, tp, total_p, total_g

def parse_glm5_content(content):
    """GLM-5 outputs a JSON list string, sometimes with markdown code fences."""
    if not content:
        return []
    content = content.strip()
    # strip markdown fences
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        arr = json.loads(content)
        if isinstance(arr, list):
            out = []
            for t in arr:
                if isinstance(t, dict):
                    out.append({
                        "aspect":    t.get("aspect", ""),
                        "opinion":   t.get("opinion", ""),
                        "sentiment": (t.get("sentiment", "") or "").upper().replace("POSITIVE","POS").replace("NEGATIVE","NEG").replace("NEUTRAL","NEU"),
                    })
            return out
    except Exception:
        pass
    return []

def load_gold_by_id(dataset):
    path = f"{ASTE_DATA}/{dataset}_test.jsonl"
    data = load_jsonl(path)
    gold_map = {}
    # try several id/idx keys
    for i, d in enumerate(data):
        key = d.get("id") or d.get("idx") or f"{dataset}_test_{i}"
        gold_map[key] = d.get("triplets") or d.get("gold_triplets") or []
    return gold_map, data

def eval_glm5(dataset, shot):
    """dataset in {rest14,rest15,rest16,lap14}, shot in {zero-shot, few-shot}"""
    pred_file = f"{GLM5_DIR}/{dataset}_{shot}.jsonl"
    if not os.path.exists(pred_file):
        return None
    pred_data = load_jsonl(pred_file)
    gold_map, gold_data = load_gold_by_id(dataset)

    # Match by order (most robust)
    preds_list = []
    golds_list = []
    for i, p in enumerate(pred_data):
        preds_list.append(parse_glm5_content(p.get("content", "")))
        if i < len(gold_data):
            golds_list.append(gold_data[i].get("triplets") or gold_data[i].get("gold_triplets") or [])
    return prf1(preds_list, golds_list)

def eval_ours(pred_file):
    if not os.path.exists(pred_file):
        return None
    data = load_jsonl(pred_file)
    preds = [d["pred_triplets"] for d in data]
    golds = [d["gold_triplets"] for d in data]
    return prf1(preds, golds)

def fmt(res, name):
    if res is None:
        print(f"{name:40} NOT FOUND")
        return
    P, R, F, tp, tp_, tg = res
    print(f"{name:40} P={P:6.2f}  R={R:6.2f}  F1={F:6.2f}  (TP={tp}, Pred={tp_}, Gold={tg})")

print("=" * 95)
print("GLM-5 (API baseline) — P/R/F1 from local jsonl")
print("=" * 95)
for d in ["rest14", "lap14", "rest15", "rest16"]:
    for s in ["zero-shot", "few-shot"]:
        fmt(eval_glm5(d, s), f"GLM-5 {d} {s}")

print("\n" + "=" * 95)
print("Ours (Qwen3-8B, Stage 2) — P/R/F1 from local jsonl")
print("=" * 95)
fmt(eval_ours(f"{BASE}/full_rest14.jsonl"),       "Ours-Qwen3 rest14")
fmt(eval_ours(f"{BASE}/lap14_predictions.jsonl"), "Ours-Qwen3 lap14")
fmt(eval_ours(f"{BASE}/rest15_predictions.jsonl"),"Ours-Qwen3 rest15")
fmt(eval_ours(f"{BASE}/rest16_predictions.jsonl"),"Ours-Qwen3 rest16")

print("\n" + "=" * 95)
print("Ours (Llama3.1-8B, Stage 2) — P/R/F1 from local jsonl")
print("=" * 95)
fmt(eval_ours(f"{BASE}/llama_rest14_predictions.jsonl"), "Ours-Llama rest14")
fmt(eval_ours(f"{BASE}/llama_lap14_predictions.jsonl"),  "Ours-Llama lap14")
fmt(eval_ours(f"{BASE}/llama_rest15_predictions.jsonl"), "Ours-Llama rest15")
fmt(eval_ours(f"{BASE}/llama_rest16_predictions.jsonl"), "Ours-Llama rest16")

print("\n" + "=" * 95)
print("Ablation — no-graph rest14")
print("=" * 95)
fmt(eval_ours(f"{BASE}/no_graph_rest14.jsonl"), "w/o Graph rest14")
