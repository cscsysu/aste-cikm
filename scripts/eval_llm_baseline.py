"""
LLM Zero-shot / Few-shot baseline evaluation script.
Uses an LLM to do ABSA directly (without gold labels) and reports F1.

Usage:
    python scripts/eval_llm_baseline.py --model glm5 --mode zero-shot --dataset rest14
    python scripts/eval_llm_baseline.py --model qwen --mode few-shot --dataset rest14
    python scripts/eval_llm_baseline.py --model all --mode all --dataset all
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_client import batch_call, call_api, extract_content, MODELS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results", "llm_baselines")

# Few-shot examples: format unified as JSON, consistent with the system prompt
FEW_SHOT_EXAMPLES = """Example 1:
Sentence: "Great food but the service was dreadfully slow."
Output: [{"aspect": "food", "opinion": "Great", "sentiment": "POS"}, {"aspect": "service", "opinion": "dreadfully slow", "sentiment": "NEG"}]

Example 2:
Sentence: "The battery life is amazing."
Output: [{"aspect": "battery life", "opinion": "amazing", "sentiment": "POS"}]

Example 3:
Sentence: "The restaurant was expensive, but the menu was creative."
Output: [{"aspect": "restaurant", "opinion": "expensive", "sentiment": "NEG"}, {"aspect": "menu", "opinion": "creative", "sentiment": "POS"}]

Example 4:
Sentence: "Staff was friendly but not very attentive."
Output: [{"aspect": "Staff", "opinion": "friendly", "sentiment": "POS"}, {"aspect": "Staff", "opinion": "not very attentive", "sentiment": "NEG"}]

Example 5:
Sentence: "Average food with nothing really to write home about."
Output: [{"aspect": "food", "opinion": "Average", "sentiment": "NEU"}]
"""

ZERO_SHOT_SYSTEM = """You are an expert in Aspect-Based Sentiment Analysis (ABSA).
Given a review sentence, extract all (aspect, opinion, sentiment) triplets.

Rules:
1. Aspect: the entity being discussed (noun/noun phrase).
2. Opinion: the word/phrase expressing sentiment toward the aspect.
3. Sentiment: POS (positive), NEG (negative), or NEU (neutral).
4. Output ONLY a JSON list of triplets, nothing else.
5. Format: [{"aspect": "...", "opinion": "...", "sentiment": "POS/NEG/NEU"}]
"""

FEW_SHOT_SYSTEM = f"""You are an expert in Aspect-Based Sentiment Analysis (ABSA).
Given a review sentence, extract all (aspect, opinion, sentiment) triplets.

Rules:
1. Aspect: the entity being discussed (noun/noun phrase).
2. Opinion: the word/phrase expressing sentiment toward the aspect.
3. Sentiment: POS (positive), NEG (negative), or NEU (neutral).
4. Output ONLY a JSON list of triplets, nothing else.
5. Format: [{{"aspect": "...", "opinion": "...", "sentiment": "POS/NEG/NEU"}}]

{FEW_SHOT_EXAMPLES}
"""

DATASETS = {
    "rest14": "aste/rest14_test.jsonl",
    "lap14": "aste/lap14_test.jsonl",
    "rest15": "aste/rest15_test.jsonl",
    "rest16": "aste/rest16_test.jsonl",
}


def parse_model_output(text: str) -> list[dict]:
    """Parse triplets from the LLM output"""
    if not text:
        return []

    # try to parse JSON directly
    # find the [ ... ] portion
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if isinstance(item, dict):
                        result.append({
                            "aspect": str(item.get("aspect", "")).strip(),
                            "opinion": str(item.get("opinion", "")).strip(),
                            "sentiment": str(item.get("sentiment", "")).strip().upper(),
                        })
                    elif isinstance(item, (list, tuple)) and len(item) == 3:
                        result.append({
                            "aspect": str(item[0]).strip(),
                            "opinion": str(item[1]).strip(),
                            "sentiment": str(item[2]).strip().upper(),
                        })
                return result
        except json.JSONDecodeError:
            pass

    # Fallback: try to parse the ("aspect", "opinion", "SENTIMENT") format
    pattern = r'\("([^"]+)",\s*"([^"]+)",\s*"?(POS|NEG|NEU)"?\)'
    matches = re.findall(pattern, text)
    return [{"aspect": a, "opinion": o, "sentiment": s} for a, o, s in matches]


def triplet_to_tuple(t: dict) -> tuple:
    """Convert a triplet dict into a comparable tuple (lowercased)"""
    return (
        t["aspect"].lower().strip(),
        t["opinion"].lower().strip(),
        t["sentiment"].upper().strip(),
    )


def compute_f1(predictions: list[list[dict]], golds: list[list[dict]]) -> dict:
    """Compute exact-match F1"""
    tp, fp, fn = 0, 0, 0
    for pred_triplets, gold_triplets in zip(predictions, golds):
        pred_set = set(triplet_to_tuple(t) for t in pred_triplets)
        gold_set = set(triplet_to_tuple(t) for t in gold_triplets)
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "tp": tp, "fp": fp, "fn": fn,
    }


def load_test_data(dataset: str) -> list[dict]:
    filepath = os.path.join(DATA_DIR, DATASETS[dataset])
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def run_evaluation(model_key: str, mode: str, dataset: str):
    """Run evaluation for a (model, mode, dataset) combination"""
    print(f"\n=== {model_key} / {mode} / {dataset} ===")

    samples = load_test_data(dataset)
    system_prompt = FEW_SHOT_SYSTEM if mode == "few-shot" else ZERO_SHOT_SYSTEM

    # build prompts
    prompts = []
    for s in samples:
        prompts.append({
            "id": s["id"],
            "content": f'Sentence: "{s["sentence"]}"\n\nExtract all (aspect, opinion, sentiment) triplets:',
        })

    # call the API (supports resuming)
    out_dir = os.path.join(RESULTS_DIR, model_key)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{dataset}_{mode}.jsonl")

    batch_call(
        model_key=model_key,
        prompts=prompts,
        output_path=output_path,
        system_prompt=system_prompt,
        temperature=0.1,
        max_tokens=2048,
        delay=0.5,
    )

    # load predictions and evaluate
    pred_map = {}
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            parsed = parse_model_output(record.get("content", ""))
            pred_map[record["id"]] = parsed

    predictions = []
    golds = []
    for s in samples:
        predictions.append(pred_map.get(s["id"], []))
        golds.append(s["triplets"])

    metrics = compute_f1(predictions, golds)
    print(f"  Results: P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  F1={metrics['f1']:.2f}")

    # save metrics
    metrics_path = os.path.join(out_dir, f"{dataset}_{mode}_metrics.json")
    metrics["model"] = model_key
    metrics["mode"] = mode
    metrics["dataset"] = dataset
    metrics["num_samples"] = len(samples)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--mode", type=str, default="all",
                        choices=["zero-shot", "few-shot", "all"])
    parser.add_argument("--dataset", type=str, default="all",
                        choices=list(DATASETS.keys()) + ["all"])
    args = parser.parse_args()

    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    modes = ["zero-shot", "few-shot"] if args.mode == "all" else [args.mode]
    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    all_results = []
    for model in models:
        for mode in modes:
            for dataset in datasets:
                metrics = run_evaluation(model, mode, dataset)
                all_results.append(metrics)

    # print summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<12} {'Mode':<12} {'Dataset':<10} {'P':>8} {'R':>8} {'F1':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['model']:<12} {r['mode']:<12} {r['dataset']:<10} {r['precision']:>8.2f} {r['recall']:>8.2f} {r['f1']:>8.2f}")
    print("=" * 70)

    # save summary
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
