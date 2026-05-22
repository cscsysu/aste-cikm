"""
Gold-Guided CoT reasoning generation script.
Given a sentence + gold triplets, prompt the teacher model to generate the reasoning process.
Usage:
    python scripts/generate_cot.py --model glm5
    python scripts/generate_cot.py --model qwen
    python scripts/generate_cot.py --model all
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_client import batch_call, MODELS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASTE_DIR = os.path.join(BASE, "data", "aste")
COT_DIR = os.path.join(BASE, "data", "cot")

SYSTEM_PROMPT = """You are an expert in Aspect-Based Sentiment Analysis (ABSA).
Your task: Given a review sentence and its correct annotation (aspect-opinion-sentiment triplets), generate a detailed step-by-step reasoning process explaining WHY each triplet is correct.

Requirements:
1. Analyze each triplet one by one.
2. For each triplet, explain:
   - How the aspect term is identified (explicit mention or implicit inference).
   - How the opinion term expresses sentiment toward the aspect.
   - Why the sentiment polarity (POS/NEG/NEU) is assigned.
   - Note any linguistic cues: negation, contrast words (but/however), intensifiers, sarcasm.
3. If there are multiple triplets, explain their relationships (e.g., contrast, parallel).
4. Keep the reasoning concise but thorough (150-300 words).
5. End with: "Final Answer: [repeat the triplets exactly]"
"""


def build_user_prompt(sentence: str, triplets: list[dict]) -> str:
    """Build the user prompt: sentence + gold triplets"""
    triplet_strs = []
    for t in triplets:
        triplet_strs.append(f'("{t["aspect"]}", "{t["opinion"]}", {t["sentiment"]})')
    triplets_text = "[" + ", ".join(triplet_strs) + "]"

    return f"""Sentence: "{sentence}"

Correct Annotation (Gold Triplets): {triplets_text}

Please provide a step-by-step reasoning process explaining why these triplets are correct."""


def load_aste_train_data() -> list[dict]:
    """Load all ASTE training data"""
    prompts = []
    for dataset in ["rest14", "lap14", "rest15", "rest16"]:
        filepath = os.path.join(ASTE_DIR, f"{dataset}_train.jsonl")
        if not os.path.exists(filepath):
            print(f"  [WARN] {filepath} not found, skipping")
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                user_prompt = build_user_prompt(sample["sentence"], sample["triplets"])
                prompts.append({
                    "id": sample["id"],
                    "content": user_prompt,
                    "sentence": sample["sentence"],
                    "gold_triplets": sample["triplets"],
                    "source": sample["source"],
                })
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        choices=list(MODELS.keys()) + ["all"],
                        help="Teacher model to use")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Delay between API calls (seconds)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens (Qwen needs ~2048+ due to thinking tokens)")
    args = parser.parse_args()

    prompts = load_aste_train_data()
    print(f"Loaded {len(prompts)} training samples")

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        out_dir = os.path.join(COT_DIR, model_key)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "cot_data.jsonl")
        print(f"\n=== Generating CoT with {model_key} ===")
        batch_call(
            model_key=model_key,
            prompts=prompts,
            output_path=output_path,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=args.max_tokens,
            delay=args.delay,
        )


if __name__ == "__main__":
    main()
