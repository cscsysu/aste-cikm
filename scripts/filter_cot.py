"""
CoT reasoning quality filtering and Multi-Teacher merging.
Check whether the reasoning covers all gold triplets, filter for high-quality data, and merge into output.

Usage: python scripts/filter_cot.py
"""

import json
import os
import re
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COT_DIR = os.path.join(BASE, "data", "cot")
OUTPUT_PATH = os.path.join(BASE, "data", "distill_train.jsonl")


def check_coverage(content: str, gold_triplets: list[dict]) -> tuple[bool, float]:
    """
    Check whether the reasoning covers all aspects and opinions in the gold triplets.
    Returns (fully_covered, coverage_rate)
    """
    if not content:
        return False, 0.0

    content_lower = content.lower()
    total = 0
    covered = 0
    for t in gold_triplets:
        # check aspect
        aspect = t["aspect"].lower()
        if aspect in content_lower:
            covered += 1
        total += 1
        # check opinion
        opinion = t["opinion"].lower()
        if opinion in content_lower:
            covered += 1
        total += 1

    if total == 0:
        return True, 1.0
    rate = covered / total
    return rate >= 1.0, rate


def check_coherence(content: str) -> bool:
    """Basic logical-coherence check: the reasoning must not be too short, empty, or an error message"""
    if not content or len(content) < 50:
        return False
    # check for common error patterns
    error_patterns = ["i cannot", "i'm sorry", "error", "as an ai"]
    for pat in error_patterns:
        if pat in content.lower()[:100]:
            return False
    return True


def score_cot(content: str, gold_triplets: list[dict]) -> float:
    """Score the reasoning chain (0-1)"""
    if not content:
        return 0.0

    score = 0.0
    # coverage (weight 0.5)
    _, coverage = check_coverage(content, gold_triplets)
    score += coverage * 0.5

    # length reasonableness (weight 0.2): 100-500 words is best
    word_count = len(content.split())
    if 100 <= word_count <= 500:
        score += 0.2
    elif 50 <= word_count < 100 or 500 < word_count <= 800:
        score += 0.1

    # structuredness (weight 0.15): contains numbering / steps
    if re.search(r'\b(step|1\.|first|second|third)\b', content.lower()):
        score += 0.15

    # contains Final Answer (weight 0.15)
    if "final answer" in content.lower():
        score += 0.15

    return score


def load_cot_data(model_name: str) -> dict:
    """Load CoT data from a teacher; returns {id: record}"""
    filepath = os.path.join(COT_DIR, model_name, "cot_data.jsonl")
    data = {}
    if not os.path.exists(filepath):
        print(f"  [WARN] {filepath} not found")
        return data
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                data[record["id"]] = record
            except (json.JSONDecodeError, KeyError):
                continue
    return data


def main():
    # load all teacher data
    teachers = ["glm5", "qwen"]
    all_data = {}
    for teacher in teachers:
        data = load_cot_data(teacher)
        if data:
            all_data[teacher] = data
            print(f"  Loaded {teacher}: {len(data)} samples")

    if not all_data:
        print("No CoT data found. Run generate_cot.py first.")
        return

    # collect all sample IDs
    all_ids = set()
    for teacher_data in all_data.values():
        all_ids.update(teacher_data.keys())
    print(f"\n  Total unique samples: {len(all_ids)}")

    # for each sample, pick the best CoT across all teachers
    results = []
    stats = {"accepted": 0, "rejected_no_data": 0, "rejected_low_quality": 0}

    for sample_id in sorted(all_ids):
        best_score = -1
        best_record = None
        best_teacher = None

        for teacher in teachers:
            if teacher not in all_data or sample_id not in all_data[teacher]:
                continue
            record = all_data[teacher][sample_id]
            content = record.get("content", "")
            gold = record.get("gold_triplets", [])

            if not check_coherence(content):
                continue

            s = score_cot(content, gold)
            if s > best_score:
                best_score = s
                best_record = record
                best_teacher = teacher

        if best_record is None:
            stats["rejected_no_data"] += 1
            continue

        if best_score < 0.3:
            stats["rejected_low_quality"] += 1
            continue

        # build the final training record
        result = {
            "id": sample_id,
            "sentence": best_record.get("sentence", ""),
            "gold_triplets": best_record.get("gold_triplets", []),
            "reasoning": best_record.get("content", ""),
            "reasoning_score": round(best_score, 3),
            "teacher_model": best_teacher,
            "source": best_record.get("source", ""),
        }
        results.append(result)
        stats["accepted"] += 1

    # save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== Filtering complete ===")
    print(f"  Accepted: {stats['accepted']}")
    print(f"  Rejected (no data): {stats['rejected_no_data']}")
    print(f"  Rejected (low quality): {stats['rejected_low_quality']}")
    print(f"  Output: {OUTPUT_PATH}")

    # per-teacher statistics
    teacher_counts = defaultdict(int)
    for r in results:
        teacher_counts[r["teacher_model"]] += 1
    print(f"\n  Teacher distribution:")
    for t, c in teacher_counts.items():
        print(f"    {t}: {c}")


if __name__ == "__main__":
    main()
