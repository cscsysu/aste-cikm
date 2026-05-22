"""
Evaluation module: compute exact-match F1 for ASTE triplets.
v3: enhanced regex parsing + strips <think> tags + de-duplication.
"""

import re


def parse_triplets_from_text(text):
    """Parse triplets from model output text (enhanced, multi-strategy fallback)."""
    if not text:
        return []

    results = []

    # Strip <think>...</think> tags (residue from Qwen3 thinking mode)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # If an [Answer] marker is present, prefer parsing from after [Answer]
    answer_text = text
    answer_match = re.search(r'\[Answer\]\s*(.*)', text, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1)

    # Strategy 1: standard format ("aspect", "opinion", "POS/NEG/NEU")
    pattern1 = r'\(\s*"([^"]*?)"\s*,\s*"([^"]*?)"\s*,\s*"?(POS|NEG|NEU)"?\s*\)'
    matches = re.findall(pattern1, answer_text)
    if matches:
        for a, o, s in matches:
            results.append((a.strip().lower(), o.strip().lower(), s.strip().upper()))
        # Deduplicate
        return list(set(results))

    # Strategy 2: loose quoting — single quotes or no quotes
    pattern2 = r"""\(\s*['"]?([^'",]+?)['"]?\s*,\s*['"]?([^'",]+?)['"]?\s*,\s*['"]?(POS|NEG|NEU|[Pp]ositive|[Nn]egative|[Nn]eutral)['"]?\s*\)"""
    matches = re.findall(pattern2, answer_text, re.IGNORECASE)
    if matches:
        s_map = {"positive": "POS", "negative": "NEG", "neutral": "NEU"}
        for a, o, s in matches:
            s_norm = s_map.get(s.lower(), s.upper())
            results.append((a.strip().lower(), o.strip().lower(), s_norm))
        return list(set(results))

    # Strategy 3: if we previously sliced from [Answer], try matching the full text again
    if answer_match:
        matches = re.findall(pattern1, text)
        if not matches:
            matches = re.findall(pattern2, text, re.IGNORECASE)
        s_map = {"positive": "POS", "negative": "NEG", "neutral": "NEU"}
        for a, o, s in matches:
            s_norm = s_map.get(s.lower(), s.upper())
            results.append((a.strip().lower(), o.strip().lower(), s_norm))

    return list(set(results))


def compute_f1(predictions, golds):
    """
    Compute exact-match F1.
    predictions: list of list of (aspect, opinion, sentiment) tuples
    golds: list of list of (aspect, opinion, sentiment) tuples
    """
    tp, fp, fn = 0, 0, 0
    for pred, gold in zip(predictions, golds):
        pred_set = set(pred)
        gold_set = set(gold)
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


def gold_triplets_to_tuples(gold_triplets):
    """Convert a list of gold_triplets dicts into a list of tuples."""
    return [(t["aspect"].strip().lower(), t["opinion"].strip().lower(),
             t["sentiment"].strip().upper()) for t in gold_triplets]
