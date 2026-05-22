"""
Dependency-tree parsing script: use Stanza to parse all data.
Outputs adjacency matrices and edge types for downstream graph modules.

Usage: python scripts/parse_dependency.py
"""

import json
import os

try:
    import stanza
except ImportError:
    print("Please install stanza first: pip install stanza")
    print("Then download the English model: python -c \"import stanza; stanza.download('en')\"")
    exit(1)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
PARSED_DIR = os.path.join(DATA_DIR, "parsed")


def parse_sentences(nlp, input_path: str, output_path: str):
    """Parse all sentences in a JSONL file"""
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    results = []
    for i, sample in enumerate(samples):
        sentence = sample["sentence"]
        doc = nlp(sentence)

        # extract dependency-tree info
        for sent in doc.sentences:
            words = []
            edges = []  # (head_idx, dep_idx, relation)
            for word in sent.words:
                words.append({
                    "id": word.id,
                    "text": word.text,
                    "lemma": word.lemma,
                    "upos": word.upos,
                    "xpos": word.xpos,
                })
                if word.head > 0:  # 0 = root
                    edges.append({
                        "head": word.head - 1,  # convert to 0-indexed
                        "dep": word.id - 1,     # convert to 0-indexed
                        "relation": word.deprel,
                    })

            # build adjacency list (bidirectional)
            n = len(words)
            adj = {j: [] for j in range(n)}
            for e in edges:
                adj[e["head"]].append({"node": e["dep"], "rel": e["relation"]})
                adj[e["dep"]].append({"node": e["head"], "rel": e["relation"] + "_inv"})

            result = {
                "id": sample["id"],
                "sentence": sentence,
                "words": words,
                "edges": edges,
                "adj": {str(k): v for k, v in adj.items()},  # JSON keys must be str
                "num_words": n,
            }
            results.append(result)

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(samples)} parsed")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Done: {len(results)} sentences -> {output_path}")


def main():
    os.makedirs(PARSED_DIR, exist_ok=True)

    print("Loading Stanza English model...")
    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse",
                          tokenize_pretokenized=False, use_gpu=False)

    # ASTE data
    aste_dir = os.path.join(DATA_DIR, "aste")
    for dataset in ["rest14", "lap14", "rest15", "rest16"]:
        for split in ["train", "dev", "test"]:
            input_path = os.path.join(aste_dir, f"{dataset}_{split}.jsonl")
            output_path = os.path.join(PARSED_DIR, f"{dataset}_{split}_parsed.jsonl")
            if not os.path.exists(input_path):
                continue
            print(f"\n  Parsing {dataset}_{split}...")
            parse_sentences(nlp, input_path, output_path)

    # MAMS data
    mams_dir = os.path.join(DATA_DIR, "mams")
    for split in ["train", "dev", "test"]:
        input_path = os.path.join(mams_dir, f"mams_{split}.jsonl")
        output_path = os.path.join(PARSED_DIR, f"mams_{split}_parsed.jsonl")
        if not os.path.exists(input_path):
            continue
        print(f"\n  Parsing mams_{split}...")
        parse_sentences(nlp, input_path, output_path)

    # ACOS data
    acos_dir = os.path.join(DATA_DIR, "acos")
    for dataset in ["acos_rest", "acos_laptop"]:
        for split in ["train", "dev", "test"]:
            input_path = os.path.join(acos_dir, f"{dataset}_{split}.jsonl")
            output_path = os.path.join(PARSED_DIR, f"{dataset}_{split}_parsed.jsonl")
            if not os.path.exists(input_path):
                continue
            print(f"\n  Parsing {dataset}_{split}...")
            parse_sentences(nlp, input_path, output_path)

    print("\n=== All parsing complete ===")


if __name__ == "__main__":
    main()
