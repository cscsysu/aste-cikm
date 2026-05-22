"""
Data loading module: assembles ASTE data + CoT reasoning chains + dependency
trees into training / evaluation format.

Two graph modes are supported:
  1. use_graph=True, use_graph_encoder=False: textualised dependency tree
     (legacy mode, [Syntax] prefix).
  2. use_graph=True, use_graph_encoder=True:  GATv2 graph encoder
     (new mode, prefix embedding).

Training format:
  Input:  [Syntax] dep_tree_text [Sentence] raw_sentence [Task] Extract triplets with reasoning.
  Output: [Reasoning] cot_text [Answer] triplets_text

Evaluation format:
  Input:  same as above
  Output: free-form model generation
"""

import json
import os
import torch
from torch.utils.data import Dataset

from graph.gatv2_encoder import REL2IDX


# Dependency relations most relevant to ABSA (kept first; used by the
# textualised mode).
ABSA_IMPORTANT_RELS = {
    'nsubj', 'amod', 'nmod', 'dobj', 'obj', 'advmod', 'conj',
    'cop', 'xcomp', 'ccomp', 'neg', 'compound', 'obl', 'mark',
    'nsubj:pass', 'acl', 'acl:relcl', 'advcl',
}


def linearize_dep_tree(edges, words):
    """Convert a list of dependency-tree edges into a textual description
    (with smart truncation)."""
    if not edges or not words:
        return ""
    word_texts = [w["text"] if isinstance(w, dict) else w for w in words]

    # Group edges by importance
    important = []
    other = []
    for e in edges:
        if e["relation"] in ABSA_IMPORTANT_RELS:
            important.append(e)
        else:
            other.append(e)

    # Keep important relations first; cap total at 25 edges
    selected = important + other[:max(0, 25 - len(important))]

    parts = []
    for e in selected:
        h, d, r = e["head"], e["dep"], e["relation"]
        if h < len(word_texts) and d < len(word_texts):
            parts.append(f"{word_texts[h]} --{r}--> {word_texts[d]}")
    return "; ".join(parts)


def format_triplets(triplets):
    """Format a list of triplets as text."""
    parts = []
    for t in triplets:
        a = t["aspect"]
        o = t["opinion"]
        s = t["sentiment"]
        parts.append(f'("{a}", "{o}", "{s}")')
    return "[" + ", ".join(parts) + "]"


def parse_triplets_from_text(text):
    """Parse triplets from model output text (lenient version)."""
    import re
    results = []

    # Strategy 1: standard format ("aspect", "opinion", "POS/NEG/NEU")
    pattern1 = r'\(\s*"([^"]*?)"\s*,\s*"([^"]*?)"\s*,\s*"?(POS|NEG|NEU)"?\s*\)'
    matches = re.findall(pattern1, text)
    if matches:
        for a, o, s in matches:
            results.append({"aspect": a.strip(), "opinion": o.strip(), "sentiment": s.strip()})
        return results

    # Strategy 2: loose format
    pattern2 = r'\(\s*"?([^",]+?)"?\s*,\s*"?([^",]+?)"?\s*,\s*"?(POS|NEG|NEU|[Pp]ositive|[Nn]egative|[Nn]eutral)"?\s*\)'
    matches = re.findall(pattern2, text, re.IGNORECASE)
    if matches:
        s_map = {"positive": "POS", "negative": "NEG", "neutral": "NEU"}
        for a, o, s in matches:
            s_norm = s_map.get(s.lower(), s.upper())
            results.append({"aspect": a.strip(), "opinion": o.strip(), "sentiment": s_norm})
        return results

    # Strategy 3: try matching after the [Answer] marker
    answer_match = re.search(r'\[Answer\]\s*(.*)', text, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1)
        matches = re.findall(pattern1, answer_text)
        if not matches:
            matches = re.findall(pattern2, answer_text, re.IGNORECASE)
        s_map = {"positive": "POS", "negative": "NEG", "neutral": "NEU"}
        for a, o, s in matches:
            s_norm = s_map.get(s.lower(), s.upper())
            results.append({"aspect": a.strip(), "opinion": o.strip(), "sentiment": s_norm})

    return results


class ABSADataset(Dataset):
    """ABSA training / evaluation dataset."""

    def __init__(self, tokenizer, data_dir, dataset_name, split, parsed_dir=None,
                 distill_path=None, max_source_len=512, max_target_len=512,
                 use_cot=True, use_graph=True, use_graph_encoder=False, mode="train"):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            data_dir: path to the data/ directory
            dataset_name: rest14 / lap14 / rest15 / rest16
            split: train / dev / test
            parsed_dir: path to data/parsed/
            distill_path: path to data/distill_train.jsonl (only used for train)
            max_source_len: maximum number of input tokens
            max_target_len: maximum number of output tokens
            use_cot: whether to use the reasoning chain
            use_graph: whether to use dependency-tree information
            use_graph_encoder: whether to use the GATv2 graph encoder
                (True = prefix embedding, False = textualised)
            mode: train / eval
        """
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.use_cot = use_cot
        self.use_graph = use_graph
        self.use_graph_encoder = use_graph_encoder
        self.mode = mode
        self.samples = []

        # Load ASTE data
        aste_path = os.path.join(data_dir, "aste", f"{dataset_name}_{split}.jsonl")
        aste_data = {}
        with open(aste_path, "r") as f:
            for line in f:
                d = json.loads(line)
                aste_data[d["id"]] = d

        # Load dependency trees (needed in both modes)
        parsed_data = {}
        if use_graph and parsed_dir:
            parsed_path = os.path.join(parsed_dir, f"{dataset_name}_{split}_parsed.jsonl")
            if os.path.exists(parsed_path):
                with open(parsed_path, "r") as f:
                    for line in f:
                        d = json.loads(line)
                        parsed_data[d["id"]] = d

        # Load CoT distillation data (train only)
        distill_data = {}
        if use_cot and mode == "train" and distill_path and os.path.exists(distill_path):
            with open(distill_path, "r") as f:
                for line in f:
                    d = json.loads(line)
                    distill_data[d["id"]] = d

        # Build samples
        for sid, sample in aste_data.items():
            source = self._build_input(sample, parsed_data.get(sid))
            target = self._build_output(sample, distill_data.get(sid))
            entry = {
                "id": sid,
                "source": source,
                "target": target,
                "sentence": sample["sentence"],
                "gold_triplets": sample["triplets"],
            }
            # In graph-encoder mode, keep the raw parsed data
            if use_graph_encoder:
                entry["parsed"] = parsed_data.get(sid)
            self.samples.append(entry)

    def _build_input(self, sample, parsed):
        """Build the model input."""
        parts = []

        # Textualised dependency tree (only when not using the graph encoder)
        if self.use_graph and not self.use_graph_encoder and parsed:
            dep_text = linearize_dep_tree(parsed.get("edges", []), parsed.get("words", []))
            if dep_text:
                parts.append(f"[Syntax] {dep_text}")

        # Raw sentence
        parts.append(f"[Sentence] {sample['sentence']}")

        # Task instruction
        parts.append("[Task] Extract all (aspect, opinion, sentiment) triplets from the sentence. "
                      "First explain your reasoning step by step, then provide the final answer.")

        return "\n".join(parts)

    def _build_output(self, sample, distill):
        """Build the target output for the model."""
        parts = []

        # Reasoning chain (CoT, used at training time)
        if self.use_cot and distill and distill.get("reasoning"):
            parts.append(f"[Reasoning] {distill['reasoning']}")

        # Triplet answer
        triplet_text = format_triplets(sample["triplets"])
        parts.append(f"[Answer] {triplet_text}")

        return "\n".join(parts)

    def _build_graph_tensors(self, parsed):
        """Convert parsed dependency data into the tensors required by GATv2.

        Returns:
            dict with:
              graph_node_token_ids: [num_nodes, max_sw] — subword token ids per word
              graph_edge_index: [2, num_edges*2] — bidirectional edges (COO format)
              graph_edge_rel_ids: [num_edges*2] — edge relation-type indices
              graph_num_nodes: int
        """
        if parsed is None or not parsed.get("words"):
            # Return a dummy single-node graph
            unk_id = self.tokenizer.unk_token_id or 0
            return {
                "graph_node_token_ids": torch.tensor([[unk_id]], dtype=torch.long),
                "graph_edge_index": torch.zeros(2, 0, dtype=torch.long),
                "graph_edge_rel_ids": torch.zeros(0, dtype=torch.long),
                "graph_num_nodes": 1,
            }

        words = parsed["words"]
        edges = parsed.get("edges", [])
        num_nodes = len(words)

        # 1) Word -> subword token ids
        node_token_ids_list = []
        max_sw = 0
        for w in words:
            text = w["text"] if isinstance(w, dict) else w
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 0:
                ids = [self.tokenizer.unk_token_id or 0]
            node_token_ids_list.append(ids)
            max_sw = max(max_sw, len(ids))

        # Pad to [num_nodes, max_sw]
        pad_id = self.tokenizer.pad_token_id or 0
        padded = torch.full((num_nodes, max(max_sw, 1)), pad_id, dtype=torch.long)
        for i, ids in enumerate(node_token_ids_list):
            padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

        # 2) Build bidirectional edges
        if edges:
            src = []
            dst = []
            rels = []
            for e in edges:
                h, d = e["head"], e["dep"]
                # Make sure the indices are in range
                if h < num_nodes and d < num_nodes:
                    rel_idx = REL2IDX.get(e["relation"], REL2IDX["<UNK>"])
                    # Forward edge
                    src.append(h)
                    dst.append(d)
                    rels.append(rel_idx)
                    # Reverse edge
                    src.append(d)
                    dst.append(h)
                    rels.append(rel_idx)

            if src:
                edge_index = torch.tensor([src, dst], dtype=torch.long)
                edge_rel_ids = torch.tensor(rels, dtype=torch.long)
            else:
                edge_index = torch.zeros(2, 0, dtype=torch.long)
                edge_rel_ids = torch.zeros(0, dtype=torch.long)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_rel_ids = torch.zeros(0, dtype=torch.long)

        return {
            "graph_node_token_ids": padded,         # [num_nodes, max_sw]
            "graph_edge_index": edge_index,         # [2, num_edges*2]
            "graph_edge_rel_ids": edge_rel_ids,     # [num_edges*2]
            "graph_num_nodes": num_nodes,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        source = sample["source"]
        target = sample["target"]

        # Build chat format
        if self.mode == "train":
            # At training time, concatenate input + output as the full sequence
            messages = [
                {"role": "user", "content": source},
                {"role": "assistant", "content": target},
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            encoding = self.tokenizer(text, truncation=True,
                                       max_length=self.max_source_len + self.max_target_len,
                                       padding=False, return_tensors=None)

            # Compute labels: set the user portion to -100 (no loss)
            user_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": source}], tokenize=False, add_generation_prompt=True)
            user_len = len(self.tokenizer(user_text, truncation=True,
                                          max_length=self.max_source_len + self.max_target_len,
                                          padding=False)["input_ids"])

            labels = encoding["input_ids"].copy()
            labels[:user_len] = [-100] * user_len

            result = {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": labels,
            }
        else:
            # At evaluation time, encode only the input
            messages = [{"role": "user", "content": source}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoding = self.tokenizer(text, truncation=True,
                                       max_length=self.max_source_len,
                                       padding=False, return_tensors=None)
            result = {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "id": sample["id"],
                "gold_triplets": sample["gold_triplets"],
                "sentence": sample["sentence"],
            }

        # Graph-encoder mode: attach graph tensor fields
        if self.use_graph_encoder:
            graph_tensors = self._build_graph_tensors(sample.get("parsed"))
            result.update(graph_tensors)

        return result


def get_datasets(tokenizer, data_dir, dataset_name, parsed_dir, distill_path,
                 max_source_len=512, max_target_len=512, use_cot=True, use_graph=True,
                 use_graph_encoder=False):
    """Get the train / dev / test datasets."""
    train_ds = ABSADataset(tokenizer, data_dir, dataset_name, "train",
                           parsed_dir=parsed_dir, distill_path=distill_path,
                           max_source_len=max_source_len, max_target_len=max_target_len,
                           use_cot=use_cot, use_graph=use_graph,
                           use_graph_encoder=use_graph_encoder, mode="train")
    dev_ds = ABSADataset(tokenizer, data_dir, dataset_name, "dev",
                         parsed_dir=parsed_dir, distill_path=None,
                         max_source_len=max_source_len, max_target_len=max_target_len,
                         use_cot=False, use_graph=use_graph,
                         use_graph_encoder=use_graph_encoder, mode="train")
    test_ds = ABSADataset(tokenizer, data_dir, dataset_name, "test",
                          parsed_dir=parsed_dir, distill_path=None,
                          max_source_len=max_source_len, max_target_len=max_target_len,
                          use_cot=False, use_graph=use_graph,
                          use_graph_encoder=use_graph_encoder, mode="eval")
    return train_ds, dev_ds, test_ds
