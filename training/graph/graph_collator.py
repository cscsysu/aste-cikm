"""
Graph data collator: handles text padding and PyG graph batching at the same time.

During training, each sample yielded by the DataLoader contains:
  - input_ids, attention_mask, labels (standard text fields)
  - graph_node_token_ids: [num_nodes, max_sw] (subword ids per word)
  - graph_edge_index: [2, num_edges] (COO edge indices)
  - graph_edge_rel_ids: [num_edges] (edge relation types)
  - graph_num_nodes: int

The collator is responsible for:
  1. Text fields: padding + assembling the batch
  2. Graph fields: padding to the max number of nodes in the batch + PyG Batch
"""

import torch
from torch_geometric.data import Data, Batch as PyGBatch
from transformers import DataCollatorForSeq2Seq


class GraphDataCollator:
    """
    Custom collator that handles both text and graph data.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: maximum text length
        label_pad_token_id: padding value for labels (default -100)
    """

    def __init__(self, tokenizer, max_length, label_pad_token_id=-100):
        self.text_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            max_length=max_length,
            label_pad_token_id=label_pad_token_id,
        )
        self.pad_token_id = tokenizer.pad_token_id or 0

    def __call__(self, features):
        """
        Args:
            features: list of dicts, each containing text fields and graph_* fields

        Returns:
            batch dict containing:
              - input_ids, attention_mask, labels (padded text)
              - graph_node_token_ids: [B, max_nodes, max_sw] (padded)
              - graph_node_mask: [B, max_nodes] (1=real, 0=pad)
              - graph_batch_data: PyG Batch object
        """
        # Separate text fields and graph fields
        text_features = []
        graph_data_list = []
        max_nodes = 0
        max_subwords = 0

        for f in features:
            # Text fields
            text_feat = {
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
            }
            if "labels" in f and f["labels"] is not None:
                text_feat["labels"] = f["labels"]
            text_features.append(text_feat)

            # Graph field statistics
            n_nodes = f["graph_num_nodes"]
            max_nodes = max(max_nodes, n_nodes)
            if f["graph_node_token_ids"].dim() == 2:
                max_subwords = max(max_subwords, f["graph_node_token_ids"].shape[1])
            else:
                max_subwords = max(max_subwords, 1)

            # Build the PyG Data object
            pyg_data = Data(
                edge_index=f["graph_edge_index"],
                edge_rel_ids=f["graph_edge_rel_ids"],
                num_nodes=n_nodes,
            )
            graph_data_list.append(pyg_data)

        # 1) Collate text fields
        batch = self.text_collator(text_features)

        # 2) Batch graph data (PyG handles edge_index offsets automatically)
        pyg_batch = PyGBatch.from_data_list(graph_data_list)
        batch["graph_batch_data"] = pyg_batch

        # 3) Pad graph_node_token_ids -> [B, max_nodes, max_subwords]
        batch_size = len(features)
        if max_subwords == 0:
            max_subwords = 1

        padded_node_tokens = torch.full(
            (batch_size, max_nodes, max_subwords),
            self.pad_token_id,
            dtype=torch.long,
        )
        node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.long)

        for i, f in enumerate(features):
            n = f["graph_num_nodes"]
            node_token_ids = f["graph_node_token_ids"]
            if node_token_ids.dim() == 2:
                sw = node_token_ids.shape[1]
                padded_node_tokens[i, :n, :sw] = node_token_ids[:n]
            elif node_token_ids.dim() == 1 and n > 0:
                padded_node_tokens[i, :n, 0] = node_token_ids[:n]
            node_mask[i, :n] = 1

        batch["graph_node_token_ids"] = padded_node_tokens
        batch["graph_node_mask"] = node_mask

        return batch
