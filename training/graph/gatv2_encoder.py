"""
GATv2 graph encoder: encodes a dependency tree into dense node features.

Architecture:
  - 2-layer GATv2Conv (4 heads, hidden_dim=3584)
  - Typed edges: 49 dependency relations (48 + UNK) -> nn.Embedding
  - Residual connection + LayerNorm
  - Output projection layer

Input:  node features [N, 3584], edge_index [2, E], edge_rel_ids [E]
Output: node features [N, 3584]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


# 48 dependency relations + UNK (from actual data statistics)
DEP_RELATIONS = [
    'acl', 'acl:relcl', 'advcl', 'advcl:relcl', 'advmod', 'amod', 'appos',
    'aux', 'aux:pass', 'case', 'cc', 'cc:preconj', 'ccomp', 'compound',
    'compound:prt', 'conj', 'cop', 'csubj', 'csubj:outer', 'dep', 'det',
    'det:predet', 'discourse', 'dislocated', 'expl', 'fixed', 'flat',
    'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nmod:poss', 'nmod:unmarked',
    'nsubj', 'nsubj:outer', 'nsubj:pass', 'nummod', 'obj', 'obl', 'obl:agent',
    'obl:unmarked', 'orphan', 'parataxis', 'punct', 'reparandum', 'vocative',
    'xcomp', '<UNK>',
]
NUM_RELATIONS = len(DEP_RELATIONS)  # 49
REL2IDX = {r: i for i, r in enumerate(DEP_RELATIONS)}


class GATv2Encoder(nn.Module):
    """
    2-layer GATv2 with typed edges for dependency trees.

    Parameters:
        hidden_dim: node feature dimension (must match the LLM's hidden_size,
            default 3584)
        num_heads: number of attention heads (default 4, 896 dims per head)
        num_layers: number of GATv2 layers (default 2)
        dropout: attention and inter-layer dropout (default 0.1)
        num_relations: number of dependency relation types (default 49)
    """

    def __init__(
        self,
        hidden_dim: int = 3584,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_relations: int = NUM_RELATIONS,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Dependency-relation embedding
        self.rel_embedding = nn.Embedding(num_relations, hidden_dim)

        # GATv2 layers
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,  # 3584 // 4 = 896
                    heads=num_heads,
                    concat=True,             # 4 * 896 = 3584
                    dropout=dropout,
                    edge_dim=hidden_dim,     # edge_attr dimension
                    add_self_loops=True,
                    share_weights=False,
                )
            )

        # LayerNorm (one per layer)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_rel_ids):
        """
        Args:
            x: [total_nodes, hidden_dim] — initial node features
               (from the LLM embed_tokens)
            edge_index: [2, total_edges] — edge indices in COO format
               (the PyG batch already handles offsets)
            edge_rel_ids: [total_edges] — edge dependency-relation type indices

        Returns:
            [total_nodes, hidden_dim] — encoded node features
        """
        # Edge features
        edge_attr = self.rel_embedding(edge_rel_ids)  # [E, hidden_dim]

        for i, gat_layer in enumerate(self.gat_layers):
            residual = x
            x = gat_layer(x, edge_index, edge_attr=edge_attr)
            x = self.layer_norms[i](x + residual)  # Residual + LayerNorm
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = self.dropout(x)

        # Output projection
        x = self.output_proj(x)
        x = self.output_norm(x)
        return x
