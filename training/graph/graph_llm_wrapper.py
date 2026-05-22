"""
GraphLLMWrapper: wraps a PeftModel (LoRA Qwen3) together with a GATv2Encoder.

Core behaviour:
  1. Extract initial node features from the LLM's embed_tokens.
  2. Encode the dependency graph through GATv2.
  3. Concatenate the graph node features as a prefix embedding in front of the
     text embedding.
  4. Let the LLM transformer process the concatenated sequence.

Training:
  forward(input_ids, labels, graph_*) -> loss
  Gradient flow: loss -> LoRA layers -> graph_prefix -> GATv2 params

Inference:
  generate(input_ids, graph_*) -> generated_ids
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch as PyGBatch
from transformers.modeling_outputs import CausalLMOutputWithPast


class GraphLLMWrapper(nn.Module):
    """
    Wrap a PeftModel + GATv2Encoder and inject the graph prefix embedding.

    Args:
        llm_model: PeftModel (LoRA-adapted Qwen3)
        graph_encoder: GATv2Encoder
        pad_token_id: tokenizer pad token id
    """

    def __init__(self, llm_model, graph_encoder, pad_token_id=0):
        super().__init__()
        self.llm = llm_model
        self.graph_encoder = graph_encoder
        self.pad_token_id = pad_token_id

    # === Attribute proxies (so HF Trainer and generate work correctly) ===

    @property
    def config(self):
        return self.llm.config

    @property
    def generation_config(self):
        return self.llm.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self.llm.generation_config = value

    @property
    def device(self):
        return next(self.llm.parameters()).device

    @property
    def dtype(self):
        return torch.bfloat16  # Force a uniform bfloat16

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def print_trainable_parameters(self):
        """Print trainable parameter statistics."""
        # LLM (LoRA)
        self.llm.print_trainable_parameters()
        # Graph encoder
        graph_params = sum(p.numel() for p in self.graph_encoder.parameters())
        graph_trainable = sum(p.numel() for p in self.graph_encoder.parameters() if p.requires_grad)
        print(f"Graph encoder: trainable params: {graph_trainable:,} || all params: {graph_params:,}")

    # === Core methods ===

    def _build_graph_prefix(self, input_ids, graph_node_token_ids, graph_node_mask, graph_batch_data):
        """
        Build the graph prefix embedding.

        Returns:
            graph_prefix: [batch, max_nodes, hidden_dim]
            graph_node_mask: [batch, max_nodes] (returned unchanged)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        embed_layer = self.get_input_embeddings()

        # Step 1: get initial node features from the LLM's embed_tokens
        # graph_node_token_ids: [B, max_nodes, max_sw]
        max_nodes = graph_node_token_ids.shape[1]
        max_sw = graph_node_token_ids.shape[2]

        with torch.no_grad():
            flat_ids = graph_node_token_ids.view(-1, max_sw)  # [B*max_nodes, max_sw]
            flat_embeds = embed_layer(flat_ids)                # [B*max_nodes, max_sw, H]
            # Mean pool subwords (mask out pad tokens)
            sw_mask = (flat_ids != self.pad_token_id).unsqueeze(-1).float()  # [B*max_nodes, max_sw, 1]
            pooled = (flat_embeds * sw_mask).sum(dim=1) / sw_mask.sum(dim=1).clamp(min=1)
            node_embeds_padded = pooled.view(batch_size, max_nodes, -1)  # [B, max_nodes, H]

        # Detach: prevent gradients from flowing back into embed_tokens (frozen)
        node_embeds_padded = node_embeds_padded.detach().to(dtype=self.dtype)

        # Step 2: extract real nodes -> feed into GATv2
        # graph_batch_data.batch tells which sample each node belongs to in the batch.
        # We need to extract real nodes from the padded tensor and stack them
        # into a flat tensor.
        all_node_features = []
        for i in range(batch_size):
            n_i = graph_node_mask[i].sum().item()
            all_node_features.append(node_embeds_padded[i, :n_i])  # [n_i, H]

        flat_node_features = torch.cat(all_node_features, dim=0)  # [total_nodes, H]
        flat_node_features.requires_grad_(True)

        # Step 3: GATv2 encode
        graph_batch_data = graph_batch_data.to(device)
        graph_out = self.graph_encoder(
            x=flat_node_features,
            edge_index=graph_batch_data.edge_index,
            edge_rel_ids=graph_batch_data.edge_rel_ids,
        )  # [total_nodes, H]

        # Step 4: scatter back into padded shape [B, max_nodes, H]
        hidden_dim = graph_out.shape[-1]
        graph_prefix = torch.zeros(
            batch_size, max_nodes, hidden_dim,
            device=device, dtype=graph_out.dtype,
        )
        offset = 0
        for i in range(batch_size):
            n_i = graph_node_mask[i].sum().item()
            graph_prefix[i, :n_i] = graph_out[offset:offset + n_i]
            offset += n_i

        return graph_prefix

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        # Graph fields (from GraphDataCollator)
        graph_node_token_ids=None,
        graph_node_mask=None,
        graph_batch_data=None,
        # Fallbacks
        graph_edge_index=None,
        graph_edge_rel_ids=None,
        graph_num_nodes=None,
        **kwargs,
    ):
        """
        Forward pass with graph prefix injection.

        When there is no graph data (e.g. subsequent token steps during generate),
        fall back to the standard LLM forward.
        """
        # No graph data -> standard forward
        if graph_node_token_ids is None or graph_batch_data is None:
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        device = input_ids.device
        max_nodes = graph_node_mask.shape[1]

        # Steps 1-4: build the graph prefix
        graph_prefix = self._build_graph_prefix(
            input_ids, graph_node_token_ids, graph_node_mask, graph_batch_data
        )  # [B, max_nodes, H]

        # Step 5: get text embeddings
        embed_layer = self.get_input_embeddings()
        with torch.no_grad():
            text_embeds = embed_layer(input_ids)  # [B, seq_len, H]
        text_embeds = text_embeds.detach().to(dtype=self.dtype)

        # Step 6: concatenate [graph_prefix | text_embeds]
        combined_embeds = torch.cat([graph_prefix, text_embeds], dim=1)
        combined_embeds = combined_embeds.to(dtype=torch.bfloat16)  # Ensure dtype consistency
        # [B, max_nodes + seq_len, H]

        # Step 7: build the new attention_mask
        combined_attention_mask = torch.cat([graph_node_mask, attention_mask], dim=1)

        # Step 8: build position_ids (consecutive)
        total_len = combined_embeds.shape[1]
        position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Step 9: build the new labels (graph prefix portion set to -100)
        if labels is not None:
            prefix_labels = torch.full(
                (batch_size, max_nodes), -100, dtype=labels.dtype, device=device
            )
            combined_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            combined_labels = None

        # Step 10: run through the LLM (inputs_embeds mode skips the embedding layer)
        return self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            position_ids=position_ids,
            labels=combined_labels,
            **kwargs,
        )

    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        graph_node_token_ids=None,
        graph_node_mask=None,
        graph_batch_data=None,
        **kwargs,
    ):
        """
        Generation with the graph prefix.

        Strategy: convert input_ids to inputs_embeds, prepend the graph prefix,
        then call the LLM's generate(inputs_embeds=...).
        """
        # No graph data -> standard generate
        if graph_node_token_ids is None or graph_batch_data is None:
            return self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        device = input_ids.device
        max_nodes = graph_node_mask.shape[1]

        # Build the graph prefix
        graph_prefix = self._build_graph_prefix(
            input_ids, graph_node_token_ids, graph_node_mask, graph_batch_data
        )

        # Text embeddings
        embed_layer = self.get_input_embeddings()
        with torch.no_grad():
            text_embeds = embed_layer(input_ids)
        text_embeds = text_embeds.to(dtype=self.dtype)

        # Concatenate
        combined_embeds = torch.cat([graph_prefix, text_embeds], dim=1)
        combined_embeds = combined_embeds.to(dtype=torch.bfloat16)  # Ensure dtype consistency
        combined_attention_mask = torch.cat([graph_node_mask, attention_mask], dim=1)

        # Generate (using inputs_embeds instead of input_ids)
        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            **kwargs,
        )

        return outputs

    # === Save / Load ===

    def save_pretrained(self, path, **kwargs):
        """Save the LoRA adapter and the graph encoder."""
        import os
        self.llm.save_pretrained(path, **kwargs)
        torch.save(
            self.graph_encoder.state_dict(),
            os.path.join(path, "graph_encoder.pt"),
        )

    def load_graph_encoder(self, path):
        """Load the graph encoder weights."""
        import os
        state_dict = torch.load(
            os.path.join(path, "graph_encoder.pt"),
            map_location="cpu",
        )
        self.graph_encoder.load_state_dict(state_dict)
