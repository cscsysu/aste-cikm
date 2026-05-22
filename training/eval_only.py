"""
ReasonGraph-ABSA standalone evaluation script.
Load a model from a trained checkpoint and evaluate on the test set.
Supports both vanilla LoRA models and GraphLLMWrapper models.

Usage:
  # Evaluate v2 (no graph encoder)
  CUDA_VISIBLE_DEVICES=0 python training/eval_only.py \
      --config training/configs/qwen3_8b.yaml \
      --dataset rest14 \
      --checkpoint outputs/reasongraph_qwen3_v2_rest14

  # Evaluate v3 (with graph encoder)
  CUDA_VISIBLE_DEVICES=0 python training/eval_only.py \
      --config training/configs/qwen3_8b.yaml \
      --dataset rest14 \
      --checkpoint outputs/reasongraph_qwen3_v2_gatv2_rest14 \
      --use_graph_encoder
"""

import argparse
import json
import os
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from data_loader import get_datasets
from evaluate import parse_triplets_from_text, compute_f1, gold_triplets_to_tuples


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model(model, tokenizer, test_ds, output_dir, cfg, use_graph_encoder=False):
    """Evaluate on the test set."""
    model.eval()
    all_preds = []
    all_golds = []
    all_outputs = []

    device = next(model.parameters()).device

    for i in range(len(test_ds)):
        sample = test_ds[i]
        input_ids = torch.tensor([sample["input_ids"]]).to(device)
        attention_mask = torch.tensor([sample["attention_mask"]]).to(device)

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 1024,  # increased to avoid truncation
            "do_sample": False,
            "num_beams": 1,
            "temperature": 1.0,
        }

        # Graph-encoder mode: pass graph data
        if use_graph_encoder and "graph_node_token_ids" in sample:
            from torch_geometric.data import Data, Batch as PyGBatch

            n_nodes = sample["graph_num_nodes"]
            graph_node_token_ids = sample["graph_node_token_ids"].unsqueeze(0).to(device)
            graph_node_mask = torch.ones(1, n_nodes, dtype=torch.long, device=device)

            pyg_data = Data(
                edge_index=sample["graph_edge_index"],
                edge_rel_ids=sample["graph_edge_rel_ids"],
                num_nodes=n_nodes,
            )
            graph_batch = PyGBatch.from_data_list([pyg_data]).to(device)

            generate_kwargs.update({
                "graph_node_token_ids": graph_node_token_ids,
                "graph_node_mask": graph_node_mask,
                "graph_batch_data": graph_batch,
            })

        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)

        # Decode
        if use_graph_encoder and "graph_node_token_ids" in sample:
            # inputs_embeds mode: decode the full output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            generated_ids = outputs[0][input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse triplets
        pred_triplets = parse_triplets_from_text(generated_text)
        gold_triplets = gold_triplets_to_tuples(sample["gold_triplets"])

        all_preds.append(pred_triplets)
        all_golds.append(gold_triplets)
        all_outputs.append({
            "id": sample["id"],
            "generated_text": generated_text,
            "pred_triplets": [{"aspect": a, "opinion": o, "sentiment": s} for a, o, s in pred_triplets],
            "gold_triplets": sample["gold_triplets"],
        })

        if (i + 1) % 50 == 0:
            p_so_far = compute_f1(all_preds, all_golds)
            print(f"    {i+1}/{len(test_ds)} evaluated  (running F1={p_so_far['f1']:.2f})")

    # Compute final metrics
    metrics = compute_f1(all_preds, all_golds)
    print(f"\n  Results: P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  F1={metrics['f1']:.2f}")
    print(f"  (TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']})")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(output_dir, "test_predictions.jsonl"), "w") as f:
        for item in all_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Saved to {output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained ReasonGraph-ABSA model")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset to evaluate (rest14/lap14/rest15/rest16). If None, eval all.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a specific checkpoint dir")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Parent dir containing checkpoint subdirs (auto-detect per dataset)")
    parser.add_argument("--no_cot", action="store_true")
    parser.add_argument("--no_graph", action="store_true")
    parser.add_argument("--use_graph_encoder", action="store_true",
                        help="Load GraphLLMWrapper with GATv2 encoder")
    args = parser.parse_args()

    cfg = load_config(args.config)

    use_cot = not args.no_cot
    use_graph = not args.no_graph
    use_graph_encoder = args.use_graph_encoder

    if use_graph_encoder:
        use_graph = True

    # Experiment name (used to locate the checkpoint automatically)
    exp_name = cfg["exp_name"]
    if use_graph_encoder:
        exp_name += "_gatv2"
    if args.no_cot:
        exp_name += "_no_cot"
    if args.no_graph:
        exp_name += "_no_graph"

    datasets_to_eval = [args.dataset] if args.dataset else cfg["datasets"]

    for dataset_name in datasets_to_eval:
        # Determine the checkpoint path
        if args.checkpoint:
            ckpt_path = args.checkpoint
        elif args.checkpoint_dir:
            ckpt_path = os.path.join(args.checkpoint_dir, f"{exp_name}_{dataset_name}")
        else:
            ckpt_path = os.path.join(cfg["output_dir"], f"{exp_name}_{dataset_name}")

        if not os.path.exists(ckpt_path):
            print(f"\n  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {exp_name} / {dataset_name}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  CoT: {use_cot}, Graph: {use_graph}, GraphEncoder: {use_graph_encoder}")
        print(f"{'='*60}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_path"], trust_remote_code=True, padding_side="right")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load data (test set, eval mode)
        _, _, test_ds = get_datasets(
            tokenizer=tokenizer,
            data_dir=cfg["data_dir"],
            dataset_name=dataset_name,
            parsed_dir=cfg["parsed_dir"],
            distill_path=cfg["distill_path"],
            max_source_len=cfg["max_source_len"],
            max_target_len=cfg["max_target_len"],
            use_cot=use_cot,
            use_graph=use_graph,
            use_graph_encoder=use_graph_encoder,
        )
        print(f"  Test: {len(test_ds)} samples")

        # Load base model + LoRA
        print(f"  Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["model_path"],
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        print(f"  Loading LoRA adapter from {ckpt_path}...")
        model = PeftModel.from_pretrained(base_model, ckpt_path)

        # Graph-encoder mode: wrap with GraphLLMWrapper + load graph_encoder weights
        if use_graph_encoder:
            from graph.gatv2_encoder import GATv2Encoder
            from graph.graph_llm_wrapper import GraphLLMWrapper

            graph_encoder = GATv2Encoder(
                hidden_dim=base_model.config.hidden_size,
                num_heads=cfg.get("gat_num_heads", 4),
                num_layers=cfg.get("gat_num_layers", 2),
                dropout=cfg.get("gat_dropout", 0.1),
            )
            wrapped_model = GraphLLMWrapper(
                llm_model=model,
                graph_encoder=graph_encoder,
                pad_token_id=tokenizer.pad_token_id or 0,
            )

            # Load the trained graph_encoder weights
            graph_ckpt = os.path.join(ckpt_path, "graph_encoder.pt")
            if os.path.exists(graph_ckpt):
                print(f"  Loading graph encoder from {graph_ckpt}...")
                wrapped_model.load_graph_encoder(ckpt_path)
            else:
                print(f"  WARNING: graph_encoder.pt not found at {ckpt_path}")

            # Move to GPU + bfloat16
            wrapped_model.graph_encoder = wrapped_model.graph_encoder.to(
                device=next(model.parameters()).device,
                dtype=torch.bfloat16,
            )
            model = wrapped_model

        model.eval()

        # Evaluate
        print(f"  Running evaluation on {dataset_name} test set...")
        metrics = evaluate_model(model, tokenizer, test_ds, ckpt_path, cfg,
                                 use_graph_encoder=use_graph_encoder)

        # Free GPU memory
        del model
        del base_model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("All evaluations done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
