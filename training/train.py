"""
ReasonGraph: main training script (Stage 1, with optional GATv2 graph encoder).

LoRA SFT for Qwen3-8B / Llama-3.1-8B / Qwen3-14B backbones.

Examples
--------
  # Two-GPU + GATv2 graph encoder (recommended)
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 training/train.py \
      --config training/configs/qwen3_8b.yaml --use_graph_encoder

  # No graph encoder (textual dependency-tree linearisation)
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 training/train.py \
      --config training/configs/qwen3_8b.yaml

  # Ablation: GATv2 + no CoT
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 training/train.py \
      --config training/configs/qwen3_8b.yaml --use_graph_encoder --no_cot

  # Ablation: no graph
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 training/train.py \
      --config training/configs/qwen3_8b.yaml --no_graph

  # Ablation: direct FT (no CoT, no graph)
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 training/train.py \
      --config training/configs/qwen3_8b.yaml --no_cot --no_graph
"""

import argparse
import json
import os
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

from data_loader import get_datasets
from evaluate import parse_triplets_from_text, compute_f1, gold_triplets_to_tuples


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class GraphTrainer(Trainer):
    """
    Custom Trainer:
    1. Split LR: graph_encoder uses a higher learning rate (trained from scratch),
       LoRA uses the standard lr.
    2. Ensure PyG Batch objects are correctly moved to GPU.
    """

    def __init__(self, *args, gat_lr_multiplier=5.0, **kwargs):
        self.gat_lr_multiplier = gat_lr_multiplier
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """Use separate learning rates for the graph encoder and LoRA."""
        model = self.model

        # Check whether the model is a GraphLLMWrapper
        if hasattr(model, 'graph_encoder'):
            graph_params = [p for p in model.graph_encoder.parameters() if p.requires_grad]
            lora_params = [p for n, p in model.llm.named_parameters() if p.requires_grad]

            self.optimizer = torch.optim.AdamW([
                {"params": graph_params, "lr": self.args.learning_rate * self.gat_lr_multiplier},
                {"params": lora_params, "lr": self.args.learning_rate},
            ], weight_decay=self.args.weight_decay)
        else:
            # Standard mode: fall back to the parent implementation
            return super().create_optimizer()

        return self.optimizer

    def _prepare_inputs(self, inputs):
        """Ensure PyG Batch objects are moved to the correct device."""
        prepared = super()._prepare_inputs(inputs)

        # PyG Batch objects are not auto-moved by Trainer; handle them manually
        if "graph_batch_data" in prepared and prepared["graph_batch_data"] is not None:
            prepared["graph_batch_data"] = prepared["graph_batch_data"].to(self.args.device)

        return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset (rest14/lap14/rest15/rest16). If None, run all.")
    parser.add_argument("--no_cot", action="store_true", help="Ablation: disable CoT")
    parser.add_argument("--no_graph", action="store_true", help="Ablation: disable graph")
    parser.add_argument("--use_graph_encoder", action="store_true",
                        help="Use GATv2 graph encoder instead of linearized syntax text")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from a trained checkpoint (for stage-2 training). "
                             "Path to checkpoint dir, e.g., outputs/reasongraph_qwen3_v2_gatv2_rest14")
    parser.add_argument("--stage2_epochs", type=int, default=3,
                        help="Number of epochs for stage-2 training (default: 3)")
    parser.add_argument("--stage2_lr", type=float, default=5e-5,
                        help="Learning rate for stage-2 training (default: 5e-5, lower than stage-1)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply ablation overrides
    use_cot = not args.no_cot
    use_graph = not args.no_graph
    use_graph_encoder = args.use_graph_encoder

    # If the graph encoder is enabled, graph must also be enabled
    if use_graph_encoder:
        use_graph = True

    # Experiment name
    exp_name = cfg["exp_name"]
    if use_graph_encoder:
        exp_name += "_gatv2"
    if args.resume_from:
        exp_name += "_stage2"
    if args.no_cot:
        exp_name += "_no_cot"
    if args.no_graph:
        exp_name += "_no_graph"

    datasets_to_run = [args.dataset] if args.dataset else cfg["datasets"]

    for dataset_name in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"Training: {exp_name} / {dataset_name}")
        print(f"  CoT: {use_cot}, Graph: {use_graph}, GraphEncoder: {use_graph_encoder}")
        print(f"{'='*60}")

        run_name = f"{exp_name}_{dataset_name}"
        output_dir = os.path.join(cfg["output_dir"], run_name)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_path"], trust_remote_code=True, padding_side="right")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load data
        train_ds, dev_ds, test_ds = get_datasets(
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
        print(f"  Train: {len(train_ds)}, Dev: {len(dev_ds)}, Test: {len(test_ds)}")

        # Load model
        # In torchrun DDP mode do NOT use device_map; in single-GPU mode use device_map="auto"
        is_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        if not is_ddp:
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], **model_kwargs)
        model.config.use_cache = False

        if args.resume_from:
            # === Two-stage training: load LoRA + GraphEncoder from a trained checkpoint ===
            resume_ckpt = args.resume_from
            # If resume_from does not contain the dataset name, try as-is
            if not os.path.exists(resume_ckpt):
                # Try appending the dataset name (compat with `--resume_from outputs/xxx`)
                print(f"  [WARN] {resume_ckpt} not found, trying as-is")

            print(f"  Stage-2: Loading LoRA from {resume_ckpt}...")
            from peft import PeftModel as PeftModelLoader
            model = PeftModelLoader.from_pretrained(model, resume_ckpt)
            # Unfreeze LoRA parameters so they can continue to train
            for name, param in model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
            model.print_trainable_parameters()
        else:
            # === Normal training: create LoRA from scratch ===
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg["lora_rank"],
                lora_alpha=cfg["lora_alpha"],
                lora_dropout=cfg["lora_dropout"],
                target_modules=cfg["lora_target_modules"],
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

        # Graph encoder (optional)
        if use_graph_encoder:
            from graph.gatv2_encoder import GATv2Encoder
            from graph.graph_llm_wrapper import GraphLLMWrapper
            from graph.graph_collator import GraphDataCollator

            graph_encoder = GATv2Encoder(
                hidden_dim=model.config.hidden_size,  # 3584 for Qwen3-8B
                num_heads=cfg.get("gat_num_heads", 4),
                num_layers=cfg.get("gat_num_layers", 2),
                dropout=cfg.get("gat_dropout", 0.1),
            )
            model = GraphLLMWrapper(
                llm_model=model,
                graph_encoder=graph_encoder,
                pad_token_id=tokenizer.pad_token_id or 0,
            )

            # If stage-2, load the trained graph_encoder weights
            if args.resume_from:
                graph_ckpt = os.path.join(args.resume_from, "graph_encoder.pt")
                if os.path.exists(graph_ckpt):
                    print(f"  Stage-2: Loading graph encoder from {graph_ckpt}...")
                    model.load_graph_encoder(args.resume_from)
                    # Move to correct device and dtype
                    device = next(model.llm.parameters()).device
                    model.graph_encoder = model.graph_encoder.to(device=device, dtype=torch.bfloat16)

            model.print_trainable_parameters()

        # warmup_ratio support
        warmup_ratio = cfg.get("warmup_ratio", 0.05)

        # Stage-2 overrides for training hyperparameters
        train_epochs = cfg["epochs"]
        train_lr = cfg["learning_rate"]
        if args.resume_from:
            train_epochs = args.stage2_epochs   # default 3
            train_lr = args.stage2_lr           # default 5e-5 (lower than stage-1's 2e-4)
            warmup_ratio = 0.1
            print(f"  Stage-2 config: epochs={train_epochs}, lr={train_lr}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=train_epochs,
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation"],
            learning_rate=train_lr,
            weight_decay=cfg["weight_decay"],
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=False,   # loading a checkpoint under DDP can OOM, disable
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            dataloader_num_workers=2,
            remove_unused_columns=False,
            ddp_find_unused_parameters=True if use_graph_encoder else False,
            gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        )

        # Data collator
        if use_graph_encoder:
            collator = GraphDataCollator(
                tokenizer=tokenizer,
                max_length=cfg["max_source_len"] + cfg["max_target_len"],
            )
        else:
            collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                padding=True,
                max_length=cfg["max_source_len"] + cfg["max_target_len"],
                label_pad_token_id=-100,
            )

        # Trainer
        gat_lr_mult = cfg.get("gat_lr_multiplier", 5.0)
        TrainerClass = GraphTrainer if use_graph_encoder else Trainer
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_ds,
            "eval_dataset": dev_ds,
            "data_collator": collator,
            "callbacks": [],  # No early stopping; run all 10 epochs and pick best checkpoint manually
        }
        if use_graph_encoder:
            trainer_kwargs["gat_lr_multiplier"] = gat_lr_mult

        trainer = TrainerClass(**trainer_kwargs)

        # Train
        trainer.train()

        # Save
        if use_graph_encoder:
            model.save_pretrained(output_dir)
        else:
            trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Evaluate
        print(f"\n  Evaluating on test set...")
        # Ensure the graph encoder stays bfloat16 (Trainer may upcast it to float32 during training)
        if use_graph_encoder and hasattr(model, 'graph_encoder'):
            model.graph_encoder = model.graph_encoder.to(dtype=torch.bfloat16)
        evaluate_model(model, tokenizer, test_ds, output_dir, cfg,
                       use_graph_encoder=use_graph_encoder)

        # Free GPU memory
        del model
        del trainer
        torch.cuda.empty_cache()


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
            "max_new_tokens": cfg["max_target_len"],
            "do_sample": False,
            "num_beams": 1,
            "temperature": 1.0,
        }

        # Graph encoder mode: pass graph data
        if use_graph_encoder and "graph_node_token_ids" in sample:
            from torch_geometric.data import Data, Batch as PyGBatch

            n_nodes = sample["graph_num_nodes"]
            graph_node_token_ids = sample["graph_node_token_ids"].unsqueeze(0).to(device)
            graph_node_mask = torch.zeros(1, n_nodes, dtype=torch.long, device=device)
            graph_node_mask[0, :n_nodes] = 1

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

        # Decode (only the generated portion)
        # In graph-encoder mode, the sequence returned by generate may include the prefix.
        # HF generate with inputs_embeds returns length = prefix + input + generated
        if use_graph_encoder and "graph_node_token_ids" in sample:
            # inputs_embeds mode: outputs contains only generated tokens (no prefix)
            # but exact behaviour depends on the HF version; safest is to subtract input length
            input_len = input_ids.shape[1]
            # generate with inputs_embeds typically does not return the input portion
            # so just decode the entire output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Try to take only the portion after [Answer]
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
            print(f"    {i+1}/{len(test_ds)} evaluated")

    # Compute metrics
    metrics = compute_f1(all_preds, all_golds)
    print(f"  Results: P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  F1={metrics['f1']:.2f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(output_dir, "test_predictions.jsonl"), "w") as f:
        for item in all_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Saved to {output_dir}")
    return metrics


if __name__ == "__main__":
    main()
