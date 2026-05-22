# ReasonGraph: Graph-Enhanced Reasoning Distillation for Aspect Sentiment Triplet Extraction

This repository accompanies our submission and contains the full training
pipeline, evaluation utilities, and figure-generation scripts for
**ReasonGraph**.  It is sufficient to reproduce every number reported in
the paper from scratch on a single 48 GB GPU (8B backbone) or two 24 GB
GPUs (Qwen3-14B with `gradient_checkpointing=true`).

> **Anonymity notice.**  This is the anonymous reviewer copy.  Personal
> identifiers, server paths, and API keys have been stripped.

---

## TL;DR

```bash
# 1. Install
pip install -r training/requirements.txt
python -c "import stanza; stanza.download('en')"

# 2. Prepare data (see data/README.md)
#    -> writes data/aste/{rest14,lap14,rest15,rest16}_{train,dev,test}.jsonl

# 3. Pre-compute Stanza dependency parses
python scripts/parse_dependency.py --aste_dir data/aste --out_dir data/parsed

# 4. Generate gold-guided CoT rationales (set LLM_API_KEY first)
export LLM_API_KEY=sk-...
export LLM_API_BASE=https://api.openai.com/v1/chat/completions   # or any OpenAI-compatible gateway
python scripts/generate_cot.py --model glm5
python scripts/filter_cot.py

# 5. Train + evaluate (Qwen3-8B, all four datasets, two stages)
bash run_qwen3_8b.sh
# or:  bash run_llama31_8b.sh   /   bash run_qwen3_14b.sh
```

---

## Repository layout

```
training/
├── train.py                Stage-1 / Stage-2 LoRA SFT (with optional GATv2)
├── eval_only.py            Standalone evaluation from a checkpoint dir
├── evaluate.py             ASTE triplet extraction + relaxed F1
├── data_loader.py          ASTE + CoT + dependency-tree assembly
├── graph/
│   ├── gatv2_encoder.py    GATv2 stack producing the continuous prefix
│   ├── graph_collator.py   Joint text-padding + PyG Batch collator
│   └── graph_llm_wrapper.py  Wires GATv2 prefix into the LLM input embeds
├── configs/
│   ├── qwen3_8b.yaml       LoRA + GATv2 hyper-params for Qwen3-8B
│   ├── qwen3_14b.yaml      ... Qwen3-14B (gradient_checkpointing on)
│   └── llama31_8b.yaml     ... Llama-3.1-8B-Instruct
└── run_all.sh              Six-experiment ablation suite (8B + Llama)

scripts/
├── parse_dependency.py     Stanza dependency parse for every sentence
├── generate_cot.py         Gold-guided CoT rationale generation (teacher = LLM API)
├── filter_cot.py           Drop rationales whose final answer != gold
├── eval_llm_baseline.py    GLM-5 / DeepSeek / etc. zero/few-shot baselines
├── compute_all_prf1.py             P/R/F1, strict span match
└── compute_all_prf1_relaxed.py     P/R/F1, relaxed span match (paper metric)

prompts/
└── teacher_prompt.md       Exact teacher system + user prompt template

api_client.py               OpenAI-compatible LLM API client (used for CoT
                            generation and zero/few-shot LLM baselines)

run_qwen3_8b.sh             Full Stage-1 + Stage-2 pipeline (Qwen3-8B)
run_qwen3_14b.sh            Full Stage-1 + Stage-2 pipeline (Qwen3-14B)
run_llama31_8b.sh           Full Stage-1 + Stage-2 pipeline (Llama-3.1-8B)
```

---

## Reproducing the main table

The pipeline is deterministic up to GPU non-determinism in attention.
Expected end-to-end deviation from the reported F1 is ≤ 1 point per
dataset (most of the variance is driven by which teacher LLM you use to
generate rationales — see "Notes" below).

| Backbone | Rest14 | Lap14 | Rest15 | Rest16 |
|----------|--------|-------|--------|--------|
| Qwen3-8B    | 80.22 | 70.15 | 77.19 | 80.86 |
| Llama-3.1-8B| 81.53 | 68.85 | 77.12 | 79.51 |
| Qwen3-14B   | 81.91 | 71.39 | 79.40 | 82.84 |

Steps to reproduce one row:

1. **Backbone.** Download once with `huggingface-cli`:
   ```bash
   huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B
   ```
   Update `model_path` in `training/configs/qwen3_8b.yaml` to point at the
   local directory (or just use the HF id directly if your network can
   reach it during training).

2. **Distillation set.** Run `scripts/generate_cot.py` once across the
   four `*_train.jsonl` files; this produces `data/distill_train.jsonl`.

3. **Train both stages.** Run `bash run_qwen3_8b.sh`. Stage-1 takes ~3 h
   on 2× A40, Stage-2 another ~30 min. Final test metrics are written to
   `outputs/reasongraph_qwen3_8b_gatv2_stage2_<dataset>/test_metrics.json`.

4. **(Optional) consolidate.** `python scripts/compute_all_prf1_relaxed.py`
   re-evaluates every prediction file in the repo with the relaxed-match
   metric used in the paper.

---

## Notes on reproducibility

- We use **relaxed-matching** (substring-tolerant aspect/opinion + exact
  sentiment) for all reported numbers; see
  `scripts/compute_all_prf1_relaxed.py`.
- F1 differences of ±0.5 across reruns of the same configuration are
  normal and stem from CUDA non-determinism in attention.
- F1 differences of ±1-2 are expected if you switch the teacher LLM
  (we use GLM-5; DeepSeek-v4-flash gives ~0.6 lower on Rest14 in our
  pilot runs).
- The graph encoder is a 2-layer GATv2 with hidden dim 256 projected up
  to the LLM hidden size; `training/configs/*.yaml` documents every
  hyperparameter.

---

## Citation

(Will be added on acceptance.)

## License

See `LICENSE`.
