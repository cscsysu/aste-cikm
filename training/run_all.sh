#!/bin/bash
# Run the full experiment suite end-to-end.
# Usage:  bash training/run_all.sh
# Prerequisites:
#   - data/distill_train.jsonl already produced (see scripts/generate_cot.py
#     and scripts/filter_cot.py)
#   - Pre-trained backbones downloaded to the model_path entries in
#     training/configs/*.yaml
#   - At least 2 GPUs (8B), or set CUDA_VISIBLE_DEVICES to a single device.

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo "ReasonGraph: full experiment suite"
echo "============================================"

# ============ 1. Qwen3-8B Full Model (CoT + Graph) ============
echo ""
echo ">>> [1/6] Qwen3-8B Full Model (CoT + Graph)"
CUDA_VISIBLE_DEVICES=0,1 python training/train.py \
    --config training/configs/qwen3_8b.yaml

# ============ 2. Qwen3-8B ablation - no CoT ============
echo ""
echo ">>> [2/6] Qwen3-8B w/o CoT"
CUDA_VISIBLE_DEVICES=0,1 python training/train.py \
    --config training/configs/qwen3_8b.yaml --no_cot

# ============ 3. Qwen3-8B ablation - no Graph ============
echo ""
echo ">>> [3/6] Qwen3-8B w/o Graph"
CUDA_VISIBLE_DEVICES=0,1 python training/train.py \
    --config training/configs/qwen3_8b.yaml --no_graph

# ============ 4. Qwen3-8B ablation - Direct FT (no CoT, no Graph) ============
echo ""
echo ">>> [4/6] Qwen3-8B Direct FT (no CoT, no Graph)"
CUDA_VISIBLE_DEVICES=0,1 python training/train.py \
    --config training/configs/qwen3_8b.yaml --no_cot --no_graph

# ============ 5. Llama3.1-8B Full Model ============
echo ""
echo ">>> [5/6] Llama3.1-8B Full Model (CoT + Graph)"
CUDA_VISIBLE_DEVICES=0 python training/train.py \
    --config training/configs/llama31_8b.yaml

# ============ 6. Llama3.1-8B Direct FT ============
echo ""
echo ">>> [6/6] Llama3.1-8B Direct FT (no CoT, no Graph)"
CUDA_VISIBLE_DEVICES=0 python training/train.py \
    --config training/configs/llama31_8b.yaml --no_cot --no_graph

echo ""
echo "============================================"
echo "Done. Results are written to outputs/."
echo "============================================"
