#!/bin/bash
# ==============================================================================
# Llama-3.1-8B end-to-end training (Stage-1 + Stage-2, all four ASTE benchmarks)
# Mirrors run_qwen3_14b.sh / run_qwen3_8b.sh but uses the Llama config.
# ==============================================================================

set -e
set -o pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CONFIG=training/configs/llama31_8b.yaml
LOGDIR=logs
mkdir -p $LOGDIR outputs

DATASETS=(rest14 lap14 rest15 rest16)

T0=$(date +%s)
echo "=================================================================="
echo "Llama-3.1-8B full pipeline starting: $(date)"
echo "=================================================================="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "[$DATASET] Stage-1: $(date)"
    python training/train.py \
        --config $CONFIG \
        --dataset $DATASET \
        --use_graph_encoder \
        2>&1 | tee $LOGDIR/llama31_8b_stage1_${DATASET}.log

    echo ""
    echo "[$DATASET] Stage-2: $(date)"
    python training/train.py \
        --config $CONFIG \
        --dataset $DATASET \
        --use_graph_encoder \
        --no_cot \
        --resume_from outputs/reasongraph_llama31_8b_gatv2_${DATASET} \
        --stage2_epochs 5 \
        --stage2_lr 1e-4 \
        2>&1 | tee $LOGDIR/llama31_8b_stage2_${DATASET}.log
done

T1=$(date +%s)
echo "=================================================================="
echo "All done: $(date)   elapsed: $(( (T1 - T0) / 60 )) min"
echo "=================================================================="

echo ""
echo "=== Result summary (Llama-3.1-8B) ==="
for DATASET in "${DATASETS[@]}"; do
    echo "--- $DATASET ---"
    grep "Results:" $LOGDIR/llama31_8b_stage2_${DATASET}.log 2>/dev/null | tail -1 || echo "  (not found)"
done
