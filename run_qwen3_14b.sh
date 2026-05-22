#!/bin/bash
# ==============================================================================
# Qwen3-14B end-to-end training (Stage-1 + Stage-2, all four ASTE benchmarks)
# Usage:
#   cd <repo_root>
#   bash run_qwen3_14b.sh
#
# Or run in the background:
#   nohup bash run_qwen3_14b.sh > logs/qwen3_14b_all.log 2>&1 &
#   tail -f logs/qwen3_14b_all.log
#
# Pipeline (per dataset):
#   Stage-1: LoRA + GATv2, joint CoT + Answer training, 10 epochs, lr=2e-4
#   Stage-2: resume from Stage-1, Answer-only, 5 epochs, lr=1e-4
#
# After completion, each dataset will have two output dirs under outputs/:
#   reasongraph_qwen3_14b_gatv2_{dataset}           <- Stage-1 (CoT + Ans)
#   reasongraph_qwen3_14b_gatv2_stage2_{dataset}    <- Stage-2 (final ASTE)
# ==============================================================================

set -e
set -o pipefail

# Run from the repo root (this script's parent directory).
cd "$(dirname "$0")"

# -------- environment --------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CONFIG=training/configs/qwen3_14b.yaml
LOGDIR=logs
mkdir -p $LOGDIR outputs

DATASETS=(rest14 lap14 rest15 rest16)

T0=$(date +%s)
echo "=================================================================="
echo "Qwen3-14B full pipeline starting: $(date)"
echo "Config:   $CONFIG"
echo "Datasets: ${DATASETS[*]}"
echo "GPU:      CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=================================================================="

# -------- per-dataset: Stage-1 -> Stage-2 --------
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "=================================================================="
    echo "[$DATASET] Stage-1 starting: $(date)"
    echo "=================================================================="

    python training/train.py \
        --config $CONFIG \
        --dataset $DATASET \
        --use_graph_encoder \
        2>&1 | tee $LOGDIR/qwen3_14b_stage1_${DATASET}.log

    echo "[$DATASET] Stage-1 done: $(date)"

    echo ""
    echo "=================================================================="
    echo "[$DATASET] Stage-2 starting: $(date)"
    echo "=================================================================="

    python training/train.py \
        --config $CONFIG \
        --dataset $DATASET \
        --use_graph_encoder \
        --no_cot \
        --resume_from outputs/reasongraph_qwen3_14b_gatv2_${DATASET} \
        --stage2_epochs 5 \
        --stage2_lr 1e-4 \
        2>&1 | tee $LOGDIR/qwen3_14b_stage2_${DATASET}.log

    echo "[$DATASET] Stage-2 done: $(date)"
done

T1=$(date +%s)
ELAPSED=$(( (T1 - T0) / 60 ))

echo ""
echo "=================================================================="
echo "All done: $(date)   elapsed: ${ELAPSED} min"
echo "=================================================================="

# -------- result summary --------
echo ""
echo "=== Result summary (Qwen3-14B) ==="
for DATASET in "${DATASETS[@]}"; do
    echo "--- $DATASET ---"
    echo "  Stage-1 (CoT + Ans):"
    grep "Results:" $LOGDIR/qwen3_14b_stage1_${DATASET}.log 2>/dev/null | tail -1 || echo "    (not found)"
    echo "  Stage-2 (Ans-only, final):"
    grep "Results:" $LOGDIR/qwen3_14b_stage2_${DATASET}.log 2>/dev/null | tail -1 || echo "    (not found)"
done

echo ""
echo "Final test_metrics.json locations:"
for DATASET in "${DATASETS[@]}"; do
    S2_DIR="outputs/reasongraph_qwen3_14b_gatv2_stage2_${DATASET}"
    if [ -f "$S2_DIR/test_metrics.json" ]; then
        echo "  $DATASET: $S2_DIR/test_metrics.json"
    fi
done
