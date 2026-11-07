#!/bin/bash
# Evaluate a trained model: generate responses + score with beaver cost/reward models
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_eval.sh outputs/catdpo_eta0.5_eps0.05/final
#
# Arguments:
#   $1 — path to trained model checkpoint (required)
#   $2 — dataset subset (default: alpaca2-7b)

set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="${1:?Usage: bash scripts/run_eval.sh <model_path> [subset]}"
SUBSET="${2:-alpaca2-7b}"

echo "============================================"
echo "Evaluation"
echo "  Model:  ${MODEL_PATH}"
echo "  Subset: ${SUBSET}"
echo "============================================"

cd "${PROJECT_ROOT}/src"

python -u evaluate.py \
    --model_path "${MODEL_PATH}" \
    --subset "${SUBSET}" \
    --max_new_tokens 1024 \
    --gen_batch_size 64 \
    --score_batch_size 64
