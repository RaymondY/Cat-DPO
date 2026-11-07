#!/bin/bash
# Train Cat-DPO on PKU-SafeRLHF
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_train.sh
#
# Key hyperparameters (modify below or override via environment):
#   ETA_DUAL     — dual ascent step size (default: 0.5); 0 disables the margin and
#                  reproduces the matched no-margin DPO baseline
#   EPSILON      — per-category safety tolerance (default: 0.02)
#   MARGIN_AGG   — max (paper) | sum (strict Lagrangian)
#   LAMBDA_CAP   — clip every lambda_k to this ceiling (0 = uncapped)
#   CATEGORY_MAP — JSON from make_cluster_categories.py, to run with label-free categories
#   SEED         — 1 | 2 | 42 are the three seeds reported in the paper

set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Configuration ---
MODEL="${MODEL:-PKU-Alignment/alpaca-7b-reproduced-llama-2}"
SUBSET="${SUBSET:-alpaca2-7b}"
DATA_MODE="${DATA_MODE:-bettersafe}"
ETA_DUAL="${ETA_DUAL:-0.5}"
EPSILON="${EPSILON:-0.02}"
MARGIN_AGG="${MARGIN_AGG:-max}"
LAMBDA_CAP="${LAMBDA_CAP:-0}"
CATEGORY_MAP="${CATEGORY_MAP:-}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-catdpo}"

RUN_NAME="${RUN_NAME:-catdpo_eta${ETA_DUAL}_eps${EPSILON}_seed${SEED}}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${RUN_NAME}"
REF_CACHE_DIR="${PROJECT_ROOT}/outputs/ref_logps"

echo "============================================"
echo "Cat-DPO Training"
echo "  Model:     ${MODEL}"
echo "  Subset:    ${SUBSET}"
echo "  Data mode: ${DATA_MODE}"
echo "  eta_dual:  ${ETA_DUAL}$([ "${ETA_DUAL}" = "0" ] && echo '  (no margin: matched DPO baseline)')"
echo "  epsilon:   ${EPSILON}"
echo "  margin:    agg=${MARGIN_AGG} cap=${LAMBDA_CAP}"
[ -n "${CATEGORY_MAP}" ] && echo "  categories: ${CATEGORY_MAP} (label-free)"
echo "  LR:        ${LR}"
echo "  Batch:     ${BATCH_SIZE} x ${GRAD_ACCUM}"
echo "  Output:    ${OUTPUT_DIR}"
echo "============================================"

cd "${PROJECT_ROOT}/src"

python -u train_catdpo.py \
    --model_name "${MODEL}" \
    --subset "${SUBSET}" \
    --data_mode "${DATA_MODE}" \
    --output_dir "${OUTPUT_DIR}" \
    --ref_cache_dir "${REF_CACHE_DIR}" \
    --beta 0.1 \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --max_length "${MAX_LENGTH}" \
    --warmup_ratio 0.03 \
    --weight_decay 0.05 \
    --eta_dual "${ETA_DUAL}" \
    --epsilon "${EPSILON}" \
    --margin_agg "${MARGIN_AGG}" \
    --lambda_cap "${LAMBDA_CAP}" \
    ${CATEGORY_MAP:+--category_map "${CATEGORY_MAP}"} \
    --wandb_project "${WANDB_PROJECT}" \
    --run_name "${RUN_NAME}" \
    --seed "${SEED}"
