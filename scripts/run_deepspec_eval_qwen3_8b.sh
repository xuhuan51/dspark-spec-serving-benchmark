#!/usr/bin/env bash
set -euo pipefail

DEEPSPEC_ROOT="${DEEPSPEC_ROOT:-$HOME/learn-ai-infra/external/DeepSpec}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
OUT_DIR="${OUT_DIR:-$PWD/outputs/deepspec-runs}"

mkdir -p "$OUT_DIR"

cd "$DEEPSPEC_ROOT"
source "${DEEPSPEC_VENV:-$HOME/venvs/deepspec/bin/activate}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29502}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# A30 24GB needs this guard for ultra-long prompt outliers.
export DEEPSPEC_MAX_PROMPT_CHARS="${DEEPSPEC_MAX_PROMPT_CHARS:-15000}"

for draft in dspark_qwen3_8b_block7 eagle3_qwen3_8b_ttt7 dflash_qwen3_8b_block7; do
  echo "===== RUN START: ${draft} ====="
  python eval.py \
    --target_name_or_path "${MODEL_ROOT}/Qwen3-8B" \
    --draft_name_or_path "${MODEL_ROOT}/${draft}" \
    > "${OUT_DIR}/eval_${draft}.log" 2>&1
  echo "===== RUN END: ${draft} ====="
done
