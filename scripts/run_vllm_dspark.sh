#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8551}"
GPUS="${GPUS:-${GPU:-0}}"
TP="${TP:-1}"
DRAFT_TP="${DRAFT_TP:-${TP}}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEM="${GPU_MEM:-0.98}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MODEL_PATH="${MODEL_PATH:-/home/liuguangli/models/Qwen3-8B}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/home/liuguangli/models/dspark_qwen3_8b_block7}"
SPEC_TOKENS="${SPEC_TOKENS:-7}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen}"
VLLM_VENV="${VLLM_VENV:-/home/liuguangli/venvs/vllm-dspark-026}"

export PATH="${VLLM_VENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

exec "${VLLM_VENV}/bin/vllm" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --generation-config vllm \
  --speculative-config \
    "{\"model\":\"${DRAFT_MODEL_PATH}\",\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS},\"draft_tensor_parallel_size\":${DRAFT_TP}}"
