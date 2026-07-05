#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-vllm-spec}"
PORT="${PORT:-8550}"
TP="${TP:-1}"
GPUS="${GPUS:-0}"
GPU_MEM="${GPU_MEM:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3-8B}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/models/Qwen3-8B_eagle3_angelslim}"
SPEC_TOKENS="${SPEC_TOKENS:-4}"
DRAFT_TP="${DRAFT_TP:-1}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"

SPEC_JSON="{\"method\":\"draft_model\",\"model\":\"${DRAFT_MODEL_PATH}\",\"num_speculative_tokens\":${SPEC_TOKENS},\"draft_tensor_parallel_size\":${DRAFT_TP}}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
sleep 1

docker run -d --name "$NAME" \
  --gpus "\"device=${GPUS}\"" \
  --network host --ipc host --shm-size 32g \
  -v "${MODEL_MOUNT:-/home/liuguangli/models}:/models" \
  "$VLLM_IMAGE" \
    --model "$MODEL_PATH" \
    --served-model-name qwen \
    --tensor-parallel-size "$TP" \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --generation-config vllm \
    --speculative-config "$SPEC_JSON" \
    --port "$PORT" >/dev/null

echo "spec server starting: ${NAME}, port=${PORT}, gpus=${GPUS}, tp=${TP}"
echo "speculative_config=${SPEC_JSON}"
start=$(date +%s)
until curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; do
  if [ $(( $(date +%s) - start )) -gt 600 ]; then
    docker logs "$NAME" 2>&1 | tail -80
    exit 1
  fi
  sleep 5
done
echo "ready: http://localhost:${PORT}/v1"
