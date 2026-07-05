#!/usr/bin/env bash
set -euo pipefail

TAG="${TAG:-spec_decode}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8550/v1}"
CONCURRENCIES="${CONCURRENCIES:-1,2,4,8,16,24,32}"
MAX_TOKENS="${MAX_TOKENS:-256}"
OUT_DIR="${OUT_DIR:-outputs/${TAG}}"

python3 benchmark/spec_decode_microbench.py \
  --tag "$TAG" \
  --base-url "$BASE_URL" \
  --out-dir "$OUT_DIR" \
  --concurrencies "$CONCURRENCIES" \
  --max-tokens "$MAX_TOKENS" \
  --ignore-eos
