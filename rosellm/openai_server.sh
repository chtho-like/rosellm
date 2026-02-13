#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

HF_MODEL_ID="${ROSEINFER_HF_MODEL_ID:-gpt2}"
HOST="${ROSEINFER_HOST:-127.0.0.1}"
PORT="${ROSEINFER_PORT:-8888}"
KV_CACHE_MAX_CONCURRENCY="${ROSEINFER_KV_CACHE_MAX_CONCURRENCY:-16}"

python -m roseinfer.server \
  --hf-model-id "$HF_MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --kv-cache-max-concurrency "$KV_CACHE_MAX_CONCURRENCY"
