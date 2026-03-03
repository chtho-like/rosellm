#!/bin/bash

set -euo pipefail

# Examples:
# - Serve GPT-2 (default):
#     ./openai_server.sh
#
# - Serve Qwen3-0.6B from Hugging Face:
#     ROSEINFER_HF_MODEL_ID="Qwen/Qwen3-0.6B" ./openai_server.sh
#
# - If the first run takes a long time (Triton/FlashInfer JIT), increase the
#   multiprocess engine init timeout:
#     ROSEINFER_HF_MODEL_ID="Qwen/Qwen3-0.6B" \
#     ROSEINFER_MP_START_TIMEOUT_S="900" \
#       ./openai_server.sh
#
# - vLLM-style KV cache sizing (optional):
#     ROSEINFER_HF_MODEL_ID="Qwen/Qwen3-0.6B" \
#     ROSEINFER_GPU_MEMORY_UTILIZATION="0.9" \
#       ./openai_server.sh
#
# - Or directly cap KV cache by a fraction of *free* GPU memory:
#     ROSEINFER_HF_MODEL_ID="Qwen/Qwen3-0.6B" \
#     ROSEINFER_KV_CACHE_MEM_FRACTION="0.2" \
#       ./openai_server.sh
#
# - Chat with the OpenAI-compatible endpoint:
#     ROSEINFER_MODEL="Qwen/Qwen3-0.6B" ./openai_client_chat.sh

cd "$(dirname "$0")/.."

HF_MODEL_ID="${ROSEINFER_HF_MODEL_ID:-gpt2}"
HOST="${ROSEINFER_HOST:-127.0.0.1}"
PORT="${ROSEINFER_PORT:-8888}"
KV_CACHE_MAX_CONCURRENCY="${ROSEINFER_KV_CACHE_MAX_CONCURRENCY:-16}"

EXTRA_ARGS=()
if [[ -n "${ROSEINFER_KV_CACHE_MAX_TOKENS:-}" ]]; then
  EXTRA_ARGS+=(--kv-cache-max-tokens "${ROSEINFER_KV_CACHE_MAX_TOKENS}")
fi
if [[ -n "${ROSEINFER_KV_CACHE_MEM_FRACTION:-}" ]]; then
  EXTRA_ARGS+=(--kv-cache-mem-fraction "${ROSEINFER_KV_CACHE_MEM_FRACTION}")
fi
if [[ -n "${ROSEINFER_GPU_MEMORY_UTILIZATION:-}" ]]; then
  EXTRA_ARGS+=(--gpu-memory-utilization "${ROSEINFER_GPU_MEMORY_UTILIZATION}")
fi
if [[ -n "${ROSEINFER_MP_START_TIMEOUT_S:-}" ]]; then
  EXTRA_ARGS+=(--mp-start-timeout-s "${ROSEINFER_MP_START_TIMEOUT_S}")
fi

python -m rosellm.roseinfer.server \
  --hf-model-id "$HF_MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --kv-cache-max-concurrency "$KV_CACHE_MAX_CONCURRENCY" \
  "${EXTRA_ARGS[@]}"
