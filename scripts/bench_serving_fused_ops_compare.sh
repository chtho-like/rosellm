#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-gpt2}"
GPU="${GPU:-0}"
OUTDIR="${OUTDIR:-outputs/benchmarks/serving}"
OFFLINE_MAX_BATCH_SIZE="${OFFLINE_MAX_BATCH_SIZE:-32}"

# roseinfer defaults (align with "industry default" optimizations)
ROSEINFER_PREFILL_ATTN_BACKEND="${ROSEINFER_PREFILL_ATTN_BACKEND:-auto}"
ROSEINFER_DECODE_ATTN_BACKEND="${ROSEINFER_DECODE_ATTN_BACKEND:-auto}"
ROSEINFER_PAGED_ATTN="${ROSEINFER_PAGED_ATTN:-1}"
ROSEINFER_CUDA_GRAPH="${ROSEINFER_CUDA_GRAPH:-1}"
ROSEINFER_CHUNKED_PREFILL="${ROSEINFER_CHUNKED_PREFILL:-1}"
ROSEINFER_PREFILL_CHUNK_SIZE="${ROSEINFER_PREFILL_CHUNK_SIZE:-256}"
ROSEINFER_PREFIX_CACHE="${ROSEINFER_PREFIX_CACHE:-1}"

# New optimization under test: fused ops (default on). Set ROSEINFER_COMPARE_FUSED_OPS=1 to run A/B.
ROSEINFER_FUSED_OPS="${ROSEINFER_FUSED_OPS:-1}"
ROSEINFER_COMPARE_FUSED_OPS="${ROSEINFER_COMPARE_FUSED_OPS:-1}"

# New optimizations under test (default on). Set *_COMPARE_*=1 to run A/B.
ROSEINFER_FUSED_MLP="${ROSEINFER_FUSED_MLP:-1}"
ROSEINFER_COMPARE_FUSED_MLP="${ROSEINFER_COMPARE_FUSED_MLP:-0}"
ROSEINFER_FUSED_SAMPLER="${ROSEINFER_FUSED_SAMPLER:-1}"
ROSEINFER_COMPARE_FUSED_SAMPLER="${ROSEINFER_COMPARE_FUSED_SAMPLER:-0}"
ROSEINFER_FUSED_KV_APPEND="${ROSEINFER_FUSED_KV_APPEND:-1}"
ROSEINFER_COMPARE_FUSED_KV_APPEND="${ROSEINFER_COMPARE_FUSED_KV_APPEND:-0}"

ROSEINFER_ARGS=(
  --roseinfer-prefill-attn-backend "${ROSEINFER_PREFILL_ATTN_BACKEND}"
  --roseinfer-decode-attn-backend "${ROSEINFER_DECODE_ATTN_BACKEND}"
  --roseinfer-prefill-chunk-size "${ROSEINFER_PREFILL_CHUNK_SIZE}"
)

if [[ "${ROSEINFER_PAGED_ATTN}" == "1" ]]; then
  ROSEINFER_ARGS+=(--roseinfer-paged-attn)
else
  ROSEINFER_ARGS+=(--roseinfer-no-paged-attn)
fi
if [[ "${ROSEINFER_CUDA_GRAPH}" == "1" ]]; then
  ROSEINFER_ARGS+=(--roseinfer-cuda-graph)
else
  ROSEINFER_ARGS+=(--roseinfer-no-cuda-graph)
fi
if [[ "${ROSEINFER_CHUNKED_PREFILL}" == "1" ]]; then
  ROSEINFER_ARGS+=(--roseinfer-chunked-prefill)
else
  ROSEINFER_ARGS+=(--roseinfer-no-chunked-prefill)
fi
if [[ "${ROSEINFER_PREFIX_CACHE}" == "1" ]]; then
  ROSEINFER_ARGS+=(--roseinfer-prefix-cache)
else
  ROSEINFER_ARGS+=(--roseinfer-no-prefix-cache)
fi
if [[ "${ROSEINFER_FUSED_OPS}" == "1" ]]; then
  ROSEINFER_ARGS+=(--roseinfer-fused-ops)
else
  ROSEINFER_ARGS+=(--roseinfer-no-fused-ops)
fi
if [[ "${ROSEINFER_COMPARE_FUSED_OPS}" == "1" ]]; then
  ROSEINFER_ARGS+=(--roseinfer-compare-fused-ops)
fi

if [[ "${ROSEINFER_FUSED_MLP}" == "1" ]]; then
  ROSEINFER_ARGS+=(--roseinfer-fused-mlp)
else
  ROSEINFER_ARGS+=(--roseinfer-no-fused-mlp)
fi
if [[ "${ROSEINFER_COMPARE_FUSED_MLP}" == "1" ]]; then
  ROSEINFER_ARGS+=(--roseinfer-compare-fused-mlp)
fi

if [[ "${ROSEINFER_FUSED_SAMPLER}" == "1" ]]; then
  ROSEINFER_ARGS+=(--roseinfer-fused-sampler)
else
  ROSEINFER_ARGS+=(--roseinfer-no-fused-sampler)
fi
if [[ "${ROSEINFER_COMPARE_FUSED_SAMPLER}" == "1" ]]; then
  ROSEINFER_ARGS+=(--roseinfer-compare-fused-sampler)
fi

if [[ "${ROSEINFER_FUSED_KV_APPEND}" == "1" ]]; then
  ROSEINFER_ARGS+=(--roseinfer-fused-kv-append)
else
  ROSEINFER_ARGS+=(--roseinfer-no-fused-kv-append)
fi
if [[ "${ROSEINFER_COMPARE_FUSED_KV_APPEND}" == "1" ]]; then
  ROSEINFER_ARGS+=(--roseinfer-compare-fused-kv-append)
fi

python benchmarks/serving/online_compare.py \
  --model "${MODEL}" \
  --gpu "${GPU}" \
  --output-dir "${OUTDIR}" \
  "${ROSEINFER_ARGS[@]}"

ONLINE_JSON="$(ls -t "${OUTDIR}"/online_*/online_results.json | head -n 1)"

python benchmarks/serving/offline_compare.py \
  --model "${MODEL}" \
  --gpu "${GPU}" \
  --output-dir "${OUTDIR}" \
  --max-batch-size "${OFFLINE_MAX_BATCH_SIZE}" \
  "${ROSEINFER_ARGS[@]}"

OFFLINE_JSON="$(ls -t "${OUTDIR}"/offline_*/offline_results.json | head -n 1)"

FIG_DIR="${OUTDIR}/figures/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${FIG_DIR}"

python benchmarks/serving/plot_compare.py \
  --online "${ONLINE_JSON}" \
  --offline "${OFFLINE_JSON}" \
  --output-dir "${FIG_DIR}"

echo "Online results: ${ONLINE_JSON}"
echo "Offline results: ${OFFLINE_JSON}"
echo "Figures: ${FIG_DIR}"
