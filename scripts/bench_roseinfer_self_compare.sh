#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-gpt2}"
GPU="${GPU:-0}"
OUTDIR="${OUTDIR:-outputs/benchmarks/serving}"

# Comma-separated list. Missing optional deps are auto-skipped by the python scripts.
ROSEINFER_PREFILL_ATTN_BACKENDS="${ROSEINFER_PREFILL_ATTN_BACKENDS:-auto,naive,flashinfer,flashattn}"
ROSEINFER_CHUNKED_PREFILL="${ROSEINFER_CHUNKED_PREFILL:-1}"
ROSEINFER_PREFILL_CHUNK_SIZE="${ROSEINFER_PREFILL_CHUNK_SIZE:-256}"

ROSEINFER_OFFLINE_NUM_PROMPTS="${ROSEINFER_OFFLINE_NUM_PROMPTS:-128}"
ROSEINFER_OFFLINE_INPUT_LEN="${ROSEINFER_OFFLINE_INPUT_LEN:-256}"
ROSEINFER_OFFLINE_OUTPUT_LEN="${ROSEINFER_OFFLINE_OUTPUT_LEN:-64}"
ROSEINFER_OFFLINE_MAX_BATCH_SIZE="${ROSEINFER_OFFLINE_MAX_BATCH_SIZE:-128}"
ROSEINFER_OFFLINE_WARMUP_PROMPTS="${ROSEINFER_OFFLINE_WARMUP_PROMPTS:-8}"

CHUNK_ARGS=()
if [[ "${ROSEINFER_CHUNKED_PREFILL}" == "1" ]]; then
  CHUNK_ARGS=(--roseinfer-chunked-prefill --roseinfer-prefill-chunk-size "${ROSEINFER_PREFILL_CHUNK_SIZE}")
else
  CHUNK_ARGS=(--roseinfer-no-chunked-prefill)
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "./.conda/bin/python" ]]; then
    PYTHON_BIN="./.conda/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

OFFLINE_EXTRA_ARGS=()
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[i]}" in
    --ignore-eos)
      OFFLINE_EXTRA_ARGS+=("--ignore-eos")
      ;;
    --dtype)
      if ((i + 1 < ${#ARGS[@]})); then
        OFFLINE_EXTRA_ARGS+=("--dtype" "${ARGS[i + 1]}")
        ((i++))
      fi
      ;;
  esac
done

"${PYTHON_BIN}" benchmarks/serving/online_compare.py \
  --model "${MODEL}" \
  --gpu "${GPU}" \
  --backends "roseinfer" \
  --roseinfer-prefill-attn-backends "${ROSEINFER_PREFILL_ATTN_BACKENDS}" \
  --roseinfer-paged-attn \
  --roseinfer-cuda-graph \
  "${CHUNK_ARGS[@]}" \
  --output-dir "${OUTDIR}" \
  "$@"

ONLINE_JSON="$(ls -t "${OUTDIR}"/online_*/online_results.json | head -n 1)"

"${PYTHON_BIN}" benchmarks/serving/offline_compare.py \
  --model "${MODEL}" \
  --gpu "${GPU}" \
  --backends "roseinfer" \
  --num-prompts "${ROSEINFER_OFFLINE_NUM_PROMPTS}" \
  --input-len "${ROSEINFER_OFFLINE_INPUT_LEN}" \
  --output-len "${ROSEINFER_OFFLINE_OUTPUT_LEN}" \
  --max-batch-size "${ROSEINFER_OFFLINE_MAX_BATCH_SIZE}" \
  --warmup-prompts "${ROSEINFER_OFFLINE_WARMUP_PROMPTS}" \
  --roseinfer-prefill-attn-backends "${ROSEINFER_PREFILL_ATTN_BACKENDS}" \
  --roseinfer-paged-attn \
  --roseinfer-cuda-graph \
  "${CHUNK_ARGS[@]}" \
  --output-dir "${OUTDIR}" \
  "${OFFLINE_EXTRA_ARGS[@]}"

OFFLINE_JSON="$(ls -t "${OUTDIR}"/offline_*/offline_results.json | head -n 1)"

FIG_DIR="${OUTDIR}/figures/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${FIG_DIR}"

"${PYTHON_BIN}" benchmarks/serving/plot_compare.py \
  --online "${ONLINE_JSON}" \
  --offline "${OFFLINE_JSON}" \
  --output-dir "${FIG_DIR}"

echo "Online results: ${ONLINE_JSON}"
echo "Offline results: ${OFFLINE_JSON}"
echo "Figures: ${FIG_DIR}"
