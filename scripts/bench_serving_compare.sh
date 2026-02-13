#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-gpt2}"
GPU="${GPU:-0}"
OUTDIR="${OUTDIR:-outputs/benchmarks/serving}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "./.conda-bench/bin/python" ]]; then
    PYTHON_BIN="./.conda-bench/bin/python"
  elif [[ -x "./.conda/bin/python" ]]; then
    PYTHON_BIN="./.conda/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

"${PYTHON_BIN}" benchmarks/serving/online_compare.py \
  --model "${MODEL}" \
  --gpu "${GPU}" \
  --output-dir "${OUTDIR}" \
  "$@"

ONLINE_JSON="$(ls -t "${OUTDIR}"/online_*/online_results.json | head -n 1)"

"${PYTHON_BIN}" benchmarks/serving/offline_compare.py \
  --model "${MODEL}" \
  --gpu "${GPU}" \
  --output-dir "${OUTDIR}" \
  "$@"

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
