#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-gpt2}"
GPU="${GPU:-0}"
OUTDIR="${OUTDIR:-outputs/benchmarks/serving}"

python benchmarks/serving/online_compare.py \
  --model "${MODEL}" \
  --gpu "${GPU}" \
  --output-dir "${OUTDIR}" \
  "$@"

ONLINE_JSON="$(ls -t "${OUTDIR}"/online_*/online_results.json | head -n 1)"

python benchmarks/serving/offline_compare.py \
  --model "${MODEL}" \
  --gpu "${GPU}" \
  --output-dir "${OUTDIR}"

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

