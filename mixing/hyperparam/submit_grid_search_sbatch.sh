#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-ml-1m}"
BPR_BATCH="${2:-1000000}"
TOPKS="${3:-[20]}"

mkdir -p logs

sbatch \
  --job-name="grid-${DATASET}" \
  --partition=gpu \
  --gres="gpu:v100-sxm2:1" \
  --cpus-per-task=1 \
  --mem=32G \
  --time=2:00:00 \
  --output="logs/grid_search_${DATASET}_%j.out" \
  --error="logs/grid_search_${DATASET}_%j.err" \
  --export=ALL,DATASET="${DATASET}",BPR_BATCH="${BPR_BATCH}",TOPKS="${TOPKS}" \
  --wrap='mamba run -n fair-ranking python mixing/hyperparam/grid_search.py --dataset ${DATASET} --bpr-batch ${BPR_BATCH} --topks "${TOPKS}"'