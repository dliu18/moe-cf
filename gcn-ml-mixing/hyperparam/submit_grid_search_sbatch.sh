#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-ml-1m}"

mkdir -p logs

sbatch \
  --job-name="grid-${DATASET}" \
  --partition=dean \
  --gres="gpu:nvidia_rtx_6000_ada_generation:1" \
  --cpus-per-task=1 \
  --mem=32G \
  --time=24:00:00 \
  --output="logs/grid_search_${DATASET}_%j.out" \
  --error="logs/grid_search_${DATASET}_%j.err" \
  --export=ALL,DATASET="${DATASET}" \
  --wrap='mamba run -n moe-cf python gcn-ml-mixing/hyperparam/grid_search.py --dataset ${DATASET}'
