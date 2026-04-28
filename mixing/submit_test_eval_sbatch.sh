#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-lgn}"
LR="${2:-0.001}"
DECAY="${3:-1e-4}"
DATASET="${4:-ml-1m}"
RECDIM="${5:-64}"

if [[ "${MODEL}" != "lgn" && "${MODEL}" != "mf" ]]; then
  echo "Usage: $0 [model:lgn|mf] [lr] [decay] [dataset] [recdim]"
  echo "Example: $0 lgn 0.001 1e-4 ml-1m 64"
  exit 1
fi

labels=(1 18 25 56)

mkdir -p logs

for label in "${labels[@]}"; do
  sbatch \
    --job-name="test-${MODEL}-${label}" \
    --partition=gpu \
    --gres="gpu:v100-sxm2:1" \
    --cpus-per-task=1 \
    --mem=32G \
    --time=8:00:00 \
    --output="logs/test_eval_${DATASET}_${MODEL}_d${RECDIM}_${label}_%j.out" \
    --error="logs/test_eval_${DATASET}_${MODEL}_d${RECDIM}_${label}_%j.err" \
    --export=ALL,SRC_LABEL="${label}",MODEL="${MODEL}",LR="${LR}",DECAY="${DECAY}",DATASET="${DATASET}",RECDIM="${RECDIM}" \
    --wrap='mamba run -n fair-ranking python mixing/evaluate_top_mixes_test.py --dataset ${DATASET} --feature-name Age --source-label ${SRC_LABEL} --model ${MODEL} --lr ${LR} --decay ${DECAY} --recdim ${RECDIM} --top-k 300 --epochs 40 --layer 1'
done
