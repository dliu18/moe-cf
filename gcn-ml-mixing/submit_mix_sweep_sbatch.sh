#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-lgn}"
LR="${2:-0.001}"
DECAY="${3:-1e-4}"

if [[ "${MODEL}" != "lgn" && "${MODEL}" != "mf" ]]; then
  echo "Usage: $0 [model:lgn|mf] [lr] [decay]"
  echo "Example: $0 lgn 0.001 1e-4"
  exit 1
fi

labels=(18 25 35 45 56)

# labels=(1)

mkdir -p logs

for label in "${labels[@]}"; do
  sbatch \
    --job-name="mix-${MODEL}-${label}" \
    --partition=dean \
    --gres="gpu:nvidia_rtx_6000_ada_generation:1" \
    --cpus-per-task=1 \
    --mem=32G \
    --time=24:00:00 \
    --output="logs/mix_sweep_${MODEL}_${label}_%j.out" \
    --error="logs/mix_sweep_${MODEL}_${label}_%j.err" \
    --export=ALL,SRC_LABEL="${label}",MODEL="${MODEL}",LR="${LR}",DECAY="${DECAY}" \
    --wrap='mamba run -n moe-cf python gcn-ml-mixing/mix_sweep.py --feature-name Age --source-label ${SRC_LABEL} --model ${MODEL} --lr ${LR} --decay ${DECAY} --trials 300 --epochs 40 --layer 1 --num-folds 5'
done
