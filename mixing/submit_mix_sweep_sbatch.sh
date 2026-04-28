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

# labels=(1)

mkdir -p logs

for label in "${labels[@]}"; do
  sbatch \
    --job-name="mix-${MODEL}-${label}" \
    --partition=gpu \
    --gres="gpu:v100-sxm2:1" \
    --mem=32G \
    --time=8:00:00 \
    --cpus-per-task=1 \
    --output="logs/mix_sweep_${DATASET}_${MODEL}_d${RECDIM}_${label}_%j.out" \
    --error="logs/mix_sweep_${DATASET}_${MODEL}_d${RECDIM}_${label}_%j.err" \
    --export=ALL,SRC_LABEL="${label}",MODEL="${MODEL}",LR="${LR}",DECAY="${DECAY}",DATASET="${DATASET}",RECDIM="${RECDIM}" \
    --wrap='mamba run -n fair-ranking python mixing/mix_sweep.py --dataset ${DATASET} --feature-name Age --source-label ${SRC_LABEL} --model ${MODEL} --lr ${LR} --decay ${DECAY} --recdim ${RECDIM} --trials 300 --epochs 40 --layer 1 --num-folds 5'
done

    # --partition=dean \
    # --gres="gpu:nvidia_rtx_6000_ada_generation:1" \
    # --mem=32G \
    # --time=24:00:00 \
    # --wrap='mamba run -n moe-cf python mixing/mix_sweep.py --dataset ${DATASET} --feature-name Age --source-label ${SRC_LABEL} --model ${MODEL} --lr ${LR} --decay ${DECAY} --recdim ${RECDIM} --trials 300 --epochs 40 --layer 1 --num-folds 5'
