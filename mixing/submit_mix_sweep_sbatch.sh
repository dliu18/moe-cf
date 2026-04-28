#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-lgn}"
LR="${2:-0.001}"
DECAY="${3:-1e-4}"
DATASET="${4:-ml-1m}"
RECDIM="${5:-64}"
FEATURE_NAME="Age"
NUM_FOLDS=5
TOPKS="[20]"

if [[ "${MODEL}" != "lgn" && "${MODEL}" != "mf" ]]; then
  echo "Usage: $0 [model:lgn|mf] [lr] [decay] [dataset] [recdim]"
  echo "Example: $0 lgn 0.001 1e-4 movielens 64"
  exit 1
fi

labels=(1 18 25 56)

if [[ "${DATASET}" == "lastfm-asia" ]]; then
  FEATURE_NAME="Country"
  labels=(17 10 0 3)
  NUM_FOLDS=4
  TOPKS="[100]"
elif [[ "${DATASET}" == "movielens" || "${DATASET}" == "ml-1m" ]]; then
  FEATURE_NAME="Age"
  labels=(1 18 25 35 45 50 56)
  NUM_FOLDS=5
  TOPKS="[20]"
fi

# labels=(1)

mkdir -p logs

for label in "${labels[@]}"; do
  sbatch \
    --job-name="mix-${MODEL}-${label}" \
    --partition=dean \
    --gres="gpu:nvidia_rtx_6000_ada_generation:1" \
    --mem=32G \
    --time=8:00:00 \
    --cpus-per-task=1 \
    --output="logs/mix_sweep_${DATASET}_${MODEL}_d${RECDIM}_${label}_%j.out" \
    --error="logs/mix_sweep_${DATASET}_${MODEL}_d${RECDIM}_${label}_%j.err" \
    --export=ALL,SRC_LABEL="${label}",MODEL="${MODEL}",LR="${LR}",DECAY="${DECAY}",DATASET="${DATASET}",RECDIM="${RECDIM}",FEATURE_NAME="${FEATURE_NAME}",NUM_FOLDS="${NUM_FOLDS}",TOPKS="${TOPKS}" \
    --wrap='mamba run -n moe-cf python mixing/mix_sweep.py --dataset ${DATASET} --feature-name ${FEATURE_NAME} --source-label ${SRC_LABEL} --model ${MODEL} --lr ${LR} --decay ${DECAY} --recdim ${RECDIM} --topks ${TOPKS} --trials 300 --epochs 40 --layer 1 --num-folds ${NUM_FOLDS}'
done

    # --partition=dean \
    # --gres="gpu:nvidia_rtx_6000_ada_generation:1" \
    # --mem=32G \
    # --time=24:00:00 \
    # --wrap='mamba run -n moe-cf python mixing/mix_sweep.py --dataset ${DATASET} --feature-name ${FEATURE_NAME} --source-label ${SRC_LABEL} --model ${MODEL} --lr ${LR} --decay ${DECAY} --recdim ${RECDIM} --topks ${TOPKS} --trials 300 --epochs 40 --layer 1 --num-folds ${NUM_FOLDS}'

    # --gres="gpu:v100-sxm2:1" \
