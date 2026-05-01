#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-lgn}"
LR="${2:-0.001}"
DECAY="${3:-1e-4}"
DATASET="${4:-ml-1m}"
SOURCE_LABEL="${5:-1}"
TRIALS="${6:-5}"

if [[ "${MODEL}" != "lgn" && "${MODEL}" != "mf" ]]; then
  echo "Usage: $0 [model:lgn|mf] [lr] [decay] [dataset] [source_label] [trials]"
  echo "Example: $0 lgn 0.001 1e-4 ml-1m 1 5"
  exit 1
fi

mkdir -p logs

sbatch \
  --job-name="propvstrat-${MODEL}-${DATASET}-src${SOURCE_LABEL}" \
  --partition=gpu \
  --gres="gpu:v100-sxm2:1" \
  --cpus-per-task=1 \
  --mem=32G \
  --time=4:00:00 \
  --output="logs/prop_vs_strat_${DATASET}_${MODEL}_src${SOURCE_LABEL}_t${TRIALS}_%j.out" \
  --error="logs/prop_vs_strat_${DATASET}_${MODEL}_src${SOURCE_LABEL}_t${TRIALS}_%j.err" \
  --export=ALL,MODEL="${MODEL}",LR="${LR}",DECAY="${DECAY}",DATASET="${DATASET}",SOURCE_LABEL="${SOURCE_LABEL}",TRIALS="${TRIALS}" \
  --wrap='mamba run -n fair-ranking python mixing/experiments/prop_vs_stratified/train.py --dataset ${DATASET} --model ${MODEL} --source-label ${SOURCE_LABEL} --lr ${LR} --decay ${DECAY} --trials ${TRIALS}'
