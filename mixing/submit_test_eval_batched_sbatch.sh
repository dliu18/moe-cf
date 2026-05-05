#!/usr/bin/env bash
set -euo pipefail

LR="${1:-0.001}"
DECAY="${2:-1e-4}"
DATASET="${3:-ml-1m}"
MODEL="${4:-lgn}"

if [[ "${MODEL}" != "lgn" && "${MODEL}" != "mf" ]]; then
  echo "Usage: $0 [lr] [decay] [dataset] [model:lgn|mf]"
  echo "Example: $0 0.001 1e-4 movielens lgn"
  exit 1
fi

FEATURE_NAME="Age"
TOPKS="[20]"
TOP_K="75"
TRIALS_PER_MIX="25"

# RECDIM=64 runs all source groups.
LABELS_64=(1 18 25 35 45 50 56)
# RECDIM=4 runs a dataset-specific subset.
LABELS_4=(45 56)

if [[ "${DATASET}" == "lastfm-asia" ]]; then
  FEATURE_NAME="Country"
  TOPKS="[100]"
  TOP_K="1"
  LABELS_64=(17 10 0 3)
  LABELS_4=(17 0)
elif [[ "${DATASET}" == "movielens" || "${DATASET}" == "ml-1m" ]]; then
  FEATURE_NAME="Age"
  TOPKS="[20]"
  TOP_K="1"
  LABELS_64=(1 18 25 35 45 50 56)
  LABELS_4=(45 56)
fi

mkdir -p logs

# Join arrays for export into single sbatch job.
LABELS_64_CSV="$(IFS=,; echo "${LABELS_64[*]}")"
LABELS_4_CSV="$(IFS=,; echo "${LABELS_4[*]}")"

sbatch \
  --job-name="test-batch-${MODEL}-${DATASET}" \
  --partition=gpu \
  --gres="gpu:v100-sxm2:1" \
  --cpus-per-task=1 \
  --mem=32G \
  --time=8:00:00 \
  --output="logs/test_eval_batch_${DATASET}_${MODEL}_%j.out" \
  --error="logs/test_eval_batch_${DATASET}_${MODEL}_%j.err" \
  --export=ALL,MODEL="${MODEL}",LR="${LR}",DECAY="${DECAY}",DATASET="${DATASET}",FEATURE_NAME="${FEATURE_NAME}",TOPKS="${TOPKS}",TOP_K="${TOP_K}",TRIALS_PER_MIX="${TRIALS_PER_MIX}",LABELS_64_CSV="${LABELS_64_CSV}",LABELS_4_CSV="${LABELS_4_CSV}" \
  --wrap='bash -lc '\''
set -euo pipefail

run_for_recdim () {
  local recdim="$1"
  local labels_csv="$2"
  IFS="," read -r -a labels <<< "$labels_csv"
  for label in "${labels[@]}"; do
    echo "=== Running dataset=${DATASET} model=${MODEL} recdim=${recdim} source_label=${label} ==="
    mamba run -n fair-ranking python mixing/evaluate_top_mixes_test.py \
      --dataset "${DATASET}" \
      --feature-name "${FEATURE_NAME}" \
      --source-label "${label}" \
      --model "${MODEL}" \
      --lr "${LR}" \
      --decay "${DECAY}" \
      --recdim "${recdim}" \
      --topks "${TOPKS}" \
      --top-k "${TOP_K}" \
      --epochs 40 \
      --layer 1 \
      --trials-per-mix "${TRIALS_PER_MIX}"
  done
}

run_for_recdim 64 "${LABELS_64_CSV}"
run_for_recdim 4 "${LABELS_4_CSV}"
'\'''
