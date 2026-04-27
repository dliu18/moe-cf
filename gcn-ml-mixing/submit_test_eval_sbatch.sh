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

labels=(45 56)

mkdir -p logs

for label in "${labels[@]}"; do
  sbatch \
    --job-name="test-${MODEL}-${label}" \
    --partition=dean \
    --gres="gpu:nvidia_rtx_6000_ada_generation:1" \
    --cpus-per-task=1 \
    --mem=32G \
    --time=8:00:00 \
    --output="logs/test_eval_${MODEL}_${label}_%j.out" \
    --error="logs/test_eval_${MODEL}_${label}_%j.err" \
    --export=ALL,SRC_LABEL="${label}",MODEL="${MODEL}",LR="${LR}",DECAY="${DECAY}" \
    --wrap='mamba run -n moe-cf python gcn-ml-mixing/evaluate_top_mixes_test.py --feature-name Age --source-label ${SRC_LABEL} --model ${MODEL} --lr ${LR} --decay ${DECAY} --top-k 300 --epochs 40 --layer 1'
done

    # --partition=gpu \
    # --gres="gpu:v100-sxm2:1" \

    # --partition=dean \
    # --gres="gpu:nvidia_rtx_6000_ada_generation:1" \
    # --wrap='mamba run -n moe-cf python gcn-ml-mixing/evaluate_top_mixes_test.py --feature-name Age --source-label ${SRC_LABEL} --model ${MODEL} --lr ${LR} --decay ${DECAY} --top-k 300 --epochs 40 --layer 1'
