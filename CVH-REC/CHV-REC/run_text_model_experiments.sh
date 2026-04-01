#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_text_model_experiments.sh all CVH-REC
#   bash run_text_model_experiments.sh fold1 CVH-REC
#   bash run_text_model_experiments.sh fold1,fold2,fold3 CVH-REC
# Optional env vars:
#   COMMON_ARGS="--patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6"

DATASET_ARG="${1:-all}"
METHOD="${2:-CVH-REC}"
COMMON_ARGS="${COMMON_ARGS:---patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6}"
RUN_METRICS="${RUN_METRICS:-1}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

if [[ "${DATASET_ARG}" == "all" ]]; then
  FOLDS=()
  for i in {1..10}; do
    FOLDS+=("fold${i}")
  done
else
  IFS=',' read -r -a FOLDS <<< "${DATASET_ARG}"
fi

echo "Using Python: ${PYTHON_BIN}"
echo "Folds: ${FOLDS[*]}"
mkdir -p logs

# Generate embeddings for all requested models into model-specific folders under data/.
"${PYTHON_BIN}" generate_dataset.py --text_model all

for DATASET in "${FOLDS[@]}"; do
  for MODEL in bge-base e5-base gte-base bge-large e5-large gte-large; do
    EMBED_SIZE=768
    if [[ "${MODEL}" == "bge-large" || "${MODEL}" == "e5-large" || "${MODEL}" == "gte-large" ]]; then
      EMBED_SIZE=1024
    fi

    echo "Running base model training for ${MODEL} on ${DATASET}"
    "${PYTHON_BIN}" main.py --dataset "${DATASET}" --text_model "${MODEL}" --embed_size "${EMBED_SIZE}" ${COMMON_ARGS}

    echo "Running text encoder and saving recommendations for ${MODEL} on ${DATASET}"
    "${PYTHON_BIN}" main_text_encoder.py --dataset "${DATASET}" --method "${METHOD}" --text_model "${MODEL}" --embed_size "${EMBED_SIZE}" --pretrain 0 --lr 0.002 --epoch 200 --patience 5

    if [[ "${RUN_METRICS}" == "1" ]]; then
      MODEL_TAG="${MODEL//-/_}"
      METRICS_METHOD="${METHOD}_${MODEL_TAG}"
      METRICS_LOG="logs/metrics_${METRICS_METHOD}_${DATASET}.txt"
      echo "Running metrics for ${METRICS_METHOD}; writing to ${METRICS_LOG}"
      "${PYTHON_BIN}" metrics.py --method "${METRICS_METHOD}" > "${METRICS_LOG}" 2>&1
    fi
  done
done

cat <<'EOF'
Completed all runs.
Results are saved under output/ as:
  <METHOD>_<model_tag>_<dataset>.json
Logs are saved under logs/ as:
  main_CVH-REC_<model_tag>_<dataset>.log
  main_text_encoder_<METHOD>_<model_tag>_<dataset>.log
  metrics_<METHOD>_<model_tag>_<dataset>.txt
EOF
