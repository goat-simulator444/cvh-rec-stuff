#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_gte_large_only.sh fold1 CVH-REC

DATASET="${1:-fold1}"
METHOD="${2:-CVH-REC}"
COMMON_ARGS="${COMMON_ARGS:---patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6}"
RUN_METRICS="${RUN_METRICS:-1}"
MODEL="gte-large"
EMBED_SIZE=1024
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

echo "Using Python: ${PYTHON_BIN}"
echo "Dataset: ${DATASET}"
mkdir -p logs

"${PYTHON_BIN}" generate_dataset.py --text_model "${MODEL}"
"${PYTHON_BIN}" main.py --dataset "${DATASET}" --text_model "${MODEL}" --embed_size "${EMBED_SIZE}" ${COMMON_ARGS}
"${PYTHON_BIN}" main_text_encoder.py --dataset "${DATASET}" --method "${METHOD}" --text_model "${MODEL}" --embed_size "${EMBED_SIZE}" --pretrain 0 --lr 0.002 --epoch 200 --patience 5

if [[ "${RUN_METRICS}" == "1" ]]; then
  MODEL_TAG="${MODEL//-/_}"
  METRICS_METHOD="${METHOD}_${MODEL_TAG}"
  METRICS_LOG="logs/metrics_${METRICS_METHOD}_${DATASET}.txt"
  echo "Running metrics for ${METRICS_METHOD}; writing to ${METRICS_LOG}"
  "${PYTHON_BIN}" metrics.py --method "${METRICS_METHOD}" > "${METRICS_LOG}" 2>&1
fi

cat <<'EOF'
Completed gte-large run.
Outputs:
  output/<METHOD>_gte_large_<dataset>.json
Logs:
  logs/main_CVH-REC_gte_large_<dataset>.log
  logs/main_text_encoder_<METHOD>_gte_large_<dataset>.log
  logs/metrics_<METHOD>_gte_large_<dataset>.txt
EOF
