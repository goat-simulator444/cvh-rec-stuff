#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_bge_large_only.sh fold1 CVH-REC
#   bash run_bge_large_only.sh all CVH-REC

# Optional env vars:
#   ATTENTION_MODE=on|off   (default: on)
#   ATTN_ARGS="..."         (overrides auto-generated attention args)

DATASET="${1:-fold1}"

DATASET="${1:-all}"
METHOD="${2:-CVH-REC}"
COMMON_ARGS="${COMMON_ARGS:---patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6}"
ATTENTION_MODE="${ATTENTION_MODE:-on}"
if [[ "${ATTENTION_MODE}" == "off" ]]; then
  DEFAULT_ATTN_ARGS="--use_hypergraph_attention 0"
else
  DEFAULT_ATTN_ARGS="--use_hypergraph_attention 1 --attn_temperature 0.7 --attn_prior_weight 0.2 --attn_consistency_weight 0.05 --attn_global_local_weight 0.03"
fi
ATTN_ARGS="${ATTN_ARGS:-${DEFAULT_ATTN_ARGS}}"
RUN_METRICS="${RUN_METRICS:-1}"
MODEL="bge-large"
EMBED_SIZE=1024
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

echo "Using Python: ${PYTHON_BIN}"
echo "Dataset: ${DATASET}"
echo "Attention mode: ${ATTENTION_MODE}"
echo "Attention args: ${ATTN_ARGS}"

if [[ "${DATASET}" == "all" ]]; then
  FOLDS=()
  for i in {1..10}; do
    FOLDS+=("fold${i}")
  done
else
  FOLDS=("${DATASET}")
fi

mkdir -p logs

"${PYTHON_BIN}" generate_dataset.py --text_model "${MODEL}"
for FOLD in "${FOLDS[@]}"; do
  echo "=============================================="
  echo "Running ${MODEL} on ${FOLD}"
  echo "=============================================="

  "${PYTHON_BIN}" main.py --dataset "${FOLD}" --text_model "${MODEL}" --embed_size "${EMBED_SIZE}" ${COMMON_ARGS} ${ATTN_ARGS}
  "${PYTHON_BIN}" main_text_encoder.py --dataset "${FOLD}" --method "${METHOD}" --text_model "${MODEL}" --embed_size "${EMBED_SIZE}" --pretrain 0 --lr 0.002 --epoch 200 --patience 5
done

if [[ "${RUN_METRICS}" == "1" ]]; then
  MODEL_TAG="${MODEL//-/_}"
  METRICS_METHOD="${METHOD}_${MODEL_TAG}"
  METRICS_DATASET_TAG="${DATASET}"
  METRICS_LOG="logs/metrics_${METRICS_METHOD}_${METRICS_DATASET_TAG}.txt"
  echo "Running metrics for ${METRICS_METHOD}; writing to ${METRICS_LOG}"
  "${PYTHON_BIN}" metrics.py --method "${METRICS_METHOD}" > "${METRICS_LOG}" 2>&1
fi

cat <<'EOF'
Completed bge-large run.
Outputs:
  output/<METHOD>_bge_large_<dataset>.json
Logs:
  logs/main_CVH-REC_bge_large_<dataset>.log
  logs/main_text_encoder_<METHOD>_bge_large_<dataset>.log
  logs/metrics_<METHOD>_bge_large_<dataset>.txt
EOF
