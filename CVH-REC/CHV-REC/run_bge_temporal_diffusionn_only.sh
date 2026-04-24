#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_bge_temporal_diffusionn_only.sh fold1 CVH-REC
#   bash run_bge_temporal_diffusionn_only.sh all CVH-REC
#
# Optional env vars:
#   PYTHON_BIN=/path/to/python
#   PREPARE_DATA=1
#   RUN_MAIN=1
#   RUN_TEXT_ENCODER=1
#   RUN_METRICS=1
#   OOM_RETRY=1
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   OUTPUT_BASE=../output/ablation_runs
#   LOG_BASE=logs/ablation_runs
#   STATS_BASE=stats/ablation_runs
#   COMMON_MAIN_ARGS="--patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6 --save_flag 1 --batch_size 64 --temporal_window_size 2592000 --temporal_stride 4 --temporal_max_steps 6 --temporal_min_edges_per_slice 20 --temporal_recent_pool_multiplier 4 --edge_diffusion_hidden_dim 128 --edge_diffusion_steps 2 --edge_diffusion_max_pos_per_batch 128 --edge_gen_threshold 0.95 --edge_gen_max_edges_per_node 4 --edge_gen_topk_per_node 48 --edge_gen_max_total_edges 15000 --edge_gen_max_ratio_to_real 1.0 --edge_gen_refresh_interval 20 --edge_gen_warmup_epochs 20 --edge_gen_chunk_size 512"
#   MEMORY_SAFE_MAIN_ARGS="--batch_size 64 --temporal_window_size 2592000 --temporal_stride 4 --temporal_max_steps 6 --temporal_min_edges_per_slice 20 --temporal_recent_pool_multiplier 4 --edge_diffusion_hidden_dim 96 --edge_diffusion_steps 2 --edge_diffusion_max_pos_per_batch 96 --edge_gen_threshold 0.97 --edge_gen_max_edges_per_node 3 --edge_gen_topk_per_node 32 --edge_gen_max_total_edges 12000 --edge_gen_max_ratio_to_real 0.8 --edge_gen_refresh_interval 25 --edge_gen_warmup_epochs 25 --edge_gen_chunk_size 384"
#   COMMON_TEXT_ARGS="--pretrain 0 --lr 0.002 --epoch 200 --patience 5 --save_flag 1 --temporal_window_size 2592000 --temporal_stride 4 --temporal_max_steps 6 --temporal_min_edges_per_slice 20 --temporal_recent_pool_multiplier 4"

DATASET="${1:-fold1}"
METHOD="${2:-CVH-REC}"

MODEL="bge-large"
MODEL_TAG="bge_large"
COMBO_ID="bge1_temporal1_diffusion1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${SCRIPT_DIR}/../../.venv/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/../../.venv/bin/python"
  elif [[ -x "${SCRIPT_DIR}/../.venv/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/../.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[FATAL] Cannot find a Python executable. Set PYTHON_BIN=/path/to/python."
    exit 1
  fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[FATAL] PYTHON_BIN is not executable: ${PYTHON_BIN}"
  exit 1
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PREPARE_DATA="${PREPARE_DATA:-1}"
RUN_MAIN="${RUN_MAIN:-1}"
RUN_TEXT_ENCODER="${RUN_TEXT_ENCODER:-1}"
RUN_METRICS="${RUN_METRICS:-1}"
OOM_RETRY="${OOM_RETRY:-1}"

OUTPUT_BASE="${OUTPUT_BASE:-../output/ablation_runs}"
LOG_BASE="${LOG_BASE:-logs/ablation_runs}"
STATS_BASE="${STATS_BASE:-stats/ablation_runs}"

COMMON_MAIN_ARGS="${COMMON_MAIN_ARGS:---patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6 --save_flag 1 --batch_size 64 --temporal_window_size 2592000 --temporal_stride 4 --temporal_max_steps 6 --temporal_min_edges_per_slice 20 --temporal_recent_pool_multiplier 4 --edge_diffusion_hidden_dim 128 --edge_diffusion_steps 2 --edge_diffusion_max_pos_per_batch 128 --edge_gen_threshold 0.95 --edge_gen_max_edges_per_node 4 --edge_gen_topk_per_node 48 --edge_gen_max_total_edges 15000 --edge_gen_max_ratio_to_real 1.0 --edge_gen_refresh_interval 20 --edge_gen_warmup_epochs 20 --edge_gen_chunk_size 512}"
MEMORY_SAFE_MAIN_ARGS="${MEMORY_SAFE_MAIN_ARGS:---batch_size 64 --temporal_window_size 2592000 --temporal_stride 4 --temporal_max_steps 6 --temporal_min_edges_per_slice 20 --temporal_recent_pool_multiplier 4 --edge_diffusion_hidden_dim 96 --edge_diffusion_steps 2 --edge_diffusion_max_pos_per_batch 96 --edge_gen_threshold 0.97 --edge_gen_max_edges_per_node 3 --edge_gen_topk_per_node 32 --edge_gen_max_total_edges 12000 --edge_gen_max_ratio_to_real 0.8 --edge_gen_refresh_interval 25 --edge_gen_warmup_epochs 25 --edge_gen_chunk_size 384}"
COMMON_TEXT_ARGS="${COMMON_TEXT_ARGS:---pretrain 0 --lr 0.002 --epoch 200 --patience 5 --save_flag 1 --temporal_window_size 2592000 --temporal_stride 4 --temporal_max_steps 6 --temporal_min_edges_per_slice 20 --temporal_recent_pool_multiplier 4}"

read -r -a MAIN_ARGS_ARR <<< "${COMMON_MAIN_ARGS}"
read -r -a SAFE_MAIN_ARGS_ARR <<< "${MEMORY_SAFE_MAIN_ARGS}"
read -r -a TEXT_ARGS_ARR <<< "${COMMON_TEXT_ARGS}"

run_logged() {
  local log_file="$1"
  shift

  set +e
  "$@" 2>&1 | tee "${log_file}"
  local rc=${PIPESTATUS[0]}
  set -e

  return "${rc}"
}

if [[ "${DATASET}" == "all" ]]; then
  FOLDS=()
  for i in {1..10}; do
    FOLDS+=("fold${i}")
  done
else
  FOLDS=("${DATASET}")
fi

combo_output_dir="${OUTPUT_BASE}/${COMBO_ID}"
combo_log_dir="${LOG_BASE}/${COMBO_ID}"
combo_stats_dir="${STATS_BASE}/${COMBO_ID}"
mkdir -p "${combo_output_dir}" "${combo_log_dir}" "${combo_stats_dir}"

if [[ "${PREPARE_DATA}" == "1" ]]; then
  vectors_file="data/${MODEL_TAG}/vectors.h5"
  api_file="data/${MODEL_TAG}/API_vectors.h5"
  tag_file="data/${MODEL_TAG}/tag_vectors.h5"

  if [[ ! -f "${vectors_file}" || ! -f "${api_file}" || ! -f "${tag_file}" ]]; then
    prep_log="${combo_log_dir}/prepare_${MODEL_TAG}.log"
    echo "[DATA] Preparing embeddings for ${MODEL} (log: ${prep_log})"
    "${PYTHON_BIN}" generate_dataset.py --text_model "${MODEL}" 2>&1 | tee "${prep_log}"
  else
    echo "[DATA] Reusing existing embeddings for ${MODEL}"
  fi
fi

for fold in "${FOLDS[@]}"; do
  echo "=============================================================="
  echo "[RUN] combo=${COMBO_ID} fold=${fold} method=${METHOD} text_model=${MODEL}"
  echo "=============================================================="

  if [[ "${RUN_MAIN}" == "1" ]]; then
    main_log="${combo_log_dir}/main_${fold}.log"
    main_cmd=(
      "${PYTHON_BIN}" main.py
      --dataset "${fold}" \
      --method "${METHOD}" \
      --text_model "${MODEL}" \
      --use_bge 1 \
      --use_temporal 1 \
      --use_diffusion 1 \
      --temporal_hypergraph 1 \
      --edge_diffusion 1 \
      "${MAIN_ARGS_ARR[@]}"
    )

    if ! run_logged "${main_log}" "${main_cmd[@]}"; then
      if [[ "${OOM_RETRY}" == "1" ]] && grep -Eiq "torch\.OutOfMemoryError|CUDA out of memory" "${main_log}"; then
        retry_log="${combo_log_dir}/main_${fold}_oom_retry.log"
        echo "[OOM][WARN] main.py failed with CUDA OOM on fold=${fold}."
        echo "[OOM][WARN] Retrying with MEMORY_SAFE_MAIN_ARGS: ${MEMORY_SAFE_MAIN_ARGS}"

        retry_cmd=(
          "${PYTHON_BIN}" main.py
          --dataset "${fold}"
          --method "${METHOD}"
          --text_model "${MODEL}"
          --use_bge 1
          --use_temporal 1
          --use_diffusion 1
          --temporal_hypergraph 1
          --edge_diffusion 1
          "${MAIN_ARGS_ARR[@]}"
          "${SAFE_MAIN_ARGS_ARR[@]}"
        )

        if ! run_logged "${retry_log}" "${retry_cmd[@]}"; then
          echo "[RUN][ERROR] main.py retry failed for fold=${fold}. See ${retry_log}"
          exit 1
        fi
      else
        echo "[RUN][ERROR] main.py failed for fold=${fold}. See ${main_log}"
        exit 1
      fi
    fi
  fi

  if [[ "${RUN_TEXT_ENCODER}" == "1" ]]; then
    text_log="${combo_log_dir}/main_text_encoder_${fold}.log"
    text_cmd=(
      "${PYTHON_BIN}" main_text_encoder.py
      --dataset "${fold}" \
      --method "${METHOD}" \
      --text_model "${MODEL}" \
      --output "${combo_output_dir}" \
      --use_bge 1 \
      --use_temporal 1 \
      --use_diffusion 1 \
      --temporal_hypergraph 1 \
      "${TEXT_ARGS_ARR[@]}"
    )

    if ! run_logged "${text_log}" "${text_cmd[@]}"; then
      echo "[RUN][ERROR] main_text_encoder.py failed for fold=${fold}. See ${text_log}"
      exit 1
    fi
  fi
done

stats_file="${combo_stats_dir}/metrics_${COMBO_ID}.txt"
if [[ "${RUN_METRICS}" == "1" ]]; then
  metrics_method="${METHOD}_${MODEL_TAG}"
  echo "[RUN] metrics.py method=${metrics_method} output=${combo_output_dir}"
  "${PYTHON_BIN}" metrics.py --method "${metrics_method}" --output "${combo_output_dir}" > "${stats_file}" 2>&1
fi

status_file="${combo_stats_dir}/status_${COMBO_ID}.txt"
{
  echo "combo=${COMBO_ID}"
  echo "status=ok"
  echo "dataset=${DATASET}"
  echo "method=${METHOD}"
  echo "text_model=${MODEL}"
  echo "python_bin=${PYTHON_BIN}"
  echo "output_dir=${combo_output_dir}"
  echo "logs_dir=${combo_log_dir}"
  echo "stats_file=${stats_file}"
} > "${status_file}"

echo "Completed ${COMBO_ID}."
echo "output=${combo_output_dir}"
echo "logs=${combo_log_dir}"
echo "stats=${stats_file}"