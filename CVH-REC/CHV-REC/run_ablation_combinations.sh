#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_ablation_combinations.sh fold1 CVH-REC
#   bash run_ablation_combinations.sh all CVH-REC
#
# Optional env vars:
#   PYTHON_BIN=/path/to/python
#   BASE_TEXT_MODEL=mpnet
#   RUN_MAIN=1
#   RUN_TEXT_ENCODER=1
#   RUN_METRICS=1
#   PREPARE_DATA=1
#   OUTPUT_BASE=../output/ablation_runs
#   LOG_BASE=logs/ablation_runs
#   STATS_BASE=stats/ablation_runs
#   RESET_SUMMARY=1
#   SEGFAULT_RETRIES=1
#   FAIL_FAST=0
#   PYTHONUNBUFFERED=1
#   OMP_NUM_THREADS=1
#   MKL_NUM_THREADS=1
#   COMMON_MAIN_ARGS="--patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6 --save_flag 0"
#   COMMON_TEXT_ARGS="--pretrain 0 --lr 0.002 --epoch 200 --patience 5 --save_flag 0"

DATASET="${1:-fold1}"
METHOD="${2:-CVH-REC}"

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

SEGFAULT_RETRIES="${SEGFAULT_RETRIES:-1}"
FAIL_FAST="${FAIL_FAST:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

BASE_TEXT_MODEL="${BASE_TEXT_MODEL:-mpnet}"

RUN_MAIN="${RUN_MAIN:-1}"
RUN_TEXT_ENCODER="${RUN_TEXT_ENCODER:-1}"
RUN_METRICS="${RUN_METRICS:-1}"
PREPARE_DATA="${PREPARE_DATA:-1}"

OUTPUT_BASE="${OUTPUT_BASE:-../output/ablation_runs}"
LOG_BASE="${LOG_BASE:-logs/ablation_runs}"
STATS_BASE="${STATS_BASE:-stats/ablation_runs}"
RESET_SUMMARY="${RESET_SUMMARY:-1}"

COMMON_MAIN_ARGS="${COMMON_MAIN_ARGS:---patience 5 --epoch 200 --lr 0.002 --alpha1 0.6 --alpha2 0.6 --alpha3 0.6 --alpha4 0.6 --save_flag 0}"
COMMON_TEXT_ARGS="${COMMON_TEXT_ARGS:---pretrain 0 --lr 0.002 --epoch 200 --patience 5 --save_flag 0}"

read -r -a MAIN_ARGS_ARR <<< "${COMMON_MAIN_ARGS}"
read -r -a TEXT_ARGS_ARR <<< "${COMMON_TEXT_ARGS}"

if [[ "${DATASET}" == "all" ]]; then
  FOLDS=()
  for i in {1..10}; do
    FOLDS+=("fold${i}")
  done
else
  FOLDS=("${DATASET}")
fi

mkdir -p "${OUTPUT_BASE}" "${LOG_BASE}" "${STATS_BASE}"

summary_file="${STATS_BASE}/run_index.tsv"
summary_header="combo\tstatus\ttext_model\toutput_dir\tstats_file\tlogs_dir\tnotes"
if [[ ! -f "${summary_file}" || "${RESET_SUMMARY}" == "1" ]]; then
  echo -e "${summary_header}" > "${summary_file}"
else
  current_header="$(head -n 1 "${summary_file}" || true)"
  if [[ "${current_header}" != "${summary_header}" ]]; then
    echo "[RUN][WARN] Existing summary header is incompatible; resetting ${summary_file}."
    echo -e "${summary_header}" > "${summary_file}"
  fi
fi

model_tag() {
  local model="$1"
  local key
  key="$(echo "${model}" | tr '[:upper:]' '[:lower:]')"
  case "${key}" in
    mpnet|all-mpnet-base-v2|sentence-transformers/all-mpnet-base-v2)
      echo "mpnet"
      ;;
    bge-large|bgelarge|baai/bge-large-en-v1.5)
      echo "bge_large"
      ;;
    bge-base|bgebase|baai/bge-base-en-v1.5)
      echo "bge_base"
      ;;
    e5-large|e5large|intfloat/e5-large-v2)
      echo "e5_large"
      ;;
    e5-base|e5base|intfloat/e5-base-v2)
      echo "e5_base"
      ;;
    gte-large|gtelarge|thenlper/gte-large)
      echo "gte_large"
      ;;
    gte-base|gtebase|thenlper/gte-base)
      echo "gte_base"
      ;;
    *)
      echo "${key//[^a-z0-9]/_}"
      ;;
  esac
}

declare -A PREPARED_MODELS

run_cmd_logged() {
  local stage="$1"
  local log_file="$2"
  shift 2

  local attempt=1
  local max_attempts=$((SEGFAULT_RETRIES + 1))
  local rc=0

  : > "${log_file}"

  while ((attempt <= max_attempts)); do
    {
      echo "[RUN][${stage}] attempt=${attempt}/${max_attempts}"
      echo "[RUN][${stage}] cmd: $*"
    } >> "${log_file}"

    set +e
    "$@" 2>&1 | tee -a "${log_file}"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ ${rc} -eq 0 ]]; then
      return 0
    fi

    if [[ ${rc} -eq 139 || ${rc} -eq 134 || ${rc} -eq 132 || ${rc} -eq 136 ]]; then
      echo "[RUN][${stage}][WARN] Native crash (exit=${rc}) on attempt ${attempt}/${max_attempts}." | tee -a "${log_file}"
      attempt=$((attempt + 1))
      continue
    fi

    echo "[RUN][${stage}][ERROR] Failed with exit=${rc}." | tee -a "${log_file}"
    return "${rc}"
  done

  echo "[RUN][${stage}][ERROR] Failed after ${max_attempts} attempts. Last exit=${rc}." | tee -a "${log_file}"
  return "${rc}"
}

ensure_model_data() {
  local model="$1"
  local tag vectors_file api_file tag_file

  tag="$(model_tag "${model}")"

  if [[ -n "${PREPARED_MODELS[${tag}]:-}" ]]; then
    return
  fi

  if [[ "${tag}" == "mpnet" ]]; then
    vectors_file="data/vectors.h5"
    api_file="data/API_vectors.h5"
    tag_file="data/tag_vectors.h5"
  else
    vectors_file="data/${tag}/vectors.h5"
    api_file="data/${tag}/API_vectors.h5"
    tag_file="data/${tag}/tag_vectors.h5"
  fi

  if [[ "${PREPARE_DATA}" == "1" ]]; then
    if [[ ! -f "${vectors_file}" || ! -f "${api_file}" || ! -f "${tag_file}" ]]; then
      local prep_log
      prep_log="${LOG_BASE}/prepare_${tag}.log"
      echo "[DATA] Preparing embeddings for model=${model} (log: ${prep_log})"
      if ! "${PYTHON_BIN}" generate_dataset.py --text_model "${model}" 2>&1 | tee "${prep_log}"; then
        echo "[DATA][ERROR] Embedding preparation failed for model=${model}."
        return 1
      fi
    else
      echo "[DATA] Reusing existing embeddings for model=${model}"
    fi
  else
    echo "[DATA] PREPARE_DATA=0, skipping embedding generation for model=${model}"
  fi

  PREPARED_MODELS["${tag}"]=1
}

run_main_stage() {
  local fold="$1"
  local combo_log_dir="$2"
  local model="$3"
  local use_bge="$4"
  local use_temporal="$5"
  local use_diffusion="$6"

  local main_log
  main_log="${combo_log_dir}/main_${fold}.log"
  local -a cmd

  cmd=(
    "${PYTHON_BIN}" main.py
    --dataset "${fold}"
    --method "${METHOD}"
    --text_model "${model}"
    --use_bge "${use_bge}"
    --use_temporal "${use_temporal}"
    --use_diffusion "${use_diffusion}"
    --temporal_hypergraph "${use_temporal}"
    --edge_diffusion "${use_diffusion}"
    "${MAIN_ARGS_ARR[@]}"
  )

  echo "[RUN][main.py] fold=${fold} log=${main_log}"
  run_cmd_logged "main.py/${fold}" "${main_log}" "${cmd[@]}"
}

run_text_stage() {
  local fold="$1"
  local combo_log_dir="$2"
  local combo_output_dir="$3"
  local model="$4"
  local use_bge="$5"
  local use_temporal="$6"
  local use_diffusion="$7"

  local text_log
  text_log="${combo_log_dir}/main_text_encoder_${fold}.log"
  local -a cmd

  cmd=(
    "${PYTHON_BIN}" main_text_encoder.py
    --dataset "${fold}"
    --method "${METHOD}"
    --text_model "${model}"
    --output "${combo_output_dir}"
    --use_bge "${use_bge}"
    --use_temporal "${use_temporal}"
    --use_diffusion "${use_diffusion}"
    --temporal_hypergraph "${use_temporal}"
    "${TEXT_ARGS_ARR[@]}"
  )

  echo "[RUN][main_text_encoder.py] fold=${fold} log=${text_log}"
  run_cmd_logged "main_text_encoder.py/${fold}" "${text_log}" "${cmd[@]}"
}

run_metrics_stage() {
  local combo_id="$1"
  local combo_output_dir="$2"
  local combo_log_dir="$3"
  local combo_stats_dir="$4"
  local metrics_method="$5"

  local stats_file metrics_log rc
  stats_file="${combo_stats_dir}/metrics_${combo_id}.txt"
  metrics_log="${combo_log_dir}/metrics_${combo_id}.log"

  : > "${metrics_log}"
  echo "[RUN][metrics.py] method=${metrics_method} output=${combo_output_dir}"
  set +e
  "${PYTHON_BIN}" metrics.py --method "${metrics_method}" --output "${combo_output_dir}" > "${stats_file}" 2>&1
  rc=$?
  set -e
  cp "${stats_file}" "${metrics_log}"

  if [[ ${rc} -ne 0 ]]; then
    echo "[RUN][metrics.py][ERROR] Failed with exit=${rc}" | tee -a "${metrics_log}"
    return "${rc}"
  fi

  return 0
}

join_notes() {
  local IFS=';'
  echo "$*"
}

total_combos=0
failed_combos=0

for use_bge in 0 1; do
  for use_temporal in 0 1; do
    for use_diffusion in 0 1; do
      combo_id="bge${use_bge}_temporal${use_temporal}_diffusion${use_diffusion}"

      if [[ "${use_bge}" == "1" ]]; then
        text_model="bge-large"
      else
        text_model="${BASE_TEXT_MODEL}"
      fi

      text_tag="$(model_tag "${text_model}")"
      metrics_method="${METHOD}_${text_tag}"

      combo_output_dir="${OUTPUT_BASE}/${combo_id}"
      combo_log_dir="${LOG_BASE}/${combo_id}"
      combo_stats_dir="${STATS_BASE}/${combo_id}"
      mkdir -p "${combo_output_dir}" "${combo_log_dir}" "${combo_stats_dir}"

      total_combos=$((total_combos + 1))
      combo_ok=1
      combo_notes=()
      stats_file_path="${combo_stats_dir}/metrics_${combo_id}.txt"

      echo "=============================================================="
      echo "[COMBO] ${combo_id} | text_model=${text_model} | method=${METHOD}"
      echo "[COMBO] output=${combo_output_dir}"
      echo "[COMBO] logs=${combo_log_dir}"
      echo "=============================================================="

      if ! ensure_model_data "${text_model}"; then
        combo_ok=0
        combo_notes+=("prepare_failed")
      fi

      for fold in "${FOLDS[@]}"; do
        if [[ ${combo_ok} -eq 1 && "${RUN_MAIN}" == "1" ]]; then
          if ! run_main_stage "${fold}" "${combo_log_dir}" "${text_model}" "${use_bge}" "${use_temporal}" "${use_diffusion}"; then
            combo_ok=0
            combo_notes+=("main_failed:${fold}")
            if [[ "${FAIL_FAST}" == "1" ]]; then
              echo "[FAIL_FAST] Stopping due to main.py failure in combo=${combo_id}, fold=${fold}."
              exit 1
            fi
            continue
          fi
        fi

        if [[ ${combo_ok} -eq 1 && "${RUN_TEXT_ENCODER}" == "1" ]]; then
          if ! run_text_stage "${fold}" "${combo_log_dir}" "${combo_output_dir}" "${text_model}" "${use_bge}" "${use_temporal}" "${use_diffusion}"; then
            combo_ok=0
            combo_notes+=("text_failed:${fold}")
            if [[ "${FAIL_FAST}" == "1" ]]; then
              echo "[FAIL_FAST] Stopping due to main_text_encoder.py failure in combo=${combo_id}, fold=${fold}."
              exit 1
            fi
            continue
          fi
        fi
      done

      if [[ ${combo_ok} -eq 1 && "${RUN_METRICS}" == "1" ]]; then
        if ! run_metrics_stage "${combo_id}" "${combo_output_dir}" "${combo_log_dir}" "${combo_stats_dir}" "${metrics_method}"; then
          combo_ok=0
          combo_notes+=("metrics_failed")
          if [[ "${FAIL_FAST}" == "1" ]]; then
            echo "[FAIL_FAST] Stopping due to metrics.py failure in combo=${combo_id}."
            exit 1
          fi
        fi
      fi

      if [[ ${combo_ok} -eq 1 ]]; then
        combo_status="ok"
        combo_note_value="none"
      else
        combo_status="failed"
        combo_note_value="$(join_notes "${combo_notes[@]}")"
        failed_combos=$((failed_combos + 1))
      fi

      {
        echo "combo=${combo_id}"
        echo "status=${combo_status}"
        echo "text_model=${text_model}"
        echo "notes=${combo_note_value}"
        echo "python_bin=${PYTHON_BIN}"
      } > "${combo_stats_dir}/status_${combo_id}.txt"

      echo -e "${combo_id}\t${combo_status}\t${text_model}\t${combo_output_dir}\t${stats_file_path}\t${combo_log_dir}\t${combo_note_value}" >> "${summary_file}"
    done
  done
done

echo "All ablation combinations completed."
echo "Run index: ${summary_file}"
echo "Summary: total_combos=${total_combos}, failed_combos=${failed_combos}, segfault_retries=${SEGFAULT_RETRIES}, python=${PYTHON_BIN}"
