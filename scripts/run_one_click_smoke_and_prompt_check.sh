#!/usr/bin/env bash
set -euo pipefail

# One-command smoke runner for Vision-RAG-CXR.
# 기본 실행:
#   bash scripts/run_one_click_smoke_and_prompt_check.sh
#
# 자주 바꾸는 옵션:
#   GPU_IDS=1,2,3,4,6 SAMPLE_SIZE=8 bash scripts/run_one_click_smoke_and_prompt_check.sh
#   RUN_REAL_SMOKE=1 bash scripts/run_one_click_smoke_and_prompt_check.sh

RUN_NAME="${RUN_NAME:-one_click_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-logs/${RUN_NAME}}"
BASE_PY="${BASE_PY:-python}"
LLM_PY="${LLM_PY:-/root/miniconda3/envs/llm_rag/bin/python}"
GPU_IDS="${GPU_IDS:-1,2,3,4,6}"
CHEXBERT_GPU="${CHEXBERT_GPU:-1}"
SAMPLE_SIZE="${SAMPLE_SIZE:-8}"
ABNORMAL_FRACTION="${ABNORMAL_FRACTION:-0.5}"
MARGIN="${MARGIN:-0.05}"
RUN_FAST_SMOKE="${RUN_FAST_SMOKE:-1}"
RUN_PROMPT_PRESERVATION="${RUN_PROMPT_PRESERVATION:-1}"
RUN_REAL_SMOKE="${RUN_REAL_SMOKE:-0}"

PAIRED_CSV="outputs/e2e_smoke/preprocessed/indiana_paired_samples.csv"
CHEXBERT_INPUT="outputs/production/preprocessed/chexbert_input_reports.csv"
CHEXBERT_OUTPUT="outputs/production/preprocessed/chexbert_output_labels.csv"
CHEXBERT_MERGED="outputs/production/preprocessed/indiana_paired_samples_chexbert.csv"
PRODUCTION_SPLIT_DIR="outputs/production/splits"
PRODUCTION_SPLIT="${PRODUCTION_SPLIT_DIR}/split_seed_0.csv"
PROMPT_OUT_DIR="${PROMPT_OUT_DIR:-outputs/${RUN_NAME}/prompt_preservation_sample}"

mkdir -p "${LOG_ROOT}" outputs/production/preprocessed "${PRODUCTION_SPLIT_DIR}"

run_step() {
  local name="$1"
  shift
  local log_file="${LOG_ROOT}/${name}.log"
  echo
  echo "========== [START] ${name} ==========" | tee "${log_file}"
  echo "command: $*" | tee -a "${log_file}"
  "$@" 2>&1 | tee -a "${log_file}"
  local status=${PIPESTATUS[0]}
  if [[ ${status} -ne 0 ]]; then
    echo "========== [FAILED] ${name} status=${status} ==========" | tee -a "${log_file}"
    exit ${status}
  fi
  echo "========== [DONE] ${name} ==========" | tee -a "${log_file}"
}

check_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[MISSING] ${path}"
    return 1
  fi
  echo "[OK] ${path}"
}

echo "RUN_NAME=${RUN_NAME}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "GPU_IDS=${GPU_IDS}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "RUN_FAST_SMOKE=${RUN_FAST_SMOKE}"
echo "RUN_PROMPT_PRESERVATION=${RUN_PROMPT_PRESERVATION}"
echo "RUN_REAL_SMOKE=${RUN_REAL_SMOKE}"

if [[ "${RUN_FAST_SMOKE}" == "1" ]]; then
  run_step e2e_fast_smoke \
    "${BASE_PY}" -u scripts/00_run_e2e_pipeline.py \
    --config configs/pipeline/e2e_smoke.yaml
fi

# CheXbert production labels are expensive, so reuse them when present.
if [[ ! -f "${CHEXBERT_MERGED}" ]]; then
  echo
  echo "[INFO] ${CHEXBERT_MERGED} not found. Building CheXbert production labels first."

  if [[ ! -f "${PAIRED_CSV}" ]]; then
    echo "[INFO] paired smoke CSV missing, running fast smoke preprocessing path first."
    run_step e2e_fast_smoke_for_paired_csv \
      "${BASE_PY}" -u scripts/00_run_e2e_pipeline.py \
      --config configs/pipeline/e2e_smoke.yaml
  fi

  run_step export_chexbert_input \
    "${BASE_PY}" -u scripts/11_export_chexbert_input.py \
    --paired_csv "${PAIRED_CSV}" \
    --out_csv "${CHEXBERT_INPUT}" \
    --source impression

  if [[ ! -f "/root/CheXbert/run_chexbert.py" ]]; then
    echo "[ERROR] /root/CheXbert/run_chexbert.py not found. Cannot build production CheXbert labels."
    exit 2
  fi
  if [[ ! -f "/root/CheXbert/chexbert.pth" ]]; then
    echo "[ERROR] /root/CheXbert/chexbert.pth not found. Cannot build production CheXbert labels."
    exit 2
  fi

  run_step run_chexbert \
    env CUDA_VISIBLE_DEVICES="${CHEXBERT_GPU}" \
    "${LLM_PY}" -u /root/CheXbert/run_chexbert.py \
    --input "$(pwd)/${CHEXBERT_INPUT}" \
    --output "$(pwd)/${CHEXBERT_OUTPUT}" \
    --model_path /root/CheXbert/chexbert.pth

  run_step merge_chexbert_labels \
    "${BASE_PY}" -u scripts/10_merge_chexbert_labels.py \
    --paired_csv "${PAIRED_CSV}" \
    --chexbert_csv "${CHEXBERT_OUTPUT}" \
    --out_csv "${CHEXBERT_MERGED}" \
    --uncertain_policy u_one
else
  echo
  echo "[SKIP] CheXbert merged CSV already exists: ${CHEXBERT_MERGED}"
fi

run_step create_production_split \
  "${BASE_PY}" -u scripts/02_create_stratified_split.py \
  --data_csv "${CHEXBERT_MERGED}" \
  --config configs/split/stratified_70_30.yaml \
  --out_dir "${PRODUCTION_SPLIT_DIR}"

if [[ "${RUN_PROMPT_PRESERVATION}" == "1" ]]; then
  run_step prompt_preservation_sampling_ttest \
    env CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=src \
    "${LLM_PY}" -u scripts/12_sample_prompt_preservation_ttest.py \
    --split_csv "${PRODUCTION_SPLIT}" \
    --generator_config configs/models/medgemma_real.yaml \
    --out_dir "${PROMPT_OUT_DIR}" \
    --sample_size "${SAMPLE_SIZE}" \
    --abnormal_fraction "${ABNORMAL_FRACTION}" \
    --margin "${MARGIN}"
fi

# Optional: current real smoke checks real MedGemma/BiomedCLIP wiring.
# It is not yet the final MeSH/Problems-controlled k-shot ablation design.
if [[ "${RUN_REAL_SMOKE}" == "1" ]]; then
  run_step e2e_real_smoke_optional \
    env CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=src \
    "${LLM_PY}" -u scripts/00_run_e2e_pipeline.py \
    --config configs/pipeline/e2e_real_smoke.yaml
fi

echo
echo "========== [SUMMARY] =========="
echo "logs: ${LOG_ROOT}"
if [[ -f outputs/e2e_smoke/summary/final_report.md ]]; then
  echo "fast smoke report: outputs/e2e_smoke/summary/final_report.md"
fi
if [[ -f "${PROMPT_OUT_DIR}/sample_metric_summary.json" ]]; then
  echo "prompt preservation metrics: ${PROMPT_OUT_DIR}/sample_metric_summary.json"
fi
if [[ -f "${PROMPT_OUT_DIR}/sample_audit.html" ]]; then
  echo "prompt preservation audit html: ${PROMPT_OUT_DIR}/sample_audit.html"
fi
check_file "${CHEXBERT_MERGED}" >/dev/null
check_file "${PRODUCTION_SPLIT}" >/dev/null
echo "All requested smoke checks completed."
