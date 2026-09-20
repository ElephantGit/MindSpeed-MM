#!/usr/bin/env bash
# Evaluate a raw-pretrained Lance checkpoint without any chat template.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"
# The Lance evaluator imports both mindspeed_mm from this checkout and the
# adjacent MindSpeed source package.  Training already exports the same path;
# keep standalone checkpoint evaluation consistent with it.
export PYTHONPATH="${REPO_ROOT}/MindSpeed:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
CHECKPOINT="${CHECKPOINT:-/mnt/models/outputs/lance-mobile-o-small-pt-2epoch/iter_0000010}"
QWEN_PATH="${QWEN_PATH:-/mnt/models/MODELS/Qwen3-0.6B}"
VAE_PATH="${VAE_PATH:-/mnt/models/MODELS/bytedance-research/Lance/Wan2.2_VAE.pth}"
PREPARED_DATA="${PREPARED_DATA:-/mnt/models/DATA_INIT/MULTI/T2I/Mobile-O-Pre-Train-preencoded-raw}"
PACKED_DATA="${PACKED_DATA:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/models/outputs/lance-mobile-o-small-pt-2epoch/eval}"
DEVICE="${DEVICE:-npu:0}"
PROMPT_FILE="${PROMPT_FILE:-${HERE}/lance_pretrain_eval_prompts.json}"
WEIGHT_ARGS=()
TEXT_DATA_ARGS=(--prepared-data "${PREPARED_DATA}")
if [[ "${MODEL_WEIGHTS:-0}" == "1" ]]; then
  WEIGHT_ARGS+=(--model-weights)
fi
if [[ -n "${PACKED_DATA}" ]]; then
  TEXT_DATA_ARGS=(--packed-data "${PACKED_DATA}")
fi

cd "${REPO_ROOT}"
python inference_lance_native.py \
  --checkpoint "${CHECKPOINT}" \
  --qwen-path "${QWEN_PATH}" \
  --vae-path "${VAE_PATH}" \
  --output-dir "${OUTPUT_DIR}/t2i" \
  --task t2i \
  --prompt-file "${PROMPT_FILE}" \
  --device "${DEVICE}" \
  --height 768 --width 768 --num-frames 1 \
  --num-steps "${T2I_STEPS:-20}" \
  --cfg-text-scale 4.0 --seed 2025 \
  "${WEIGHT_ARGS[@]}"

python "${HERE}/evaluate_lance_pretrain_text.py" \
  --checkpoint "${CHECKPOINT}" \
  --qwen-path "${QWEN_PATH}" \
  "${TEXT_DATA_ARGS[@]}" \
  --output "${OUTPUT_DIR}/text_results.json" \
  --device "${DEVICE}" \
  --samples "${TEXT_SAMPLES:-5}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-48}" \
  "${WEIGHT_ARGS[@]}"

echo "Pretraining-format evaluation complete: ${OUTPUT_DIR}"
