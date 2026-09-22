#!/usr/bin/env bash
# Generate and score all Stage-C training images on one selectable physical NPU.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"

sample_count="${LANCE_OVERFIT_MULTI_SAMPLES:-32}"
device_id="${DEVICE_ID:-0}"
if [[ ! "${device_id}" =~ ^[0-9]+$ ]]; then
  echo "DEVICE_ID must be a non-negative integer, got: ${device_id}" >&2
  exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES="${device_id}"
export PYTHONPATH="${REPO_ROOT}/MindSpeed:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NON_MEGATRON=true
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

CHECKPOINT="${CHECKPOINT:-/mnt/models/outputs/lance-qwen3-06b-t2i-overfit-stage-c-uniform-${sample_count}/iter_0005000}"
QWEN_PATH="${QWEN_PATH:-/mnt/models/MODELS/Qwen3-0.6B}"
VAE_PATH="${VAE_PATH:-/mnt/models/MODELS/bytedance-research/Lance/Wan2.2_VAE.pth}"
DATASET_ROOT="${LANCE_OVERFIT_DATASET_ROOT:-/mnt/models/DATA_INIT/MULTI/T2I/Mobile-O-Pre-Train-GEN}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
CHECKPOINT_ROOT="$(dirname "${CHECKPOINT}")"
EXAMPLE_DIR="${EXAMPLE_DIR:-${CHECKPOINT_ROOT}/training-examples-${sample_count}}"
MANIFEST="${MANIFEST:-${EXAMPLE_DIR}/training_examples.json}"
NUM_STEPS="${NUM_STEPS:-50}"
TIMESTEP_SCHEDULE="${TIMESTEP_SCHEDULE:-linear}"
WEIGHTS="${WEIGHTS:-model}"
BASE_SEED="${BASE_SEED:-2025}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/eval-${CHECKPOINT_NAME}-training-${sample_count}-${NUM_STEPS}step-${TIMESTEP_SCHEDULE}-${WEIGHTS}}"

if [[ "${WEIGHTS}" != "model" && "${WEIGHTS}" != "ema" ]]; then
  echo "WEIGHTS must be model or ema, got: ${WEIGHTS}" >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  python "${HERE}/resolve_lance_t2i_overfit_examples.py" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${EXAMPLE_DIR}" \
    --sample-count "${sample_count}"
fi

args=(
  --checkpoint "${CHECKPOINT}"
  --manifest "${MANIFEST}"
  --qwen-path "${QWEN_PATH}"
  --vae-path "${VAE_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --device npu:0
  --num-steps "${NUM_STEPS}"
  --timestep-shift 1.0
  --timestep-schedule "${TIMESTEP_SCHEDULE}"
  --seed "${BASE_SEED}"
  --latent-patch-size 1 2 2
  --max-latent-size 64
  --max-num-frames 121
)
if [[ "${WEIGHTS}" == "ema" ]]; then
  args+=(--ema-weights)
fi

echo "Stage C inference physical NPU: ${device_id} (logical npu:0)"
echo "Training examples: ${MANIFEST}"
echo "Output: ${OUTPUT_DIR}"
cd "${REPO_ROOT}"
exec python inference_lance_stage_c_training_set.py "${args[@]}"
