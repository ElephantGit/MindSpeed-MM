#!/usr/bin/env bash
# Standard comparable evaluation: exact training prompt/geometry, 50 steps, 3 seeds.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/MindSpeed:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NON_MEGATRON=true
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

CHECKPOINT="${CHECKPOINT:-/mnt/models/outputs/lance-qwen3-06b-t2i-overfit-stage-b/iter_0002000}"
QWEN_PATH="${QWEN_PATH:-/mnt/models/MODELS/Qwen3-0.6B}"
VAE_PATH="${VAE_PATH:-/mnt/models/MODELS/bytedance-research/Lance/Wan2.2_VAE.pth}"
EXAMPLE_DIR="${EXAMPLE_DIR:-/mnt/models/outputs/lance-overfit-train-example-input}"
METADATA="${METADATA:-${EXAMPLE_DIR}/training_example.json}"
PROMPT_FILE="${PROMPT_FILE:-${EXAMPLE_DIR}/prompt.json}"
TARGET_IMAGE="${TARGET_IMAGE:-${EXAMPLE_DIR}/training_target.png}"
DEVICE="${DEVICE:-npu:0}"
NUM_STEPS="${NUM_STEPS:-50}"
WEIGHTS="${WEIGHTS:-model}"
SEEDS="${SEEDS:-2025 2026 2027}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
CHECKPOINT_ROOT="$(dirname "${CHECKPOINT}")"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/eval-${CHECKPOINT_NAME}-${NUM_STEPS}step-${WEIGHTS}}"

if [[ ! -f "${METADATA}" || ! -f "${PROMPT_FILE}" || ! -f "${TARGET_IMAGE}" ]]; then
  echo "Missing training-example metadata, prompt, or target under ${EXAMPLE_DIR}." >&2
  echo "Create them with resolve_lance_t2i_overfit_example.py; it does not import MindSpeed-MM or Torch." >&2
  exit 1
fi
if [[ "${WEIGHTS}" != "model" && "${WEIGHTS}" != "ema" ]]; then
  echo "WEIGHTS must be model or ema, got: ${WEIGHTS}" >&2
  exit 1
fi

read -r HEIGHT WIDTH < <(
  python -c 'import json,sys; x=json.load(open(sys.argv[1], encoding="utf-8")); print(x["height"], x["width"])' "${METADATA}"
)

mkdir -p "${OUTPUT_ROOT}"
predictions=()
for seed in ${SEEDS}; do
  output_dir="${OUTPUT_ROOT}/seed-${seed}"
  args=(
    --checkpoint "${CHECKPOINT}"
    --qwen-path "${QWEN_PATH}"
    --vae-path "${VAE_PATH}"
    --task t2i
    --prompt-file "${PROMPT_FILE}"
    --output-dir "${output_dir}"
    --device "${DEVICE}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --num-steps "${NUM_STEPS}"
    --timestep-shift 1.0
    --cfg-text-scale 1.0
    --cfg-renorm-type none
    --seed "${seed}"
    --latent-patch-size 1 2 2
    --max-latent-size 64
    --max-num-frames 121
  )
  if [[ "${WEIGHTS}" == "model" ]]; then
    args+=(--model-weights)
  fi
  python "${REPO_ROOT}/inference_lance_native.py" "${args[@]}"
  predictions+=(--prediction "${output_dir}/000000.png")
done

python "${REPO_ROOT}/score_lance_t2i_overfit.py" \
  --target "${TARGET_IMAGE}" \
  "${predictions[@]}" \
  --output "${OUTPUT_ROOT}/metrics.json"
