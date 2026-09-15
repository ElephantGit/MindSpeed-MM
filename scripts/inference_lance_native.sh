#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LANCE_CHECKPOINT="${LANCE_CHECKPOINT:-${REPO_ROOT}/outputs/lance-native-pt}"
QWEN_PATH="${QWEN_PATH:-/mnt/qs/models/Qwen/Qwen2.5-VL-3B-Instruct}"
VAE_PATH="${VAE_PATH:-/mnt/qs/models/bytedance-research/Lance/Wan2.2_VAE.pth}"
LANCE_TASK="${LANCE_TASK:-t2i}"
LANCE_PROMPT="${LANCE_PROMPT:-A red panda wearing sunglasses, cinematic lighting, highly detailed.}"
LANCE_OUTPUT_DIR="${LANCE_OUTPUT_DIR:-${REPO_ROOT}/outputs/lance-native-inference}"
LANCE_DEVICE="${LANCE_DEVICE:-npu:0}"

export NON_MEGATRON=true
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

ARGS=(
    --checkpoint "${LANCE_CHECKPOINT}"
    --qwen-path "${QWEN_PATH}"
    --vae-path "${VAE_PATH}"
    --task "${LANCE_TASK}"
    --output-dir "${LANCE_OUTPUT_DIR}"
    --device "${LANCE_DEVICE}"
    --latent-patch-size 1 2 2
    --max-latent-size 64
    --max-num-frames 121
)

if [[ -n "${LANCE_PROMPT_FILE:-}" ]]; then
    ARGS+=(--prompt-file "${LANCE_PROMPT_FILE}")
else
    ARGS+=(--prompt "${LANCE_PROMPT}")
fi
if [[ "${LANCE_USE_MODEL_WEIGHTS:-0}" == "1" ]]; then
    ARGS+=(--model-weights)
fi
if [[ "${LANCE_DRY_RUN:-0}" == "1" ]]; then
    ARGS+=(--dry-run)
fi

python "${REPO_ROOT}/inference_lance_native.py" "${ARGS[@]}" "$@"
