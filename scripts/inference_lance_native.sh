#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LANCE_CHECKPOINT="${LANCE_CHECKPOINT:-${REPO_ROOT}/outputs/lance-native-pt}"
QWEN_PATH="${QWEN_PATH:-/mnt/qs/models/Qwen/Qwen2.5-VL-3B-Instruct}"
VAE_PATH="${VAE_PATH:-/mnt/qs/models/bytedance-research/Lance/Wan2.2_VAE.pth}"
VIT_PATH="${VIT_PATH:-/mnt/qs/models/bytedance-research/Lance/Qwen2.5-VL-ViT}"
LANCE_OFFICIAL_ROOT="${LANCE_OFFICIAL_ROOT:-${REPO_ROOT}/../Lance}"
LANCE_TASK="${LANCE_TASK:-t2i}"
LANCE_PROMPT="${LANCE_PROMPT:-A red panda wearing sunglasses, cinematic lighting, highly detailed.}"
LANCE_CONFIG_PATH="${LANCE_CONFIG_PATH:-${LANCE_OFFICIAL_ROOT}/config/examples/x2t_image_example.json}"
LANCE_OUTPUT_DIR="${LANCE_OUTPUT_DIR:-${REPO_ROOT}/outputs/lance-native-inference}"
LANCE_DEVICE="${LANCE_DEVICE:-npu:0}"

export NON_MEGATRON=true
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

ARGS=(
    --checkpoint "${LANCE_CHECKPOINT}"
    --qwen-path "${QWEN_PATH}"
    --task "${LANCE_TASK}"
    --output-dir "${LANCE_OUTPUT_DIR}"
    --device "${LANCE_DEVICE}"
    --latent-patch-size 1 2 2
    --max-latent-size 64
    --max-num-frames 121
)

if [[ "${LANCE_TASK}" == "i2t" ]]; then
    ARGS+=(--vit-path "${VIT_PATH}" --config-path "${LANCE_CONFIG_PATH}")
else
    ARGS+=(--vae-path "${VAE_PATH}")
    if [[ -n "${LANCE_PROMPT_FILE:-}" ]]; then
        ARGS+=(--prompt-file "${LANCE_PROMPT_FILE}")
    else
        ARGS+=(--prompt "${LANCE_PROMPT}")
    fi
fi
if [[ "${LANCE_USE_MODEL_WEIGHTS:-0}" == "1" ]]; then
    ARGS+=(--model-weights)
fi
if [[ "${LANCE_DRY_RUN:-0}" == "1" ]]; then
    ARGS+=(--dry-run)
fi

python "${REPO_ROOT}/inference_lance_native.py" "${ARGS[@]}" "$@"
