#!/usr/bin/env bash
# Prepare pure-Qwen3 T2I data for the three overfit stages.
#   one   -> one image, shared by stages A and B
#   multi -> 32 images by default, used by stage C
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"
MODE="${1:-}"

case "${MODE}" in
    one)
        max_samples="${LANCE_OVERFIT_ONE_SAMPLES:-1}"
        prepared_default="/mnt/models/DATA_INIT/MULTI/T2I/Qwen3-0.6B-overfit-1-prepared"
        packed_default="/mnt/models/DATA_INIT/MULTI/T2I/Qwen3-0.6B-overfit-1-packed"
        ;;
    multi)
        max_samples="${LANCE_OVERFIT_MULTI_SAMPLES:-32}"
        prepared_default="/mnt/models/DATA_INIT/MULTI/T2I/Qwen3-0.6B-overfit-${max_samples}-prepared"
        packed_default="/mnt/models/DATA_INIT/MULTI/T2I/Qwen3-0.6B-overfit-${max_samples}-packed"
        ;;
    *)
        echo "Usage: $0 {one|multi}" >&2
        exit 2
        ;;
esac

QWEN_PATH="${QWEN_PATH:-/mnt/models/MODELS/Qwen3-0.6B}"
DATASET_ROOT="${LANCE_OVERFIT_DATASET_ROOT:-/mnt/models/DATA_INIT/MULTI/T2I/Mobile-O-Pre-Train-GEN}"
VIT_PATH="${VIT_PATH:-/mnt/models/MODELS/Qwen2.5-VL-3B-Instruct}"
VAE_PATH="${VAE_PATH:-/mnt/models/MODELS/bytedance-research/Lance/Wan2.2_VAE.pth}"
PREPARED_OUTPUT="${LANCE_OVERFIT_PREPARED_OUTPUT:-${prepared_default}}"
PACKED_OUTPUT="${LANCE_OVERFIT_PACKED_OUTPUT:-${packed_default}}"
PREPARE_DEVICE="${LANCE_OVERFIT_PREPARE_DEVICE:-0}"

for path in "${QWEN_PATH}" "${DATASET_ROOT}" "${VIT_PATH}" "${VAE_PATH}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required input: ${path}" >&2
        exit 1
    fi
done
for output in "${PREPARED_OUTPUT}" "${PACKED_OUTPUT}"; do
    if [[ -e "${output}" ]]; then
        echo "Refusing to overwrite existing output: ${output}" >&2
        exit 1
    fi
done

cd "${REPO_ROOT}"
ASCEND_RT_VISIBLE_DEVICES="${PREPARE_DEVICE}" torchrun \
    --nproc_per_node 1 \
    --master_addr 127.0.0.1 \
    --master_port "${LANCE_OVERFIT_PREPARE_PORT:-6010}" \
    scripts/prepare_lance_native_data.py \
    --dataset-root "${DATASET_ROOT}" \
    --qwen-path "${QWEN_PATH}" \
    --vit-path "${VIT_PATH}" \
    --vae-path "${VAE_PATH}" \
    --output "${PREPARED_OUTPUT}" \
    --variant video \
    --latent-patch-size 1 2 2 \
    --max-latent-size 64 \
    --max-num-frames 121 \
    --max-samples "${max_samples}" \
    --seed 2025 \
    --text-cond-dropout-prob 0.0 \
    --emit-tasks t2i \
    --fail-fast

# expected-tokens=1 deliberately emits one packed file per source image.  It
# gives stage C independently shuffled prompts and makes the world-size check
# in the launchers meaningful.
python scripts/pack_lance_native_data.py \
    --input "${PREPARED_OUTPUT}" \
    --output "${PACKED_OUTPUT}" \
    --llm-config "${QWEN_PATH}/config.json" \
    --variant video \
    --latent-patch-size 1 2 2 \
    --max-latent-size 64 \
    --max-num-frames 121 \
    --expected-tokens 1 \
    --max-tokens 50000 \
    --max-sample-tokens 40000 \
    --seed 2025 \
    --task-weights t2i=1

echo "Prepared ${max_samples} pure-T2I samples in ${PACKED_OUTPUT}"
