#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-all}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6010}"

LANCE_MODEL_ROOT="${LANCE_MODEL_ROOT:-/mnt/qs/models/bytedance-research/Lance}"
QWEN_PATH="${QWEN_PATH:-/mnt/qs/models/Qwen/Qwen2.5-VL-3B-Instruct}"
VIT_PATH="${VIT_PATH:-${LANCE_MODEL_ROOT}/Qwen2.5-VL-ViT}"
VAE_PATH="${VAE_PATH:-${LANCE_MODEL_ROOT}/Wan2.2_VAE.pth}"
LANCE_EXAMPLE_DATA="${LANCE_EXAMPLE_DATA:-/mnt/qs/datasets/bytedance-research/Lance_example_dataset}"

export LANCE_INIT_DCP="${LANCE_INIT_DCP:-${REPO_ROOT}/checkpoints/lance-qwen-init-dcp}"
export LANCE_PREPARED_SAMPLES="${LANCE_PREPARED_SAMPLES:-${REPO_ROOT}/datasets/lance-prepared-samples}"
export LANCE_PREENCODED_DATA="${LANCE_PREENCODED_DATA:-${REPO_ROOT}/datasets/lance-preencoded}"
LANCE_PREPARE_SEED="${LANCE_PREPARE_SEED:-2025}"
LANCE_TEXT_COND_DROPOUT="${LANCE_TEXT_COND_DROPOUT:-0.1}"
LANCE_EXPECTED_TOKENS="${LANCE_EXPECTED_TOKENS:-44000}"
LANCE_MAX_TOKENS="${LANCE_MAX_TOKENS:-50000}"
LANCE_MAX_SAMPLE_TOKENS="${LANCE_MAX_SAMPLE_TOKENS:-40000}"
LANCE_LATENT_PATCH_T="${LANCE_LATENT_PATCH_T:-1}"
LANCE_LATENT_PATCH_H="${LANCE_LATENT_PATCH_H:-2}"
LANCE_LATENT_PATCH_W="${LANCE_LATENT_PATCH_W:-2}"
LANCE_MAX_LATENT_SIZE="${LANCE_MAX_LATENT_SIZE:-64}"
LANCE_MAX_NUM_FRAMES="${LANCE_MAX_NUM_FRAMES:-121}"
# train_local/unified.yaml declares six independent groups with equal weight.
LANCE_TASK_WEIGHTS="${LANCE_TASK_WEIGHTS:-t2i=1,t2v=1,i2i=1,v2v=1,i2t=1,v2t=1}"

check_path() {
    if [[ ! -e "$1" ]]; then
        echo "Missing required path: $1" >&2
        exit 1
    fi
}

prepare_init() {
    check_path "${QWEN_PATH}"
    python "${REPO_ROOT}/scripts/prepare_lance_native_init.py" \
        --qwen-path "${QWEN_PATH}" \
        --output "${LANCE_INIT_DCP}" \
        --variant video \
        --latent-patch-size \
            "${LANCE_LATENT_PATCH_T}" "${LANCE_LATENT_PATCH_H}" "${LANCE_LATENT_PATCH_W}" \
        --max-latent-size "${LANCE_MAX_LATENT_SIZE}" \
        --max-num-frames "${LANCE_MAX_NUM_FRAMES}"
}

prepare_samples() {
    check_path "${QWEN_PATH}"
    check_path "${VIT_PATH}"
    check_path "${VAE_PATH}"
    check_path "${LANCE_EXAMPLE_DATA}"
    local extra_args=()
    if [[ -n "${LANCE_MAX_SAMPLES_PER_RANK:-}" ]]; then
        extra_args+=(--max-samples "${LANCE_MAX_SAMPLES_PER_RANK}")
    fi
    if [[ -n "${LANCE_MAX_SAMPLES_PER_TASK:-}" ]]; then
        extra_args+=(--max-samples-per-task "${LANCE_MAX_SAMPLES_PER_TASK}")
    fi
    torchrun \
        --nproc_per_node "${NPROC_PER_NODE}" \
        --master_addr "${MASTER_ADDR}" \
        --master_port "${MASTER_PORT}" \
        "${REPO_ROOT}/scripts/prepare_lance_native_data.py" \
        --dataset-root "${LANCE_EXAMPLE_DATA}" \
        --qwen-path "${QWEN_PATH}" \
        --vit-path "${VIT_PATH}" \
        --vae-path "${VAE_PATH}" \
        --output "${LANCE_PREPARED_SAMPLES}" \
        --variant video \
        --latent-patch-size \
            "${LANCE_LATENT_PATCH_T}" "${LANCE_LATENT_PATCH_H}" "${LANCE_LATENT_PATCH_W}" \
        --max-latent-size "${LANCE_MAX_LATENT_SIZE}" \
        --max-num-frames "${LANCE_MAX_NUM_FRAMES}" \
        --seed "${LANCE_PREPARE_SEED}" \
        --text-cond-dropout-prob "${LANCE_TEXT_COND_DROPOUT}" \
        "${extra_args[@]}"
}

pack_samples() {
    check_path "${LANCE_PREPARED_SAMPLES}"
    python "${REPO_ROOT}/scripts/pack_lance_native_data.py" \
        --input "${LANCE_PREPARED_SAMPLES}" \
        --output "${LANCE_PREENCODED_DATA}" \
        --llm-config "${QWEN_PATH}/config.json" \
        --variant video \
        --latent-patch-size \
            "${LANCE_LATENT_PATCH_T}" "${LANCE_LATENT_PATCH_H}" "${LANCE_LATENT_PATCH_W}" \
        --max-latent-size "${LANCE_MAX_LATENT_SIZE}" \
        --max-num-frames "${LANCE_MAX_NUM_FRAMES}" \
        --expected-tokens "${LANCE_EXPECTED_TOKENS}" \
        --max-tokens "${LANCE_MAX_TOKENS}" \
        --max-sample-tokens "${LANCE_MAX_SAMPLE_TOKENS}" \
        --seed "${LANCE_PREPARE_SEED}" \
        --task-weights "${LANCE_TASK_WEIGHTS}"
}

case "${MODE}" in
    init)
        prepare_init
        ;;
    encode)
        prepare_samples
        ;;
    pack)
        pack_samples
        ;;
    all)
        prepare_init
        prepare_samples
        pack_samples
        ;;
    *)
        echo "Usage: $0 {init|encode|pack|all}" >&2
        exit 2
        ;;
esac
