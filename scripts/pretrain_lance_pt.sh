#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LANCE_SOURCE_ROOT="${LANCE_SOURCE_ROOT:-/mnt/qs/Lance}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/qs/models/bytedance-research/Lance}"
QWEN_PATH="${QWEN_PATH:-}"
VIT_PATH="${VIT_PATH:-${MODEL_ROOT}/Qwen2.5-VL-ViT}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${MODEL_ROOT}/Wan2.2_VAE.pth}"
LANCE_IMAGE_MODEL_PATH="${LANCE_IMAGE_MODEL_PATH:-${MODEL_ROOT}/Lance_3B}"
LANCE_VIDEO_MODEL_PATH="${LANCE_VIDEO_MODEL_PATH:-${MODEL_ROOT}/Lance_3B_Video}"
DATASET_CONFIG_FILE="${DATASET_CONFIG_FILE:-${LANCE_SOURCE_ROOT}/config/train_local/unified.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_REPLICATE="${NUM_REPLICATE:-1}"
NUM_SHARD="${NUM_SHARD:-${NPROC_PER_NODE}}"
TRAINING_MANIFEST="${TRAINING_MANIFEST:-${REPO_ROOT}/results/lance-pt-manifest.json}"
RUN_MANIFEST="${RUN_MANIFEST:-${REPO_ROOT}/results/lance-pt-run.json}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${REPO_ROOT}/outputs}"
WANDB_NAME="${WANDB_NAME:-lance-pt-ascend}"
TOTAL_STEPS=350000
WARMUP_STEPS=2500
EXPECTED_NUM_TOKENS=44000
MAX_NUM_TOKENS=50000
MAX_NUM_TOKENS_PER_SAMPLE=40000
ADAPTER_FLAGS=()

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
    ADAPTER_FLAGS+=(--smoke-test)
    TOTAL_STEPS=20
    WARMUP_STEPS=2
    EXPECTED_NUM_TOKENS=1024
    MAX_NUM_TOKENS=1280
    MAX_NUM_TOKENS_PER_SAMPLE=768
fi
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    ADAPTER_FLAGS+=(--preflight-only)
fi

require_directory() {
    local name="$1"
    local path="$2"
    if [[ ! -d "${path}" ]]; then
        echo "Missing ${name} directory: ${path}" >&2
        return 1
    fi
}

require_file() {
    local name="$1"
    local path="$2"
    if [[ ! -f "${path}" ]]; then
        echo "Missing ${name} file: ${path}" >&2
        return 1
    fi
}

if [[ -z "${QWEN_PATH}" ]]; then
    echo "QWEN_PATH is required for paper PT initialization." >&2
    echo "It is not included in ${MODEL_ROOT}; do not use Lance_3B or Lance_3B_Video instead." >&2
    exit 1
fi
if (( NUM_REPLICATE * NUM_SHARD != NPROC_PER_NODE )); then
    echo "NUM_REPLICATE * NUM_SHARD must equal NPROC_PER_NODE for this single-node launcher." >&2
    exit 1
fi

require_directory "Lance source" "${LANCE_SOURCE_ROOT}"
require_directory "model root" "${MODEL_ROOT}"
require_directory "Qwen2.5-VL initialization" "${QWEN_PATH}"
require_directory "Qwen2.5-VL ViT" "${VIT_PATH}"
require_file "Qwen2.5-VL ViT config" "${VIT_PATH}/config.json"
require_file "Qwen2.5-VL ViT weights" "${VIT_PATH}/vit.safetensors"
require_file "Wan2.2 VAE" "${WAN_VAE_PATH}"
require_file "PackedDataset config" "${DATASET_CONFIG_FILE}"

# The released Lance VAE loader reads config/path_default.yaml instead of a CLI
# argument. Expose the configured weight through its gitignored downloads path
# without modifying tracked files in the clean Lance checkout.
LANCE_WAN_VAE_PATH="${LANCE_SOURCE_ROOT}/downloads/Wan2.2_VAE.pth"
if [[ ! -e "${LANCE_WAN_VAE_PATH}" && ! -L "${LANCE_WAN_VAE_PATH}" ]]; then
    mkdir -p "$(dirname "${LANCE_WAN_VAE_PATH}")"
    ln -s "${WAN_VAE_PATH}" "${LANCE_WAN_VAE_PATH}"
elif [[ ! -f "${LANCE_WAN_VAE_PATH}" ]]; then
    echo "Invalid Lance Wan2.2 VAE path: ${LANCE_WAN_VAE_PATH}" >&2
    exit 1
fi

if [[ ! -f "${TRAINING_MANIFEST}" || "${REGENERATE_TRAINING_MANIFEST:-0}" == "1" ]]; then
    python "${REPO_ROOT}/prepare_lance_training.py" \
        --stage pt \
        --init-mode qwen2_5_vl \
        --init-path "${QWEN_PATH}" \
        --variant video \
        --world-size "${NPROC_PER_NODE}" \
        --dataset-manifest "${DATASET_CONFIG_FILE}" \
        --output "${TRAINING_MANIFEST}"
fi

echo "Lance source: ${LANCE_SOURCE_ROOT}"
echo "Model root: ${MODEL_ROOT}"
echo "Qwen initialization: ${QWEN_PATH}"
echo "ViT: ${VIT_PATH}"
echo "Wan2.2 VAE: ${WAN_VAE_PATH}"
echo "Lance image checkpoint (not used by PT): ${LANCE_IMAGE_MODEL_PATH}"
echo "Lance video checkpoint (not used by PT): ${LANCE_VIDEO_MODEL_PATH}"
echo "Dataset config: ${DATASET_CONFIG_FILE}"
echo "Training manifest: ${TRAINING_MANIFEST}"

torchrun --nproc_per_node "${NPROC_PER_NODE}" \
    "${REPO_ROOT}/pretrain_lance.py" \
    --training-manifest "${TRAINING_MANIFEST}" \
    --lance-source-root "${LANCE_SOURCE_ROOT}" \
    --run-manifest "${RUN_MANIFEST}" \
    "${ADAPTER_FLAGS[@]}" \
    -- \
    --llm_path "${QWEN_PATH}" \
    --vit_path "${VIT_PATH}" \
    --init_from_vlm_checkpoint true \
    --load_from_lance_checkpoint false \
    --copy_init_moe true \
    --layer_module Qwen2MoTDecoderLayer \
    --vit_type qwen2_5_vl \
    --vae_model_type wan \
    --max_num_frames 121 \
    --max_latent_size 64 \
    --latent_patch_size 1 1 1 \
    --visual_gen true \
    --visual_und true \
    --freeze_vit true \
    --freeze_vae true \
    --freeze_llm false \
    --freeze_llm_embed_tokens false \
    --freeze_vit_connector false \
    --freeze_und_params false \
    --freeze_und false \
    --use_ema true \
    --use_flex false \
    --cpu_offload false \
    --sharding_strategy HYBRID_SHARD \
    --backward_prefetch BACKWARD_PRE \
    --dataset_config_file "${DATASET_CONFIG_FILE}" \
    --total_steps "${TOTAL_STEPS}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --lr 1e-4 \
    --lr_scheduler constant \
    --expected_num_tokens "${EXPECTED_NUM_TOKENS}" \
    --max_num_tokens "${MAX_NUM_TOKENS}" \
    --max_num_tokens_per_sample "${MAX_NUM_TOKENS_PER_SAMPLE}" \
    --timestep_shift 1.0 \
    --ce_weight 0.25 \
    --mse_weight 1.0 \
    --text_cond_dropout_prob 0.1 \
    --vae_cond_dropout_prob 0.0 \
    --vit_cond_dropout_prob 0.0 \
    --global_seed 2025 \
    --beta1 0.9 \
    --beta2 0.95 \
    --eps 1e-15 \
    --max_grad_norm 1.0 \
    --ema 0.9999 \
    --min_lr 1e-7 \
    --num_replicate "${NUM_REPLICATE}" \
    --num_shard "${NUM_SHARD}" \
    --outputs_dir "${OUTPUTS_DIR}" \
    --wandb_name "${WANDB_NAME}" \
    --wandb_offline true
