#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_CONFIG_FILE_WAS_SET=0
TRAINING_MANIFEST_WAS_SET=0
if [[ -n "${DATASET_CONFIG_FILE:-}" ]]; then
    DATASET_CONFIG_FILE_WAS_SET=1
fi
if [[ -n "${TRAINING_MANIFEST:-}" ]]; then
    TRAINING_MANIFEST_WAS_SET=1
fi

LANCE_SOURCE_ROOT="${LANCE_SOURCE_ROOT:-/mnt/qs/Lance}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/qs/models/bytedance-research/Lance}"
QWEN_PATH="${QWEN_PATH:-/mnt/qs/models/Qwen/Qwen2.5-VL-3B-Instruct/}"
VIT_PATH="${VIT_PATH:-${MODEL_ROOT}/Qwen2.5-VL-ViT}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${MODEL_ROOT}/Wan2.2_VAE.pth}"
LANCE_IMAGE_MODEL_PATH="${LANCE_IMAGE_MODEL_PATH:-${MODEL_ROOT}/Lance_3B}"
LANCE_VIDEO_MODEL_PATH="${LANCE_VIDEO_MODEL_PATH:-${MODEL_ROOT}/Lance_3B_Video}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/qs/dataset/bytedance-research/Lance_example_dataset/}"
DATASET_CONFIG_FILE="${DATASET_CONFIG_FILE:-${LANCE_SOURCE_ROOT}/config/train_local/unified.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_REPLICATE="${NUM_REPLICATE:-1}"
NUM_SHARD="${NUM_SHARD:-${NPROC_PER_NODE}}"
NUM_WORKERS="${NUM_WORKERS:-}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
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
    if [[ "${DATASET_CONFIG_FILE_WAS_SET}" == "0" ]]; then
        DATASET_CONFIG_FILE="${LANCE_SOURCE_ROOT}/config/train_local/t2i_local.yaml"
    fi
    NUM_WORKERS="${NUM_WORKERS:-0}"
    TOTAL_STEPS=20
    WARMUP_STEPS=2
    EXPECTED_NUM_TOKENS=4096
    MAX_NUM_TOKENS=8192
    MAX_NUM_TOKENS_PER_SAMPLE=4096
else
    NUM_WORKERS="${NUM_WORKERS:-8}"
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

# train_local YAML files resolve their datasets/... entries relative to the
# Lance checkout. Hugging Face snapshots may either expose the task folders at
# their root or wrap them in one additional datasets/ directory.
DATASET_TREE_ROOT="${DATASET_ROOT}"
LANCE_DATASET_PATH="${LANCE_SOURCE_ROOT}/datasets"
if [[ "${DATASET_CONFIG_FILE}" == "${LANCE_SOURCE_ROOT}/config/train_local/"* ]]; then
    require_directory "Lance example dataset" "${DATASET_ROOT}"
    if [[ ! -d "${DATASET_TREE_ROOT}/text2image" && -d "${DATASET_ROOT}/datasets/text2image" ]]; then
        DATASET_TREE_ROOT="${DATASET_ROOT}/datasets"
    fi

    if [[ ! -e "${LANCE_DATASET_PATH}" && ! -L "${LANCE_DATASET_PATH}" ]]; then
        ln -s "${DATASET_TREE_ROOT}" "${LANCE_DATASET_PATH}"
    elif [[ ! -d "${LANCE_DATASET_PATH}" ]]; then
        echo "Invalid Lance dataset path: ${LANCE_DATASET_PATH}" >&2
        exit 1
    fi
fi

EXPECTED_DATASET_FILES=()
case "${DATASET_CONFIG_FILE}" in
    "${LANCE_SOURCE_ROOT}/config/train_local/unified.yaml")
        EXPECTED_DATASET_FILES=(
            text2image/local_256.parquet
            text2video/local_128.parquet
            image2image/local_256.parquet
            video2video/local_64.parquet
            image2text/local_256.parquet
            video2text/local_256.parquet
        )
        ;;
    "${LANCE_SOURCE_ROOT}/config/train_local/t2i_local.yaml")
        EXPECTED_DATASET_FILES=(text2image/local_256.parquet)
        ;;
esac
for relative_path in "${EXPECTED_DATASET_FILES[@]}"; do
    require_file "Lance example dataset" "${LANCE_DATASET_PATH}/${relative_path}"
done

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

if [[ "${TRAINING_MANIFEST_WAS_SET}" == "0" || ! -f "${TRAINING_MANIFEST}" || "${REGENERATE_TRAINING_MANIFEST:-0}" == "1" ]]; then
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
echo "Dataset root: ${DATASET_TREE_ROOT}"
echo "Lance dataset view: ${LANCE_DATASET_PATH}"
echo "Dataset config: ${DATASET_CONFIG_FILE}"
echo "DataLoader workers per rank: ${NUM_WORKERS}"
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
    --use_flex true \
    --cpu_offload false \
    --sharding_strategy HYBRID_SHARD \
    --backward_prefetch BACKWARD_PRE \
    --dataset_config_file "${DATASET_CONFIG_FILE}" \
    --num_workers "${NUM_WORKERS}" \
    --prefetch_factor "${PREFETCH_FACTOR}" \
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
