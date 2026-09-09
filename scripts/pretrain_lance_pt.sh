#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LANCE_SOURCE_ROOT="${LANCE_SOURCE_ROOT:-${REPO_ROOT}/../Lance}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_REPLICATE="${NUM_REPLICATE:-1}"
NUM_SHARD="${NUM_SHARD:-${NPROC_PER_NODE}}"
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

: "${TRAINING_MANIFEST:?Set TRAINING_MANIFEST to the prepared PT manifest}"
: "${QWEN_PATH:?Set QWEN_PATH to the Qwen2.5-VL initialization directory}"
: "${VIT_PATH:?Set VIT_PATH to the extracted Qwen2.5-VL ViT directory}"
: "${DATASET_CONFIG_FILE:?Set DATASET_CONFIG_FILE to the upstream PackedDataset YAML}"

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
    --wandb_name "${WANDB_NAME}"
