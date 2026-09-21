#!/usr/bin/env bash
# Shared launcher for the Qwen3-0.6B T2I overfit diagnostics.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"

: "${LANCE_OVERFIT_STAGE:?stage launcher must set LANCE_OVERFIT_STAGE}"
: "${LANCE_PREENCODED_DATA:?stage launcher must set LANCE_PREENCODED_DATA}"
: "${LANCE_OUTPUT_DIR:?stage launcher must set LANCE_OUTPUT_DIR}"
: "${LANCE_OVERFIT_RESAMPLE_TIMESTEPS:?stage launcher must set timestep policy}"
: "${LANCE_OVERFIT_FIXED_NOISE_SEED:?stage launcher must set fixed noise seed or null}"
: "${LANCE_OVERFIT_DISABLE_POSTERIOR_SAMPLING:?stage launcher must set posterior policy}"
: "${LANCE_OVERFIT_SHUFFLE:?stage launcher must set shuffle policy}"

export CONFIG_FILE="${CONFIG_FILE:-${HERE}/fsdp2_t2i_overfit_qwen3_06b.yaml}"
export QWEN_PATH="${QWEN_PATH:-/mnt/models/MODELS/Qwen3-0.6B}"
export LANCE_LOAD_DCP="${LANCE_LOAD_DCP:-/mnt/models/MODELS/lance-qwen3-06b-init-dcp}"

# A single rank makes the one-packed-batch A/B experiments valid.  Stage C
# may override these, but it must provide at least world_size packed files.
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-6011}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

export LANCE_TRAIN_TOKENS=0
export LANCE_SAVE_INTERVAL_TOKENS=0
export LANCE_GRADIENT_CHECKPOINTING="${LANCE_GRADIENT_CHECKPOINTING:-false}"
export LANCE_RESHARD_AFTER_FORWARD="${LANCE_RESHARD_AFTER_FORWARD:-false}"
export LANCE_GRADIENT_ACCUMULATION_STEPS="${LANCE_GRADIENT_ACCUMULATION_STEPS:-1}"
export LANCE_NUM_WORKERS="${LANCE_NUM_WORKERS:-0}"
export LANCE_LATENT_PATCH_T="${LANCE_LATENT_PATCH_T:-1}"
export LANCE_LATENT_PATCH_H="${LANCE_LATENT_PATCH_H:-2}"
export LANCE_LATENT_PATCH_W="${LANCE_LATENT_PATCH_W:-2}"
export LANCE_MAX_LATENT_SIZE="${LANCE_MAX_LATENT_SIZE:-64}"
export LANCE_MAX_NUM_FRAMES="${LANCE_MAX_NUM_FRAMES:-121}"
# Full-parameter checksum traces are intentionally opt-in: computing one every
# step is useful for continuity debugging but unnecessarily expensive here.
export LANCE_TRACE_FILE="${LANCE_TRACE_FILE:-}"

world_size=$((NNODES * NPROC_PER_NODE))
if [[ -d "${LANCE_PREENCODED_DATA}" ]]; then
    packed_count="$(find "${LANCE_PREENCODED_DATA}" -maxdepth 1 -type f -name '*.pt' | wc -l)"
elif [[ -f "${LANCE_PREENCODED_DATA}" ]]; then
    packed_count=1
else
    echo "Missing packed overfit data: ${LANCE_PREENCODED_DATA}" >&2
    echo "Create it with ${HERE}/prepare_lance_t2i_overfit_qwen3_06b.sh one|multi" >&2
    exit 1
fi
if (( packed_count < world_size )); then
    echo "Packed batches (${packed_count}) must be >= data-parallel world size (${world_size})." >&2
    echo "Use one rank for stage A/B or prepare more stage-C packed batches." >&2
    exit 1
fi
if [[ -d "${LANCE_OUTPUT_DIR}" ]] && [[ -n "$(find "${LANCE_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to write a diagnostic run into non-empty output: ${LANCE_OUTPUT_DIR}" >&2
    echo "Choose a new LANCE_OUTPUT_DIR.  These model-only checkpoints are not configured for optimizer resume." >&2
    exit 1
fi

echo "T2I overfit stage: ${LANCE_OVERFIT_STAGE}"
echo "Initialization: ${LANCE_LOAD_DCP}"
echo "Packed data: ${LANCE_PREENCODED_DATA} (${packed_count} batches)"
echo "Output: ${LANCE_OUTPUT_DIR}"
echo "Randomness: resample_t=${LANCE_OVERFIT_RESAMPLE_TIMESTEPS}, fixed_noise=${LANCE_OVERFIT_FIXED_NOISE_SEED}, disable_posterior=${LANCE_OVERFIT_DISABLE_POSTERIOR_SAMPLING}"
echo "Optimization: iters=${LANCE_TRAIN_ITERS}, warmup=${LANCE_WARMUP_STEPS}, lr=${LANCE_OVERFIT_LR}"

cd "${REPO_ROOT}"
exec bash scripts/pretrain_lance_native.sh "$@"
