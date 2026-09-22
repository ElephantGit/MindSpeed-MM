#!/usr/bin/env bash
# Optimization-only follow-up: original Stage-B data distribution, longer cosine run, EMA.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LANCE_OVERFIT_STAGE=stage-b-optimized-original-distribution
export LANCE_PREENCODED_DATA="${LANCE_PREENCODED_DATA:-/mnt/models/DATA_INIT/MULTI/T2I/Qwen3-0.6B-overfit-1-packed}"
export LANCE_OUTPUT_DIR="${LANCE_OUTPUT_DIR:-/mnt/models/outputs/lance-qwen3-06b-t2i-overfit-stage-b-optimized}"
export LANCE_OVERFIT_RESAMPLE_TIMESTEPS=true
export LANCE_OVERFIT_TIMESTEP_SAMPLING=sigmoid_normal
export LANCE_OVERFIT_TIMESTEP_UNIFORM_PROBABILITY=0.0
export LANCE_OVERFIT_FIXED_NOISE_SEED=null
export LANCE_OVERFIT_DISABLE_POSTERIOR_SAMPLING="${LANCE_OVERFIT_DISABLE_POSTERIOR_SAMPLING:-false}"
export LANCE_OVERFIT_SHUFFLE=false
export LANCE_OVERFIT_LR="${LANCE_OVERFIT_LR:-1.0e-4}"
export LANCE_OVERFIT_LR_MIN="${LANCE_OVERFIT_LR_MIN:-1.0e-5}"
export LANCE_OVERFIT_LR_DECAY_STYLE=cosine
export LANCE_OVERFIT_USE_EMA=true
export LANCE_OVERFIT_EMA_DECAY="${LANCE_OVERFIT_EMA_DECAY:-0.999}"
export LANCE_TRAIN_ITERS="${LANCE_TRAIN_ITERS:-4000}"
export LANCE_STOP_AFTER_ITERS="${LANCE_STOP_AFTER_ITERS:-${LANCE_TRAIN_ITERS}}"
export LANCE_WARMUP_STEPS="${LANCE_WARMUP_STEPS:-50}"
export LANCE_SAVE_INTERVAL="${LANCE_SAVE_INTERVAL:-500}"
export MASTER_PORT="${MASTER_PORT:-6017}"

exec bash "${HERE}/pretrain_lance_t2i_overfit_common.sh" "$@"
