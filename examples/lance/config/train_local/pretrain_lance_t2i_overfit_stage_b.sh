#!/usr/bin/env bash
# Stage B: one image with per-visit posterior, noise, and timestep sampling.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LANCE_OVERFIT_STAGE=stage-b-stochastic-single-image
export LANCE_PREENCODED_DATA="${LANCE_PREENCODED_DATA:-/mnt/models/DATA_INIT/MULTI/T2I/Qwen3-0.6B-overfit-1-packed}"
export LANCE_OUTPUT_DIR="${LANCE_OUTPUT_DIR:-/mnt/models/outputs/lance-qwen3-06b-t2i-overfit-stage-b}"
export LANCE_OVERFIT_RESAMPLE_TIMESTEPS=true
export LANCE_OVERFIT_FIXED_NOISE_SEED=null
export LANCE_OVERFIT_DISABLE_POSTERIOR_SAMPLING=false
export LANCE_OVERFIT_SHUFFLE=false
export LANCE_OVERFIT_LR="${LANCE_OVERFIT_LR:-1.0e-4}"
export LANCE_TRAIN_ITERS="${LANCE_TRAIN_ITERS:-2000}"
export LANCE_STOP_AFTER_ITERS="${LANCE_STOP_AFTER_ITERS:-${LANCE_TRAIN_ITERS}}"
export LANCE_WARMUP_STEPS="${LANCE_WARMUP_STEPS:-50}"
export LANCE_SAVE_INTERVAL="${LANCE_SAVE_INTERVAL:-200}"
export MASTER_PORT="${MASTER_PORT:-6012}"

exec bash "${HERE}/pretrain_lance_t2i_overfit_common.sh" "$@"
