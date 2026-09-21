#!/usr/bin/env bash
# Stage C: 8-32 distinct image-caption pairs with stochastic flow matching.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
multi_samples="${LANCE_OVERFIT_MULTI_SAMPLES:-32}"

export LANCE_OVERFIT_STAGE=stage-c-stochastic-multi-image
export LANCE_PREENCODED_DATA="${LANCE_PREENCODED_DATA:-/mnt/models/DATA_INIT/MULTI/T2I/Qwen3-0.6B-overfit-${multi_samples}-packed}"
export LANCE_OUTPUT_DIR="${LANCE_OUTPUT_DIR:-/mnt/models/outputs/lance-qwen3-06b-t2i-overfit-stage-c}"
export LANCE_OVERFIT_RESAMPLE_TIMESTEPS=true
export LANCE_OVERFIT_FIXED_NOISE_SEED=null
export LANCE_OVERFIT_DISABLE_POSTERIOR_SAMPLING=false
export LANCE_OVERFIT_SHUFFLE=true
export LANCE_OVERFIT_LR="${LANCE_OVERFIT_LR:-1.0e-4}"
export LANCE_TRAIN_ITERS="${LANCE_TRAIN_ITERS:-5000}"
export LANCE_STOP_AFTER_ITERS="${LANCE_STOP_AFTER_ITERS:-${LANCE_TRAIN_ITERS}}"
export LANCE_WARMUP_STEPS="${LANCE_WARMUP_STEPS:-100}"
export LANCE_SAVE_INTERVAL="${LANCE_SAVE_INTERVAL:-500}"
export MASTER_PORT="${MASTER_PORT:-6013}"

exec bash "${HERE}/pretrain_lance_t2i_overfit_common.sh" "$@"
