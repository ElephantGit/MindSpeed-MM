#!/usr/bin/env bash
# Stage C uniform: 8-32 image-caption pairs on one selectable physical NPU.
# Usage:
#   DEVICE_ID=3 bash examples/lance/config/train_local/pretrain_lance_t2i_overfit_stage_c_uniform.sh
# or:
#   bash examples/lance/config/train_local/pretrain_lance_t2i_overfit_stage_c_uniform.sh 3
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
multi_samples="${LANCE_OVERFIT_MULTI_SAMPLES:-32}"

if (( $# > 1 )); then
  echo "Usage: DEVICE_ID=<physical_npu> $0 [physical_npu]" >&2
  exit 2
fi
device_id="${1:-${DEVICE_ID:-0}}"
if [[ ! "${device_id}" =~ ^[0-9]+$ ]]; then
  echo "DEVICE_ID must be a non-negative integer, got: ${device_id}" >&2
  exit 2
fi

# Expose exactly one physical NPU.  torchrun uses logical local rank 0 inside
# this visibility mask, while device_id selects the physical card.
export ASCEND_RT_VISIBLE_DEVICES="${device_id}"
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=1

export LANCE_OVERFIT_STAGE=stage-c-uniform-multi-image
export LANCE_PREENCODED_DATA="${LANCE_PREENCODED_DATA:-/mnt/models/DATA_INIT/MULTI/T2I/Qwen3-0.6B-overfit-${multi_samples}-packed}"
export LANCE_OUTPUT_DIR="${LANCE_OUTPUT_DIR:-/mnt/models/outputs/lance-qwen3-06b-t2i-overfit-stage-c-uniform-${multi_samples}}"
export LANCE_OVERFIT_RESAMPLE_TIMESTEPS=true
export LANCE_OVERFIT_TIMESTEP_SAMPLING=uniform
export LANCE_OVERFIT_TIMESTEP_UNIFORM_PROBABILITY=0.0
export LANCE_OVERFIT_FIXED_NOISE_SEED=null
export LANCE_OVERFIT_DISABLE_POSTERIOR_SAMPLING=false
export LANCE_OVERFIT_SHUFFLE=true
export LANCE_OVERFIT_LR="${LANCE_OVERFIT_LR:-1.0e-4}"
export LANCE_TRAIN_ITERS="${LANCE_TRAIN_ITERS:-5000}"
export LANCE_STOP_AFTER_ITERS="${LANCE_STOP_AFTER_ITERS:-${LANCE_TRAIN_ITERS}}"
export LANCE_WARMUP_STEPS="${LANCE_WARMUP_STEPS:-100}"
export LANCE_SAVE_INTERVAL="${LANCE_SAVE_INTERVAL:-500}"
export MASTER_PORT="${MASTER_PORT:-6018}"

echo "Stage C physical NPU: ${device_id} (logical local rank 0)"
exec bash "${HERE}/pretrain_lance_t2i_overfit_common.sh"
