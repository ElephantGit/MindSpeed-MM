#!/usr/bin/env bash
# Stage D: paired 32 T2I + 32 I2T overfit with 90% sigmoid-normal / 10% uniform
# T2I timestep sampling on one selectable physical NPU.
#
# By default 60K optimizer steps give each task about 30K expected exposures,
# matching the single-task 30K Stage-C comparison.  Use LANCE_TRAIN_ITERS=30000
# for a fixed-total-compute comparison instead.
#
# Usage:
#   DEVICE_ID=3 bash examples/lance/config/train_local/pretrain_lance_t2i_i2t_overfit_stage_d_mixture.sh
# or:
#   bash examples/lance/config/train_local/pretrain_lance_t2i_i2t_overfit_stage_d_mixture.sh 3
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pair_count="${LANCE_OVERFIT_MULTI_SAMPLES:-32}"

if (( $# > 1 )); then
  echo "Usage: DEVICE_ID=<physical_npu> $0 [physical_npu]" >&2
  exit 2
fi
device_id="${1:-${DEVICE_ID:-0}}"
if [[ ! "${device_id}" =~ ^[0-9]+$ ]]; then
  echo "DEVICE_ID must be a non-negative integer, got: ${device_id}" >&2
  exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES="${device_id}"
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=1

export LANCE_OVERFIT_STAGE=stage-d-mixture-p010-paired-t2i-i2t
export LANCE_PREENCODED_DATA="${LANCE_PREENCODED_DATA:-/mnt/models/DATA_INIT/MULTI/T2I/Qwen3-0.6B-overfit-${pair_count}t2i-${pair_count}i2t-packed}"
export LANCE_OUTPUT_DIR="${LANCE_OUTPUT_DIR:-/mnt/models/outputs/lance-qwen3-06b-t2i-i2t-overfit-stage-d-mixture-p010-${pair_count}x${pair_count}}"

# Only T2I documents own latent timesteps; I2T documents are unaffected by
# these flow-matching controls.
export LANCE_OVERFIT_RESAMPLE_TIMESTEPS=true
export LANCE_OVERFIT_TIMESTEP_SAMPLING=mixture
export LANCE_OVERFIT_TIMESTEP_UNIFORM_PROBABILITY="${LANCE_OVERFIT_TIMESTEP_UNIFORM_PROBABILITY:-0.1}"
export LANCE_OVERFIT_FIXED_NOISE_SEED=null
export LANCE_OVERFIT_DISABLE_POSTERIOR_SAMPLING=false
export LANCE_OVERFIT_SHUFFLE=true

# expected_tokens=1 preparation emits one task per packed file.  Weighting
# both objectives by one therefore preserves the single-task gradient scale;
# 0.5/0.5 would silently halve the effective learning rate of each task.
export LANCE_OVERFIT_CE_WEIGHT="${LANCE_OVERFIT_CE_WEIGHT:-1.0}"
export LANCE_OVERFIT_MSE_WEIGHT="${LANCE_OVERFIT_MSE_WEIGHT:-1.0}"
export LANCE_OVERFIT_LR="${LANCE_OVERFIT_LR:-1.0e-4}"
export LANCE_TRAIN_ITERS="${LANCE_TRAIN_ITERS:-60000}"
export LANCE_STOP_AFTER_ITERS="${LANCE_STOP_AFTER_ITERS:-${LANCE_TRAIN_ITERS}}"
export LANCE_WARMUP_STEPS="${LANCE_WARMUP_STEPS:-100}"
export LANCE_SAVE_INTERVAL="${LANCE_SAVE_INTERVAL:-5000}"
export MASTER_PORT="${MASTER_PORT:-6021}"

manifest="${LANCE_PREENCODED_DATA}/manifest.json"
if [[ ! -f "${manifest}" ]]; then
  echo "Missing packed-data manifest: ${manifest}" >&2
  echo "Prepare the paired 32+32 dataset before starting Stage D." >&2
  exit 1
fi
python - "${manifest}" "${pair_count}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))
counts = payload.get("accepted_task_counts", {})
actual = {"t2i": int(counts.get("t2i", 0)), "i2t": int(counts.get("i2t", 0))}
required = {"t2i": expected, "i2t": expected}
if payload.get("status") != "completed" or actual != required:
    raise SystemExit(
        "invalid paired packed data {}: status={}, task_counts={}, expected={}".format(
            path, payload.get("status"), actual, required
        )
    )
print("Validated paired packed data: {} T2I + {} I2T".format(expected, expected))
PY

echo "Stage D physical NPU: ${device_id} (logical local rank 0)"
comparison_mode=custom-total-steps
if [[ "${LANCE_TRAIN_ITERS}" == "60000" ]]; then
  comparison_mode=per-task-exposure-matched
fi
echo "Comparison mode: ${comparison_mode}"
exec bash "${HERE}/pretrain_lance_t2i_overfit_common.sh"
