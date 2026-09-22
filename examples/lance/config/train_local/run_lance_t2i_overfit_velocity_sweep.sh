#!/usr/bin/env bash
# Diagnose where along t=1 -> 0 the learned Stage-B velocity field is weakest.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/MindSpeed:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
CHECKPOINT="${CHECKPOINT:-/mnt/models/outputs/lance-qwen3-06b-t2i-overfit-stage-b/iter_0002000}"
PACKED_BATCH="${PACKED_BATCH:-/mnt/models/DATA_INIT/MULTI/T2I/Qwen3-0.6B-overfit-1-packed/batch-00000000.pt}"
QWEN_PATH="${QWEN_PATH:-/mnt/models/MODELS/Qwen3-0.6B}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
CHECKPOINT_ROOT="$(dirname "${CHECKPOINT}")"
OUTPUT="${OUTPUT:-${CHECKPOINT_ROOT}/eval-${CHECKPOINT_NAME}-velocity-sweep.json}"
DEVICE="${DEVICE:-npu:0}"

args=(
  --checkpoint "${CHECKPOINT}"
  --packed-batch "${PACKED_BATCH}"
  --qwen-path "${QWEN_PATH}"
  --output "${OUTPUT}"
  --device "${DEVICE}"
  --posterior-mode "${POSTERIOR_MODE:-both}"
  --timestep-shift "${TIMESTEP_SHIFT:-1.0}"
  --latent-patch-size 1 2 2
  --max-latent-size 64
  --max-num-frames 121
)
if [[ "${WEIGHTS:-model}" == "ema" ]]; then
  args+=(--ema-weights)
fi

cd "${REPO_ROOT}"
exec python evaluate_lance_stage_b_velocity.py "${args[@]}" "$@"
