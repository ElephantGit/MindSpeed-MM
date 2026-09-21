#!/usr/bin/env bash
# Reconstruct the exact fixed-noise, fixed-timestep Stage-A training point.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/MindSpeed:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
CHECKPOINT="${CHECKPOINT:-/mnt/models/outputs/lance-qwen3-06b-t2i-overfit-stage-a/iter_0000500}"
PACKED_BATCH="${PACKED_BATCH:-/mnt/models/DATA_INIT/MULTI/T2I/Qwen3-0.6B-overfit-1-packed/batch-00000000.pt}"
QWEN_PATH="${QWEN_PATH:-/mnt/models/MODELS/Qwen3-0.6B}"
VAE_PATH="${VAE_PATH:-/mnt/models/MODELS/bytedance-research/Lance/Wan2.2_VAE.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/models/outputs/lance-qwen3-06b-t2i-overfit-stage-a/eval-exact-training-point}"
DEVICE="${DEVICE:-npu:0}"

cd "${REPO_ROOT}"
exec python reconstruct_lance_stage_a.py \
  --checkpoint "${CHECKPOINT}" \
  --packed-batch "${PACKED_BATCH}" \
  --qwen-path "${QWEN_PATH}" \
  --vae-path "${VAE_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --noise-seed "${NOISE_SEED:-2025}" \
  --dataset-index "${DATASET_INDEX:-0}" \
  --timestep "${TIMESTEP:-0.5}" \
  --timestep-shift "${TIMESTEP_SHIFT:-1.0}" \
  --latent-patch-size 1 2 2 \
  --max-latent-size 64 \
  --max-num-frames 121
