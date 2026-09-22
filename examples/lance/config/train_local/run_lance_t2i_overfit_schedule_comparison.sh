#!/usr/bin/env bash
# Compare the released linear Euler grid with training-density quantile points.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${CHECKPOINT:-/mnt/models/outputs/lance-qwen3-06b-t2i-overfit-stage-b/iter_0002000}"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
CHECKPOINT_ROOT="$(dirname "${CHECKPOINT}")"
NUM_STEPS="${NUM_STEPS:-50}"
WEIGHTS="${WEIGHTS:-model}"

for schedule in linear sigmoid_normal; do
  CHECKPOINT="${CHECKPOINT}" \
  NUM_STEPS="${NUM_STEPS}" \
  TIMESTEP_SCHEDULE="${schedule}" \
  WEIGHTS="${WEIGHTS}" \
  OUTPUT_ROOT="${CHECKPOINT_ROOT}/eval-${CHECKPOINT_NAME}-${NUM_STEPS}step-${schedule}-${WEIGHTS}" \
  bash "${HERE}/run_lance_t2i_overfit_eval.sh"
done

python -c 'import json,sys; a=json.load(open(sys.argv[1], encoding="utf-8")); b=json.load(open(sys.argv[2], encoding="utf-8")); print(json.dumps({"linear": a["average"], "sigmoid_normal": b["average"], "delta_sigmoid_minus_linear": {"psnr_db": b["average"]["psnr_db"]-a["average"]["psnr_db"], "correlation": b["average"]["correlation"]-a["average"]["correlation"], "mse": b["average"]["mse"]-a["average"]["mse"]}}, indent=2))' \
  "${CHECKPOINT_ROOT}/eval-${CHECKPOINT_NAME}-${NUM_STEPS}step-linear-${WEIGHTS}/metrics.json" \
  "${CHECKPOINT_ROOT}/eval-${CHECKPOINT_NAME}-${NUM_STEPS}step-sigmoid_normal-${WEIGHTS}/metrics.json"
