#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export NON_MEGATRON=true
python examples/tuna2/prepare_smoke.py --output outputs/tuna2-smoke
python -m mindspeed_mm.models.tuna2.train --config outputs/tuna2-smoke/smoke.yaml --max-steps 8
cp outputs/tuna2-smoke/checkpoints/metrics.json outputs/tuna2-smoke/initial-metrics.json
python -m mindspeed_mm.models.tuna2.train --config outputs/tuna2-smoke/smoke.yaml \
  --resume outputs/tuna2-smoke/checkpoints/last.pt --max-steps 12
