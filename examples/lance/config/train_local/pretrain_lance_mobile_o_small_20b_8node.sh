#!/usr/bin/env bash
# One-node entry point for the eight-node/64-card Lance 6B run.
# Training remains continuous for the full 6B budget.  Checkpoint evaluation
# is managed by an independent watcher and does not stop the 64 training ranks.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"
# The latest image has MindSpeed-MM installed from /workspace, but this Lance
# adaptation uses the shared checkout and its adjacent MindSpeed source tree.
export PYTHONPATH="${REPO_ROOT}/MindSpeed:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export NNODES="${NNODES:-8}"
export NODE_RANK="${NODE_RANK:?NODE_RANK (0..7) is required}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-110.129.0.5}"
export MASTER_PORT="${MASTER_PORT:-6002}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp189s0f0}"
export RANK_TABLE_FILE="${RANK_TABLE_FILE:-/mnt/models/rank_table_generated.json}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-0}"
export DISTRIBUTED_BACKEND_TIMEOUT="${DISTRIBUTED_BACKEND_TIMEOUT:-7200}"
unset HCCL_IF_IP

export CONFIG_FILE="${HERE}/fsdp2_pt_mobile_o_small_20b_8node.yaml"
export QWEN_PATH="${QWEN_PATH:-/mnt/models/MODELS/Qwen3-0.6B}"
export LANCE_PREENCODED_DATA="/mnt/models/DATA_INIT/MULTI/T2I/Mobile-O-Pre-Train-packed-16k"
export LANCE_OUTPUT_DIR="${LANCE_OUTPUT_DIR:-/mnt/models/outputs/lance-mobile-o-small-pt-6b-8node-planA}"
export LANCE_LOAD_DCP="${LANCE_LOAD_DCP:-/mnt/models/MODELS/lance-qwen3-06b-init-dcp}"
if [[ "${LANCE_AUTO_RESUME:-1}" == "1" && -f "${LANCE_OUTPUT_DIR}/latest_checkpointed_iteration.txt" ]]; then
  export LANCE_LOAD_DCP="${LANCE_OUTPUT_DIR}"
fi

export LANCE_TRAIN_TOKENS="${LANCE_TRAIN_TOKENS:-6000000000}"
export LANCE_SAVE_INTERVAL_TOKENS=2000000000
export LANCE_SAVE_INTERVAL=0
export LANCE_WARMUP_STEPS="${LANCE_WARMUP_STEPS:-2500}"
export LANCE_GRADIENT_ACCUMULATION_STEPS="${LANCE_GRADIENT_ACCUMULATION_STEPS:-1}"
export LANCE_ESTIMATED_TOKENS_PER_RANK="${LANCE_ESTIMATED_TOKENS_PER_RANK:-14254}"
# One 6B scheduler horizon plus 5% packed-length safety headroom.
export LANCE_TRAIN_ITERS="${LANCE_TRAIN_ITERS:-6906}"
# Plan A: spend the available HBM to avoid decoder recomputation and the
# second parameter all-gather before backward.  The 64-way shard topology is
# deliberately unchanged for this controlled comparison.
export LANCE_GRADIENT_CHECKPOINTING="${LANCE_GRADIENT_CHECKPOINTING:-false}"
export LANCE_RESHARD_AFTER_FORWARD="${LANCE_RESHARD_AFTER_FORWARD:-false}"
export LANCE_NUM_WORKERS="${LANCE_NUM_WORKERS:-0}"
export LANCE_TRACE_FILE="${LANCE_TRACE_FILE:-}"

cd "${REPO_ROOT}"
exec bash scripts/pretrain_lance_native.sh "$@"
