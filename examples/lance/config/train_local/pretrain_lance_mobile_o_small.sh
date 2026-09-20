#!/usr/bin/env bash
# 从零预训练（Mobile-O raw PT，Qwen3-0.6B 小模型，t2i+i2t 双任务）。
#
# 用法（容器 mindspeed-mm 内）:
#   cd /mnt/models/CODE/MindSpeed-MM
#   bash examples/lance/config/train_local/pretrain_lance_mobile_o_small.sh   # 默认 20B token
#   LANCE_STOP_AFTER_ITERS=10 LANCE_SAVE_INTERVAL=10000 \
#     bash examples/lance/config/train_local/pretrain_lance_mobile_o_small.sh # 10 步 smoke
#
# 两机示例（两台机器使用相同代码、模型、数据和共享输出路径）：
#   # node 0
#   NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 MASTER_ADDR=<node0-ip> MASTER_PORT=6002 \
#     bash examples/lance/config/train_local/pretrain_lance_mobile_o_small.sh
#   # node 1
#   NNODES=2 NODE_RANK=1 NPROC_PER_NODE=8 MASTER_ADDR=<node0-ip> MASTER_PORT=6002 \
#     bash examples/lance/config/train_local/pretrain_lance_mobile_o_small.sh
#
# 与 pretrain_lance_mobile_o.sh 的差异：
#   - 基座 Qwen2.5-VL-3B (6.45B) -> Qwen3-0.6B (~1.1B MoT)
#   - 数据为 raw PT 格式（无 chat template / system prompt），t2i+i2t 1:1
#   - init DCP 需先用 prepare_lance_native_init.py --qwen-path Qwen3-0.6B 生成
#   - 数据需先用 prepare_lance_native_data.py --emit-tasks t2i,i2t 重编码打包
# 注意：Qwen3-0.6B tokenizer effective_vocab_size=151669（比 Qwen2.5-VL 多 4），
#   数据编码 / init DCP / 本启动脚本的 QWEN_PATH 必须同为 Qwen3-0.6B。

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"

# 小模型专用配置（数据路径已在 yaml 内写死）
export CONFIG_FILE="${CONFIG_FILE:-${HERE}/fsdp2_pt_mobile_o_small.yaml}"

# 权重与数据（本机 /mnt/models 布局）
export QWEN_PATH="${QWEN_PATH:-/mnt/models/MODELS/Qwen3-0.6B}"
export LANCE_LOAD_DCP="${LANCE_LOAD_DCP:-/mnt/models/MODELS/lance-qwen3-06b-init-dcp}"
# The 16K pack is the measured throughput optimum on this 8x910B3 host
# (about 37.3K multimodal tokens/s over stable steps).  Override this path to
# reproduce another packing point; the model-side micro batch remains one
# packed sequence per rank.
export LANCE_PREENCODED_DATA="${LANCE_PREENCODED_DATA:-/mnt/models/DATA_INIT/MULTI/T2I/Mobile-O-Pre-Train-packed-16k}"
export LANCE_OUTPUT_DIR="${LANCE_OUTPUT_DIR:-/mnt/models/outputs/lance-mobile-o-small-pt}"

# Token budget and checkpointing.  The current 2,646 raw image-caption pairs
# produce 2,883,850 positions per full dual-task epoch, or about 1,089.89
# positions per raw pair.  A 20B-token run is therefore about 6,935 repeated
# epochs / 18.35M pair exposures; it does not create additional data diversity.
export LANCE_TRAIN_TOKENS="${LANCE_TRAIN_TOKENS:-20000000000}"
export LANCE_SAVE_INTERVAL_TOKENS="${LANCE_SAVE_INTERVAL_TOKENS:-2000000000}"
# Disable legacy step checkpoints by default; the final checkpoint is always
# saved and token checkpoints are emitted at every crossed 2B boundary.
export LANCE_SAVE_INTERVAL="${LANCE_SAVE_INTERVAL:-0}"
export LANCE_WARMUP_STEPS="${LANCE_WARMUP_STEPS:-2500}"
export LANCE_GRADIENT_ACCUMULATION_STEPS="${LANCE_GRADIENT_ACCUMULATION_STEPS:-1}"
# Stable 16K measurements averaged about 14,277 positions/rank/microstep.  The
# common launcher uses this only to derive a scheduler horizon/safety cap; the
# engine counts actual tokens for stopping and checkpointing.
export LANCE_ESTIMATED_TOKENS_PER_RANK="${LANCE_ESTIMATED_TOKENS_PER_RANK:-14277}"
export LANCE_ESTIMATED_PAIR_EXPOSURES="${LANCE_ESTIMATED_PAIR_EXPOSURES:-18350469}"
export LANCE_ESTIMATED_DATA_EPOCHS="${LANCE_ESTIMATED_DATA_EPOCHS:-6935.17}"

# Performance knobs.  Decoder checkpointing and forward resharding preserve the
# measured memory-conservative recipe. EMA is disabled in the YAML for this
# finite 20B-token run; ordinary optimizer weights are the inference weights.
export LANCE_GRADIENT_CHECKPOINTING="${LANCE_GRADIENT_CHECKPOINTING:-true}"
export LANCE_RESHARD_AFTER_FORWARD="${LANCE_RESHARD_AFTER_FORWARD:-true}"

# 端口：避开 6000/6001，允许同机并行
export MASTER_PORT="${MASTER_PORT:-6002}"

exec bash "${REPO_ROOT}/scripts/pretrain_lance_native.sh" "$@"
