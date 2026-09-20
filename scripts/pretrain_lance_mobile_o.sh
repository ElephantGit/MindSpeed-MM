#!/usr/bin/env bash
# 从头预训练（Mobile-O t2i）：基于 Mobile-O-Pre-Train-packed 数据的全新 Lance native PT run。
#
# 用法（容器 mindspeed-mm 内）:
#   docker exec -it mindspeed-mm bash
#   cd /mnt/models/CODE/MindSpeed-MM
#   bash scripts/pretrain_lance_mobile_o.sh                     # 默认 1000 iters
#   LANCE_TRAIN_ITERS=5000 bash scripts/pretrain_lance_mobile_o.sh   # 覆盖步数
#
# 本脚本只设置 Mobile-O 相关默认值，随后 exec 官方 pretrain_lance_native.sh
# （含启动前三件套契约校验 check_lance_native_artifacts.py）。所有默认值均可被
# 外部环境变量覆盖；NPROC_PER_NODE、LANCE_NUM_WORKERS 等通用变量透传。
#
# 数据规模注意：Mobile-O-Pre-Train-packed 仅 39 个 batch（1,687,341 multimodal
# tokens，源数据集 2250 个 tar 分片中的 00000）。官方 350K 步 schedule 面向
# ~17.5B token 语料，这里默认 1000 iters（8 卡 × 1 batch/步 ≈ 8000 batch 消费，
# 数据重复约 200 遍）——适合管线验证与小规模实验。扩大数据后按比例上调。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Mobile-O 专用配置（数据路径已在 yaml 内写死）
export CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/examples/lance/config/train_local/fsdp2_pt_mobile_o.yaml}"

# 权重与数据（本机 /mnt/models 布局）
export QWEN_PATH="${QWEN_PATH:-/mnt/models/MODELS/Qwen2.5-VL-3B-Instruct}"
export LANCE_LOAD_DCP="${LANCE_LOAD_DCP:-/mnt/models/MODELS/lance-qwen-init-dcp}"
export LANCE_PREENCODED_DATA="${LANCE_PREENCODED_DATA:-/mnt/models/DATA_INIT/MULTI/T2I/Mobile-O-Pre-Train-packed}"
export LANCE_OUTPUT_DIR="${LANCE_OUTPUT_DIR:-/mnt/models/outputs/lance-mobile-o-pt}"

# 步数与保存：小数据默认值（每 ckpt 约 94G = model + optimizer + EMA）
export LANCE_TRAIN_ITERS="${LANCE_TRAIN_ITERS:-1000}"
export LANCE_SAVE_INTERVAL="${LANCE_SAVE_INTERVAL:-500}"

# 端口：避开主 PT 默认 6000，允许同机并行
export MASTER_PORT="${MASTER_PORT:-6001}"

exec bash "${REPO_ROOT}/scripts/pretrain_lance_native.sh" "$@"
