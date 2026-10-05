# Tuna-2 迁移与小数据训练

本分支将 Meta 官方 `facebookresearch/tuna-2` 模型源码纳入
`mindspeed_mm/models/tuna2/upstream`。来源提交与修改清单见
[PROVENANCE.json](../../mindspeed_mm/models/tuna2/PROVENANCE.json)，原始 Apache-2.0
许可证见 [LICENSE.upstream](../../mindspeed_mm/models/tuna2/LICENSE.upstream)。

## 模型结构

原始 RGB 像素 → Conv2d patch embedding → Qwen2.5 主干 → 文本预测头与
带时间调制的扩散头。保留文本/图像混合注意力、3D RoPE、DeTok 随机掩码、
JiT 噪声调度、x0 预测与加权生成损失；Tuna-2 不需要 VAE 或 SigLIP。

模型由官方 `Tuna2PixelModel` 构建，生产配置是 7B；小数据验证使用相同
模型类和损失函数，将 Qwen 缩小为 2 层、hidden_size=64、4 个注意力头，
扩散头为 2 层，图像 32×32。模型共 329,344 参数，随机初始化，仅用于
验证训练链路，不代表预训练模型质量。

## CPU 一键验证

Python 3.12：

```bash
python -m venv .venv-tuna2
source .venv-tuna2/bin/activate
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r examples/tuna2/requirements-cpu.txt
bash examples/tuna2/smoke.sh
NON_MEGATRON=true python -m pytest -q tests/ut/tuna2 --confcutdir=tests/ut/tuna2
```

生成器在 `outputs/tuna2-smoke/` 创建本地 tokenizer、Qwen 配置、红蓝方块
图片及 4 个 JSONL 文件，每路 2 条数据，共 8 条。无需下载模型权重或数据集。
训练按 t2i → edit → mmu → text 顺序循环 8 步，再加载检查点训练到 12 步。
日志记录总损失、文本损失、生成损失与梯度范数；保存模型、AdamW、调度器、
EMA、随机数状态和数据读取位置。所有检查点均在优化器更新边界保存。

[validation.json](validation.json) 记录实际运行数据。已有验证包括：四路
前向/反向与更新、有限损失/梯度、检查点加载、以及恢复到 12 步与连续 12 步的
模型/EMA/调度器完全一致。GPU、NPU 和原生 FSDP2 未在当前无加速卡环境验证。

## 完整数据与权重的训练流程

[train_7b.yaml](train_7b.yaml) 保留官方 7B 架构和四路采样比例
0.50 / 0.20 / 0.20 / 0.10。修改本地 `llm_model_path`、可选
`load_stage1_model`、四个 `jsonl_path` 与 `image_root` 后运行：

```bash
export NON_MEGATRON=true
python -m mindspeed_mm.models.tuna2.train \
  --config examples/tuna2/train_7b.yaml --device npu --max-steps 8
# 多卡 DDP，同步选择数据流，每个 rank 使用独立的数据分片
NON_MEGATRON=true torchrun --nproc_per_node=8 \
  -m mindspeed_mm.models.tuna2.train \
  --config examples/tuna2/train_7b.yaml --device npu
```

设备为 NPU 时，需要匹配的 CANN、torch_npu 与 PyTorch；GPU 可选 `--device cuda`。
JSONL 格式沿用官方实现：

```jsonl
{"image": "red.png", "caption": "A red square."}
{"raw_image": "red.png", "out_image": "blue.png", "instruction": "make it blue"}
{"image": "red.png", "conversations": [{"from":"human","value":"<image> Describe the image."},{"from":"gpt","value":"A red square."}]}
{"caption": "A red square."}
```

四行分别属于 t2i、edit、mmu、text 文件。纯文本使用 `caption` 字段
（或在配置中指定 `text_field`）。开启多分辨率时 batch_size=1。
训练支持 AdamW、warmup、梯度累积、梯度裁剪、bf16、EMA 和断点恢复。
DDP 检查点为 `last-rankN.pt`，恢复时路径可写为 `last-rank{rank}.pt`。
检查点仅加载自己生成或可信来源的文件。

原官方训练实现与配置也保留在 `upstream/training`、`upstream/scripts`、
`configs/train` 中，使用这些 TorchTNT 入口需额外安装 `torchtnt`、`tensorboard`、
`setuptools<81`。迁移后的主入口使用上述 portable runner 或下述原生入口。

## MindSpeed-MM 原生 FSDP2

[pretrain_tuna2.py](../../pretrain_tuna2.py) 提供框架标准回调：model_provider、
get_batch、forward_step、loss_func、train_valid_test_datasets_provider。
[MindSpeedTuna2](../../mindspeed_mm/models/tuna2/native.py) 继承
MultiModalModule 与 FSDP2Mixin，由 MindSpeed-MM 管理优化器、调度、分片和 DCP。

按仓库安装指南安装 MindSpeed、Megatron、CANN 与 torch_npu，然后使用与
官方 Tuna 源码匹配的 Transformers 4.52.4 / Diffusers 0.35.1；这两个版本与
仓库通用依赖不同，Tuna 请使用独立环境并以 `--no-deps` 安装本仓。
当前未验证该组合下的 NPU 框架兼容性。

```bash
TUNA_CONFIG=examples/tuna2/train_7b.yaml NPUS_PER_NODE=8 TRAIN_ITERS=8 \
  bash examples/tuna2/pretrain.sh
# 框架检查点恢复
TUNA_CONFIG=examples/tuna2/train_7b.yaml TRAIN_ITERS=12 \
  bash examples/tuna2/pretrain.sh --load outputs/tuna2-native
```

原生入口要求各数据流 batch_size 与 --micro-batch-size 一致，使用
SDPA 和加载的 CPU 权重；不支持 meta 初始化、张量/流水线/上下文并行。
原生优化器超参数由 shell 参数指定，YAML training 段仅用于 portable runner。
原生入口的 EMA 能力取决于框架；portable runner 单独实现并验证了 EMA。
FSDP2 配置按 Qwen 层及扩散层分片，注意力保持整数 token 类型。

这里迁移的是官方公开模型和训练路径。官方 README 将公开训练指南定位为
foundation checkpoint 的微调；本分支不声称重现论文全部大规模预训练实验，
也不以 CPU 小模型测试证明 7B 昇腾精度/性能已对齐。
