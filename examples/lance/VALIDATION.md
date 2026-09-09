# Lance 适配验证记录

本记录区分三类结果：已在当前工作区实际执行、只验证元数据，以及必须在 Ascend 节点执行。
记录日期为 2026-09-08。

## 已实际执行并通过

Lance 官方源码处于 clean revision：

```text
4baeee086648996f6ab12e673cbe461b0b149997
```

四个随官方源码发布的数据协议均通过 `evaluate_lance.py validate`：

| Benchmark | 实际条数 | SHA-256 | 结果 |
|---|---:|---|---|
| GenEval | 553 | `2b455ba7255c5289da8586f3547acc1da6590a022ea817122c2898718dfa703f` | valid |
| DPG-Bench | 1065 | `da4fb02b04b51ee42053a570bfc621def56be977e358c638b018c531dad2eee3` | valid |
| GEdit-Bench | 606 | `5b33a3198106095374e7ab4e0c1c63e8107f4dc7cded4370c3d24c6de5c40e2e` | valid |
| VBench recaption | 946 | `a40ce0fc8a765d925e2dfe5a6205d19e8b1b158ca1a968918a4ec2f882450747` | valid |
| VBench temporal prompts | 75 | `bedc9e5a6fbfd0a9ce78baa9a377f16e44ae36b0e161c147cbc0eab033d0ea05` | valid |

执行命令：

```bash
python evaluate_lance.py validate --benchmark geneval --lance-source-root ../Lance
python evaluate_lance.py validate --benchmark dpgbench --lance-source-root ../Lance
python evaluate_lance.py validate --benchmark gedit --lance-source-root ../Lance
python evaluate_lance.py validate --benchmark vbench --lance-source-root ../Lance
```

本机纯 Python 回归覆盖 evaluation 协议、scorer 归一化、MVBench、NPU shim 调用契约、官方
checkpoint 契约、packed attention 和 flow matching 方向。最终执行结果见本文末尾的最新验证。

## 官方 checkpoint 元数据验证

通过 Hugging Face 官方文件的 safetensors header 实际执行 `convert_lance_checkpoint.py inspect
--metadata-only`，所有 key、shape、BF16 dtype 和 data offset 均通过：

| 变体 | tensors | 元素 | tensor 数据字节 | 结果 |
|---|---:|---:|---:|---|
| Lance_3B | 1021 | 6,185,205,808 | 12,370,411,616 | valid metadata |
| Lance_3B_Video | 1411 | 7,105,548,336 | 14,211,096,672 | valid metadata |

这只证明发布权重的结构契约与适配器一致。完整文件 SHA 和 payload 长度会在 NPU sampling
preflight 再检查；不能把 metadata-only 结果表述为已完成 checkpoint 推理。

## CPU 结构与算法验证

当前项目默认 Python 环境没有 PyTorch。为继续验证原生实现，另在 `/tmp/lance_torch_runtime` 安装
CPU PyTorch 2.2.2 和 safetensors 0.4.5，并由 Python 3.10.18 临时加入 import path。该环境完成：

- image/video 完整配置在 meta device 上的参数名和 shape 精确契约；
- tiny MoT 模型 understanding/generation 双专家 forward/backward；
- safetensors→DCP 写入及 DCP metadata 回读；
- NPU block 调度器通过 fake fusion-attention 与 dense oracle 对齐；
- joint CE/flow-matching step、Qwen 初始化映射、generation expert copy；
- 3D sin/cos 位置表与官方 NumPy 公式数值对齐；
- Euler timestep/方向、文本+视觉 CFG、renorm、edit 子集更新和原生 tiny-model 采样。
- Qwen2.5-VL ViT window/full schedule、visual RoPE、merger reorder 和 window 隔离；
- full/causal 原生 KV-cache 与完整序列逐层等价，缓存版 Euler latent 与完整重算一致；
- 官方 PackedDataset 的 mixed-expert segment、online/offline VAE/ViT 和 joint backward；
- 逐 tensor BF16 safetensors streaming load 的完整参数往返；
- decoder activation checkpointing 开关前后的输出、输入梯度和参数梯度一致。

这些是 CPU 上的结构、梯度和算法语义验证，不能替代目标 PyTorch 2.7.1、torch-npu、CANN 与
Ascend 芯片上的数值和性能验收。

## 当前环境阻塞项

当前登录机的项目环境没有 `torch_npu`、CANN、Ascend 设备、Lance 完整 checkpoint 和外部 scorer
权重。因此以下结果尚未产生：

- NPU varlen/GQA/KV-cache 数值误差；
- 七任务真实生成结果；
- GenEval、DPG-Bench、GEdit-Bench、VBench、MVBench 的模型得分；
- 多卡吞吐、显存、确定性和原生模型梯度一致性。

当前机执行：

```bash
python inference_lance.py --runtime-check
```

会返回机器可读的 `status=blocked`，原因是缺少 PyTorch/NPU 环境。将同一工作区放到 Ascend
节点并激活 MindSpeed-MM 镜像后，应首先运行该命令；只有 `status=passed` 且
`max_abs_error <= 0.08` 才进入 checkpoint 推理和全量 evaluation。

## 上游 PT 桥 preflight

在无需导入 torch 的路径上，已使用同级真实 Lance checkout 和其
`config/train_local/unified.yaml` 完成 PT preflight：

- Lance revision：`4baeee086648996f6ab12e673cbe461b0b149997`，checkout clean；
- `train/unified_train.py` SHA-256：`7c537026be87962396171a36b85068554c6dbb02dd8f34fa3eb0fd5c837fef77`；
- dataset YAML SHA-256：`f45cf0b175d8270fcdb2c0ef8dfe6f7c6ca43125f5839f559e6b2f7f971ac92c`；
- PT 350K steps、warmup、token budget、loss/dropout、AdamW/EMA、冻结策略和 1x8 FSDP 拓扑均通过；
- fail-fast AST 门禁精确识别 1 个官方 step exception handler，未修改上游文件。

该结果只证明源码和启动契约闭合；训练态 NPU attention backward、FSDP/HCCL、EMA 内存和 checkpoint
恢复仍必须在目标 8 NPU 环境验证。

## 最新验证

```text
115 passed (temporary CPU torch validation runtime)
Python compilation: passed
CLI help/preflight: passed
Upstream PT bridge preflight: ready
git diff --check: passed
Released dataset preflight: 4/4 valid
Official checkpoint header contract: image/video valid metadata
Ascend numerical/runtime evaluation: blocked by unavailable hardware/runtime
```
