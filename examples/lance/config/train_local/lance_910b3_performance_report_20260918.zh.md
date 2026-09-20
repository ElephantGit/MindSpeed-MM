# Lance Qwen3-0.6B 性能探测（8 × 昇腾 910B3）

日期：2026-09-18

所有成功的实验行均使用相同的初始化、相同的 5,292 份已准备的 T2I/I2T 任务文档、
BF16 参数、FP32 归约、FSDP2 size 8、`micro_batch_size=1`、
`gradient_accumulation_steps=1`、解码器梯度检查点（gradient checkpointing）、
EMA，以及 `reshard_after_forward=true`。第 1 步被排除。末尾不完整（ragged）的
打包文件仅在明确标注处被排除。

| 打包设置 | 每步聚合 token 数 | 稳态 tokens/s | 估算稠密 TFLOPS（8 NPU）| 结果 |
|---|---:|---:|---:|---|
| 8K（实际每 rank 7.0-7.6K）| ~58K | 22.4K | 79.0 | 固定 FSDP/优化器开销占主导 |
| 16K（实际每 rank 14.0-14.6K）| ~114K | **37.3K** | **131.5** | 最佳实测点（10 步确认）|
| 24K（实际每 rank 22.0-22.6K）| ~178K | 31.6K | 111.3 | 比 16K 慢 |
| 32K（实际每 rank 30.0-30.6K）| ~243K | 29.1K | 102.4 | 比 24K 慢 |
| 40K（实际每 rank 38.0-38.5K）| ~306K | 26.5K | 93.2 | 排除一个 ragged 步 |
| 50K（实际每 rank 44.0-44.6K）| ~354K | 25.2K | 88.7 | 排除一个 ragged 步 |
| 74K（实际每 rank 72.0-72.6K）| ~578K | 17.3K | 60.8 | token 打包量最高，吞吐很差 |

TFLOPS 估算是一个透明的模型 FLOP 下界，并非硬件计数器。此处 Qwen3-0.6B 的
hidden size 为 1024、intermediate size 为 3072、28 层、16 个 query 头、显式
head dimension 为 128。每 token：

```
forward_dense = 28 * 2 * (6 * 1024^2 + 3 * 1024 * 3072)
              = 0.8808 GFLOP/token
train_with_full_decoder_recompute ~= 4 * forward_dense
                                  = 3.523 GFLOP/token
estimated_TFLOPS = measured_tokens_per_second * 3.523e9 / 1e12
```

该估算不包含 attention-score 矩阵乘、bridge（模态连接层）、LM head、通信和
优化器的工作量，因此不能标注为精确的 MFU。16K 确认运行期间 `npu-smi` 显示
rank 0 的 HBM 使用率约 17%；AICore 采样值在矩阵计算阶段达到 89-99%，在通信
或优化器阶段降至接近零。因此这个粗粒度采样器可以解释阶段性行为，但不能替代
由 profiler 得出的 MFU。

补充探测：

- 在 50K 和 32K 下关闭解码器 checkpointing，均在第一次 forward 时 OOM。64GB
  设备上的活跃显存分配达到约 57.6-59.4 GiB。
- `reshard_after_forward=false` 提高了采样到的 HBM 使用率（50K 下约 29%，对
  比 26%），但降低了 token 吞吐。
- 关闭 EMA 对稳态步吞吐的改变仅在短程运行噪声范围内；它主要减少了 checkpoint
  的大小/保存时间。它对短程评估语义有用，并非已验证的计算优化。

证据保留在 `/mnt/models/outputs/lance-bs-probe-*` 目录下。最终确认为
`lance-bs-probe-16k-10step.log` 及其八个 rank 的 trace。

## 20B token 正式训练预算（2026-09-18 配置更新）

当前 Mobile-O 数据包含 2,646 个唯一图文对，展开为 T2I/I2T 后每个完整 epoch
包含 5,292 个任务文档和 2,883,850 个 multimodal positions。因此：

- 每个原始图文对的双任务平均长度约为 1,089.89 positions；
- 20B token 对应约 6,935.17 个当前数据 epoch；
- 对应约 18.35M 次原始图文对曝光，但仍只有 2,646 个唯一图文对；
- 按 16K pack、8 NPU、约 114,212 token/update 估算，需要约 175,113 个
  optimizer steps；launcher 额外保留 5% 的 step 安全余量，实际由精确 token
  计数在达到 20B 后停止；
- 按实测 37.3K token/s，纯训练计算时间约 149 小时（约 6.2 天），不含启动、
  checkpoint 保存、评估和故障恢复。

正式配置关闭 EMA，将联合损失改为
`0.5 * CE + 0.5 * MSE`，使用 Lance PT 的 2,500 个固定 warmup steps，之后保持
`1e-4` constant learning rate。checkpoint 按全局实际累计 token 触发，跨过每个
2B token 边界时保存；由于一次 optimizer step 不可拆分，实际保存量最多比边界
多一个 step 的 token。

需要强调，18.35M 是重复采样后的“图文对曝光次数”，不是 18.35M 个唯一图文对。
反复训练当前 2,646 对数据只能满足计算预算，不能提供与大规模预训练相同的数据
多样性。

## 多机多卡启动

公共 launcher 已支持 `NNODES`、`NODE_RANK`、`NPROC_PER_NODE`、`MASTER_ADDR` 和
`MASTER_PORT`。例如两机各 8 卡时，两台机器分别执行：

```bash
# node 0
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=<node0-ip> MASTER_PORT=6002 \
bash examples/lance/config/train_local/pretrain_lance_mobile_o_small.sh

# node 1
NNODES=2 NODE_RANK=1 NPROC_PER_NODE=8 \
MASTER_ADDR=<node0-ip> MASTER_PORT=6002 \
bash examples/lance/config/train_local/pretrain_lance_mobile_o_small.sh
```

两台机器必须使用相同版本的代码和相同可见路径；DCP 输出目录应位于所有节点均
可读写的共享文件系统。`MASTER_ADDR` 必须是其他节点可访问的 node 0 地址，不能
使用默认的 `127.0.0.1`。当前 `fully_shard_parallel_size: auto` 会把完整 world
size 用作 FSDP shard/data-parallel group；TP/PP 仍未在这个 native FSDP Lance
路径中实现。
