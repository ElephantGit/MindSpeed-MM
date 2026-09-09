# Lance 在 MindSpeed-MM 上从推理到预训练的实施与验收计划

## 目标与边界

最终目标是在 Ascend 上支持同一套 Lance 模型完成七类推理、论文五项 evaluation、PT、CT、
SFT，以及可选 RL。实现分成两个连续阶段：先以官方 checkpoint 和官方数据语义完成推理数值
对齐，再把模型、数据和 checkpoint 完整原生化到 MindSpeed-MM。任何阶段都不能用“脚本能启动”
代替数值验收。

论文的“training from scratch”仍使用 Qwen2.5-VL 3B 权重初始化语义理解编码器和两个 LLM
专家。原生训练需同时提供 `init_mode=qwen2_5_vl`（论文复现）和 `init_mode=random`（严格随机
初始化实验），两者的实验结果不能混报。

## 阶段 I：官方 checkpoint 推理对齐

### I-1 运行时与权重（代码已落地，待 NPU 实机验收）

- `inference_lance.py` 作为 MindSpeed-MM 入口，官方 Lance 参数原样透传；
- checkpoint 保持 `model.safetensors`/`ema.safetensors`，数值对齐前不做 DCP 转换；
- `torch_npu.contrib.transfer_to_npu` 处理第三方 CUDA API 拼写；
- NCCL 映射 HCCL，FlashAttention varlen 映射为
  `torch_npu.npu_fusion_attention(input_layout="TND")`；
- KV-cache 使用 `sparse_mode=3` 的 right-down causal 语义，覆盖 `q_len != kv_len`；
- 执行期间工作目录切到 Lance 源码根目录，保证 ViT/VAE 相对路径与官方一致；
- 支持 `t2i`、`t2v`、`i2v`、`image_edit`、`video_edit`、`x2t_image`、`x2t_video`。

验收顺序：

1. `python inference_lance.py --runtime-check`：16Q/2KV、BF16、非等长 causal attention
   相对 CPU FP32 参考最大绝对误差不超过 0.08；
2. checkpoint 载入只允许官方已知的固定位置编码忽略项，不允许未解释的 missing/unexpected key；
3. 七种任务各跑 2 个固定样例，保存 prompt、seed、源码 revision、checkpoint SHA-256；
4. 连续运行两次，文本 token 完全一致，生成 latent 统计与文件数量一致；
5. 记录首 token/首 step 延迟、峰值 NPU 内存和端到端吞吐，作为原生化前基线。

### I-2 Evaluation（协议与适配器已落地，待全量实跑）

| Benchmark | 官方目标 | 采样协议 | 完整输出 | Scorer |
|---|---:|---|---:|---|
| GenEval | 0.90 | 50 steps, shift 3.5, CFG 4, 768², 4/提示词 | 2212 图 | 官方 GenEval |
| DPG-Bench | 84.67 | 50 steps, shift 3.5, CFG 4, 768², 4 图网格 | 1065 网格 | 官方 mPLUG |
| GEdit-Bench | 7.30 | 50 steps, shift 3.5, CFG 4, 1/样本 | 606 WebP | GPT-4.1 `G_O` |
| VBench | 85.11 | 30 steps, shift 3.0, CFG 4, 480×848, 50 帧, 12 fps | 6230 视频 | 官方 VBench + NPU patch |
| MVBench | 62.0 | 论文 Table 8 的 19-task 子集，多选题，带边界裁剪 | 3800 答案 | 首选项匹配、任务宏平均 |

VBench 的 6230 个视频由 871 个普通提示词各 5 个和 75 个 temporal-flickering 提示词各
25 个组成。GEdit 的 Qwen 离线裁判只能标记为 `Q_O`，不能用于论文 `G_O` 对齐。MVBench
官方完整版虽为 20 个任务，但 Lance Table 8 缺少 Fine-grained Pose；其余 19 个任务的
论文分数宏平均为 62.0，所以论文复现必须使用 19-task/3800 条，20-task/4000 条另行报告。

每个全量任务执行以下门禁：

1. `validate` 检查数据数量、SHA-256、checkpoint、tokenizer、ViT 和 VAE；
2. sampling 自动写 `lance_eval_run.json`；
3. `audit` 检查 scorer-facing 文件数量；
4. 固定版本的官方 scorer 运行后，由 `geneval-score`、`dpgbench-score`、`gedit-score`、
   `vbench-score` 或 `mvbench-score` 归一化为带 provenance 的 `metrics.json`；
5. `report` 输出相对论文值的 delta。默认容差：GenEval 0.02、DPG 1.0、GEdit 0.2、
   VBench 1.0、MVBench 1.0；容差用于发现移植回归，不代表统计显著性结论。

## 阶段 II：MindSpeed-MM 原生模型

### II-1 共享模型结构

新增原生 `LanceModel`，训练与推理共享以下组件：

- Qwen2.5 3B decoder：hidden 2048、36 层、16 attention heads、2 KV heads、FFN 11008；
- understanding/generation 两套 Attention、MLP、RMSNorm 和 QK-Norm 参数；
- Qwen2.5-VL ViT 语义 token 路径与连接器；
- Lance 专用 Wan2.2 causal VAE（空间下采样 16、时间下采样 4、48 latent channels）；
- 3D latent patchify/unpatchify、timestep embedding、VAE↔LLM projection；
- MaPE：ViT、clean VAE、noisy VAE 三组在时间轴使用 1000 的固定 offset；
- understanding CE head 与 generation flow head。

原生 `LanceNativeModel` 已按官方名称注册完整参数树，并提供小模型参考 forward/backward。完整
image/video 配置可在 meta device 构建并与发布契约逐 key/shape 对齐；两套 MoT Attention、MLP、
RMSNorm/QK-Norm、bridge heads、timestep embedder 和冻结 3D sin/cos 位置表均已实现。位置表数值已
与发布版 NumPy 公式对照。Qwen2.5-VL ViT 的 patch embed、visual RoPE、window reorder、
window/full block schedule、spatial merger 和 NPU TND varlen backend 均已原生实现，并通过窗口隔离
和梯度测试。

`convert_lance_checkpoint.py inspect/plan` 无需 torch 即可审计完整 safetensors header。官方真实
header 已通过逐 key 契约：image 为 1021 tensors / 6,185,205,808 BF16 元素，video 为 1411
tensors / 7,105,548,336 BF16 元素。`to-dcp` 在完整 payload 审计后按 identity mapping 写入
MindSpeed DCP，`verify-dcp` 无需分配模型即可回读名称、shape 和 dtype；小型 BF16 checkpoint
已执行 safetensors→DCP metadata 往返测试。另提供原生逐 tensor streaming loader：在任何参数写入
前强制检查完整 payload/shape/dtype，加载峰值只增加一个 CPU tensor。全量 12--14 GB 转换和加载
仍需在目标训练节点执行。

### II-2 长序列广义 3D causal attention

训练序列上限 70K，禁止构造 `[L,L]` dense mask。实现两级路径：

1. 参考路径：按 sample、segment 分解。每个 segment 的 Q 只访问此前 clean segment 和自身；
   text 自身使用 causal，visual 自身使用 full attention；
2. 性能路径：把相同 segment 类型的 Q/KV 打包为 TND，传
   `actual_seq_qlen/actual_seq_kvlen`，减少 kernel launch；
3. CP 路径：复用 MindSpeed ring/ulysses 通信原语，Q 按 token 切分，clean K/V 按块流动；
4. noisy target 不写入后续 clean KV-cache，推理每个 diffusion step 只更新 query。

验收包含 256 token 小模型的显式 dense-mask golden test、1K/8K/70K 长度压力测试、GQA、
多 sample packing、空 generation/understanding segment，以及 DP/CP 不同切分下的输出与梯度对齐。

当前已完成语义 oracle、block compiler 与 NPU block backend：原生 `LancePackedSequence` 固定
`causal/full/noise/full_noise/full_noise_target` 的精确关系、document 隔离和 MoT token routing；
`noise` target 只允许自身 full attention，禁止被其他 segment 当作 KV。70K 测试只构造常数级
block 描述，dense oracle 超过 4096 tokens 会主动报错。Ascend backend 按 query segment 拼接
历史 clean KV 和自身 KV，使用 TND fusion attention，测试替身已与 dense oracle 数值对齐且确认
不会传入 `[L,L]` mask。真实 NPU kernel 输出/梯度以及 CP 通信仍需在 Ascend 环境验收。

原生推理现已支持逐层 post-RoPE K/V cache。条件前缀只投影一次，每个 diffusion step 仅重算
noisy VAE query；full/causal 两种 `q_len != kv_len` 路径均与完整序列 oracle 对齐。NPU backend
使用 TND fusion attention 与 right-down causal 语义，fake-NPU 数值测试已通过。缓存版 Euler/CFG
与逐步完整重算的 velocity 和最终 latent 一致；非后缀、非等价 editing 布局会拒绝缓存并保留完整
重算路径。

### II-3 数据与损失

统一样本 schema 保留 ordered segments、modality、clean/noisy/target、3D grid、loss mask 和
sample id。实现 I2T、V2T、T2I、T2V、I2V、image/video edit、subject-driven generation、
interleaved X2T/X2I/X2V。

- ViT/VAE 在 PT 冻结，数据侧预编码与在线编码可切换；
- CE:MSE 权重为 PT `0.25:1`、CT `0.5:1`、SFT `0.25:1`；
- PT T2I text dropout 10%；CT/SFT multimodal full-condition dropout 5%，额外 text-only
  dropout 5%；
- 官方代码采用 `(1-t)*clean+t*noise` 与 `noise-clean` velocity。尽管论文用反向记号，
  checkpoint 对齐路径必须遵循代码参数化，并用单元测试固定 sampler 方向；
- packing 在每个 rank 保持目标 token 数范围，损失按全局有效 CE/MSE token 数归一化。

原生 joint step 已支持预编码 ViT/VAE 输入、`(1-t)clean+t*noise`、`noise-clean` target、CE/MSE
全局 denominator 和真实 backward。原生推理 sampler 已固定 shifted timestep、
`x <- x - v*dt`、文本/视觉三分支 CFG、global/channel renorm、CFG interval 以及 edit 子区域更新，
并通过 tiny 模型端到端确定性测试。官方 `PackedDataset` 后处理适配器已支持 online/offline Wan-VAE、
原生/离线 ViT、timestep logits、1D/3D position IDs、CE/MSE 选择以及 attention segment 内逐 token
MoT 路由。尚待在目标镜像加载生产 tokenizer/Wan-VAE 权重并跑真实 parquet 数据。

### II-4 并行、优化器与 checkpoint

- 默认 FSDP2 + HCCL；模型参数、梯度、优化器 state 使用 DCP；
- ViT/VAE 冻结且不进入 optimizer，generation/understanding expert 分组可独立冻结；
- 支持 activation checkpoint、BF16、distributed optimizer、DP/CP，TP/PP 在基线稳定后加入；
- checkpoint 必须保存 dataloader cursor、混合采样 RNG、noise RNG、optimizer、scheduler、EMA；
- 断点续训测试比较连续 20 step 与 10+resume+10 step 的 sample id、loss 和权重。

当前已落地 AdamW、论文 warmup/constant/5-cycle cosine scheduler、gradient clipping、独立 EMA 模型、
decoder activation checkpointing，以及包含 model/optimizer/EMA/数据游标/混合 RNG/noise RNG/
scheduler 的 DCP state contract。EMA 以顶层 sharded model 保存，不会复制进每个 rank 的 extra state。

为先获得可归因的训练基线，已落地官方 `train/unified_train.py` 的 Ascend 桥：启动前绑定 clean Lance
revision、训练入口 SHA、数据 YAML SHA、初始化、论文超参、冻结/EMA 与 FSDP 拓扑；训练态 runtime
补充 HCCL、NPU device mesh、autocast 和可反传 attention dropout；官方会吞掉 step 异常的 handler
通过窄范围 AST 转换改为 fail-fast，且不修改上游文件。该路径只覆盖官方监督训练入口，不把它误标为
原生 FSDP2，也不用于 strict-random 或 RL。

下一项是先在 8 NPU 上完成 PT 的 1 step/20 step/10+resume+10 闭环并记录 loss、吞吐、峰值内存和
checkpoint 可恢复性；随后把已存在的原生 model/data/joint-loss/task-mixer/DCP 组件接入 MindSpeed-MM
通用 FSDP2 TrainEngine，并以同一批次对比上游桥的 loss、梯度范数和一次参数更新。

## 阶段 III：论文训练阶段

| 阶段 | Steps | LR / Scheduler | Warmup | 每 rank 序列长度 | 最大 context | timestep shift |
|---|---:|---|---:|---|---:|---:|
| PT | 350K | 1e-4 / constant | 2500 | 44K–50K | 40K | 1.0 |
| CT | 80K | 1e-4 / constant | 2500 | 74K–80K | 70K | 4.0 |
| SFT | 15K | 2.5e-5 / cosine | 500 | 74K–80K | 70K | 4.0 |
| RL | 800 | 2e-6 / constant | 50 | 74K–80K | 70K | 4.0 |

全局 video-gen:video-und:image-gen:image-und 固定 `64:16:16:4`。PT 的 generation 子任务只含
T2I/T2V；CT-I/II/III 逐步提升 edit、subject-driven 和 I2V 占比；SFT 使用高质量配比。
每一阶段先走 1 卡 overfit、8 卡一致性、单机性能、多机稳定性，再扩大 token budget。

## 阶段 IV：持续回归门禁

- PR 级：纯 Python 协议、mask、MaPE、converter、dataset、loss 单测；
- 每日：1 卡七任务 smoke + attention numerical test；
- 每周：GenEval/MVBench 小集、VBench 指定 dimensions；
- release：五项全量 evaluation，并同时报告 checkpoint SHA、源码 revision、CANN/torch-npu、
  卡型、卡数、采样配置、scorer revision 和原始结果；
- 性能回归阈值：吞吐下降超过 5% 或峰值内存增加超过 5% 阻断合入；
- 精度回归阈值使用阶段 I-2 的默认容差，任何 scorer/provenance 变化单独建基线，禁止覆盖旧结果。

## 当前状态

- 已完成：官方推理桥接、HCCL/FlashAttention shim、七任务入口、五项论文 evaluation 的数据门禁、
  sampling/scorer/归一化/provenance/report；原生模型精确参数树、MoT forward/backward、3D 位置表、
  原生 Qwen2.5-VL ViT、packed attention 语义、NPU block/KV backend、KV-cached Euler/CFG sampler、
  joint CE/MSE step、官方 PackedDataset 适配、decoder activation checkpoint、论文与严格随机初始化、
  PT/CT/SFT/RL stage manifest、streaming 权重加载、safetensors→DCP 转换与 metadata 回读；官方
  监督训练 Ascend 基线桥、训练态 NPU runtime、严格启动 preflight 和 fail-fast 异常策略；
- 本机已验证：115 项既有 Lance 单测、Python 编译、CLI help、git diff whitespace，以及 image/video
  官方 safetensors 真实 header 和四套官方发布 evaluation 数据；torch 相关测试使用临时 CPU
  torch 2.2.2，仅用于结构/梯度/算法语义验证，不代表目标 torch 2.7.1 + torch-npu 结果；
- 本轮额外完成真实 clean Lance revision 上的 PT preflight：上游 revision、入口 SHA、dataset YAML
  SHA、8 卡拓扑及论文参数均为 `ready`；当前系统无 pytest，新增用例已通过 Python 编译但尚未在完整
  项目测试环境执行；
- 当前阻塞于实机的项目：本机无 torch_npu、CANN、Ascend 设备、完整 checkpoint 和 scorer 权重，
  因此尚未执行真实 NPU kernel numerical/gradient test、七任务生成、全量 DCP 转换及论文分数；
- 原生实现尚缺：生产 tokenizer/Wan-VAE 权重实载、CP/FSDP2 训练 runner、Ascend 10+resume 闭环、
  70K 压测和多卡性能调优；这些不能以当前 CPU/fake-NPU 结构测试替代；
- 进入原生训练开发前的硬门禁：阶段 I 的 NPU attention smoke 与至少 T2I、T2V、X2T 三条
  checkpoint 推理通过，避免在未确认底层算子语义时扩大改动面。
