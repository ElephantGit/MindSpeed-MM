# Lance on MindSpeed-MM

## 原生 FSDP2 预训练（`codex/lance-native-mindspeed-training`）

本分支新增的预训练路径是 MindSpeed-MM 原生实现，不是 Lance 官方仓库的启动桥：模型注册、MoT
forward/backward、数据集、FSDP2、融合 AdamW、梯度裁剪、EMA、DCP 保存/恢复均在
`mindspeed_mm` 内执行。训练命令不会发现、导入或运行 Lance 官方 Git checkout。

为避免每个 epoch、每个 data-parallel rank 重复运行冻结编码器，示例 parquet 先离线执行 Qwen
ViT 和 48-channel Wan2.2 VAE 编码，再按论文的 44K--50K token budget 打包。训练进程只常驻
Lance 的可训练 MoT core、bridge/head 和 connector；冻结 ViT、VAE、3D position table 不进入
optimizer 或 EMA。

原生 sample builder 使用 Lance/Qwen 的 system-user-assistant 模板、`<|video_pad|>` 视觉展开和
assistant-only CE 范围。VAE 离线保存 posterior mean/log-variance，训练每次访问重新采样 latent、
sigmoid-normal timestep 和 flow noise；不是把某一次离线随机结果重复 350K steps。不同模态 rank
即使本地 CE、MSE、ViT 或 VAE token 为空，也会以零长度输入进入相同的 FSDP2 wrapped module，
避免 mixed-task 训练时 collective 顺序分叉。

位置编码按 Qwen2.5-VL 的三轴 MRoPE 生成：ViT 使用 merger 后网格与 temporal stride 2，Lance
MaPE 把语义 ViT 条件移到 temporal band 1000；I2I/V2V 的 noisy target 复用对应 clean VAE 条件
的位置。视频抽帧也按官方 `MultiClipsFrameSampler(assert_seconds=false, truncate=false)` 在完整视频
范围均匀采样，短低帧率视频允许重复帧，以保持 `kn+1` 时间长度契约。

### 1. 原生运行时 smoke

以下命令使用 tiny Lance，但会真实执行 joint CE+MSE 前向、反向、FSDP2 参数更新、EMA 和 DCP：

```bash
NATIVE_SMOKE_TEST=1 NPROC_PER_NODE=8 bash scripts/pretrain_lance_native.sh
```

该 smoke 只验证训练引擎和分布式图，不代表真实 tokenizer、初始化权重和媒体编码已经验收。
`outputs/lance-native-synthetic` 必须是新的输出路径；可用 `LANCE_SYNTHETIC_OUTPUT` 指向另一个目录。

### 2. 准备 Qwen 初始化 DCP 与 350K 示例数据

脚本已写入当前机器的默认路径：

```text
Qwen:   /mnt/qs/models/Qwen/Qwen2.5-VL-3B-Instruct
ViT:    /mnt/qs/models/bytedance-research/Lance/Qwen2.5-VL-ViT
VAE:    /mnt/qs/models/bytedance-research/Lance/Wan2.2_VAE.pth
Data:   /mnt/qs/datasets/bytedance-research/Lance_example_dataset
```

分阶段执行便于失败后定位；每个输出目录都拒绝覆盖非空内容：

```bash
bash scripts/prepare_lance_native_pt.sh init
NPROC_PER_NODE=8 bash scripts/prepare_lance_native_pt.sh encode
bash scripts/prepare_lance_native_pt.sh pack
```

`init` 流式读取 Qwen safetensors，初始化 understanding expert，再复制到 generation expert，最后
直接写 MindSpeed-MM release DCP。`encode` 原生识别示例 parquet 中的 T2I/T2V、I2T/V2T、
I2I/V2V schema，并把各 rank 结果写到独立目录。`pack` 在 40K 单样本上限下生成 44K--50K
Ascend packed sequences；过长样本只在 manifest 中明确计数，不会出现训练时无限 `skip`。

原生 PT 默认显式使用 `latent_patch_size=1 2 2`，对应官方 PT 配置的空间 latent patching；这是让
V2V 示例保持在 40K 单样本上限内的必要配置。初始化、预编码、packing 和训练会使用同一几何契约。
如果要做发布 checkpoint 的 `1 1 1` 结构实验，必须在所有命令中同时设置
`LANCE_LATENT_PATCH_H=1 LANCE_LATENT_PATCH_W=1`，且 packer 会在某个必需任务全部超长时直接失败，
不会悄悄丢掉整个任务。

先做少量真实数据 smoke 时，不要占用完整数据输出目录：

```bash
LANCE_PREPARED_SAMPLES=datasets/lance-prepared-smoke \
LANCE_PREENCODED_DATA=datasets/lance-preencoded-smoke \
LANCE_MAX_SAMPLES_PER_TASK=16 \
LANCE_EXPECTED_TOKENS=1 \
LANCE_MAX_TOKENS=40000 \
NPROC_PER_NODE=8 bash scripts/prepare_lance_native_pt.sh encode

LANCE_PREPARED_SAMPLES=datasets/lance-prepared-smoke \
LANCE_PREENCODED_DATA=datasets/lance-preencoded-smoke \
LANCE_EXPECTED_TOKENS=1 \
LANCE_MAX_TOKENS=40000 \
bash scripts/prepare_lance_native_pt.sh pack
```

其中 `expected=1` 让每个合格样本单独成 batch，便于确保 8 个 DP rank 都有输入；它不是完整训练
packing 参数。完整准备应换回新的目录并使用默认 44K/50K。

封装脚本默认使用本地 `train_local/unified.yaml` 的六组等权配比
`t2i:t2v:i2i:v2v:i2t:v2t=1:1:1:1:1:1`。packer 会确定性过采样较小组，使后续 uniform
stateful sampler 真正得到目标分布，而不只是改变一次文件排列。如需按论文四大类配比，可直接调用
packer 并指定权重，例如：

```bash
python scripts/pack_lance_native_data.py \
  --input datasets/lance-prepared-samples \
  --output datasets/lance-preencoded-paper-mix \
  --llm-config /mnt/qs/models/Qwen/Qwen2.5-VL-3B-Instruct/config.json \
  --task-weights t2v=64,v2t=16,t2i=16,i2t=4
```

### 3. 真实前向/反向与断点连续性

先在真实 packed 数据上跑 1 step，保留完整 350K scheduler horizon：

```bash
LANCE_PREENCODED_DATA=datasets/lance-preencoded-smoke \
LANCE_OUTPUT_DIR=outputs/lance-native-real-1step \
LANCE_TRAIN_ITERS=350000 LANCE_STOP_AFTER_ITERS=1 LANCE_SAVE_INTERVAL=1 \
NPROC_PER_NODE=8 bash scripts/pretrain_lance_native.sh
```

原生 packed batch 最多包含 50K multimodal tokens，并带有较大的 ViT/VAE tensor。启动脚本因此
默认 `LANCE_NUM_WORKERS=0`，在各 rank 主进程内读取，避免 8 卡 worker prefetch 再次耗尽容器
`/dev/shm`。完成 1-step 正确性验证后，如果容器通过 `--shm-size` 或 `--ipc=host` 提供了足够
共享内存，可分别测试 `LANCE_NUM_WORKERS=1` 和 `2`；保留吞吐更高且没有 bus error 的设置，
不建议直接恢复为每 rank 4 个 worker。worker 模式会使用 pinned memory 和 non-blocking H2D。

完成真实 20-step 后，再做连续与恢复等价测试。以下所有 output/trace 路径在执行前必须不存在或
为空；第二段恢复是唯一会继续写同一个 split 输出目录的命令：

```bash
# 连续 20 step
LANCE_OUTPUT_DIR=outputs/lance-native-continuous-20 \
LANCE_TRAIN_ITERS=20 LANCE_STOP_AFTER_ITERS=20 LANCE_SAVE_INTERVAL=10 \
LANCE_TRACE_FILE=outputs/lance-native-continuous-20/trace.jsonl \
NPROC_PER_NODE=8 bash scripts/pretrain_lance_native.sh

# 先跑到 step 10，scheduler 的总 horizon 仍是 20
LANCE_OUTPUT_DIR=outputs/lance-native-resume-20 \
LANCE_TRAIN_ITERS=20 LANCE_STOP_AFTER_ITERS=10 LANCE_SAVE_INTERVAL=10 \
LANCE_TRACE_FILE=outputs/lance-native-resume-20/trace.jsonl \
NPROC_PER_NODE=8 bash scripts/pretrain_lance_native.sh

# 从同一 DCP 恢复到 step 20
LANCE_LOAD_DCP=outputs/lance-native-resume-20 \
LANCE_OUTPUT_DIR=outputs/lance-native-resume-20 \
LANCE_TRAIN_ITERS=20 LANCE_STOP_AFTER_ITERS=20 LANCE_SAVE_INTERVAL=10 \
LANCE_TRACE_FILE=outputs/lance-native-resume-20/trace.jsonl \
NPROC_PER_NODE=8 bash scripts/pretrain_lance_native.sh

python scripts/compare_lance_native_traces.py \
  --continuous outputs/lance-native-continuous-20/trace.jsonl \
  --resumed outputs/lance-native-resume-20/trace.jsonl
```

比较器逐 rank 检查 iteration、实际 batch path、loss、CE/MSE、token 数、grad norm 和 LR。
默认完整训练不设置 `LANCE_TRACE_FILE`，不会承担逐 step 文件写入。

### 4. 启动完整 PT

```bash
NPROC_PER_NODE=8 bash scripts/pretrain_lance_native.sh
```

默认配置固定 350K steps、2500 warmup、LR `1e-4` constant、CE:MSE=`0.25:1`、AdamW
`beta=(0.9,0.95)`/`eps=1e-15`、clip `1.0`、EMA `0.9999`。输出为可严格恢复 optimizer、
scheduler、dataloader cursor、EMA 的 MindSpeed-MM DCP。

训练入口启动前会检查 DCP tracker/release、初始化配置、tokenizer 有效词表、packed manifest、
模型 variant 和 batch 数；任何不一致都会在创建 8 个训练进程前失败。当前原生 PT 配置固定
TP=CP=1、FSDP2=8，I2V/subject/interleaved 以及 CP/70K 属于后续 CT/SFT 扩展，不应混入这次
350K example PT 的验收结论。

当前开发机没有 torch/torch-npu，因此这里完成的是静态编译和契约测试；上述三条 NPU 命令仍需
在已配置的容器内执行后，才能把原生路径标记为实机验收通过。

## 原生 T2I/T2V/I2T 推理

`inference_lance_native.py` 直接加载原生训练产生的 DCP 和 `ema_state.parameters`，不会发现、
导入或执行 Lance 官方源码，也不需要先转换为 safetensors。当前入口面向单卡 Ascend 推理，支持
`t2i`、`t2v` 和 `i2t`；默认使用 EMA，传 `--model-weights` 可改用普通模型参数。

对本文 PT 配置必须保持 `latent_patch_size=1 2 2`。这要求输出高宽均为 32 的倍数；Wan2.2 的
因果时间结构要求视频帧数为 `4k+1`。因此原生默认值是 T2I `768x768x1`，T2V
`480x864x49`，而不是发布脚本为 `1 1 1` checkpoint 使用的 `480x848x50`。

先做不分配模型/NPU 的参数和路径检查：

```bash
python inference_lance_native.py \
  --checkpoint /mnt/qs/mod/MindSpeed-MM/outputs/lance-native-pt/iter_0001500 \
  --qwen-path /mnt/qs/models/Qwen/Qwen2.5-VL-3B-Instruct \
  --vae-path /mnt/qs/models/bytedance-research/Lance/Wan2.2_VAE.pth \
  --task t2i \
  --prompt "A red panda wearing sunglasses, cinematic lighting." \
  --output-dir outputs/lance-native-t2i-1500 \
  --dry-run
```

去掉 `--dry-run` 即可执行。T2V 示例：

```bash
python inference_lance_native.py \
  --checkpoint /mnt/qs/mod/MindSpeed-MM/outputs/lance-native-pt/iter_0001500 \
  --qwen-path /mnt/qs/models/Qwen/Qwen2.5-VL-3B-Instruct \
  --vae-path /mnt/qs/models/bytedance-research/Lance/Wan2.2_VAE.pth \
  --task t2v \
  --prompt "A red panda surfing a bright seaside wave, tracking shot." \
  --height 480 --width 864 --num-frames 49 --fps 12 \
  --output-dir outputs/lance-native-t2v-1500
```

也可使用封装脚本；`LANCE_CHECKPOINT` 既可指向具体 iteration，也可指向带 tracker 的 checkpoint
根目录：

```bash
LANCE_CHECKPOINT=/mnt/qs/mod/MindSpeed-MM/outputs/lance-native-pt/iter_0001500 \
LANCE_TASK=t2i \
LANCE_PROMPT="A red panda wearing sunglasses, cinematic lighting." \
LANCE_OUTPUT_DIR=outputs/lance-native-t2i-1500 \
bash scripts/inference_lance_native.sh
```

使用 Lance 官方 image-understanding 示例运行 I2T：

```bash
LANCE_CHECKPOINT=/mnt/qs/mod/MindSpeed-MM/outputs/lance-native-pt/iter_0001500 \
LANCE_TASK=i2t \
LANCE_OFFICIAL_ROOT=/mnt/qs/mod/Lance \
LANCE_CONFIG_PATH=/mnt/qs/mod/Lance/config/examples/x2t_image_example.json \
VIT_PATH=/mnt/qs/models/bytedance-research/Lance/Qwen2.5-VL-ViT \
QWEN_PATH=/mnt/qs/models/Qwen/Qwen2.5-VL-3B-Instruct \
LANCE_OUTPUT_DIR=outputs/lance-native-i2t-1500 \
bash scripts/inference_lance_native.sh
```

I2T 复用官方 `x2t_image_example.json` 的 `interleave_array`、`element_dtype_array` 和
`istarget_in_interleave` 结构。相对图片路径会从当前目录和 JSON 的各级父目录解析，因此官方
`assets/image-understanding/...` 路径无需改写。理解模型和 PT connector 从 DCP/EMA 加载；训练时
冻结且未写入 DCP 的 Qwen2.5-VL ViT 从 `VIT_PATH` 单独加载。I2T 不加载 Wan VAE，使用 KV-cache
进行最多 256 token 的 greedy decoding。

T2I/T2V 输出目录包含 `000000.png` 或 `000000.mp4`；I2T 输出 `result.json` 和 `prompt.json`。
三者都会写入记录 checkpoint、模型/EMA 选择和推理参数的 `lance_native_inference.json`。生成任务
当前为了保持 prompt builder 的严格注意力语义使用完整序列 sampler；理解任务使用增量 KV-cache。

## 历史第一阶段：官方推理与评测基线

本目录提供 Lance 官方 checkpoint 的零转换推理桥接和论文评测协议。第一阶段保留官方
Lance 的模型、tokenizer、VAE 和数据处理语义，只在独立进程内将 CUDA/NCCL/FlashAttention
映射到 NPU/HCCL/`torch_npu.npu_fusion_attention`，用于先做数值对齐。完成对齐后再将共享
模型并入 MindSpeed-MM 的原生训练构建器。

这一节只用于保留已经完成的论文推理基线，不被上述原生预训练入口调用。

## 前置条件

- Python 3.10，MindSpeed-MM 对应的 PyTorch 2.7.1、torch-npu 和 CANN 环境；
- Lance 官方源码（默认自动查找与 `MindSpeed-MM` 同级的 `Lance`，也可设置
  `LANCE_SOURCE_ROOT`）；
- Lance 官方 checkpoint 和 Wan2.2 VAE 权重，路径配置沿用 Lance 的
  `config/path_default.yaml`；
- 多卡任务使用 `torchrun`，通信后端由适配器从 NCCL 映射为 HCCL。

为保证 scorer 不随上游仓库更新漂移，论文对齐固定以下版本；Git scorer 必须处于对应 revision
且工作区干净，否则归一化命令直接失败：

| Scorer | 固定版本 |
|---|---|
| GenEval | `djghosh13/geneval@af4902f24d3ca90ebbb446dd9891a59e0f82725f` |
| DPG-Bench | `TencentQQGYLab/ELLA@3c228f1dc6c4d3cad0a47493816151a419f14db3` |
| GEdit-Bench | `stepfun-ai/Step1X-Edit@5d350cdbeefc8108c8cd9d4134bbb0d33ee05a74` |
| VBench | PyPI `vbench==0.1.2`，再应用 MindSpeed-MM NPU patch |

先在目标昇腾环境执行数值 smoke test。它使用 Lance 的 16 个 Q 头、2 个 KV 头和
KV-cache 非等长 causal attention，将 NPU BF16 结果与 CPU FP32 参考实现比较：

```bash
python inference_lance.py --runtime-check
```

## 单样例推理

先检查解析结果，不加载模型：

```bash
python inference_lance.py --dry-run \
  --lance-source-root ../Lance \
  --task t2i \
  --model_path /path/to/Lance_3B
```

实际执行时去掉 `--dry-run`。其余参数完整透传到官方 `inference_lance.py`，因此七种任务
`t2i`、`t2v`、`i2v`、`image_edit`、`video_edit`、`x2t_image`、`x2t_video` 均使用同一入口。

## 官方权重只读审计

在加载 12--14 GB tensor 数据或写 DCP 之前，先只读 safetensors JSON header，逐 key 检查
名称、shape、BF16 dtype、offset 和总字节数：

```bash
python convert_lance_checkpoint.py inspect \
  --checkpoint /path/to/Lance_3B/model.safetensors \
  --variant image --fingerprint \
  --output /path/to/results/lance-image-checkpoint-audit.json

python convert_lance_checkpoint.py inspect \
  --checkpoint /path/to/Lance_3B_Video/model.safetensors \
  --variant video --fingerprint \
  --output /path/to/results/lance-video-checkpoint-audit.json
```

已用 Hugging Face 官方文件的真实 header 验证以下契约：

| 变体 | tensors | BF16 元素 | tensor 数据字节 | `latent_pos_embed` |
|---|---:|---:|---:|---:|
| Lance_3B image | 1021 | 6,185,205,808 | 12,370,411,616 | `[4096, 2048]` |
| Lance_3B_Video | 1411 | 7,105,548,336 | 14,211,096,672 | `[126976, 2048]` |

生成转换计划时仍会先执行同一审计。原生模型刻意保留官方参数名，所以初始映射是 lossless identity；
DP/CP/TP 分片由目标 DCP writer 完成，而不是通过含糊的 key 重命名完成：

```bash
python convert_lance_checkpoint.py plan \
  --checkpoint /path/to/Lance_3B_Video/model.safetensors \
  --variant video --fingerprint \
  --output /path/to/results/lance-video-conversion-plan.json
```

只有 `latent_pos_embed.pos_embed` 可以通过
`--allow-rebuild-position-embedding` 声明缺失；它是确定性的 3D sin/cos 表。其他 missing 或
unexpected key 一律阻断转换。

完整 safetensors 审计通过后，可以直接写成 MindSpeed 分布式 checkpoint（DCP）布局，再以
metadata-only 方式逐 key 回读，不需要分配 12--14 GB 的完整模型：

```bash
python convert_lance_checkpoint.py to-dcp \
  --checkpoint /path/to/Lance_3B_Video/model.safetensors \
  --variant video \
  --output-dir /path/to/checkpoints/lance-video-dcp \
  --output /path/to/results/lance-video-dcp-conversion.json

python convert_lance_checkpoint.py verify-dcp \
  --dcp-dir /path/to/checkpoints/lance-video-dcp \
  --variant video \
  --output /path/to/results/lance-video-dcp-audit.json
```

转换器拒绝覆盖非空目录，并同时写入 `release/.metadata`、
`latest_checkpointed_iteration.txt` 和带源文件 SHA-256 的转换 manifest。

原生模型也提供逐 tensor streaming loader。它在写入模型前执行同一完整审计，每次只从 CPU
materialize 一个源 tensor；需要先构造非 meta 的 BF16 `LanceNativeModel`，再调用
`get_native_checkpoint_loader()` 返回的加载函数。原生 evaluation 推荐同时启用
`AscendBlockAttentionBackend`、`AscendVisionAttentionBackend` 和
`AscendKVCacheAttentionBackend`；KV-cached sampler 可由 `get_native_cached_sampler()` 获取。

原生缓存路径只接受能证明等价的“静态 condition prefix + 连续 noisy-VAE suffix”。常见 T2I/T2V
可直接使用；复杂 edit 模板若目标 token 非连续后缀会主动报错，应回退到完整序列 sampler，不能
为了性能改变原注意力关系。

开发时若只通过 HTTP Range 获取了 header，可额外使用 `--metadata-only` 检查结构；该结果明确
标记为 metadata-only，不能作为推理或 evaluation 的完整文件门禁。

## 论文评测协议

检查官方数据文件的样本数和 SHA-256：

```bash
python evaluate_lance.py validate --benchmark all --lance-source-root ../Lance
```

若同时检查 checkpoint，单个模型只能对应一个发布变体；全 benchmark 预检时需要明确指定：

```bash
python evaluate_lance.py validate --benchmark all --lance-source-root ../Lance \
  --model-path /path/to/Lance_3B_Video --model-variant video
```

非 dry-run 的 `sample` 会按任务自动选择 image/video 契约，并在创建运行 manifest 前检查完整
safetensors payload 长度。header 正确但 tensor 数据残缺的文件不能进入 evaluation。

查看完整机器可读协议：

```bash
python evaluate_lance.py list
```

以 8 卡 GenEval 为例，先 dry-run，再执行：

```bash
python evaluate_lance.py sample --benchmark geneval \
  --lance-source-root ../Lance \
  --model-path /path/to/Lance_3B \
  --output-path /path/to/results/geneval \
  --world-size 8 --dry-run

torchrun --nproc_per_node 8 evaluate_lance.py sample --benchmark geneval \
  --lance-source-root ../Lance \
  --model-path /path/to/Lance_3B \
  --output-path /path/to/results/geneval \
  --world-size 8
```

非 dry-run 会在 rank 0 先写入状态为 `running` 的 `lance_eval_run.json`，正常返回后自动执行
产物数量审计并更新为 `completed`；若上游异常退出，manifest 会保留 `running`，若产物缺失则
更新为 `invalid` 并让命令失败。

`dpgbench`、`gedit`、`vbench` 的采样用法相同。VBench 严格采用发布脚本的 30 steps、shift 3.0、
480x848、50 帧、12 fps；普通提示词生成 5 个视频，官方列出的 75 个 temporal-flickering
提示词生成 25 个视频，总计 6230 个。GenEval/DPG-Bench 使用 50 steps、shift 3.5、768x768
和每提示词 4 个样本。

## 五项 scorer 与结果归一化

先按 GenEval 官方说明运行 detector，得到 2212 行 `results.jsonl`，再归一化并绑定生成 manifest：

```bash
python /path/to/geneval/evaluation/evaluate_images.py \
  /path/to/results/geneval \
  --outfile /path/to/results/geneval-score/results.jsonl \
  --model-path /path/to/mask2former

python evaluate_lance.py geneval-score \
  --results /path/to/results/geneval-score/results.jsonl \
  --scorer-root /path/to/geneval \
  --run-manifest /path/to/results/geneval/lance_eval_run.json \
  --output /path/to/results/geneval-score/metrics.json
```

DPG-Bench 的 1065 个 PNG 每个都是四张生成图拼成的 2×2 网格。按 ELLA 官方命令运行 mPLUG
scorer，并将它产生的完整文本结果交给归一化器：

```bash
(cd /path/to/ELLA && bash dpg_bench/dist_eval.sh /path/to/results/dpgbench 768)

python evaluate_lance.py dpgbench-score \
  --results /path/to/dpgbench-score/results.txt \
  --scorer-root /path/to/ELLA \
  --run-manifest /path/to/results/dpgbench/lance_eval_run.json \
  --output /path/to/results/dpgbench-score/metrics.json
```

GEdit 先按固定 revision 的 `GEdit-Bench/EVAL.md` 运行 GPT-4.1 judge，产生 11 个类别 CSV；
归一化器逐样本计算 `sqrt(semantic * quality)`，再对 11 个类别做宏平均：

```bash
python evaluate_lance.py gedit-score \
  --score-dir /path/to/gedit-score/csv \
  --model-name lance \
  --judge gpt-4.1 --language en \
  --scorer-root /path/to/Step1X-Edit \
  --run-manifest /path/to/results/gedit/lance_eval_run.json \
  --output /path/to/results/gedit-score/metrics.json
```

VBench 采样完成后可直接运行已适配的 NPU scorer，不会重新加载 Lance。命令会先强制检查
`vbench==0.1.2`、6230 个视频和全部 16 个维度：

```bash
torchrun --nproc_per_node 8 evaluate_lance.py vbench-score \
  --videos-path /path/to/results/vbench \
  --full-info-path /path/to/VBench_full_info.json \
  --output-dir /path/to/results/vbench-score \
  --run-manifest /path/to/results/vbench/lance_eval_run.json \
  --output /path/to/results/vbench-score/metrics.json
```

注意：GEdit 的论文指标是 GPT-4.1 裁判得到的 `G_O`；离线 Qwen 裁判得到的是 `Q_O`，
不能作为同一指标。`--allow-partial` 仅用于调试 scorer 输入，产生的计数不满足门禁，不能进入论文
对齐报告。

## MVBench

Lance 官方仓库未发布 MVBench sampler。本适配补充了带时间边界样本裁剪、帧目录转视频和
官方首选项匹配。需要特别区分两个协议：MVBench 官方完整版为 20 个任务/4000 条；Lance
论文 Table 8 只报告 19 个任务，遗漏 Fine-grained Pose，表中 19 项宏平均正好为 62.0。
因此默认 `--task-set paper` 生成论文口径的 19 任务/3800 条：

```bash
python evaluate_lance.py mvbench-prepare \
  --annotation-root /path/to/MVBench/json \
  --media-root /path/to/MVBench/media \
  --output-dir /path/to/results/mvbench-prepared

torchrun --nproc_per_node 8 evaluate_lance.py sample --benchmark mvbench \
  --lance-source-root ../Lance \
  --model-path /path/to/Lance_3B_Video \
  --dataset-path /path/to/results/mvbench-prepared/mvbench_lance.json \
  --output-path /path/to/results/mvbench \
  --world-size 8

python evaluate_lance.py mvbench-score \
  --metadata /path/to/results/mvbench-prepared/mvbench_metadata.json \
  --results /path/to/results/mvbench/result.json \
  --run-manifest /path/to/results/mvbench/lance_eval_run.json \
  --output /path/to/results/mvbench/metrics.json
```

评分 JSON 同时给出 Table 8 的 19 个逐任务目标值、实际值和 delta。

若要额外报告官方完整口径，在 prepare 和 score 两个命令都加 `--task-set official`，并在 sample
命令加 `--mvbench-task-set official`；该结果
会标记为 `official-mvbench-20`，不能拿来与论文 62.0 直接比较。

采样完成后可先检查结果数量，例如：

```bash
python evaluate_lance.py audit --benchmark mvbench --output-path /path/to/results/mvbench
```

## 合并并生成论文对齐报告

五项 scorer 都完成后合并标准化 JSON，再执行严格报告：

```bash
python evaluate_lance.py merge --metrics \
  /path/to/results/geneval-score/metrics.json \
  /path/to/results/dpgbench-score/metrics.json \
  /path/to/results/gedit-score/metrics.json \
  /path/to/results/vbench-score/metrics.json \
  /path/to/results/mvbench/metrics.json \
  --output /path/to/results/lance-paper-metrics.json

python evaluate_lance.py report \
  --metrics /path/to/results/lance-paper-metrics.json \
  --output /path/to/results/lance-paper-alignment.json
```

报告只有在五项分数齐全、采样数量正确、scorer 版本固定，并且每项都绑定了 checkpoint SHA-256、
Lance revision、数据 SHA-256、原始 scorer 结果 SHA-256 和运行 manifest SHA-256 时，才会将
`paper_comparable` 标为 `true`。

## 原生训练准备门禁

训练前先生成可审计配置。每个数据 manifest 的路径、大小和 SHA-256 都会写入结果；未提供数据
manifest 时命令返回 3 且状态为 `dataset-manifest-required`，不能误当作可启动训练：

```bash
python prepare_lance_training.py \
  --stage pt \
  --init-mode qwen2_5_vl \
  --init-path /path/to/Qwen2.5-VL-3B-Instruct \
  --variant video \
  --world-size 64 \
  --dataset-manifest /path/to/Lance/config/train/pt.yaml \
  --dataset-manifest /path/to/video-generation.json \
  --dataset-manifest /path/to/video-understanding.json \
  --dataset-manifest /path/to/image-generation.json \
  --dataset-manifest /path/to/image-understanding.json \
  --output /path/to/results/lance-pt-manifest.json
```

`qwen2_5_vl` 表示论文复现初始化：加载 Qwen2.5-VL understanding 参数并复制到 generation
专家；`random` 才表示所有参数严格随机初始化；`lance_checkpoint` 只用于续训/微调。三者会写入
不同 provenance 标签，PT 不允许用 Lance checkpoint 冒充 from-scratch。

## PT 上游基线桥

当前已增加 `pretrain_lance.py`，先以官方 `train/unified_train.py` 建立 Ascend 训练基线。它不复制或
修改 Lance checkout，而是在初始化 torch/NPU 前执行以下强门禁：

- 上游训练源码必须完整、处于 clean Git revision；
- 训练 manifest、初始化路径和 PackedDataset YAML 必须存在，数据 YAML 大小及 SHA-256 不得漂移；
- PT steps、token budget、loss/dropout、AdamW、EMA、冻结策略和 FSDP 拓扑必须与内置阶段契约一致；
- `WORLD_SIZE` 必须与 manifest 及 `num_replicate * num_shard` 一致；
- 官方 step 级宽泛异常处理会在内存中转换为 fail-fast，转换只匹配已审计的 AST 结构，上游文件不落盘。

manifest 中至少要包含实际传给 `--dataset_config_file` 的 PackedDataset YAML；其他数据清单可以继续
附加。单机 8 卡 PT 的准备和只读 preflight 示例：

```bash
python prepare_lance_training.py \
  --stage pt \
  --init-mode qwen2_5_vl \
  --init-path /path/to/Qwen2.5-VL-3B-Instruct \
  --variant video \
  --world-size 8 \
  --dataset-manifest /path/to/Lance/config/train/pt.yaml \
  --output /path/to/results/lance-pt-manifest.json

TRAINING_MANIFEST=/path/to/results/lance-pt-manifest.json \
QWEN_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
VIT_PATH=/path/to/Qwen2.5-VL-ViT \
DATASET_CONFIG_FILE=/path/to/Lance/config/train/pt.yaml \
PREFLIGHT_ONLY=1 \
bash scripts/pretrain_lance_pt.sh
```

正式运行前，将 Wan2.2 VAE 放到官方 `config/path_default.yaml` 解析出的路径，然后执行：

```bash
TRAINING_MANIFEST=/path/to/results/lance-pt-manifest.json \
QWEN_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
VIT_PATH=/path/to/Qwen2.5-VL-ViT \
DATASET_CONFIG_FILE=/path/to/Lance/config/train/pt.yaml \
bash scripts/pretrain_lance_pt.sh
```

Ascend 启动脚本默认使用每 rank 0 个 DataLoader worker，避免 8 卡任务中的多进程张量传输耗尽
容器 `/dev/shm`。确认容器配置了足够大的共享内存后，可以显式设置 `NUM_WORKERS` 开启并行加载。

设置 `SMOKE_TEST=1` 会进一步切换为 20 steps、适合 768px 图像样例的缩小 token budget 和
`config/train_local/t2i_local.yaml`。这样也不会让示例视频的约 6 万 token 序列阻塞轻量冒烟。可通过
`DATASET_CONFIG_FILE` 覆盖 smoke 数据配置。`--smoke-test` 只允许缩小 steps、
warmup 和三项 token budget，其余语义参数仍须严格一致。Ascend 训练固定启用 `use_flex`，让
PackedDataset 不生成 O(L²) dense mask；MindSpeed-MM 在训练进程内截获官方 FlexAttention mask 元数据，
以紧凑的分段描述调用 NPU fused attention，不会执行 NPU 不支持的 TorchInductor/FlexAttention。
当前上游桥
明确拒绝 `random` 初始化和 RL：官方入口没有严格随机初始化路径，且 `unified_train.py` 是监督训练
循环。二者继续由原生 TrainEngine 路线实现。

要覆盖 ViT、understanding expert 和 CE backward，可在 T2I smoke 通过后运行纯理解 smoke：

```bash
SMOKE_TEST=1 NUM_WORKERS=0 \
DATASET_CONFIG_FILE=/mnt/qs/Lance/config/train_local/i2t_local.yaml \
WANDB_NAME=lance-pt-i2t-smoke \
bash scripts/pretrain_lance_pt.sh
```

对 `i2t_local.yaml`、`v2t_local.yaml` 和 `multi_und.yaml`，smoke 启动器会自动设置
`visual_gen=false`。这些数据没有 VAE target；关闭 generation 分支可避免官方模型对空
`padded_latent` 执行 MSE 前向。正式 PT 和生成类 smoke 仍保持 `visual_gen=true`。

官方 `config/train_local/unified.yaml` 中的 `datasets/...` 是相对于 Lance checkout 的路径。启动脚本
默认将 `/mnt/qs/datasets/bytedance-research/Lance_example_dataset` 映射为
`/mnt/qs/Lance/datasets`，并在启动分布式进程前检查 unified 配置引用的六个 parquet。数据放在其他位置时
设置 `DATASET_ROOT=/absolute/path/to/Lance_example_dataset`；如果下载目录外层还包含一个 `datasets/`
目录，脚本会自动识别。

官方 Lance `PackedDataset` 可继续负责 tokenizer、模板和 parquet 采样。冻结 Wan2.2 VAE/ViT
编码后，使用 `prepare_upstream_lance_batch()` 转为原生 `LanceTrainingBatch`；该适配保留
`split_lens/attn_modes`，并独立计算逐 token MoT 路由，因此 `full_noise` split 内的视觉边界文本
仍走 understanding expert，VAE token 仍走 generation expert。长序列应选择 `attention_backend="ascend"`，
不得生成 dense mask。
