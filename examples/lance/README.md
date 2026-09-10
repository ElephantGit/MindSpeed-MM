# Lance 昇腾推理与评测（第一阶段）

本目录提供 Lance 官方 checkpoint 的零转换推理桥接和论文评测协议。第一阶段保留官方
Lance 的模型、tokenizer、VAE 和数据处理语义，只在独立进程内将 CUDA/NCCL/FlashAttention
映射到 NPU/HCCL/`torch_npu.npu_fusion_attention`，用于先做数值对齐。完成对齐后再将共享
模型并入 MindSpeed-MM 的原生训练构建器。

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
warmup 和三项 token budget，其余语义参数仍须严格一致。Ascend 训练固定启用 `use_flex`；在该设备上
它选择紧凑的 `SegmentedAttentionMask` 和 NPU fused attention，并不会执行 CUDA/PyTorch FlexAttention。
当前上游桥
明确拒绝 `random` 初始化和 RL：官方入口没有严格随机初始化路径，且 `unified_train.py` 是监督训练
循环。二者继续由原生 TrainEngine 路线实现。

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
