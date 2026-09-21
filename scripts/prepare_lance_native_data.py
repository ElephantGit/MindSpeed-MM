#!/usr/bin/env python3
"""Encode Lance example parquet into native MindSpeed-MM prepared samples.

The script owns media decoding, frozen Qwen ViT inference and frozen Wan2.2
VAE inference.  It never imports the standalone Lance repository.  Run it with
torchrun to shard the 350K example data across available NPUs, then feed its
output to ``pack_lance_native_data.py``.
"""

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("NON_MEGATRON", "true")

import torch
from PIL import Image

from mindspeed_mm.models.omni.lance.initialization import load_native_vit_checkpoint
from mindspeed_mm.models.omni.lance.modeling_lance import LanceVisionModel, reference_vision_sdpa
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.npu_attention import AscendVisionAttentionBackend
from mindspeed_mm.models.omni.lance.preprocessing import (
    LanceEncodedVisual,
    build_edit_sample,
    build_generation_sample,
    build_understanding_sample,
    patchify_qwen_video,
    prepare_lance_tokenizer,
)
from mindspeed_mm.models.omni.lance.wan_vae import LanceWanVAE


VIT_MEAN = (0.48145466, 0.4578275, 0.40821073)
VIT_STD = (0.26862954, 0.26130258, 0.27577711)
ASPECT_RATIOS = ((21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Prepare bridge-free native Lance data")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--qwen-path", required=True)
    parser.add_argument("--vit-path", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=("image", "video"), default="video")
    parser.add_argument("--latent-patch-size", nargs=3, type=int)
    parser.add_argument("--max-latent-size", type=int)
    parser.add_argument("--max-num-frames", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--max-samples-per-task", type=int,
        help="Encode at most this many rows from each of the six base PT tasks per rank",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--text-cond-dropout-prob", type=float, default=0.1)
    parser.add_argument(
        "--emit-tasks",
        help="Comma list of tasks to emit per generation-schema row, e.g. t2i,i2t "
             "(t2i/t2v rows can additionally emit their i2t/v2t twin). "
             "Default: only the schema-detected task",
    )
    parser.add_argument(
        "--base-tasks",
        help="Optional comma list of schema-detected source tasks to include before "
             "applying --max-samples, e.g. t2i.  This filters a mixed six-task root "
             "without changing --emit-tasks semantics.",
    )
    parser.add_argument("--sample-posterior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resample-posterior-during-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store VAE mean/log-variance so every training visit resamples like upstream Lance",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-failure-rate", type=float, default=0.05)
    return parser.parse_args()


def _device():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(local_rank)
        return torch.device("npu", local_rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    raise RuntimeError("native media preparation requires an NPU or CUDA accelerator")


def _bucket_size(width, height, resolution, stride):
    if isinstance(stride, (tuple, list)):
        if len(stride) != 2:
            raise ValueError("spatial stride must contain height and width factors")
        height_stride, width_stride = (int(item) for item in stride)
    else:
        height_stride = width_stride = int(stride)
    if height_stride <= 0 or width_stride <= 0:
        raise ValueError("spatial stride factors must be positive")
    ratio = width / height
    target_ratio = min((w / h for w, h in ASPECT_RATIOS), key=lambda item: abs(item - ratio))
    width_a = round((resolution * resolution * target_ratio) ** 0.5 / width_stride) * width_stride
    height_a = round((width_a / target_ratio) / height_stride) * height_stride
    height_b = round((resolution * resolution / target_ratio) ** 0.5 / height_stride) * height_stride
    width_b = round((height_b * target_ratio) / width_stride) * width_stride
    candidates = ((max(width_stride, width_a), max(height_stride, height_a)),
                  (max(width_stride, width_b), max(height_stride, height_b)))
    return min(candidates, key=lambda item: (abs(item[0] / item[1] - target_ratio),
                                             abs(item[0] * item[1] - resolution * resolution)))


def _transform(frames, resolution, stride, mean, std):
    from torchvision.transforms import functional as tvf
    from torchvision.transforms import InterpolationMode

    width, height = frames[0].size
    target_width, target_height = _bucket_size(width, height, resolution, stride)
    scale = max(target_width / width, target_height / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    tensors = []
    for frame in frames:
        value = tvf.resize(
            frame, (resized_height, resized_width), interpolation=InterpolationMode.LANCZOS,
            antialias=True,
        )
        value = tvf.crop(value, top, left, target_height, target_width)
        value = tvf.to_tensor(value)
        value = tvf.normalize(value, mean, std)
        tensors.append(value)
    return torch.stack(tensors).permute(1, 0, 2, 3).contiguous()


def _image_frames(value):
    image = Image.open(BytesIO(value))
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        image = background
    return [image.convert("RGB")]


def _sample_video_indices(total, origin_fps, *, sample_fps=12, max_duration=6, temporal=4):
    """Port MultiClipsFrameSampler for one full-video clip."""

    total = int(total)
    fps = int(origin_fps)
    if total <= 0:
        raise ValueError("cannot sample an empty video")
    if fps <= 0 or sample_fps <= 0 or max_duration <= 0 or temporal <= 0:
        raise ValueError("video sampling rates, duration, and temporal stride must be positive")
    duration = min(total / fps, float(max_duration))
    count = int(round(duration * sample_fps))
    if count % temporal:
        count = (count // temporal) * temporal + 1
    else:
        count = (count // temporal) * temporal + 1 - temporal
    if count <= 0:
        raise ValueError(
            "video is too short for Lance's kn+1 temporal sampler: "
            "frames={}, fps={}".format(total, fps)
        )
    # np.linspace(..., dtype=int), used upstream, truncates positive values in
    # the same way as Tensor.long().  Counts larger than ``total`` intentionally
    # repeat source frames.
    return torch.linspace(0, total - 1, count).long().tolist()


def _video_frames(value, *, sample_fps=12, max_duration=6, temporal=4):
    import decord

    reader = decord.VideoReader(BytesIO(value), ctx=decord.cpu(0))
    total = len(reader)
    # Match Lance's BaseMMParquetDataset: average FPS is rounded to an integer
    # before MultiClipsFrameSampler sees it, with 24 FPS as the decoder fallback.
    try:
        fps = int(round(float(reader.get_avg_fps())))
    except Exception:
        fps = 24
    if fps <= 0:
        fps = 24
    indexes = _sample_video_indices(
        total, fps, sample_fps=sample_fps,
        max_duration=max_duration, temporal=temporal,
    )
    return [Image.fromarray(item) for item in reader.get_batch(indexes).asnumpy()]


class FeatureEncoder:
    def __init__(
        self, config, vit_path, vae_path, device, sample_posterior,
        resample_posterior_during_training,
    ):
        self.config = config
        self.device = device
        backend = AscendVisionAttentionBackend() if device.type == "npu" else None
        self.vit = LanceVisionModel(
            config,
            attention_backend=backend or reference_vision_sdpa,
            device=device,
            dtype=torch.bfloat16,
        ).eval().requires_grad_(False)
        self.vit_report = load_native_vit_checkpoint(self.vit, vit_path)
        self.vae = LanceWanVAE(
            vae_path, device=device, dtype=torch.bfloat16,
            sample_posterior=sample_posterior,
        )
        self.resample_posterior_during_training = bool(
            resample_posterior_during_training
        )

    @torch.no_grad()
    def vit_encode(self, frames, *, resolution):
        value = _transform(frames, resolution, 28, VIT_MEAN, VIT_STD)
        if value.shape[1] == 1:
            value = value.repeat(1, 2, 1, 1)
        elif value.shape[1] % 2:
            value = torch.cat((value, value[:, -1:]), dim=1)
        patches = patchify_qwen_video(value).to(self.device, dtype=torch.bfloat16)
        grid = torch.tensor(
            [[value.shape[1] // 2, value.shape[2] // 14, value.shape[3] // 14]],
            device=self.device, dtype=torch.long,
        )
        embedding = self.vit(patches, grid).to(device="cpu", dtype=torch.bfloat16)
        return embedding, tuple(int(item) for item in grid[0].tolist())

    def vae_encode(self, frames, *, resolution):
        _, patch_h, patch_w = self.config.latent_patch_size
        stride = (
            self.vae.spatial_downsample * patch_h,
            self.vae.spatial_downsample * patch_w,
        )
        value = _transform(frames, resolution, stride, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if self.resample_posterior_during_training:
            return self.vae.encode_distribution((value,))[0]
        return self.vae.encode((value,))[0], None

    def visual(
        self, value, modality, *, need_vit, need_vae, edit=False, max_duration=6
    ):
        frames = (
            _image_frames(value)
            if modality == "image"
            else _video_frames(value, max_duration=max_duration)
        )
        vit_resolution = 672 if edit and modality == "image" else 616
        vit, vit_grid = (
            self.vit_encode(frames, resolution=vit_resolution)
            if need_vit else (None, None)
        )
        vae, log_variance = (
            self.vae_encode(frames, resolution=768 if modality == "image" else 640)
            if need_vae else (None, None)
        )
        return LanceEncodedVisual(
            modality,
            vae_latent=vae,
            vae_log_variance=log_variance,
            vit_embedding=vit,
            vit_grid_thw=vit_grid,
        )


def _task(path, row):
    keys = set(row)
    lowered = str(path).lower()
    if {"input_image_bytes", "output_image_bytes"} <= keys:
        return "i2i"
    if {"input_video_bytes", "output_video_bytes"} <= keys:
        return "v2v"
    if "image_bytes" in keys:
        return "i2t" if "image2text" in lowered or "caption_a" in keys else "t2i"
    if "video_bytes" in keys:
        return "v2t" if "video2text" in lowered or "caption_a" in keys else "t2v"
    raise ValueError("unrecognized Lance example row schema: {}".format(sorted(keys)))


def _caption_answer(row):
    if "caption_a" in row:
        return str(row.get("caption_q", "")).strip(), str(row["caption_a"])
    return "", str(row.get("caption", ""))


_UNDERSTANDING_TWIN = {"t2i": "i2t", "t2v": "v2t"}


def _stable_number(sample_id, seed, namespace):
    payload = "{}\0{}\0{}".format(seed, namespace, sample_id).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _prepare(
    path, row, sample_id, encoder, tokenizer, config, *, seed, text_dropout,
    emit_tasks=None,
):
    """Return the list of (task, sample) pairs requested for one source row.

    ``emit_tasks=None`` keeps the schema-detected task only.  Generation-schema
    rows (t2i/t2v) may additionally emit their understanding twin (i2t/v2t)
    from the same media: one frozen-encoder pass produces both the VAE target
    and the ViT condition, and caption dropout only ever applies to the
    generation half.
    """

    base_task = _task(path, row)
    requested = emit_tasks if emit_tasks is not None else {base_task}
    results = []
    if base_task in ("t2i", "t2v"):
        modality, key = ("image", "image_bytes") if base_task == "t2i" else ("video", "video_bytes")
        twin = _UNDERSTANDING_TWIN[base_task]
        visual = encoder.visual(
            row[key], modality, need_vit=twin in requested, need_vae=True
        )
        if base_task in requested:
            target = LanceEncodedVisual(
                modality,
                vae_latent=visual.vae_latent,
                vae_log_variance=visual.vae_log_variance,
            )
            dropout_draw = _stable_number(sample_id, seed, "text-dropout") / 2**64
            caption = None if dropout_draw < text_dropout else row["caption"]
            generation_task = base_task
            condition_frames = ()
            if base_task == "t2v":
                # Match train_local's task_type_rate=[0.8, 0.2] for t2v/ff2v.
                variant_draw = _stable_number(sample_id, seed, "video-task") / 2**64
                if variant_draw >= 0.8:
                    generation_task = "ff2v"
                    condition_frames = (0,)
            results.append((generation_task, build_generation_sample(
                sample_id,
                caption,
                target,
                tokenizer,
                config,
                condition_frames=condition_frames,
            )))
        if twin in requested:
            condition = LanceEncodedVisual(
                modality,
                vit_embedding=visual.vit_embedding,
                vit_grid_thw=visual.vit_grid_thw,
            )
            prompt, answer = _caption_answer(row)
            results.append((twin, build_understanding_sample(
                "{}:{}".format(sample_id, twin), prompt, answer, condition,
                tokenizer, config,
            )))
        return results
    if base_task in ("i2t", "v2t"):
        modality, key = ("image", "image_bytes") if base_task == "i2t" else ("video", "video_bytes")
        condition = encoder.visual(
            row[key], modality, need_vit=True, need_vae=False,
            max_duration=2 if modality == "video" else 6,
        )
        prompt, answer = _caption_answer(row)
        return [(base_task, build_understanding_sample(
            sample_id, prompt, answer, condition, tokenizer, config,
        ))]
    if base_task == "i2i":
        condition = encoder.visual(row["input_image_bytes"], "image", need_vit=True, need_vae=True, edit=True)
        target = encoder.visual(row["output_image_bytes"], "image", need_vit=False, need_vae=True, edit=True)
    else:
        condition = encoder.visual(row["input_video_bytes"], "video", need_vit=True, need_vae=True, edit=True)
        target = encoder.visual(row["output_video_bytes"], "video", need_vit=False, need_vae=True, edit=True)
    return [(base_task, build_edit_sample(
        sample_id, row["caption"], condition, target, tokenizer, config,
    ))]


def main():
    args = parse_arguments()
    if not 0.0 <= args.max_failure_rate <= 1.0:
        raise ValueError("max-failure-rate must be in [0, 1]")
    if not 0.0 <= args.text_cond_dropout_prob <= 1.0:
        raise ValueError("text-cond-dropout-prob must be in [0, 1]")
    if args.emit_tasks:
        emit_tasks = {item.strip() for item in args.emit_tasks.split(",") if item.strip()}
        unknown = emit_tasks - {"t2i", "t2v", "i2t", "v2t", "i2i", "v2v"}
        if unknown:
            raise ValueError("unsupported --emit-tasks entries: {}".format(sorted(unknown)))
    else:
        emit_tasks = None
    if args.base_tasks:
        base_tasks = {item.strip() for item in args.base_tasks.split(",") if item.strip()}
        unknown = base_tasks - {"t2i", "t2v", "i2t", "v2t", "i2i", "v2v"}
        if unknown:
            raise ValueError("unsupported --base-tasks entries: {}".format(sorted(unknown)))
        if not base_tasks:
            raise ValueError("--base-tasks must contain at least one task")
    else:
        base_tasks = None
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = _device()
    torch.manual_seed(args.seed + rank)
    root = Path(args.dataset_root).expanduser().resolve()
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("no parquet files found under {}".format(root))
    output = Path(args.output).expanduser().resolve() / "rank-{:05d}".format(rank)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty rank output: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    import pyarrow.parquet as parquet

    qwen = Path(args.qwen_path).expanduser().resolve()
    config = LanceNativeConfig.from_llm_config(qwen / "config.json", variant=args.variant)
    geometry = {
        name: value for name, value in {
            "latent_patch_size": args.latent_patch_size,
            "max_latent_size": args.max_latent_size,
            "max_num_frames": args.max_num_frames,
        }.items() if value is not None
    }
    config = config.with_overrides(**geometry)
    tokenizer = prepare_lance_tokenizer(
        AutoTokenizer.from_pretrained(qwen, trust_remote_code=False)
    )
    if len(tokenizer) > config.vocab_size:
        raise ValueError("Lance tokenizer exceeds the padded model vocabulary")
    encoder = FeatureEncoder(
        config, args.vit_path, args.vae_path, device, args.sample_posterior,
        args.resample_posterior_during_training,
    )
    counts = {}
    base_task_counts = {}
    failures = []
    global_index = -1
    written = 0
    stop = False
    for path in files:
        source = parquet.ParquetFile(path)
        for row_group in range(source.num_row_groups):
            rows = source.read_row_group(row_group).to_pylist()
            for row_index, row in enumerate(rows):
                global_index += 1
                if global_index % world_size != rank:
                    continue
                if args.max_samples is not None and written >= args.max_samples:
                    stop = True
                    break
                sample_id = "{}:{}:{}".format(path.relative_to(root), row_group, row_index)
                try:
                    base_task = _task(path, row)
                    if base_tasks is not None and base_task not in base_tasks:
                        continue
                    if (
                        args.max_samples_per_task is not None
                        and base_task_counts.get(base_task, 0) >= args.max_samples_per_task
                    ):
                        continue
                    prepared = _prepare(
                        path, row, sample_id, encoder, tokenizer, config,
                        seed=args.seed,
                        text_dropout=args.text_cond_dropout_prob,
                        emit_tasks=emit_tasks,
                    )
                    for task, sample in prepared:
                        sample.validate(config)
                        task_output = output / task
                        task_output.mkdir(parents=True, exist_ok=True)
                        torch.save(sample, task_output / "sample-{:08d}.pt".format(written))
                        counts[task] = counts.get(task, 0) + 1
                        written += 1
                    base_task_counts[base_task] = base_task_counts.get(base_task, 0) + 1
                except Exception as exc:
                    failures.append({"sample_id": sample_id, "error": str(exc)})
                    if args.fail_fast:
                        raise
            if stop:
                break
        if stop:
            break
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "mode": "native-mindspeed-mm-media-preprocessing",
        "rank": rank,
        "world_size": world_size,
        "seed": args.seed,
        "variant": config.variant,
        "config": config.to_dict(),
        "effective_vocab_size": len(tokenizer),
        "qwen_path": str(qwen),
        "vit_path": str(Path(args.vit_path).expanduser().resolve()),
        "vae_path": str(Path(args.vae_path).expanduser().resolve()),
        "text_cond_dropout_prob": args.text_cond_dropout_prob,
        "text_format": "raw-pt",
        "emit_tasks": sorted(emit_tasks) if emit_tasks else None,
        "base_tasks": sorted(base_tasks) if base_tasks else None,
        "resample_posterior_during_training": args.resample_posterior_during_training,
        "dataset_root": str(root),
        "parquet_files": len(files),
        "written": written,
        "task_counts": counts,
        "base_task_counts": base_task_counts,
        "failure_count": len(failures),
        "failures": failures[:100],
        "vit": encoder.vit_report,
    }
    attempted = written + len(failures)
    failure_rate = len(failures) / attempted if attempted else 1.0
    manifest["failure_rate"] = failure_rate
    if failure_rate > args.max_failure_rate:
        manifest["status"] = "invalid"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["status"] != "completed":
        raise RuntimeError(
            "native preprocessing failure rate {:.2%} exceeds {:.2%}".format(
                failure_rate, args.max_failure_rate
            )
        )


if __name__ == "__main__":
    main()
