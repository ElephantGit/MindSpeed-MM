#!/usr/bin/env python3
"""Resolve a T2I overfit source example without importing MindSpeed-MM or Torch."""

import argparse
from io import BytesIO
import json
import math
from pathlib import Path

import pyarrow.parquet as parquet
from PIL import Image


ASPECT_RATIOS = ((21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--sample-index",
        type=int,
        help="zero-based T2I sample index in the deterministic preprocessing scan",
    )
    selection.add_argument(
        "--sample-id",
        help="exact sample_id stored in LancePackedSequence document metadata",
    )
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--vae-downsample", type=int, default=16)
    parser.add_argument("--latent-patch-height", type=int, default=2)
    parser.add_argument("--latent-patch-width", type=int, default=2)
    return parser.parse_args()


def source_task(path, row):
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


def bucket_size(width, height, resolution, height_stride, width_stride):
    ratio = width / height
    target_ratio = min(
        (bucket_width / bucket_height for bucket_width, bucket_height in ASPECT_RATIOS),
        key=lambda value: abs(value - ratio),
    )
    width_a = round(
        math.sqrt(resolution * resolution * target_ratio) / width_stride
    ) * width_stride
    height_a = round((width_a / target_ratio) / height_stride) * height_stride
    height_b = round(
        math.sqrt(resolution * resolution / target_ratio) / height_stride
    ) * height_stride
    width_b = round((height_b * target_ratio) / width_stride) * width_stride
    candidates = (
        (max(width_stride, width_a), max(height_stride, height_a)),
        (max(width_stride, width_b), max(height_stride, height_b)),
    )
    return min(
        candidates,
        key=lambda value: (
            abs(value[0] / value[1] - target_ratio),
            abs(value[0] * value[1] - resolution * resolution),
        ),
    )


def normalized_rgb(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        image = background
    return image.convert("RGB")


def center_crop_for_training(image, target_width, target_height):
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    resized = image.resize((resized_width, resized_height), resample=resampling)
    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    return resized.crop((left, top, left + target_width, top + target_height))


def resolve_example(root, sample_index, requested_sample_id):
    sample_index = 0 if sample_index is None and requested_sample_id is None else sample_index
    if sample_index is not None and sample_index < 0:
        raise ValueError("sample-index must be non-negative")
    selected = 0
    for path in sorted(root.rglob("*.parquet")):
        source = parquet.ParquetFile(path)
        for row_group in range(source.num_row_groups):
            rows = source.read_row_group(row_group).to_pylist()
            for row_index, row in enumerate(rows):
                if source_task(path, row) != "t2i":
                    continue
                sample_id = "{}:{}:{}".format(
                    path.relative_to(root), row_group, row_index
                )
                if (
                    (requested_sample_id is not None and sample_id == requested_sample_id)
                    or (requested_sample_id is None and selected == sample_index)
                ):
                    return path, row_group, row_index, selected, sample_id, row
                selected += 1
    if requested_sample_id is not None:
        raise ValueError("T2I sample_id not found: {}".format(requested_sample_id))
    raise ValueError(
        "dataset contains only {} T2I samples; requested index {}".format(
            selected, sample_index
        )
    )


def main():
    args = parse_arguments()
    root = Path(args.dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("dataset root not found: {}".format(root))
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    destinations = {
        "metadata": output / "training_example.json",
        "prompt": output / "prompt.json",
        "original": output / "original.png",
        "target": output / "training_target.png",
    }
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing outputs: {}".format(existing))

    path, row_group, row_index, sample_index, sample_id, row = resolve_example(
        root, args.sample_index, args.sample_id
    )
    image = normalized_rgb(row["image_bytes"])
    stride_h = args.vae_downsample * args.latent_patch_height
    stride_w = args.vae_downsample * args.latent_patch_width
    target_width, target_height = bucket_size(
        image.width,
        image.height,
        args.resolution,
        stride_h,
        stride_w,
    )
    target = center_crop_for_training(image, target_width, target_height)
    prompt = str(row["caption"])
    metadata = {
        "schema_version": 1,
        "sample_index": sample_index,
        "sample_id": sample_id,
        "parquet": str(path),
        "row_group": row_group,
        "row_index": row_index,
        "prompt": prompt,
        "original_width": image.width,
        "original_height": image.height,
        "width": target_width,
        "height": target_height,
        "num_frames": 1,
        "resolution_policy": {
            "area_reference": args.resolution,
            "height_stride": stride_h,
            "width_stride": stride_w,
            "resize": "cover",
            "crop": "center",
        },
    }

    destinations["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    destinations["prompt"].write_text(
        json.dumps([prompt], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    image.save(destinations["original"])
    target.save(destinations["target"])
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
