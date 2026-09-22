#!/usr/bin/env python3
"""Resolve multiple T2I training examples without importing Torch or MindSpeed-MM."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as parquet

from resolve_lance_t2i_overfit_example import (
    bucket_size,
    center_crop_for_training,
    normalized_rgb,
    source_task,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--vae-downsample", type=int, default=16)
    parser.add_argument("--latent-patch-height", type=int, default=2)
    parser.add_argument("--latent-patch-width", type=int, default=2)
    return parser.parse_args()


def iter_t2i_rows(root):
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
                yield selected, path, row_group, row_index, sample_id, row
                selected += 1


def main():
    args = parse_arguments()
    if args.sample_count <= 0:
        raise ValueError("sample-count must be positive")
    root = Path(args.dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("dataset root not found: {}".format(root))
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "refusing to write training examples into non-empty output: {}".format(output)
        )
    output.mkdir(parents=True, exist_ok=True)

    stride_h = args.vae_downsample * args.latent_patch_height
    stride_w = args.vae_downsample * args.latent_patch_width
    examples = []
    for selected, path, row_group, row_index, sample_id, row in iter_t2i_rows(root):
        if len(examples) >= args.sample_count:
            break
        image = normalized_rgb(row["image_bytes"])
        target_width, target_height = bucket_size(
            image.width,
            image.height,
            args.resolution,
            stride_h,
            stride_w,
        )
        target = center_crop_for_training(image, target_width, target_height)
        sample_dir = output / "{:06d}".format(selected)
        sample_dir.mkdir(parents=True, exist_ok=False)
        original_path = sample_dir / "original.png"
        target_path = sample_dir / "training_target.png"
        metadata_path = sample_dir / "training_example.json"
        image.save(original_path)
        target.save(target_path)
        entry = {
            "index": selected,
            "sample_id": sample_id,
            "parquet": str(path),
            "row_group": row_group,
            "row_index": row_index,
            "prompt": str(row["caption"]),
            "original_width": image.width,
            "original_height": image.height,
            "width": target_width,
            "height": target_height,
            "num_frames": 1,
            "original_image": str(original_path),
            "training_target": str(target_path),
        }
        metadata_path.write_text(
            json.dumps(entry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        examples.append(entry)

    if len(examples) != args.sample_count:
        raise RuntimeError(
            "dataset contains only {} T2I samples; requested {}".format(
                len(examples), args.sample_count
            )
        )
    manifest = {
        "schema_version": 1,
        "dataset_root": str(root),
        "sample_count": len(examples),
        "resolution_policy": {
            "area_reference": args.resolution,
            "height_stride": stride_h,
            "width_stride": stride_w,
            "resize": "cover",
            "crop": "center",
        },
        "examples": examples,
    }
    manifest_path = output / "training_examples.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Resolved {} training examples: {}".format(len(examples), manifest_path))


if __name__ == "__main__":
    main()
