#!/usr/bin/env python3
"""Inspect prompts and latent geometry stored in one native Lance packed batch."""

import argparse
import json
import os
from pathlib import Path
import sys


# Must be set before importing the mindspeed_mm package.  This process-local
# value prevents the package initializer from importing Megatron/Lora patches.
os.environ.setdefault("NON_MEGATRON", "true")

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoTokenizer

from mindspeed_mm.models.omni.lance.preprocessing import prepare_lance_tokenizer
from mindspeed_mm.models.omni.lance.sequence import LancePackedSequence


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, help="batch-XXXXXXXX.pt file")
    parser.add_argument("--qwen-path", required=True)
    parser.add_argument("--output", help="optional JSON output path")
    parser.add_argument("--max-latent-size", type=int, default=64)
    parser.add_argument("--latent-patch-size", nargs=3, type=int, default=(1, 2, 2))
    parser.add_argument("--vae-spatial-downsample", type=int, default=16)
    parser.add_argument("--vae-temporal-downsample", type=int, default=4)
    return parser.parse_args()


def _decode_text(token_ids, tokenizer):
    values = [int(value) for value in token_ids]
    im_start = int(tokenizer.convert_tokens_to_ids("<|im_start|>"))
    im_end = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    if values and values[0] == im_start:
        values = values[1:]
    if values and values[-1] == im_end:
        values = values[:-1]
    return tokenizer.decode(values, skip_special_tokens=False).strip()


def _document_latent_geometry(batch, start, end, args):
    if batch.vae_indexes is None or batch.latent_position_ids is None:
        return None
    selected = (batch.vae_indexes >= start) & (batch.vae_indexes < end)
    positions = batch.latent_position_ids[selected].long()
    if not positions.numel():
        return None
    maximum = args.max_latent_size
    spatial = maximum * maximum
    grid_t = int(torch.unique(positions // spatial).numel())
    grid_h = int(torch.unique((positions % spatial) // maximum).numel())
    grid_w = int(torch.unique(positions % maximum).numel())
    patch_t, patch_h, patch_w = (int(value) for value in args.latent_patch_size)
    latent_frames = grid_t * patch_t
    latent_height = grid_h * patch_h
    latent_width = grid_w * patch_w
    return {
        "latent_token_count": int(positions.numel()),
        "latent_grid": [grid_t, grid_h, grid_w],
        "latent_shape": [latent_frames, latent_height, latent_width],
        "num_frames": (latent_frames - 1) * args.vae_temporal_downsample + 1,
        "height": latent_height * args.vae_spatial_downsample,
        "width": latent_width * args.vae_spatial_downsample,
    }


def main():
    args = parse_arguments()
    batch_path = Path(args.batch).expanduser().resolve()
    if not batch_path.is_file():
        raise FileNotFoundError("packed batch not found: {}".format(batch_path))
    qwen_path = Path(args.qwen_path).expanduser().resolve()
    if not qwen_path.is_dir():
        raise FileNotFoundError("Qwen path not found: {}".format(qwen_path))

    batch = torch.load(batch_path, map_location="cpu", weights_only=False)
    packed = batch.attention_mask
    if not isinstance(packed, LancePackedSequence):
        raise TypeError("packed batch does not contain LancePackedSequence metadata")
    tokenizer = prepare_lance_tokenizer(
        AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=False)
    )

    documents = []
    start = 0
    for document_index, document in enumerate(packed.documents):
        end = start + document.length
        selected_text = batch.text_indexes[
            (batch.text_indexes >= start) & (batch.text_indexes < end)
        ]
        prompt = _decode_text(batch.token_ids[selected_text].tolist(), tokenizer)
        documents.append({
            "document_index": document_index,
            "sample_id": document.sample_id,
            "sequence_start": start,
            "sequence_end": end,
            "sequence_length": document.length,
            "prompt": prompt,
            "segments": [dict(segment.__dict__) for segment in document.segments],
            "geometry": _document_latent_geometry(batch, start, end, args),
        })
        start = end

    result = {
        "schema_version": 1,
        "batch": str(batch_path),
        "sequence_length": int(batch.sequence_length),
        "document_count": len(documents),
        "documents": documents,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        destination = Path(args.output).expanduser().resolve()
        if destination.exists():
            raise FileExistsError("refusing to overwrite output: {}".format(destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
