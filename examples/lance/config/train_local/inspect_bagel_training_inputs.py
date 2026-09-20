#!/usr/bin/env python3
"""Audit ten Mobile-O samples through the unmodified BAGEL packing code.

Mobile-O stores ``image_bytes``/``caption`` whereas BAGEL's two readers expect
``image``/``captions`` (T2I parquet) or an image filename plus conversations
(I2T JSONL).  This script only adapts those source fields into the exact sample
dictionary those readers emit, then calls BagelMultiDataset.pack_sequence() and
to_tensor() without modifying the production implementation.
"""

import argparse
from hashlib import sha256
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("NON_MEGATRON", "true")

import pyarrow.parquet as parquet
from PIL import Image
import torch
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer


def _load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Import the real BAGEL source files without executing mindspeed_mm/data/__init__.py,
# whose legacy Megatron dependency is absent from the native-FSDP container.
for package, path in (
    ("mindspeed_mm", REPO_ROOT / "mindspeed_mm"),
    ("mindspeed_mm.data", REPO_ROOT / "mindspeed_mm/data"),
    ("mindspeed_mm.data.data_utils", REPO_ROOT / "mindspeed_mm/data/data_utils"),
    ("mindspeed_mm.data.datasets", REPO_ROOT / "mindspeed_mm/data/datasets"),
):
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(path)]
        sys.modules[package] = module

transforms = _load_source_module(
    "mindspeed_mm.data.data_utils.data_transform",
    REPO_ROOT / "mindspeed_mm/data/data_utils/data_transform.py",
)
_load_source_module(
    "mindspeed_mm.data.datasets.bagel_iterable_dataset",
    REPO_ROOT / "mindspeed_mm/data/datasets/bagel_iterable_dataset.py",
)
bagel_source = _load_source_module(
    "mindspeed_mm.data.datasets.bagel_dataset",
    REPO_ROOT / "mindspeed_mm/data/datasets/bagel_dataset.py",
)
MaxLongEdgeMinShortEdgeResize = transforms.MaxLongEdgeMinShortEdgeResize
BagelMultiDataset = bagel_source.BagelMultiDataset
add_special_tokens = bagel_source.add_special_tokens
get_flattened_position_ids_extrapolate = bagel_source.get_flattened_position_ids_extrapolate


DEFAULT_OUTPUT = Path(__file__).with_name("bagel_training_inputs_10.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="/mnt/models/DATA_INIT/MULTI/T2I/Mobile-O-Pre-Train-GEN",
    )
    parser.add_argument("--tokenizer-path", default="/mnt/models/MODELS/Qwen3-0.6B")
    parser.add_argument("--source-rows", type=int, default=5)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def iter_rows(root, limit):
    root = Path(root).expanduser().resolve()
    emitted = 0
    for path in sorted(root.rglob("*.parquet")):
        source = parquet.ParquetFile(path)
        for row_group in range(source.num_row_groups):
            for row_index, row in enumerate(source.read_row_group(row_group).to_pylist()):
                sample_id = "{}:{}:{}".format(path.relative_to(root), row_group, row_index)
                yield row, sample_id
                emitted += 1
                if emitted >= limit:
                    return


def make_bagel(tokenizer):
    # Construct only the state consumed by the production pack_sequence method.
    dataset = BagelMultiDataset.__new__(BagelMultiDataset)
    torch.utils.data.IterableDataset.__init__(dataset)
    dataset.tokenizer = tokenizer
    dataset.bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    dataset.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    dataset.start_of_image = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    dataset.end_of_image = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    dataset.get_flattened_position_ids = get_flattened_position_ids_extrapolate
    dataset.config = {
        "text_cond_dropout_prob": 0.0,
        "vit_cond_dropout_prob": 0.0,
        "vae_cond_dropout_prob": 0.0,
        "vit_patch_size": 14,
        "max_num_patch_per_side": 70,
        "vae_image_downsample": 16,
        "max_latent_size": 64,
        "use_flex": False,
    }
    return dataset


def tensor_summary(value):
    if value is None:
        return None
    cpu = value.detach().cpu().contiguous()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    floating = cpu.float()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "min": float(floating.min()) if cpu.numel() else None,
        "max": float(floating.max()) if cpu.numel() else None,
        "mean": float(floating.mean()) if cpu.numel() else None,
        "sha256": sha256(raw).hexdigest(),
    }


def index_mask(length, indexes):
    result = [0] * length
    if indexes is not None:
        for index in indexes.tolist():
            result[int(index)] = 1
    return result


def dense_sequence(length, text_indexes, text_ids, vit_indexes, vae_indexes, tokenizer):
    values = ["<unused>" for _ in range(length)]
    for index, token_id in zip(text_indexes.tolist(), text_ids.tolist()):
        values[int(index)] = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    if vit_indexes is not None:
        for index in vit_indexes.tolist():
            values[int(index)] = "<vit_patch>"
    if vae_indexes is not None:
        for index in vae_indexes.tolist():
            values[int(index)] = "<vae_latent>"
    compact = []
    cursor = 0
    while cursor < length:
        value = values[cursor]
        if value in ("<vit_patch>", "<vae_latent>"):
            end = cursor + 1
            while end < length and values[end] == value:
                end += 1
            compact.append("{}x{}".format(value, end - cursor))
            cursor = end
        else:
            compact.append(value)
            cursor += 1
    return "".join(compact)


def attention_summary(mask):
    allowed = torch.isfinite(mask)
    length = int(mask.shape[0])
    return {
        "shape": list(mask.shape),
        "allowed_entries": int(allowed.sum()),
        "blocked_entries": int(length * length - allowed.sum()),
        "row_allowed_counts": allowed.sum(dim=1).tolist(),
    }


def pack_record(dataset, tokenizer, sample, task, sample_id, caption):
    status = dataset.set_sequence_status()
    status = dataset.pack_sequence(sample, status)
    packed = dataset.to_tensor(status)
    length = int(packed["sequence_length"])
    text_indexes = packed["packed_text_indexes"]
    text_ids = packed["packed_text_ids"]
    vit_indexes = packed.get("packed_vit_token_indexes")
    vae_indexes = packed.get("packed_vae_token_indexes")
    ce_indexes = packed.get("ce_loss_indexes")
    mse_indexes = packed.get("mse_loss_indexes")
    labels = packed.get("packed_label_ids")
    weights = packed.get("ce_loss_weights")
    ce_rows = []
    if ce_indexes is not None:
        for index, label, weight in zip(ce_indexes.tolist(), labels.tolist(), weights.tolist()):
            ce_rows.append({
                "position": int(index),
                "input": dense_sequence(
                    length,
                    text_indexes[text_indexes == index],
                    text_ids[text_indexes == index],
                    None,
                    None,
                    tokenizer,
                )[int(index):int(index) + 1] if False else tokenizer.decode(
                    [int(text_ids[(text_indexes == index).nonzero()[0]])],
                    skip_special_tokens=False,
                ),
                "label_id": int(label),
                "label": tokenizer.decode([int(label)], skip_special_tokens=False),
                "weight": float(weight),
            })
    return {
        "task": task,
        "sample_id": sample_id if task == "t2i" else sample_id + ":i2t",
        "source_caption": caption,
        "adapter_note": (
            "same Mobile-O row adapted from image_bytes/caption to BAGEL reader output; "
            "production BagelMultiDataset.pack_sequence and to_tensor are unmodified"
        ),
        "input_contract": "BAGEL packed batch",
        "sequence_plan": sample["sequence_plan"],
        "sequence_length": length,
        "sample_lens": packed["sample_lens"],
        "compact_sequence": dense_sequence(
            length, text_indexes, text_ids, vit_indexes, vae_indexes, tokenizer
        ),
        "packed_text_ids": text_ids.tolist(),
        "packed_text_indexes": text_indexes.tolist(),
        "packed_position_ids": packed["packed_position_ids"].tolist(),
        "attention": attention_summary(packed["nested_attention_masks"][0]),
        "vit_token_indexes": None if vit_indexes is None else vit_indexes.tolist(),
        "vit_tokens": tensor_summary(packed.get("packed_vit_tokens")),
        "vit_position_ids": (
            None if packed.get("packed_vit_position_ids") is None
            else packed["packed_vit_position_ids"].tolist()
        ),
        "vit_token_seqlens": (
            None if packed.get("vit_token_seqlens") is None
            else packed["vit_token_seqlens"].tolist()
        ),
        "vae_token_indexes": None if vae_indexes is None else vae_indexes.tolist(),
        "padded_images": tensor_summary(packed.get("padded_images")),
        "patchified_vae_latent_shapes": packed.get("patchified_vae_latent_shapes"),
        "latent_position_ids": (
            None if packed.get("packed_latent_position_ids") is None
            else packed["packed_latent_position_ids"].tolist()
        ),
        "timesteps": (
            None if packed.get("packed_timesteps") is None
            else packed["packed_timesteps"].tolist()
        ),
        "ce_loss_mask": index_mask(length, ce_indexes),
        "ce_rows": ce_rows,
        "mse_loss_mask": index_mask(length, mse_indexes),
        "mse_loss_indexes": None if mse_indexes is None else mse_indexes.tolist(),
    }


def main():
    args = parse_args()
    tokenizer, _ = add_special_tokens(Qwen2Tokenizer.from_pretrained(args.tokenizer_path))
    dataset = make_bagel(tokenizer)
    t2i_transform = MaxLongEdgeMinShortEdgeResize(
        image_stride=16, max_image_size=1024, min_image_size=512
    )
    i2t_transform = MaxLongEdgeMinShortEdgeResize(
        image_stride=14, max_image_size=980, min_image_size=378, max_pixels=2007040
    )
    records = []
    for row, sample_id in iter_rows(args.dataset_root, args.source_rows):
        image = Image.open(BytesIO(row["image_bytes"])).convert("RGB")
        caption = row["caption"]
        caption_ids = tokenizer.encode(caption)
        records.append(pack_record(dataset, tokenizer, {
            "image_tensor_list": [t2i_transform(image)],
            "text_ids_list": [caption_ids],
            "sequence_plan": [
                {"type": "text", "enable_cfg": 1, "loss": 0,
                 "special_token_loss": 0, "special_token_label": None},
                {"type": "vae_image", "enable_cfg": 0, "loss": 1,
                 "special_token_loss": 0, "special_token_label": None},
            ],
        }, "t2i", sample_id, caption))
        records.append(pack_record(dataset, tokenizer, {
            "image_tensor_list": [i2t_transform(image)],
            "text_ids_list": [caption_ids],
            "sequence_plan": [
                {"type": "vit_image", "enable_cfg": 0, "loss": 0,
                 "special_token_loss": 0, "special_token_label": None},
                {"type": "text", "enable_cfg": 0, "loss": 1,
                 "special_token_loss": 0, "special_token_label": None},
            ],
        }, "i2t", sample_id, caption))

    expected = args.source_rows * 2
    assert len(records) == expected, (len(records), expected)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("wrote {} BAGEL reference samples to {}".format(len(records), output))


if __name__ == "__main__":
    main()
