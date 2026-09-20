#!/usr/bin/env python3
"""Build and audit ten real Lance PT samples without running model training.

The script reads real Mobile-O parquet rows, executes the production frozen
ViT/VAE encoder and the same ``_prepare``/sample builders used by data
preprocessing, then writes exact per-position inputs and loss masks as JSONL.
"""

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("NON_MEGATRON", "true")

import pyarrow.parquet as parquet
import torch
from transformers import AutoTokenizer

from mindspeed_mm.models.omni.lance.data import pack_preencoded_samples
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.preprocessing import prepare_lance_tokenizer
from scripts.prepare_lance_native_data import FeatureEncoder, _device, _prepare


DEFAULT_OUTPUT = Path(__file__).with_name("lance_training_inputs_10.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="/mnt/models/DATA_INIT/MULTI/T2I/Mobile-O-Pre-Train-GEN",
    )
    parser.add_argument("--qwen-path", default="/mnt/models/MODELS/Qwen3-0.6B")
    parser.add_argument(
        "--vit-path", default="/mnt/models/MODELS/Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument(
        "--vae-path",
        default="/mnt/models/MODELS/bytedance-research/Lance/Wan2.2_VAE.pth",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--source-rows", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--text-cond-dropout-prob", type=float, default=0.1)
    return parser.parse_args()


def _index_mask(length, indexes):
    mask = [0] * length
    if indexes is not None:
        for index in indexes.tolist():
            mask[int(index)] = 1
    return mask


def _position_values(length, indexes, values, default=None):
    result = [default] * length
    if indexes is not None:
        for index, value in zip(indexes.tolist(), values.tolist()):
            result[int(index)] = value
    return result


def _compact_tokens(token_ids, tokenizer):
    special_names = (
        "<|im_start|>", "<|im_end|>", "<|vision_start|>",
        "<|vision_end|>", "<|video_pad|>", "<|image_pad|>",
    )
    special = {
        int(tokenizer.convert_tokens_to_ids(name)): name for name in special_names
    }
    video_pad = int(tokenizer.convert_tokens_to_ids("<|video_pad|>"))
    parts = []
    ordinary = []

    def flush():
        if ordinary:
            parts.append(tokenizer.decode(ordinary, skip_special_tokens=False))
            ordinary.clear()

    cursor = 0
    while cursor < len(token_ids):
        token_id = int(token_ids[cursor])
        if token_id == video_pad:
            flush()
            end = cursor + 1
            while end < len(token_ids) and int(token_ids[end]) == video_pad:
                end += 1
            parts.append("<|video_pad|>x{}".format(end - cursor))
            cursor = end
        elif token_id in special:
            flush()
            parts.append(special[token_id])
            cursor += 1
        else:
            ordinary.append(token_id)
            cursor += 1
    flush()
    return "".join(parts)


def _sample_record(task, sample, tokenizer):
    sample.validate(CONFIG)
    packed = pack_preencoded_samples(
        (sample,), CONFIG, max_tokens=sample.length, attention_backend="ascend"
    )
    batch = packed.batch
    # LancePreencodedDataset sets this flag immediately before returning the
    # batch to the training engine.
    batch.resample_timesteps = True
    length = batch.sequence_length
    token_ids = batch.token_ids.tolist()
    ce_labels = batch.ce_labels if batch.ce_labels is not None else torch.empty(0, dtype=torch.long)
    ce_weights = batch.ce_weights if batch.ce_weights is not None else torch.empty(0)
    attention = packed.packed_sequence
    ce_rows = []
    if batch.ce_indexes is not None:
        for index, label, weight in zip(
            batch.ce_indexes.tolist(), ce_labels.tolist(), ce_weights.tolist()
        ):
            ce_rows.append({
                "position": int(index),
                "input_token_id": int(token_ids[index]),
                "input_token": tokenizer.decode([int(token_ids[index])]),
                "label_id": int(label),
                "label_token": tokenizer.decode([int(label)]),
                "weight": float(weight),
            })

    return {
        "task": task,
        "sample_id": sample.sample_id,
        "input_contract": "LanceTrainingBatch",
        "resample_timesteps": batch.resample_timesteps,
        "sequence_length": length,
        "compact_sequence": _compact_tokens(token_ids, tokenizer),
        "token_ids": token_ids,
        "segments": [
            {
                "length": segment.length,
                "attention_mode": segment.attention_mode,
                "modality": segment.modality,
                "expert": segment.expert,
            }
            for segment in sample.segments
        ],
        "attention_blocks": [block.to_dict() for block in attention.block_schedule()],
        "text_indexes": batch.text_indexes.tolist(),
        "vit_indexes": None if batch.vit_indexes is None else batch.vit_indexes.tolist(),
        "vae_indexes": None if batch.vae_indexes is None else batch.vae_indexes.tolist(),
        "understanding_mask": _index_mask(length, batch.understanding_indexes),
        "generation_mask": _index_mask(length, batch.generation_indexes),
        "ce_loss_mask": _index_mask(length, batch.ce_indexes),
        "ce_label_by_position": _position_values(
            length, batch.ce_indexes, ce_labels, default=None
        ),
        "ce_weight_by_position": _position_values(
            length, batch.ce_indexes, ce_weights, default=None
        ),
        "ce_rows": ce_rows,
        "mse_loss_mask": _index_mask(length, batch.mse_indexes),
        "mse_indexes": None if batch.mse_indexes is None else batch.mse_indexes.tolist(),
        "position_ids": batch.position_ids.tolist(),
        "vit_embeddings_shape": (
            None if batch.vit_embeddings is None else list(batch.vit_embeddings.shape)
        ),
        "clean_latents_shape": (
            None if batch.clean_latents is None else list(batch.clean_latents.shape)
        ),
        "latent_log_variance_shape": (
            None if batch.latent_log_variance is None
            else list(batch.latent_log_variance.shape)
        ),
        "timesteps": None if batch.timesteps is None else batch.timesteps.tolist(),
    }


def _assert_layout(task, record, tokenizer):
    im_start = int(tokenizer.convert_tokens_to_ids("<|im_start|>"))
    im_end = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    vision_start = int(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
    vision_end = int(tokenizer.convert_tokens_to_ids("<|vision_end|>"))
    ids = record["token_ids"]
    if task == "t2i":
        assert ids[-1] == vision_end
        assert record["ce_rows"] == []
        assert sum(record["mse_loss_mask"]) > 0
        if ids[0] == im_start:
            assert ids.index(im_end) < ids.index(vision_start)
        else:
            assert ids[0] == vision_start  # CFG caption-dropout branch.
    elif task == "i2t":
        assert ids[0] == vision_start
        vision_end_index = ids.index(vision_end)
        target_start = record["ce_rows"][0]["position"]
        assert target_start > vision_end_index
        assert ids[target_start] == im_start
        assert record["ce_rows"][-1]["label_id"] == im_end
        assert sum(record["mse_loss_mask"]) == 0
    else:
        raise AssertionError("unexpected audit task: {}".format(task))


def iter_rows(dataset_root, limit):
    root = Path(dataset_root).expanduser().resolve()
    emitted = 0
    for path in sorted(root.rglob("*.parquet")):
        source = parquet.ParquetFile(path)
        for row_group in range(source.num_row_groups):
            rows = source.read_row_group(row_group).to_pylist()
            for row_index, row in enumerate(rows):
                sample_id = "{}:{}:{}".format(
                    path.relative_to(root), row_group, row_index
                )
                yield path, row, sample_id
                emitted += 1
                if emitted >= limit:
                    return


def main():
    global CONFIG
    args = parse_args()
    if args.source_rows <= 0:
        raise ValueError("source-rows must be positive")
    device = _device()
    qwen_path = Path(args.qwen_path).expanduser().resolve()
    CONFIG = LanceNativeConfig.from_llm_config(
        qwen_path / "config.json", variant="video"
    ).with_overrides(
        latent_patch_size=(1, 2, 2), max_latent_size=64, max_num_frames=121
    )
    tokenizer = prepare_lance_tokenizer(
        AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=False)
    )
    encoder = FeatureEncoder(
        CONFIG, args.vit_path, args.vae_path, device,
        sample_posterior=True, resample_posterior_during_training=True,
    )
    records = []
    for path, row, sample_id in iter_rows(args.dataset_root, args.source_rows):
        prepared = _prepare(
            path, row, sample_id, encoder, tokenizer, CONFIG,
            seed=args.seed,
            text_dropout=args.text_cond_dropout_prob,
            emit_tasks={"t2i", "i2t"},
        )
        assert [task for task, _ in prepared] == ["t2i", "i2t"]
        for task, sample in prepared:
            record = _sample_record(task, sample, tokenizer)
            _assert_layout(task, record, tokenizer)
            records.append(record)

    expected = args.source_rows * 2
    assert len(records) == expected, "expected {} samples, got {}".format(
        expected, len(records)
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("wrote {} real pretraining samples to {}".format(len(records), output))


if __name__ == "__main__":
    main()
