#!/usr/bin/env python3
"""Raw-pretraining continuation evaluation for Lance T2T and I2T.

No chat template or role prompt is used.  T2T starts with
``<|im_start|>prefix``; I2T starts with the prepared ViT segment followed by
``<|im_start|>``.  Both autoregressively continue until ``<|im_end|>``.
"""

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("NON_MEGATRON", "true")
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoTokenizer

from mindspeed_mm.fsdp.models.lance.modeling_lance import LanceFSDPModel
from mindspeed_mm.models.omni.lance.data import LancePreparedSample
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.native_inference import load_native_generation_dcp
from mindspeed_mm.models.omni.lance.npu_attention import AscendBlockAttentionBackend
from mindspeed_mm.models.omni.lance.preprocessing import prepare_lance_tokenizer
from mindspeed_mm.models.omni.lance.sequence import LanceDocument, LancePackedSequence, LanceSegment
from mindspeed_mm.models.omni.lance.training_lance import LanceTrainingBatch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qwen-path", default="/mnt/models/MODELS/Qwen3-0.6B")
    parser.add_argument(
        "--prepared-data",
        default="/mnt/models/DATA_INIT/MULTI/T2I/Mobile-O-Pre-Train-preencoded-raw",
    )
    parser.add_argument(
        "--packed-data",
        help=(
            "read I2T references directly from packed batch-*.pt files; this "
            "takes precedence over --prepared-data"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--model-weights", action="store_true",
        help="use ordinary checkpoint weights instead of EMA",
    )
    return parser.parse_args()


def load_sample(path):
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, LancePreparedSample):
        return value
    if isinstance(value, dict):
        return LancePreparedSample(**value)
    raise TypeError("unsupported prepared sample in {}".format(path))


def load_packed_i2t_samples(root, count):
    """Recover individual I2T documents from training-time packed batches."""

    results = []
    for path in sorted(Path(root).glob("batch-*.pt")):
        value = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(value, Mapping):
            value = LanceTrainingBatch(**dict(value))
        if not isinstance(value, LanceTrainingBatch):
            raise TypeError("{} does not contain a LanceTrainingBatch".format(path))
        if not isinstance(value.attention_mask, LancePackedSequence):
            raise TypeError("{} lacks packed document metadata".format(path))
        if value.vit_indexes is None or value.ce_indexes is None:
            continue

        offset = 0
        for document in value.attention_mask.documents:
            start, end = offset, offset + document.length
            offset = end
            vit_mask = (value.vit_indexes >= start) & (value.vit_indexes < end)
            ce_mask = (value.ce_indexes >= start) & (value.ce_indexes < end)
            if not bool(vit_mask.any()) or not bool(ce_mask.any()):
                continue
            results.append(SimpleNamespace(
                sample_id=document.sample_id,
                segments=document.segments,
                token_ids=value.token_ids[start:end].clone(),
                position_ids=value.position_ids[:, start:end].clone(),
                vit_indexes=(value.vit_indexes[vit_mask] - start).clone(),
                vit_embeddings=value.vit_embeddings[vit_mask].clone(),
                ce_indexes=(value.ce_indexes[ce_mask] - start).clone(),
                ce_labels=value.ce_labels[ce_mask].clone(),
                source=str(path),
            ))
            if len(results) == count:
                return results
    raise ValueError(
        "requested {} packed I2T samples, found {} under {}".format(
            count, len(results), root
        )
    )


def choose_token(logits, args, generator):
    if not args.do_sample:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits.float() / args.temperature, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    remove = torch.cumsum(sorted_probs, dim=-1) - sorted_probs > args.top_p
    sorted_probs[remove] = 0
    selected = torch.multinomial(sorted_probs, 1, generator=generator)
    return int(sorted_ids[selected].item())


@torch.no_grad()
def continue_tokens(model, tokenizer, state, args, generator):
    token_ids = state["token_ids"].to(args.device)
    position_ids = state["position_ids"].to(args.device)
    vit_indexes = state.get("vit_indexes")
    connected_vit = state.get("connected_vit")
    if vit_indexes is not None:
        vit_indexes = vit_indexes.to(args.device)
        connected_vit = connected_vit.to(args.device)
    generated = []
    eos = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    for _ in range(args.max_new_tokens):
        length = int(token_ids.numel())
        visual_length = int(state.get("visual_length", 0))
        segments = []
        if visual_length:
            segments.append(LanceSegment(visual_length, "full", "vit", "understanding"))
        segments.append(LanceSegment(length - visual_length, "causal", "text", "understanding"))
        attention = LancePackedSequence((LanceDocument(state["sample_id"], tuple(segments)),))
        all_indexes = torch.arange(length, device=args.device, dtype=torch.long)
        embeddings = model.language_model.model.embed_tokens(token_ids)
        hidden_inputs = embeddings.clone()
        if vit_indexes is not None:
            hidden_inputs[vit_indexes] = connected_vit.to(hidden_inputs.dtype)
        hidden = model.forward_language(
            hidden_inputs,
            position_ids,
            attention,
            all_indexes,
            torch.empty(0, device=args.device, dtype=torch.long),
        )
        logits = model.language_model.lm_head(hidden[-1])[: len(tokenizer)]
        next_id = choose_token(logits, args, generator)
        if next_id == eos:
            break
        generated.append(next_id)
        token_ids = torch.cat((token_ids, token_ids.new_tensor([next_id])))
        next_position = position_ids[:, -1:] + 1
        position_ids = torch.cat((position_ids, next_position), dim=1)
    return generated


def t2t_state(tokenizer, prefix, index):
    im_start = int(tokenizer.convert_tokens_to_ids("<|im_start|>"))
    ids = [im_start] + tokenizer.encode(prefix, add_special_tokens=False)
    token_ids = torch.tensor(ids, dtype=torch.long)
    positions = torch.arange(len(ids), dtype=torch.long).repeat(3, 1)
    return {
        "sample_id": "t2t-{:06d}".format(index),
        "token_ids": token_ids,
        "position_ids": positions,
        "visual_length": 0,
    }


def i2t_state(model, sample, index, device):
    if sample.ce_indexes is None or sample.vit_indexes is None:
        raise ValueError("I2T reference sample lacks CE/ViT indexes")
    target_start = int(sample.ce_indexes[0])
    prefix_end = target_start + 1  # retain target <|im_start|>, hide caption
    visual_length = sample.segments[0].length
    if target_start != visual_length:
        raise ValueError("I2T target must immediately follow its ViT segment")
    connected = model.connector(sample.vit_embeddings.to(device=device, dtype=torch.bfloat16))
    return {
        "sample_id": "i2t-{:06d}".format(index),
        "token_ids": sample.token_ids[:prefix_end],
        "position_ids": sample.position_ids[:, :prefix_end],
        "visual_length": visual_length,
        "vit_indexes": sample.vit_indexes,
        "connected_vit": connected,
    }


def main():
    args = parse_args()
    if args.samples <= 0 or args.max_new_tokens <= 0:
        raise ValueError("samples and max-new-tokens must be positive")
    if args.do_sample and (args.temperature <= 0 or not 0 < args.top_p <= 1):
        raise ValueError("sampling requires temperature > 0 and top-p in (0,1]")
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    qwen = Path(args.qwen_path).expanduser().resolve()
    config = LanceNativeConfig.from_llm_config(qwen / "config.json", variant="video").with_overrides(
        latent_patch_size=(1, 2, 2), max_latent_size=64, max_num_frames=121
    )
    tokenizer = prepare_lance_tokenizer(
        AutoTokenizer.from_pretrained(qwen, trust_remote_code=False)
    )
    model = LanceFSDPModel(
        config,
        attention_backend=AscendBlockAttentionBackend(),
        use_vit_connector=True,
        include_vit_model=False,
        effective_vocab_size=len(tokenizer),
        device=device,
        dtype=torch.bfloat16,
    ).eval()
    report = load_native_generation_dcp(
        model, args.checkpoint, use_ema=not args.model_weights
    )

    if args.packed_data:
        samples = load_packed_i2t_samples(args.packed_data, args.samples)
        paths = [sample.source for sample in samples]
    else:
        paths = sorted(Path(args.prepared_data).glob("rank-*/i2t/sample-*.pt"))[: args.samples]
        if len(paths) != args.samples:
            raise ValueError("requested {} I2T samples, found {}".format(args.samples, len(paths)))
        samples = [load_sample(path) for path in paths]
    eos = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    results = []
    mode = "sample" if args.do_sample else "greedy"
    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
        for index, (path, sample) in enumerate(zip(paths, samples)):
            expected_ids = sample.ce_labels.tolist()
            if expected_ids and expected_ids[-1] == eos:
                expected_ids = expected_ids[:-1]
            expected = tokenizer.decode(expected_ids, skip_special_tokens=False)

            i2t_ids = continue_tokens(
                model, tokenizer, i2t_state(model, sample, index, device), args, generator
            )
            # Text-only continuation uses the first roughly one-third of the
            # same caption as a raw pretraining prefix, not a user/assistant turn.
            full_ids = tokenizer.encode(expected, add_special_tokens=False)
            prefix_count = max(1, len(full_ids) // 3)
            prefix = tokenizer.decode(full_ids[:prefix_count], skip_special_tokens=False)
            t2t_ids = continue_tokens(
                model, tokenizer, t2t_state(tokenizer, prefix, index), args, generator
            )
            results.extend((
                {
                    "task": "i2t",
                    "sample_id": sample.sample_id,
                    "input_format": "<|vision_start|>ViT<|vision_end|><|im_start|>",
                    "decode": mode,
                    "generated": tokenizer.decode(i2t_ids, skip_special_tokens=False),
                    "generated_token_ids": i2t_ids,
                    "reference_caption": expected,
                    "source_prepared_sample": str(path),
                },
                {
                    "task": "t2t",
                    "sample_id": "t2t-{:06d}".format(index),
                    "input_format": "<|im_start|>text-prefix",
                    "prefix": prefix,
                    "decode": mode,
                    "generated": tokenizer.decode(t2t_ids, skip_special_tokens=False),
                    "generated_token_ids": t2t_ids,
                    "reference_continuation": tokenizer.decode(
                        full_ids[prefix_count:], skip_special_tokens=False
                    ),
                },
            ))

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "completed",
        "checkpoint": report,
        "format": "raw pretraining continuation; no chat template",
        "decode": mode,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote {} T2T/I2T results to {}".format(len(results), output))


if __name__ == "__main__":
    main()
