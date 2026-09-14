#!/usr/bin/env python3
"""Create a bridge-free Lance pretraining initialization DCP."""

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("NON_MEGATRON", "true")

import torch

from mindspeed_mm.fsdp.models.lance.modeling_lance import LanceFSDPModel
from mindspeed_mm.models.omni.lance.dcp import write_native_state_to_dcp
from mindspeed_mm.models.omni.lance.initialization import (
    initialize_from_qwen_vl_files,
    initialize_random,
)
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.preprocessing import prepare_lance_tokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Initialize native Lance from Qwen2.5-VL and write MindSpeed-MM DCP"
    )
    parser.add_argument("--qwen-path", required=True)
    parser.add_argument("--vit-path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=("image", "video"), default="video")
    parser.add_argument("--latent-patch-size", nargs=3, type=int)
    parser.add_argument("--max-latent-size", type=int)
    parser.add_argument("--max-num-frames", type=int)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--include-vit-model", action="store_true",
        help="Include the frozen ViT in the training DCP (not recommended for pre-encoded PT)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    from transformers import AutoTokenizer

    qwen_path = Path(args.qwen_path).expanduser().resolve()
    config_path = qwen_path / "config.json"
    config = LanceNativeConfig.from_llm_config(config_path, variant=args.variant)
    geometry = {
        name: value for name, value in {
            "latent_patch_size": args.latent_patch_size,
            "max_latent_size": args.max_latent_size,
            "max_num_frames": args.max_num_frames,
        }.items() if value is not None
    }
    config = config.with_overrides(**geometry)
    model = LanceFSDPModel(
        config,
        use_vit_connector=True,
        include_vit_model=args.include_vit_model,
        device="cpu",
        dtype=torch.bfloat16,
    )
    random_report = initialize_random(model, args.seed)
    source_report = initialize_from_qwen_vl_files(
        model,
        qwen_path,
        vit_path=args.vit_path if args.include_vit_model else None,
        require_complete=not args.allow_missing,
    )
    trainable = sorted(name for name, value in model.named_parameters() if value.requires_grad)
    tokenizer = prepare_lance_tokenizer(
        AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=False)
    )
    if len(tokenizer) > config.vocab_size:
        raise ValueError("Lance tokenizer exceeds the padded model vocabulary")
    initialization = {
        "seeded_native_parameters": random_report,
        "source": source_report,
        "trainable_parameter_names": trainable,
        "config": config.to_dict(),
        "effective_vocab_size": len(tokenizer),
    }
    report = write_native_state_to_dcp(
        model.state_dict(),
        args.output,
        manifest=initialization,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
