#!/usr/bin/env python3
"""Fail-fast validation for native Lance training artifacts."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep preflight usable before importing torch/torch_npu.  Importing the top
# level mindspeed_mm package initializes accelerator-dependent modules, while
# the Lance shape contract itself is deliberately dependency-free.
_CONFIG_PATH = REPO_ROOT / "mindspeed_mm/models/omni/lance/native_config.py"
_SPEC = importlib.util.spec_from_file_location("_lance_native_config", _CONFIG_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("cannot load native Lance config contract")
_CONFIG_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CONFIG_MODULE
_SPEC.loader.exec_module(_CONFIG_MODULE)
LanceNativeConfig = _CONFIG_MODULE.LanceNativeConfig


def parse_arguments():
    parser = argparse.ArgumentParser(description="Validate native Lance DCP and packed data")
    parser.add_argument("--load", required=True, help="Initialization or resume DCP root")
    parser.add_argument("--data", required=True, help="Packed native Lance batch directory")
    parser.add_argument("--llm-config", required=True)
    parser.add_argument("--variant", choices=("image", "video"), default="video")
    parser.add_argument("--latent-patch-size", nargs=3, type=int)
    parser.add_argument("--max-latent-size", type=int)
    parser.add_argument("--max-num-frames", type=int)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--effective-vocab-size", type=int, required=True)
    return parser.parse_args()


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read JSON {}: {}".format(path, exc)) from exc


def _checkpoint(root, config, effective_vocab_size):
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise FileNotFoundError("DCP tracker is missing: {}".format(tracker))
    iteration = tracker.read_text(encoding="utf-8").strip()
    if iteration == "release":
        target = root / "release"
    else:
        try:
            target = root / "iter_{:07d}".format(int(iteration))
        except ValueError as exc:
            raise ValueError("invalid DCP tracker value: {}".format(iteration)) from exc
    if not target.is_dir() or not (target / ".metadata").is_file():
        raise FileNotFoundError("DCP iteration is incomplete: {}".format(target))

    initialization = root / "lance_native_initialization.json"
    kind = "resume"
    if initialization.is_file():
        manifest = _read_json(initialization)
        recorded = manifest.get("initialization", {}).get("config", {})
        expected = config.to_dict()
        shape_fields = (
            "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "latent_channels",
            "latent_patch_size", "max_latent_size", "max_num_frames",
        )
        mismatches = {
            name: {"checkpoint": recorded.get(name), "training": expected[name]}
            for name in shape_fields
            if recorded.get(name) != expected[name]
        }
        if mismatches:
            raise ValueError(
                "initialization DCP and training model configurations differ: {}".format(
                    json.dumps(mismatches, sort_keys=True)
                )
            )
        recorded_vocab = manifest.get("initialization", {}).get("effective_vocab_size")
        if recorded_vocab is not None and int(recorded_vocab) != effective_vocab_size:
            raise ValueError(
                "initialization effective vocabulary {} differs from training {}".format(
                    recorded_vocab, effective_vocab_size
                )
            )
        kind = "initialization"
    elif iteration == "release":
        raise FileNotFoundError(
            "release DCP lacks lance_native_initialization.json: {}".format(root)
        )
    return {"root": str(root), "iteration": iteration, "kind": kind}


def _data(root, config, effective_vocab_size, world_size):
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("packed data manifest is missing: {}".format(manifest_path))
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError("packed data manifest is not completed")
    if manifest.get("variant") != config.variant:
        raise ValueError("packed data variant differs from the training model")
    recorded = manifest.get("config")
    if not isinstance(recorded, dict):
        raise ValueError("packed data manifest lacks the native model config")
    expected = config.to_dict()
    shape_fields = (
        "vocab_size", "hidden_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "latent_channels",
        "latent_patch_size", "max_latent_size", "max_num_frames",
    )
    mismatches = [
        name for name in shape_fields if recorded.get(name) != expected[name]
    ]
    if mismatches:
        raise ValueError(
            "packed data and training configurations differ: {}".format(
                ", ".join(mismatches)
            )
        )
    recorded_vocab = manifest.get("effective_vocab_size")
    if recorded_vocab is None or int(recorded_vocab) != effective_vocab_size:
        raise ValueError(
            "packed-data effective vocabulary {} differs from training {}".format(
                recorded_vocab, effective_vocab_size
            )
        )
    batches = sorted(root.glob("batch-*.pt"))
    declared = int(manifest.get("batch_count", -1))
    if declared != len(batches):
        raise ValueError(
            "packed data batch count mismatch: manifest={}, files={}".format(
                declared, len(batches)
            )
        )
    if len(batches) < world_size:
        raise ValueError(
            "packed data has {} batches but {} data-parallel ranks require at least one each".format(
                len(batches), world_size
            )
        )
    return {
        "root": str(root),
        "batches": len(batches),
        "total_tokens": int(manifest.get("total_tokens", 0)),
        "skipped_samples": int(manifest.get("skipped_sample_count", 0)),
    }


def main():
    args = parse_arguments()
    if args.world_size <= 0:
        raise ValueError("world-size must be positive")
    config = LanceNativeConfig.from_llm_config(args.llm_config, variant=args.variant)
    geometry = {
        name: value for name, value in {
            "latent_patch_size": args.latent_patch_size,
            "max_latent_size": args.max_latent_size,
            "max_num_frames": args.max_num_frames,
        }.items() if value is not None
    }
    config = config.with_overrides(**geometry)
    if not 0 < args.effective_vocab_size <= config.vocab_size:
        raise ValueError("effective-vocab-size must be in (0, config.vocab_size]")
    result = {
        "schema_version": 1,
        "status": "valid",
        "mode": "native-mindspeed-mm-preflight",
        "config": config.to_dict(),
        "effective_vocab_size": args.effective_vocab_size,
        "checkpoint": _checkpoint(
            Path(args.load).expanduser().resolve(), config,
            args.effective_vocab_size,
        ),
        "data": _data(
            Path(args.data).expanduser().resolve(), config,
            args.effective_vocab_size, args.world_size,
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
