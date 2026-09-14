#!/usr/bin/env python3
"""Pack native Lance prepared samples into high-throughput training batches."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("NON_MEGATRON", "true")

import torch

from mindspeed_mm.models.omni.lance.data import LancePreparedSample, pack_preencoded_samples
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig


def parse_arguments():
    parser = argparse.ArgumentParser(description="Pack native Lance .pt samples")
    parser.add_argument("--input", action="append", required=True, help="Prepared .pt file or directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=("image", "video"), default="video")
    parser.add_argument("--llm-config", help="Qwen/Lance config.json used for exact dimensions")
    parser.add_argument("--latent-patch-size", nargs=3, type=int)
    parser.add_argument("--max-latent-size", type=int)
    parser.add_argument("--max-num-frames", type=int)
    parser.add_argument("--expected-tokens", type=int, default=44000)
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--max-sample-tokens", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--task-weights",
        help="Optional comma list such as t2v=64,v2t=16,t2i=16,i2t=4",
    )
    return parser.parse_args()


def _files(inputs):
    result = []
    for value in inputs:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            result.extend(sorted(path.rglob("*.pt")))
        elif path.is_file():
            result.append(path)
        else:
            raise FileNotFoundError("prepared sample input not found: {}".format(path))
    if not result:
        raise ValueError("no prepared .pt samples found")
    return result


def _prepared_manifests(inputs, config):
    """Require a complete, geometry-compatible distributed encode result."""

    manifests = []
    for value in inputs:
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(
                "native packing requires prepared-data directories with rank manifests: {}".format(
                    root
                )
            )
        candidates = sorted(root.glob("rank-*/manifest.json"))
        if not candidates and (root / "manifest.json").is_file():
            candidates = [root / "manifest.json"]
        if not candidates:
            raise FileNotFoundError("prepared-data manifests are missing under {}".format(root))
        manifests.extend(candidates)

    expected = config.to_dict()
    shape_fields = (
        "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "latent_channels", "latent_patch_size",
        "max_latent_size", "max_num_frames",
    )
    ranks_by_world = {}
    vocab_sizes = set()
    summaries = []
    for path in manifests:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("cannot read prepared-data manifest {}: {}".format(path, exc)) from exc
        if payload.get("status") != "completed":
            raise ValueError("prepared-data manifest is not completed: {}".format(path))
        recorded = payload.get("config")
        if not isinstance(recorded, dict):
            raise ValueError("prepared-data manifest lacks the native model config: {}".format(path))
        mismatches = [name for name in shape_fields if recorded.get(name) != expected[name]]
        if mismatches:
            raise ValueError(
                "prepared data and packing configurations differ at {}: {}".format(
                    path, ", ".join(mismatches)
                )
            )
        world_size = int(payload.get("world_size", 1))
        rank = int(payload.get("rank", 0))
        vocab_size = payload.get("effective_vocab_size")
        if vocab_size is None:
            raise ValueError("prepared-data manifest lacks effective_vocab_size: {}".format(path))
        vocab_sizes.add(int(vocab_size))
        ranks_by_world.setdefault(world_size, set()).add(rank)
        summaries.append({
            "path": str(path),
            "rank": rank,
            "world_size": world_size,
            "written": int(payload.get("written", 0)),
            "failure_count": int(payload.get("failure_count", 0)),
        })
    if len(ranks_by_world) != 1:
        raise ValueError("prepared-data manifests contain inconsistent world sizes")
    if len(vocab_sizes) != 1:
        raise ValueError("prepared-data manifests contain inconsistent tokenizer sizes")
    effective_vocab_size = next(iter(vocab_sizes))
    if not 0 < effective_vocab_size <= config.vocab_size:
        raise ValueError("prepared-data effective vocabulary exceeds the model vocabulary")
    world_size, ranks = next(iter(ranks_by_world.items()))
    expected_ranks = set(range(world_size))
    if ranks != expected_ranks:
        raise ValueError(
            "prepared-data rank manifests are incomplete: missing={}, unexpected={}".format(
                sorted(expected_ranks - ranks), sorted(ranks - expected_ranks)
            )
        )
    return {
        "ranks": summaries,
        "effective_vocab_size": effective_vocab_size,
    }


def _samples(path):
    value = torch.load(path, map_location="cpu", weights_only=False)
    values = value if isinstance(value, (list, tuple)) else (value,)
    for item in values:
        if isinstance(item, LancePreparedSample):
            yield item
        elif isinstance(item, dict):
            yield LancePreparedSample(**item)
        else:
            raise TypeError("{} contains unsupported value {}".format(path, type(item).__name__))


def _new_output(path):
    path = Path(path).expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError("refusing to overwrite non-empty output: {}".format(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _weighted_files(files, raw_weights, rng):
    if not raw_weights:
        rng.shuffle(files)
        return files, {}
    weights = {}
    for item in raw_weights.split(","):
        name, value = item.split("=", 1)
        weights[name.strip()] = float(value)
    if not weights or any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("task weights must be non-negative and contain a positive value")
    grouped = {name: [] for name in weights}
    ignored = []
    for path in files:
        matches = _matching_tasks(path, weights)
        if len(matches) == 1 and weights[matches[0]] > 0:
            grouped[matches[0]].append(path)
        else:
            ignored.append(path)
    for values in grouped.values():
        rng.shuffle(values)
    missing = [name for name, values in grouped.items() if weights[name] > 0 and not values]
    if missing:
        raise ValueError(
            "task weights requested groups without samples: {}".format(
                ", ".join(sorted(missing))
            )
        )
    # Preserve every matched source sample, then deterministically oversample
    # smaller groups so uniform sampling over packed batches implements the
    # requested long-run task distribution.  Merely reordering files would not
    # change probabilities once the stateful dataloader starts a new epoch.
    scale = max(
        len(grouped[name]) / weights[name]
        for name in weights if weights[name] > 0
    )
    target_counts = {
        name: int(math.ceil(scale * weights[name])) if weights[name] > 0 else 0
        for name in weights
    }
    source_counts = {name: len(values) for name, values in grouped.items()}
    expanded = {}
    for name, values in grouped.items():
        target = target_counts[name]
        expanded[name] = []
        while len(expanded[name]) < target:
            cycle = list(values)
            rng.shuffle(cycle)
            expanded[name].extend(cycle[:target - len(expanded[name])])
    grouped = expanded
    ordered = []
    while True:
        active = [name for name, values in grouped.items() if values]
        if not active:
            break
        selected = rng.choices(active, weights=[weights[name] for name in active], k=1)[0]
        ordered.append(grouped[selected].pop())
    if not ordered:
        raise ValueError("task weights did not match any prepared sample paths")
    return ordered, {
        "weights": weights,
        "source_counts": source_counts,
        "target_counts": target_counts,
        "ignored_files": len(ignored),
        "sampling": "deterministic-oversampling",
    }


def _matching_tasks(path, weights):
    path_tasks = set(path.parts)
    if "ff2v" in path_tasks and "ff2v" not in weights:
        path_tasks.add("t2v")
    return [name for name in weights if name in path_tasks]


def _save_batch(output, index, samples, config, maximum):
    packed = pack_preencoded_samples(
        samples,
        config,
        max_tokens=maximum,
        attention_backend="ascend",
    )
    target = output / "batch-{:08d}.pt".format(index)
    temporary = output / (target.name + ".tmp")
    torch.save(packed.batch, temporary)
    temporary.replace(target)
    return target, packed.batch.sequence_length


def main():
    args = parse_arguments()
    if not 0 < args.expected_tokens <= args.max_tokens:
        raise ValueError("expected-tokens must be positive and <= max-tokens")
    if not 0 < args.max_sample_tokens <= args.max_tokens:
        raise ValueError("max-sample-tokens must be positive and <= max-tokens")
    config = (
        LanceNativeConfig.from_llm_config(args.llm_config, variant=args.variant)
        if args.llm_config else LanceNativeConfig.for_variant(args.variant)
    )
    geometry = {
        name: value for name, value in {
            "latent_patch_size": args.latent_patch_size,
            "max_latent_size": args.max_latent_size,
            "max_num_frames": args.max_num_frames,
        }.items() if value is not None
    }
    config = config.with_overrides(**geometry)
    output = _new_output(args.output)
    prepared_manifests = _prepared_manifests(args.input, config)
    files = _files(args.input)
    files, mixture = _weighted_files(files, args.task_weights, random.Random(args.seed))

    current = []
    current_tokens = 0
    batch_count = 0
    accepted = 0
    skipped = []
    accepted_task_counts = {}
    skipped_task_counts = {}
    total_tokens = 0
    for path in files:
        task_group = None
        if mixture:
            matches = _matching_tasks(path, mixture["weights"])
            if len(matches) != 1:
                raise ValueError("packed source has ambiguous task group: {}".format(path))
            task_group = matches[0]
        for sample in _samples(path):
            sample.validate(config)
            # Upstream PackedDataset accepts only length < max-per-sample.
            if sample.length >= args.max_sample_tokens:
                skipped.append({
                    "sample_id": sample.sample_id,
                    "task": task_group,
                    "tokens": sample.length,
                })
                if task_group is not None:
                    skipped_task_counts[task_group] = skipped_task_counts.get(task_group, 0) + 1
                continue
            if current and current_tokens + sample.length > args.max_tokens:
                _, tokens = _save_batch(output, batch_count, current, config, args.max_tokens)
                batch_count += 1
                total_tokens += tokens
                current, current_tokens = [], 0
            current.append(sample)
            current_tokens += sample.length
            accepted += 1
            if task_group is not None:
                accepted_task_counts[task_group] = accepted_task_counts.get(task_group, 0) + 1
            if current_tokens >= args.expected_tokens:
                _, tokens = _save_batch(output, batch_count, current, config, args.max_tokens)
                batch_count += 1
                total_tokens += tokens
                current, current_tokens = [], 0
    if current:
        _, tokens = _save_batch(output, batch_count, current, config, args.max_tokens)
        batch_count += 1
        total_tokens += tokens

    file_digest = hashlib.sha256()
    for path in files:
        file_digest.update(str(path).encode("utf-8"))
        file_digest.update(b"\0")
    missing_task_groups = []
    if mixture:
        missing_task_groups = sorted(
            name for name, weight in mixture["weights"].items()
            if weight > 0 and accepted_task_counts.get(name, 0) == 0
        )
    manifest = {
        "schema_version": 1,
        "status": "completed" if batch_count and not missing_task_groups else "invalid",
        "mode": "native-mindspeed-mm-preencoded-packing",
        "variant": args.variant,
        "config": config.to_dict(),
        "effective_vocab_size": prepared_manifests["effective_vocab_size"],
        "seed": args.seed,
        "input_file_count": len(files),
        "input_path_digest": file_digest.hexdigest(),
        "accepted_samples": accepted,
        "accepted_task_counts": accepted_task_counts,
        "skipped_sample_count": len(skipped),
        "skipped_task_counts": skipped_task_counts,
        "skipped_samples": skipped[:100],
        "missing_task_groups": missing_task_groups,
        "batch_count": batch_count,
        "total_tokens": total_tokens,
        "expected_tokens": args.expected_tokens,
        "max_tokens": args.max_tokens,
        "max_sample_tokens": args.max_sample_tokens,
        "mixture": mixture,
        "prepared_manifests": prepared_manifests,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["status"] != "completed":
        if missing_task_groups:
            raise RuntimeError(
                "all samples were rejected for required task groups: {}. "
                "Check latent patch geometry and max-sample-tokens.".format(
                    ", ".join(missing_task_groups)
                )
            )
        raise RuntimeError("native Lance packing produced no batches")


if __name__ == "__main__":
    main()
