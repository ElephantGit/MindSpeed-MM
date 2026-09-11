"""Auditable preflight for running upstream Lance training on Ascend.

This module deliberately has no torch dependency.  It validates the immutable
training manifest, the upstream Lance checkout, and the critical arguments that
will be forwarded to ``train/unified_train.py`` before any distributed runtime
is initialized.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from .training_contract import PAPER_OPTIMIZER, STAGES
from .upstream_training import validate_strict_training_entrypoint


class LanceTrainingBridgeError(ValueError):
    """Raised when an upstream training launch violates its declared contract."""


UPSTREAM_TRAINING_ENTRYPOINT = "train/unified_train.py"
UNDERSTANDING_ONLY_SMOKE_CONFIGS = frozenset(
    {"i2t_local.yaml", "v2t_local.yaml", "multi_und.yaml"}
)
UPSTREAM_TRAINING_FILES = (
    UPSTREAM_TRAINING_ENTRYPOINT,
    "train/train_utils.py",
    "train/fsdp_utils.py",
    "data/dataset_base_train.py",
    "data/data_utils.py",
    "modeling/lance/lance.py",
    "modeling/lance/qwen2_navit.py",
    "config/config_factory.py",
)


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(source: Path, *arguments: str) -> Optional[str]:
    result = subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def validate_upstream_training_source(
    source_root: Union[str, Path],
    entrypoint: str = UPSTREAM_TRAINING_ENTRYPOINT,
) -> Dict[str, Any]:
    """Check that the selected checkout contains the released training stack."""

    source = Path(source_root).expanduser().resolve()
    candidate = (source / entrypoint).resolve()
    issues = []
    try:
        candidate.relative_to(source)
    except ValueError:
        issues.append("training entrypoint escapes the Lance source root")

    required = list(UPSTREAM_TRAINING_FILES)
    if entrypoint not in required:
        required[0] = entrypoint
    missing = [value for value in required if not (source / value).is_file()]
    issues.extend("missing upstream training file: {}".format(value) for value in missing)
    revision = _git_value(source, "rev-parse", "HEAD")
    if revision is None:
        issues.append("Lance source is not a readable Git checkout")
    dirty_value = _git_value(source, "status", "--porcelain")
    dirty = bool(dirty_value) if dirty_value is not None else None
    if dirty:
        issues.append("Lance source checkout has uncommitted changes")
    return {
        "status": "valid" if not issues else "invalid",
        "path": str(source),
        "entrypoint": str(candidate),
        "revision": revision,
        "dirty": dirty,
        "missing": missing,
        "issues": issues,
    }


def load_training_manifest(path: Union[str, Path]) -> Tuple[Path, Dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LanceTrainingBridgeError(
            "could not read training manifest {}: {}".format(manifest_path, exc)
        ) from exc
    if not isinstance(payload, Mapping):
        raise LanceTrainingBridgeError("training manifest must contain a JSON object")
    return manifest_path, dict(payload)


def _resolve_launch_path(value: str, source_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source_root / path
    return path.resolve()


def validate_training_manifest(
    manifest_path: Union[str, Path],
    source_root: Union[str, Path],
) -> Dict[str, Any]:
    """Revalidate every path and digest captured by prepare_lance_training.py."""

    try:
        path, manifest = load_training_manifest(manifest_path)
    except LanceTrainingBridgeError as exc:
        return {
            "status": "invalid",
            "path": str(Path(manifest_path).expanduser().resolve()),
            "sha256": None,
            "stage": None,
            "initialization_mode": None,
            "initialization_path": None,
            "world_size": None,
            "datasets": [],
            "payload": {},
            "issues": [str(exc)],
        }
    source = Path(source_root).expanduser().resolve()
    issues = []
    if manifest.get("schema_version") != 1:
        issues.append("unsupported training manifest schema_version")
    if manifest.get("status") != "ready-for-runtime-preflight":
        issues.append("training manifest is not ready-for-runtime-preflight")

    stage = manifest.get("stage")
    stage_name = stage.get("name") if isinstance(stage, Mapping) else None
    if stage_name not in STAGES:
        issues.append("training manifest has an unknown stage")
    elif dict(stage) != STAGES[stage_name].to_dict():
        issues.append("training stage values differ from the built-in Lance contract")

    initialization = manifest.get("initialization")
    init_mode = initialization.get("mode") if isinstance(initialization, Mapping) else None
    init_path_value = initialization.get("path") if isinstance(initialization, Mapping) else None
    if init_mode not in ("qwen2_5_vl", "random", "lance_checkpoint"):
        issues.append("training manifest has an invalid initialization mode")
    if init_mode in ("qwen2_5_vl", "lance_checkpoint"):
        if not init_path_value:
            issues.append("training manifest initialization path is missing")
        else:
            init_path = _resolve_launch_path(str(init_path_value), source)
            if not init_path.exists():
                issues.append("training initialization path does not exist: {}".format(init_path))

    datasets = manifest.get("dataset_manifests")
    if not isinstance(datasets, list) or not datasets:
        issues.append("training manifest must contain at least one dataset manifest")
        datasets = []
    checked_datasets = []
    for item in datasets:
        if not isinstance(item, Mapping) or not item.get("path") or not item.get("sha256"):
            issues.append("dataset manifest entry is incomplete")
            continue
        dataset_path = _resolve_launch_path(str(item["path"]), source)
        if not dataset_path.is_file():
            issues.append("dataset manifest does not exist: {}".format(dataset_path))
            continue
        actual_size = dataset_path.stat().st_size
        actual_digest = sha256_file(dataset_path)
        checked_datasets.append(
            {"path": str(dataset_path), "bytes": actual_size, "sha256": actual_digest}
        )
        if actual_size != item.get("bytes") or actual_digest != item.get("sha256"):
            issues.append("dataset manifest changed after preparation: {}".format(dataset_path))

    world_size = manifest.get("world_size")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        issues.append("training manifest world_size must be a positive integer")
    return {
        "status": "valid" if not issues else "invalid",
        "path": str(path),
        "sha256": sha256_file(path),
        "stage": stage_name,
        "initialization_mode": init_mode,
        "initialization_path": init_path_value,
        "world_size": world_size,
        "datasets": checked_datasets,
        "payload": manifest,
        "issues": issues,
    }


def _option_values(arguments: Sequence[str]) -> Dict[str, Sequence[str]]:
    """Parse forwarded ``--name value...`` arguments without owning their schema."""

    values: Dict[str, list] = {}
    current: Optional[str] = None
    for argument in arguments:
        if argument == "--":
            continue
        if argument.startswith("--"):
            rendered = argument[2:]
            if "=" in rendered:
                name, value = rendered.split("=", 1)
                values[name.replace("-", "_")] = [value]
                current = None
            else:
                current = rendered.replace("-", "_")
                values[current] = []
        elif current is not None:
            values[current].append(argument)
    return values


def _single_option(options: Mapping[str, Sequence[str]], name: str) -> Optional[str]:
    values = options.get(name)
    if not values:
        return None
    return str(values[-1])


def _bool_option(options: Mapping[str, Sequence[str]], name: str, default: bool = False) -> bool:
    value = _single_option(options, name)
    if value is None:
        return default
    rendered = value.lower()
    if rendered in ("true", "1", "yes", "on"):
        return True
    if rendered in ("false", "0", "no", "off"):
        return False
    raise LanceTrainingBridgeError("--{} must be a boolean value".format(name))


def _required_bool_option(
    options: Mapping[str, Sequence[str]],
    name: str,
    issues: list,
) -> Optional[bool]:
    if _single_option(options, name) is None:
        issues.append("critical upstream argument is missing: --{}".format(name))
        return None
    try:
        return _bool_option(options, name)
    except LanceTrainingBridgeError as exc:
        issues.append(str(exc))
        return None


def _number_option(
    options: Mapping[str, Sequence[str]],
    name: str,
    number_type: Any,
    issues: list,
) -> Optional[Union[int, float]]:
    value = _single_option(options, name)
    if value is None:
        issues.append("critical upstream argument is missing: --{}".format(name))
        return None
    try:
        return number_type(value)
    except ValueError:
        issues.append("upstream argument --{} is not a valid number".format(name))
        return None


def _matches_number(actual: Union[int, float], expected: Union[int, float]) -> bool:
    if isinstance(expected, int):
        return actual == expected
    return abs(float(actual) - float(expected)) <= max(1e-12, abs(float(expected)) * 1e-9)


def validate_forwarded_training_arguments(
    arguments: Sequence[str],
    manifest_result: Mapping[str, Any],
    source_root: Union[str, Path],
    *,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Bind upstream CLI arguments to the declared Lance stage and assets."""

    source = Path(source_root).expanduser().resolve()
    options = _option_values(arguments)
    issues = []
    warnings = []
    manifest = manifest_result["payload"]
    stage = manifest["stage"]
    init_mode = manifest["initialization"]["mode"]

    if stage["name"] == "rl":
        issues.append(
            "the supervised upstream unified_train.py entrypoint cannot implement the RL stage"
        )

    load_lance = _required_bool_option(options, "load_from_lance_checkpoint", issues)
    init_vlm = _required_bool_option(options, "init_from_vlm_checkpoint", issues)
    copy_moe = _required_bool_option(options, "copy_init_moe", issues)

    if init_mode == "qwen2_5_vl":
        if load_lance is not False or init_vlm is not True or copy_moe is not True:
            issues.append(
                "qwen2_5_vl initialization requires load_from_lance_checkpoint=false, "
                "init_from_vlm_checkpoint=true, and copy_init_moe=true"
            )
        upstream_init = _single_option(options, "llm_path")
    elif init_mode == "lance_checkpoint":
        if load_lance is not True or init_vlm is not False or copy_moe is not False:
            issues.append(
                "lance_checkpoint initialization requires load_from_lance_checkpoint=true "
                "and init_from_vlm_checkpoint=false and copy_init_moe=false"
            )
        upstream_init = _single_option(options, "model_path")
    else:
        upstream_init = None
        issues.append(
            "the released upstream training entrypoint has no strict-random initialization path; "
            "use the native Lance training path for init_mode=random"
        )

    declared_init = manifest["initialization"].get("path")
    if declared_init and upstream_init:
        expected_path = _resolve_launch_path(str(declared_init), source)
        actual_path = _resolve_launch_path(upstream_init, source)
        if actual_path != expected_path:
            issues.append(
                "forwarded initialization path does not match the training manifest: {} != {}".format(
                    actual_path, expected_path
                )
            )
    elif init_mode != "random" and upstream_init is None:
        issues.append("forwarded initialization path is missing")

    vit_path_value = _single_option(options, "vit_path")
    if vit_path_value is None:
        issues.append("critical upstream argument is missing: --vit_path")
        vit_path = None
    else:
        vit_path = _resolve_launch_path(vit_path_value, source)
        if not vit_path.is_dir():
            issues.append("upstream ViT path is not a directory: {}".format(vit_path))

    dataset_value = _single_option(options, "dataset_config_file")
    if dataset_value is None:
        issues.append("critical upstream argument is missing: --dataset_config_file")
        dataset_path = None
    else:
        dataset_path = _resolve_launch_path(dataset_value, source)
        declared_datasets = {Path(item["path"]).resolve() for item in manifest_result["datasets"]}
        if not dataset_path.is_file():
            issues.append("upstream dataset config does not exist: {}".format(dataset_path))
        elif dataset_path not in declared_datasets:
            issues.append("upstream dataset config is not bound by the training manifest")

    expected_numbers = {
        "total_steps": (int(stage["steps"]), int),
        "warmup_steps": (int(stage["warmup_steps"]), int),
        "lr": (float(stage["learning_rate"]), float),
        "expected_num_tokens": (int(stage["expected_tokens_per_rank"]), int),
        "max_num_tokens": (int(stage["max_tokens_per_rank"]), int),
        "max_num_tokens_per_sample": (int(stage["max_context"]), int),
        "timestep_shift": (float(stage["timestep_shift"]), float),
        "ce_weight": (float(stage["ce_weight"]), float),
        "mse_weight": (float(stage["mse_weight"]), float),
        "text_cond_dropout_prob": (float(stage["text_dropout"]), float),
        "vae_cond_dropout_prob": (float(stage["multimodal_dropout"]), float),
        "vit_cond_dropout_prob": (float(stage["multimodal_dropout"]), float),
        "global_seed": (int(manifest["global_seed"]), int),
        "beta1": (float(PAPER_OPTIMIZER["beta1"]), float),
        "beta2": (float(PAPER_OPTIMIZER["beta2"]), float),
        "eps": (float(PAPER_OPTIMIZER["epsilon"]), float),
        "max_grad_norm": (float(PAPER_OPTIMIZER["max_grad_norm"]), float),
        "ema": (float(PAPER_OPTIMIZER["ema_decay"]), float),
        "min_lr": (float(PAPER_OPTIMIZER["min_learning_rate"]), float),
    }
    checked_numbers = {}
    for name, (expected, number_type) in expected_numbers.items():
        actual = _number_option(options, name, number_type, issues)
        checked_numbers[name] = actual
        if actual is None:
            continue
        if smoke_test and name in {
            "total_steps",
            "warmup_steps",
            "expected_num_tokens",
            "max_num_tokens",
            "max_num_tokens_per_sample",
        }:
            if actual <= 0 or actual > expected:
                issues.append("smoke-test --{} must be in (0, {}]".format(name, expected))
            elif actual != expected:
                warnings.append("smoke-test overrides --{}: {} -> {}".format(name, expected, actual))
        elif not _matches_number(actual, expected):
            issues.append("--{}={} differs from stage contract {}".format(name, actual, expected))

    total_steps = checked_numbers.get("total_steps")
    warmup_steps = checked_numbers.get("warmup_steps")
    expected_tokens = checked_numbers.get("expected_num_tokens")
    max_tokens = checked_numbers.get("max_num_tokens")
    max_sample_tokens = checked_numbers.get("max_num_tokens_per_sample")
    if total_steps is not None and warmup_steps is not None and warmup_steps >= total_steps:
        issues.append("--warmup_steps must be smaller than --total_steps")
    if expected_tokens is not None and max_tokens is not None and expected_tokens > max_tokens:
        issues.append("--expected_num_tokens must not exceed --max_num_tokens")
    if max_sample_tokens is not None and max_tokens is not None and max_sample_tokens > max_tokens:
        issues.append("--max_num_tokens_per_sample must not exceed --max_num_tokens")

    scheduler = _single_option(options, "lr_scheduler")
    if scheduler is None:
        issues.append("critical upstream argument is missing: --lr_scheduler")
    elif scheduler != stage["scheduler"]:
        issues.append("--lr_scheduler={} differs from stage contract {}".format(scheduler, stage["scheduler"]))

    expected_strings = {
        "layer_module": "Qwen2MoTDecoderLayer",
        "vit_type": "qwen2_5_vl",
        "vae_model_type": "wan",
        "sharding_strategy": "HYBRID_SHARD",
        "backward_prefetch": "BACKWARD_PRE",
    }
    for name, expected in expected_strings.items():
        actual = _single_option(options, name)
        if actual is None:
            issues.append("critical upstream argument is missing: --{}".format(name))
        elif actual != expected:
            issues.append("--{}={} differs from bridge contract {}".format(name, actual, expected))

    model_contract = manifest.get("model")
    checked_model_values = {}
    if not isinstance(model_contract, Mapping):
        issues.append("training manifest model contract is missing")
    else:
        for name in ("max_num_frames", "max_latent_size"):
            expected = model_contract.get(name)
            if not isinstance(expected, int) or isinstance(expected, bool):
                issues.append("training manifest model.{} must be an integer".format(name))
                continue
            actual = _number_option(options, name, int, issues)
            checked_model_values[name] = actual
            if actual is not None and actual != expected:
                issues.append("--{}={} differs from model contract {}".format(name, actual, expected))

        expected_patch_size = model_contract.get("latent_patch_size")
        patch_values = options.get("latent_patch_size")
        actual_patch_size = None
        if not isinstance(expected_patch_size, list) or len(expected_patch_size) != 3:
            issues.append("training manifest model.latent_patch_size must contain three integers")
        elif not patch_values:
            issues.append("critical upstream argument is missing: --latent_patch_size")
        else:
            try:
                actual_patch_size = [int(value) for value in patch_values]
            except ValueError:
                issues.append("upstream argument --latent_patch_size must contain integers")
            else:
                if actual_patch_size != expected_patch_size:
                    issues.append(
                        "--latent_patch_size={} differs from model contract {}".format(
                            actual_patch_size, expected_patch_size
                        )
                    )
        checked_model_values["latent_patch_size"] = actual_patch_size

    understanding_only_smoke = (
        smoke_test
        and dataset_path is not None
        and dataset_path.name in UNDERSTANDING_ONLY_SMOKE_CONFIGS
    )
    expected_booleans = {
        "visual_gen": not understanding_only_smoke,
        "visual_und": True,
        "freeze_vit": True,
        "freeze_vae": True,
        "freeze_llm": False,
        "freeze_llm_embed_tokens": False,
        "freeze_vit_connector": False,
        "freeze_und_params": False,
        "freeze_und": False,
        "use_ema": True,
        # This keeps PackedDataset from materialising dense O(L^2) masks. The
        # process-local Ascend runtime preserves the split metadata and routes
        # the upstream FlexAttention call to segmented NPU fused attention.
        "use_flex": True,
        "cpu_offload": False,
    }
    checked_booleans = {}
    for name, expected in expected_booleans.items():
        actual = _required_bool_option(options, name, issues)
        checked_booleans[name] = actual
        if actual is not None and actual is not expected:
            issues.append("--{}={} differs from bridge contract {}".format(name, actual, expected))
        elif name == "visual_gen" and understanding_only_smoke and actual is False:
            warnings.append("understanding-only smoke test overrides --visual_gen: True -> False")

    world_size = manifest_result["world_size"]
    environment_world_size = os.environ.get("WORLD_SIZE")
    if environment_world_size is not None:
        try:
            actual_world_size = int(environment_world_size)
        except ValueError:
            issues.append("WORLD_SIZE is not an integer")
        else:
            if actual_world_size != world_size:
                issues.append(
                    "WORLD_SIZE={} differs from training manifest {}".format(
                        actual_world_size, world_size
                    )
                )
    replicate = _number_option(options, "num_replicate", int, issues)
    shard = _number_option(options, "num_shard", int, issues)
    if replicate is not None and shard is not None:
        if replicate <= 0 or shard <= 0 or replicate * shard != world_size:
            issues.append(
                "num_replicate * num_shard must equal manifest world_size {}".format(world_size)
            )

    return {
        "status": "valid" if not issues else "invalid",
        "smoke_test": smoke_test,
        "smoke_profile": "understanding-only" if understanding_only_smoke else "joint",
        "dataset_config_file": str(dataset_path) if dataset_path is not None else None,
        "vit_path": str(vit_path) if vit_path is not None else None,
        "initialization_mode": init_mode,
        "checked_stage_values": checked_numbers,
        "checked_model_values": checked_model_values,
        "checked_booleans": checked_booleans,
        "num_replicate": replicate,
        "num_shard": shard,
        "options": {name: list(values) for name, values in sorted(options.items())},
        "warnings": warnings,
        "issues": issues,
    }


def build_training_bridge_preflight(
    source_root: Union[str, Path],
    training_manifest: Union[str, Path],
    arguments: Sequence[str],
    *,
    entrypoint: str = UPSTREAM_TRAINING_ENTRYPOINT,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    source_result = validate_upstream_training_source(source_root, entrypoint)
    manifest_result = validate_training_manifest(training_manifest, source_root)
    failure_policy_result = validate_strict_training_entrypoint(
        source_result["entrypoint"]
    )
    payload = manifest_result["payload"]
    can_validate_arguments = manifest_result["status"] == "valid"
    if can_validate_arguments:
        arguments_result = validate_forwarded_training_arguments(
            arguments,
            manifest_result,
            source_root,
            smoke_test=smoke_test,
        )
    else:
        arguments_result = {
            "status": "invalid",
            "smoke_test": smoke_test,
            "warnings": [],
            "issues": ["forwarded arguments cannot be validated against an invalid manifest structure"],
        }
    issues = (
        list(source_result["issues"])
        + list(manifest_result["issues"])
        + list(failure_policy_result["issues"])
        + list(arguments_result["issues"])
    )
    return {
        "schema_version": 1,
        "status": "ready" if not issues else "invalid",
        "mode": "upstream-lance-ascend-training-bridge",
        "source": source_result,
        "failure_policy": failure_policy_result,
        "training_manifest": {
            key: value for key, value in manifest_result.items() if key != "payload"
        },
        "forwarded_arguments": list(arguments),
        "argument_validation": arguments_result,
        "issues": issues,
    }
