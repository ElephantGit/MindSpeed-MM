"""Lance safetensors to MindSpeed-MM distributed-checkpoint conversion.

Torch and safetensors imports are deliberately lazy so metadata-only evaluation
commands continue to work on login nodes without the training environment.
"""

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from .checkpoint import (
    LanceCheckpointError,
    audit_checkpoint_metadata,
    expected_state_shapes,
    read_safetensors_header,
    sha256_file,
)
from .native_config import LanceNativeConfig


class LanceDCPError(RuntimeError):
    """Raised when lossless Lance DCP conversion or verification fails."""


def _require_conversion_dependencies() -> Tuple[Any, Any, Any]:
    try:
        import torch
        from safetensors.torch import load_file
        from torch.distributed.checkpoint import FileSystemWriter
        from torch.distributed.checkpoint.state_dict_saver import _save_state_dict
    except ImportError as exc:
        raise LanceDCPError(
            "DCP conversion requires torch, safetensors, and torch.distributed.checkpoint"
        ) from exc
    return torch, load_file, (FileSystemWriter, _save_state_dict)


def _require_reader() -> Tuple[Any, Any]:
    try:
        import torch
        from torch.distributed.checkpoint import FileSystemReader
    except ImportError as exc:
        raise LanceDCPError("DCP verification requires torch.distributed.checkpoint") from exc
    return torch, FileSystemReader


def _shape_tuple(value: Any) -> Tuple[int, ...]:
    return tuple(int(item) for item in value)


def _dtype_name(dtype: Any) -> str:
    rendered = str(dtype).replace("torch.", "")
    aliases = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}
    return aliases.get(rendered, rendered.upper())


def _ensure_new_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise LanceDCPError("DCP output path exists and is not a directory: {}".format(path))
        if any(path.iterdir()):
            raise LanceDCPError("refusing to overwrite non-empty DCP directory: {}".format(path))
    path.mkdir(parents=True, exist_ok=True)


def _validate_loaded_state(
    state_dict: Mapping[str, Any],
    expected: Mapping[str, Sequence[int]],
) -> None:
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    mismatched = [
        name
        for name in sorted(set(expected) & set(state_dict))
        if tuple(state_dict[name].shape) != tuple(expected[name])
    ]
    bad_dtype = [name for name, value in state_dict.items() if _dtype_name(value.dtype) != "BF16"]
    if missing or unexpected or mismatched or bad_dtype:
        raise LanceDCPError(
            "loaded tensor validation failed: missing={}, unexpected={}, shape={}, dtype={}".format(
                len(missing), len(unexpected), len(mismatched), len(bad_dtype)
            )
        )


def convert_safetensors_to_dcp(
    checkpoint: Union[str, Path],
    output_dir: Union[str, Path],
    config: LanceNativeConfig,
    iteration: str = "release",
    fingerprint: bool = True,
) -> Dict[str, Any]:
    """Losslessly write one official Lance safetensors file as MindSpeed DCP."""

    source = Path(checkpoint).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()
    header = read_safetensors_header(source)
    audit = audit_checkpoint_metadata(header, config)
    if not audit["valid"]:
        raise LanceCheckpointError("source checkpoint failed the Lance {} contract".format(config.variant))

    torch, load_file, writer_api = _require_conversion_dependencies()
    FileSystemWriter, save_state_dict = writer_api
    expected = expected_state_shapes(config)
    state_dict = load_file(str(source), device="cpu")
    _validate_loaded_state(state_dict, expected)

    _ensure_new_directory(root)
    checkpoint_dir = root / iteration
    checkpoint_dir.mkdir(parents=False, exist_ok=False)
    in_progress = root / "lance_conversion.json"
    conversion: Dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "variant": config.variant,
        "source": str(source),
        "source_sha256": sha256_file(source) if fingerprint else None,
        "source_audit": audit,
        "target": str(checkpoint_dir),
        "iteration": iteration,
        "mapping": "identity",
        "lossless": True,
    }
    in_progress.write_text(json.dumps(conversion, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    save_payload = {"model": state_dict, "checkpoint_version": 3.0}
    save_state_dict(
        save_payload,
        storage_writer=FileSystemWriter(str(checkpoint_dir)),
        no_dist=True,
    )
    (root / "latest_checkpointed_iteration.txt").write_text(iteration, encoding="utf-8")
    verification = verify_dcp_metadata(checkpoint_dir, config)
    if not verification["valid"]:
        conversion["status"] = "invalid"
        conversion["verification"] = verification
        in_progress.write_text(json.dumps(conversion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise LanceDCPError("DCP metadata did not round-trip to the Lance contract")

    files = []
    for path in sorted(checkpoint_dir.iterdir()):
        if path.is_file():
            files.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path) if fingerprint else None,
                }
            )
    conversion.update({"status": "completed", "verification": verification, "files": files})
    in_progress.write_text(json.dumps(conversion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del state_dict
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return conversion


def _tensor_metadata_shape_and_dtype(metadata: Any) -> Tuple[Tuple[int, ...], str]:
    size = getattr(metadata, "size", None)
    properties = getattr(metadata, "properties", None)
    dtype = getattr(properties, "dtype", None) if properties is not None else None
    if size is None or dtype is None:
        raise LanceDCPError("DCP tensor metadata is missing size or dtype")
    return _shape_tuple(size), _dtype_name(dtype)


def verify_dcp_metadata(
    checkpoint_dir: Union[str, Path],
    config: LanceNativeConfig,
) -> Dict[str, Any]:
    """Validate DCP names/shapes/dtypes without allocating the model tensors."""

    _, FileSystemReader = _require_reader()
    directory = Path(checkpoint_dir).expanduser().resolve()
    if not directory.is_dir():
        raise LanceDCPError("DCP checkpoint directory does not exist: {}".format(directory))
    metadata = FileSystemReader(str(directory)).read_metadata()
    entries = getattr(metadata, "state_dict_metadata", {})
    prefix = "model."
    actual: Dict[str, Tuple[Tuple[int, ...], str]] = {}
    for name, tensor_metadata in entries.items():
        if not name.startswith(prefix):
            continue
        try:
            actual[name[len(prefix) :]] = _tensor_metadata_shape_and_dtype(tensor_metadata)
        except LanceDCPError:
            # Non-tensor model entries are not valid in the released Lance tree.
            actual[name[len(prefix) :]] = ((), "NON_TENSOR")

    expected = expected_state_shapes(config)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    shape_mismatches = [
        {
            "name": name,
            "expected": list(expected[name]),
            "actual": list(actual[name][0]),
        }
        for name in sorted(set(expected) & set(actual))
        if tuple(expected[name]) != actual[name][0]
    ]
    dtype_mismatches = [
        {"name": name, "expected": "BF16", "actual": actual[name][1]}
        for name in sorted(set(expected) & set(actual))
        if actual[name][1] != "BF16"
    ]
    valid = not any((missing, unexpected, shape_mismatches, dtype_mismatches))
    return {
        "status": "valid" if valid else "invalid",
        "valid": valid,
        "variant": config.variant,
        "checkpoint_dir": str(directory),
        "tensor_count": len(actual),
        "expected_tensor_count": len(expected),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
    }
