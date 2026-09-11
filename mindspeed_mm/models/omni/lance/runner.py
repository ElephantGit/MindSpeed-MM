"""Execution bridge from MindSpeed-MM to the released Lance source tree."""

import importlib
import os
from pathlib import Path
import runpy
import sys
from typing import Dict, List, Optional, Union


class LanceSourceError(RuntimeError):
    pass


def _default_source_candidates() -> List[Path]:
    mindspeed_root = Path(__file__).resolve().parents[4]
    candidates = []
    if os.environ.get("LANCE_SOURCE_ROOT"):
        candidates.append(Path(os.environ["LANCE_SOURCE_ROOT"]))
    candidates.extend(
        [
            mindspeed_root / "third_party" / "Lance",
            mindspeed_root.parent / "Lance",
        ]
    )
    return candidates


def resolve_lance_source(source_root: Optional[Union[str, Path]] = None) -> Path:
    candidates = [Path(source_root)] if source_root else _default_source_candidates()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "inference_lance.py").is_file() and (resolved / "modeling" / "lance").is_dir():
            return resolved
    searched = ", ".join(str(item) for item in candidates)
    raise LanceSourceError(
        "Could not locate a compatible Lance checkout. Set --lance-source-root or "
        "LANCE_SOURCE_ROOT. Searched: {}".format(searched)
    )


def resolve_entrypoint(source_root: Path, entrypoint: str) -> Path:
    candidate = (source_root / entrypoint).resolve()
    try:
        candidate.relative_to(source_root)
    except ValueError as exc:
        raise LanceSourceError("Lance entrypoint must stay inside the source checkout") from exc
    if not candidate.is_file() or candidate.suffix != ".py":
        raise LanceSourceError("Lance entrypoint does not exist: {}".format(candidate))
    return candidate


def describe_run(
    source_root: Path,
    entrypoint: str,
    arguments: List[str],
    execution_mode: str = "inference",
    strict_training: bool = False,
) -> Dict[str, object]:
    if execution_mode not in ("inference", "training"):
        raise LanceSourceError("execution_mode must be 'inference' or 'training'")
    if strict_training and execution_mode != "training":
        raise LanceSourceError("strict_training is only valid for training execution")
    return {
        "source_root": str(source_root),
        "entrypoint": str(resolve_entrypoint(source_root, entrypoint)),
        "arguments": list(arguments),
        "accelerator": "ascend-npu",
        "distributed_backend": "hccl",
        "attention_backend": "torch_npu.npu_fusion_attention:TND",
        "execution_mode": execution_mode,
        "failure_policy": (
            "fail-fast-on-training-step-exception" if strict_training else "upstream-default"
        ),
    }


def run_lance_entrypoint(
    source_root: Union[str, Path],
    entrypoint: str,
    arguments: List[str],
    dry_run: bool = False,
    execution_mode: str = "inference",
    strict_training: bool = False,
) -> Dict[str, object]:
    source = resolve_lance_source(source_root)
    description = describe_run(
        source,
        entrypoint,
        arguments,
        execution_mode,
        strict_training,
    )
    script = resolve_entrypoint(source, entrypoint)
    compiled_entrypoint = None
    if strict_training:
        from .upstream_training import compile_strict_training_entrypoint

        compiled_entrypoint, transform = compile_strict_training_entrypoint(script)
        description["source_transform"] = transform
    if dry_run:
        return description

    from .ascend_runtime import (
        enable_lance_ascend_runtime,
        patch_upstream_lance_fsdp_optimizer_resume,
        patch_upstream_lance_training_attention,
    )

    original_argv = sys.argv[:]
    original_cwd = Path.cwd()
    inserted = str(source) not in sys.path
    try:
        if inserted:
            sys.path.insert(0, str(source))
        os.chdir(str(source))
        runtime = enable_lance_ascend_runtime(execution_mode=execution_mode)
        description["runtime"] = runtime.to_dict()
        if execution_mode == "training":
            training_attention_runtime = patch_upstream_lance_training_attention(
                importlib.import_module("torch_npu")
            )
            optimizer_resume_runtime = patch_upstream_lance_fsdp_optimizer_resume()
            description["training_attention_runtime"] = training_attention_runtime
            description["optimizer_resume_runtime"] = optimizer_resume_runtime
            if os.environ.get("RANK", "0") == "0":
                sys.stdout.write(
                    "Lance training attention bridge: {} -> {}\n".format(
                        training_attention_runtime["mask_backend"],
                        training_attention_runtime["attention_backend"],
                    )
                )
                sys.stdout.write(
                    "Lance optimizer resume bridge: {} -> {}\n".format(
                        optimizer_resume_runtime["save_format"],
                        optimizer_resume_runtime["load_conversion"],
                    )
                )
        sys.argv = [str(script)] + list(arguments)
        if compiled_entrypoint is None:
            runpy.run_path(str(script), run_name="__main__")
        else:
            namespace = {
                "__name__": "__main__",
                "__file__": str(script),
                "__cached__": None,
                "__package__": None,
                "__spec__": None,
            }
            exec(compiled_entrypoint, namespace)
    finally:
        os.chdir(str(original_cwd))
        sys.argv = original_argv
        if inserted and sys.path and sys.path[0] == str(source):
            sys.path.pop(0)
    return description
