"""Execution bridge from MindSpeed-MM to the released Lance source tree."""

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


def describe_run(source_root: Path, entrypoint: str, arguments: List[str]) -> Dict[str, object]:
    return {
        "source_root": str(source_root),
        "entrypoint": str(resolve_entrypoint(source_root, entrypoint)),
        "arguments": list(arguments),
        "accelerator": "ascend-npu",
        "distributed_backend": "hccl",
        "attention_backend": "torch_npu.npu_fusion_attention:TND",
    }


def run_lance_entrypoint(
    source_root: Union[str, Path],
    entrypoint: str,
    arguments: List[str],
    dry_run: bool = False,
) -> Dict[str, object]:
    source = resolve_lance_source(source_root)
    description = describe_run(source, entrypoint, arguments)
    if dry_run:
        return description

    from .ascend_runtime import enable_lance_ascend_runtime

    runtime = enable_lance_ascend_runtime()
    description["runtime"] = runtime.to_dict()
    script = resolve_entrypoint(source, entrypoint)
    original_argv = sys.argv[:]
    original_cwd = Path.cwd()
    inserted = str(source) not in sys.path
    if inserted:
        sys.path.insert(0, str(source))
    try:
        sys.argv = [str(script)] + list(arguments)
        os.chdir(str(source))
        runpy.run_path(str(script), run_name="__main__")
    finally:
        os.chdir(str(original_cwd))
        sys.argv = original_argv
        if inserted and sys.path and sys.path[0] == str(source):
            sys.path.pop(0)
    return description
