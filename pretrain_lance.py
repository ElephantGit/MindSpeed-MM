#!/usr/bin/env python3
"""Run released Lance training through the audited Ascend compatibility bridge."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("NON_MEGATRON", "true")

from mindspeed_mm.models.omni.lance.runner import describe_run, resolve_lance_source, run_lance_entrypoint
from mindspeed_mm.models.omni.lance.training_bridge import (
    UPSTREAM_TRAINING_ENTRYPOINT,
    build_training_bridge_preflight,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--training-manifest", required=False)
    parser.add_argument("--lance-source-root")
    parser.add_argument("--lance-entrypoint", default=UPSTREAM_TRAINING_ENTRYPOINT)
    parser.add_argument("--run-manifest")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--adapter-help", action="store_true")
    return parser


def _rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def _write_manifest(path: str, payload: dict) -> None:
    if _rank() != 0:
        return
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _emit(payload: dict) -> None:
    if _rank() == 0:
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")


def _help() -> None:
    print(
        "Lance Ascend training bridge options:\n"
        "  --training-manifest PATH  output of prepare_lance_training.py\n"
        "  --lance-source-root PATH  released Lance source checkout\n"
        "  --run-manifest PATH        write preflight and terminal run status\n"
        "  --smoke-test               allow reduced steps and token budgets\n"
        "  --preflight-only           validate without importing torch/torch_npu\n"
        "  --dry-run                  print the validated execution without running it\n"
        "All remaining arguments are forwarded to Lance train/unified_train.py.\n"
        "Use '--' before upstream arguments to make the boundary explicit."
    )


def main(argv=None) -> int:
    adapter, forwarded = build_parser().parse_known_args(
        sys.argv[1:] if argv is None else argv
    )
    if adapter.adapter_help:
        _help()
        return 0
    if not adapter.training_manifest:
        raise SystemExit("--training-manifest is required")
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]

    source = resolve_lance_source(adapter.lance_source_root)
    training_manifest_path = Path(adapter.training_manifest).expanduser().resolve()
    if adapter.run_manifest:
        run_manifest_path = Path(adapter.run_manifest).expanduser().resolve()
        if run_manifest_path == training_manifest_path:
            raise SystemExit("--run-manifest must not overwrite --training-manifest")

    preflight = build_training_bridge_preflight(
        source,
        training_manifest_path,
        forwarded,
        entrypoint=adapter.lance_entrypoint,
        smoke_test=adapter.smoke_test,
    )
    if preflight["source"]["status"] == "valid":
        preflight["execution"] = describe_run(
            source,
            adapter.lance_entrypoint,
            forwarded,
            execution_mode="training",
            strict_training=True,
        )
    else:
        preflight["execution"] = {
            "status": "invalid",
            "issues": list(preflight["source"]["issues"]),
        }
    if adapter.run_manifest:
        _write_manifest(adapter.run_manifest, preflight)
    if preflight["status"] != "ready":
        _emit(preflight)
        return 2
    if adapter.preflight_only or adapter.dry_run:
        _emit(preflight)
        return 0

    running = deepcopy(preflight)
    running["status"] = "running"
    if adapter.run_manifest:
        _write_manifest(adapter.run_manifest, running)

    os.environ["LANCE_ASCEND_TRAINING_BRIDGE"] = "1"
    try:
        execution = run_lance_entrypoint(
            source_root=source,
            entrypoint=adapter.lance_entrypoint,
            arguments=forwarded,
            execution_mode="training",
            strict_training=True,
        )
    except BaseException as exc:
        failed = deepcopy(running)
        failed["status"] = "failed"
        failed["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if adapter.run_manifest:
            _write_manifest(adapter.run_manifest, failed)
        raise

    completed = deepcopy(running)
    completed["status"] = "completed"
    completed["execution"] = execution
    if adapter.run_manifest:
        _write_manifest(adapter.run_manifest, completed)
    _emit(completed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
