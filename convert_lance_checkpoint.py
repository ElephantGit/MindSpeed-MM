#!/usr/bin/env python3
"""Inspect and plan conversion of official Lance checkpoints."""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict

os.environ.setdefault("NON_MEGATRON", "true")

from mindspeed_mm.models.omni.lance.checkpoint import (
    LanceCheckpointError,
    audit_checkpoint_metadata,
    build_checkpoint_conversion_plan,
    read_safetensors_header,
    sha256_file,
)
from mindspeed_mm.models.omni.lance.native_config import LanceConfigError, LanceNativeConfig


def _config(args: argparse.Namespace) -> LanceNativeConfig:
    if args.llm_config:
        return LanceNativeConfig.from_llm_config(args.llm_config, variant=args.variant)
    return LanceNativeConfig.for_variant(args.variant)


def _write(payload: Dict[str, Any], output: str) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output == "-":
        sys.stdout.write(serialized)
    else:
        destination = Path(output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized, encoding="utf-8")
        print(destination)


def _inspect(args: argparse.Namespace) -> int:
    config = _config(args)
    header = read_safetensors_header(args.checkpoint)
    audit = audit_checkpoint_metadata(
        header,
        config,
        args.allow_rebuild_position_embedding,
        metadata_only=args.metadata_only,
    )
    payload: Dict[str, Any] = {"config": config.to_dict(), "checkpoint": str(Path(args.checkpoint).resolve())}
    payload["audit"] = audit
    if args.fingerprint:
        payload["sha256"] = sha256_file(args.checkpoint)
    _write(payload, args.output)
    return 0 if audit["valid"] else 2


def _plan(args: argparse.Namespace) -> int:
    config = _config(args)
    header = read_safetensors_header(args.checkpoint)
    plan = build_checkpoint_conversion_plan(
        header,
        config,
        args.allow_rebuild_position_embedding,
        metadata_only=args.metadata_only,
    )
    plan["checkpoint"] = str(Path(args.checkpoint).resolve())
    if args.fingerprint:
        plan["sha256"] = sha256_file(args.checkpoint)
    _write(plan, args.output)
    return 0


def _to_dcp(args: argparse.Namespace) -> int:
    from mindspeed_mm.models.omni.lance.dcp import convert_safetensors_to_dcp

    config = _config(args)
    result = convert_safetensors_to_dcp(
        args.checkpoint,
        args.output_dir,
        config,
        iteration=args.iteration,
        fingerprint=not args.no_fingerprint,
    )
    _write(result, args.output)
    return 0


def _verify_dcp(args: argparse.Namespace) -> int:
    from mindspeed_mm.models.omni.lance.dcp import verify_dcp_metadata

    config = _config(args)
    result = verify_dcp_metadata(args.dcp_dir, config)
    _write(result, args.output)
    return 0 if result["valid"] else 2


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True, help="official model.safetensors path")
    parser.add_argument("--variant", choices=("image", "video"), required=True)
    parser.add_argument("--llm-config", help="optional released llm_config.json")
    parser.add_argument(
        "--allow-rebuild-position-embedding",
        action="store_true",
        help="allow only latent_pos_embed.pos_embed to be absent and rebuilt deterministically",
    )
    parser.add_argument("--fingerprint", action="store_true", help="stream the full file to compute SHA-256")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="validate only the header contract; do not assert that the tensor payload is present",
    )
    parser.add_argument("--output", default="-", help="JSON output path, or - for stdout")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="audit safetensors metadata without loading tensors")
    _add_common(inspect_parser)
    inspect_parser.set_defaults(handler=_inspect)
    plan_parser = subparsers.add_parser("plan", help="emit a validated HF-to-DCP conversion plan")
    _add_common(plan_parser)
    plan_parser.set_defaults(handler=_plan)
    dcp_parser = subparsers.add_parser("to-dcp", help="losslessly convert a validated checkpoint to DCP")
    dcp_parser.add_argument("--checkpoint", required=True)
    dcp_parser.add_argument("--variant", choices=("image", "video"), required=True)
    dcp_parser.add_argument("--llm-config")
    dcp_parser.add_argument("--output-dir", required=True)
    dcp_parser.add_argument("--iteration", default="release")
    dcp_parser.add_argument("--no-fingerprint", action="store_true")
    dcp_parser.add_argument("--output", default="-")
    dcp_parser.set_defaults(handler=_to_dcp)
    verify_parser = subparsers.add_parser("verify-dcp", help="audit DCP metadata without model allocation")
    verify_parser.add_argument("--dcp-dir", required=True)
    verify_parser.add_argument("--variant", choices=("image", "video"), required=True)
    verify_parser.add_argument("--llm-config")
    verify_parser.add_argument("--output", default="-")
    verify_parser.set_defaults(handler=_verify_dcp)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except (LanceCheckpointError, LanceConfigError, RuntimeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
