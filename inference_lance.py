#!/usr/bin/env python3
"""Run the official Lance inference contract through MindSpeed-MM on Ascend."""

import argparse
import json
import os
import sys

os.environ.setdefault("NON_MEGATRON", "true")

from mindspeed_mm.models.omni.lance import resolve_lance_source, run_lance_entrypoint


def parse_adapter_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lance-source-root")
    parser.add_argument("--lance-entrypoint", default="inference_lance.py")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runtime-check", action="store_true")
    parser.add_argument("--adapter-help", action="store_true")
    return parser.parse_known_args(argv)


def main(argv=None):
    adapter_args, lance_args = parse_adapter_args(sys.argv[1:] if argv is None else argv)
    if adapter_args.adapter_help:
        print(
            "MindSpeed-MM Lance adapter options:\n"
            "  --lance-source-root PATH   upstream Lance checkout\n"
            "  --lance-entrypoint PATH    Python entrypoint within that checkout\n"
            "  --dry-run                  validate and print the resolved execution\n"
            "  --runtime-check             run NPU varlen-attention numerical smoke test\n"
            "All other arguments are forwarded unchanged to Lance."
        )
        return 0

    if adapter_args.runtime_check:
        from mindspeed_mm.models.omni.lance.ascend_runtime import attention_smoke_test

        try:
            result = attention_smoke_test()
        except RuntimeError as exc:
            result = {
                "status": "blocked",
                "check": "ascend-varlen-attention",
                "reason": str(exc),
            }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "passed" else 2

    source_root = resolve_lance_source(adapter_args.lance_source_root)
    result = run_lance_entrypoint(
        source_root=source_root,
        entrypoint=adapter_args.lance_entrypoint,
        arguments=lance_args,
        dry_run=adapter_args.dry_run,
    )
    if adapter_args.dry_run:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
