#!/usr/bin/env python3
"""Emit an auditable Lance PT/CT/SFT/RL launch contract."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("NON_MEGATRON", "true")

from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.training_contract import STAGES, training_manifest


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument(
        "--init-mode",
        choices=("qwen2_5_vl", "random", "lance_checkpoint"),
        required=True,
    )
    parser.add_argument("--init-path")
    parser.add_argument("--variant", choices=("image", "video"), default="video")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--dataset-manifest", action="append", default=[])
    parser.add_argument("--output")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.init_mode in ("qwen2_5_vl", "lance_checkpoint") and not args.init_path:
        raise SystemExit("--init-path is required for --init-mode={}".format(args.init_mode))
    if args.init_mode == "random" and args.init_path:
        raise SystemExit("strict random initialization must not specify --init-path")

    result = training_manifest(args.stage, args.init_mode, args.world_size, seed=args.seed)
    result["model"] = LanceNativeConfig.for_variant(args.variant).to_dict()
    result["initialization"]["path"] = str(Path(args.init_path).expanduser().resolve()) if args.init_path else None
    datasets = []
    for value in args.dataset_manifest:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise SystemExit("dataset manifest does not exist: {}".format(path))
        datasets.append({"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    result["dataset_manifests"] = datasets
    result["status"] = "ready-for-runtime-preflight" if datasets else "dataset-manifest-required"

    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(output)
    else:
        sys.stdout.write(rendered)
    return 0 if datasets else 3


if __name__ == "__main__":
    raise SystemExit(main())

