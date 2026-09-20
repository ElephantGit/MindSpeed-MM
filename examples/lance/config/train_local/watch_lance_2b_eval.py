#!/usr/bin/env python3
"""Evaluate every completed 2B Lance checkpoint while training continues."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--packed-data", required=True)
    parser.add_argument("--interval", type=int, default=2_000_000_000)
    parser.add_argument("--target", type=int, default=6_000_000_000)
    parser.add_argument("--device-id", default="7")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--training-failed", required=True)
    return parser.parse_args()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main():
    args = parse_args()
    if args.interval <= 0 or args.target <= 0 or args.target % args.interval:
        raise ValueError("target must be a positive multiple of interval")
    here = Path(__file__).resolve().parent
    root = Path(args.checkpoint_root).resolve()
    eval_root = root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    failed = Path(args.training_failed)
    completed = set()

    while args.target not in completed:
        for marker in sorted(root.glob("iter_*/lance_checkpoint.json")):
            checkpoint = json.loads(marker.read_text(encoding="utf-8"))
            consumed = int(checkpoint["consumed_train_tokens"])
            milestone = min(consumed // args.interval * args.interval, args.target)
            if milestone <= 0 or milestone in completed:
                continue
            output = eval_root / "token_{:012d}".format(milestone)
            status_path = output / "eval_status.json"
            if status_path.is_file():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("status") == "completed":
                    completed.add(milestone)
                    continue
            output.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update({
                "ASCEND_RT_VISIBLE_DEVICES": args.device_id,
                "CHECKPOINT": str(marker.parent),
                "OUTPUT_DIR": str(output),
                "MODEL_WEIGHTS": "1",
                "PACKED_DATA": args.packed_data,
                "PROMPT_FILE": str(here / "lance_pretrain_eval_prompts_10task.json"),
                "TEXT_SAMPLES": "4",
                "DEVICE": "npu:0",
            })
            started = time.time()
            with (output / "eval.log").open("a", encoding="utf-8") as stream:
                result = subprocess.run(
                    ["bash", str(here / "run_lance_pretrain_eval.sh")],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            write_json(status_path, {
                "status": "completed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "checkpoint": str(marker.parent),
                "checkpoint_iteration": int(checkpoint["iteration"]),
                "consumed_train_tokens": consumed,
                "milestone_tokens": milestone,
                "tasks": {"t2i": 2, "i2t": 4, "t2t": 4},
                "elapsed_seconds": round(time.time() - started, 3),
            })
            if result.returncode:
                raise RuntimeError(
                    "evaluation failed for {}; see {}".format(
                        marker.parent, output / "eval.log"
                    )
                )
            completed.add(milestone)
        if failed.exists():
            raise RuntimeError("training failed before all checkpoint evaluations completed")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
