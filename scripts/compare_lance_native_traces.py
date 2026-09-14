#!/usr/bin/env python3
"""Compare native Lance continuous and checkpoint-resumed validation traces."""

import argparse
import json
import math
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare 20-step and 10+resume+10 native Lance traces"
    )
    parser.add_argument("--continuous", required=True, help="Continuous trace base path")
    parser.add_argument("--resumed", required=True, help="Resumed trace base path")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


def _rank_files(base):
    path = Path(base).expanduser().resolve()
    suffix = path.suffix or ".jsonl"
    stem = path.name[:-len(path.suffix)] if path.suffix else path.name
    files = sorted(path.parent.glob("{}.rank?????{}".format(stem, suffix)))
    if not files:
        raise FileNotFoundError("no rank traces found for {}".format(path))
    return files


def _load(base):
    result = {}
    for path in _rank_files(base):
        records = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise ValueError("invalid JSON at {}:{}".format(path, number)) from exc
        if not records:
            raise ValueError("empty rank trace: {}".format(path))
        rank = int(records[0]["rank"])
        if any(int(record["rank"]) != rank for record in records):
            raise ValueError("rank changes within trace: {}".format(path))
        iterations = [int(record["iteration"]) for record in records]
        if len(iterations) != len(set(iterations)) or iterations != sorted(iterations):
            raise ValueError("trace iterations are duplicated or unordered: {}".format(path))
        result[rank] = records
    return result


def _close(left, right, atol, rtol):
    return math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol)


def main():
    args = parse_arguments()
    continuous = _load(args.continuous)
    resumed = _load(args.resumed)
    issues = []
    if set(continuous) != set(resumed):
        issues.append("rank sets differ")
    for rank in sorted(set(continuous) & set(resumed)):
        left = continuous[rank]
        right = resumed[rank]
        if len(left) != len(right):
            issues.append("rank {} record counts differ: {} != {}".format(rank, len(left), len(right)))
            continue
        for expected, actual in zip(left, right):
            iteration = int(expected["iteration"])
            if iteration != int(actual["iteration"]):
                issues.append("rank {} iteration sequence differs".format(rank))
                break
            if expected.get("batch_paths") != actual.get("batch_paths"):
                issues.append("rank {} iteration {} consumed different batches".format(rank, iteration))
            for name in ("loss", "grad_norm", "learning_rate"):
                if expected.get(name) is None or actual.get(name) is None:
                    if expected.get(name) != actual.get(name):
                        issues.append("rank {} iteration {} {} differs".format(rank, iteration, name))
                elif not _close(expected[name], actual[name], args.atol, args.rtol):
                    issues.append(
                        "rank {} iteration {} {} differs: {} != {}".format(
                            rank, iteration, name, expected[name], actual[name]
                        )
                    )
            for name in ("ce", "mse", "ce_tokens", "mse_tokens"):
                left_metric = expected.get("metrics", {}).get(name)
                right_metric = actual.get("metrics", {}).get(name)
                if left_metric is None or right_metric is None or not _close(
                    left_metric, right_metric, args.atol, args.rtol
                ):
                    issues.append("rank {} iteration {} metric {} differs".format(rank, iteration, name))
            for name in ("sum", "square_sum", "global_numel", "max_abs"):
                left_value = expected.get("parameter_checksum", {}).get(name)
                right_value = actual.get("parameter_checksum", {}).get(name)
                if left_value is None or right_value is None or not _close(
                    left_value, right_value, args.atol, args.rtol
                ):
                    issues.append(
                        "rank {} iteration {} parameter checksum {} differs".format(
                            rank, iteration, name
                        )
                    )
    report = {
        "schema_version": 1,
        "status": "valid" if not issues else "invalid",
        "continuous": str(Path(args.continuous).expanduser().resolve()),
        "resumed": str(Path(args.resumed).expanduser().resolve()),
        "ranks": len(continuous),
        "iterations_per_rank": {
            str(rank): len(records) for rank, records in continuous.items()
        },
        "atol": args.atol,
        "rtol": args.rtol,
        "issues": issues[:100],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
