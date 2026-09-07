#!/usr/bin/env python3
"""Validate, sample, and report the paper-aligned Lance evaluation suite."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("NON_MEGATRON", "true")

from mindspeed_mm.models.omni.lance import resolve_lance_source, run_lance_entrypoint
from mindspeed_mm.tasks.evaluation.lance import (
    BENCHMARKS,
    attach_run_manifest,
    attach_scorer_checkout,
    audit_outputs,
    build_alignment_report,
    build_sample_arguments,
    merge_metric_files,
    prepare_mvbench,
    score_dpgbench_results,
    score_gedit_results,
    score_geneval_results,
    score_mvbench,
    score_vbench,
    validate_dataset,
)
from mindspeed_mm.tasks.evaluation.lance.protocol import (
    get_benchmark,
    protocol_manifest,
    validate_model_path,
    validate_runtime_assets,
    validate_lance_source,
)


def _names(value):
    return list(BENCHMARKS) if value == "all" else [value]


def _spec(name, mvbench_task_set="paper"):
    spec = get_benchmark(name)
    if name == "mvbench" and mvbench_task_set == "official":
        return replace(
            spec,
            expected_samples=4000,
            scorer="Official MVBench 20-task multiple-choice evaluator",
            paper_comparability="Official 20-task/4000-sample result; not comparable with Lance Table 8.",
        )
    return spec


def _model_variant(spec):
    return "video" if spec.task in ("t2v", "i2v", "video_edit", "x2t_video") else "image"


def _dump(value, output=None):
    rendered = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="print the frozen benchmark manifest")
    list_parser.add_argument("--benchmark", choices=["all"] + list(BENCHMARKS), default="all")
    list_parser.add_argument("--output")

    validate_parser = subparsers.add_parser("validate", help="validate source data and checkpoint layout")
    validate_parser.add_argument("--benchmark", choices=["all"] + list(BENCHMARKS), default="all")
    validate_parser.add_argument("--lance-source-root")
    validate_parser.add_argument("--dataset-path", help="override dataset path; only valid for one benchmark")
    validate_parser.add_argument("--model-path")
    validate_parser.add_argument(
        "--model-variant",
        choices=["image", "video"],
        help="checkpoint contract to validate; required with --model-path when --benchmark=all",
    )
    validate_parser.add_argument("--mvbench-task-set", choices=["paper", "official"], default="paper")
    validate_parser.add_argument("--output")

    sample_parser = subparsers.add_parser("sample", help="run an upstream benchmark sampler through the NPU adapter")
    sample_parser.add_argument("--benchmark", choices=list(BENCHMARKS), required=True)
    sample_parser.add_argument("--lance-source-root")
    sample_parser.add_argument("--dataset-path")
    sample_parser.add_argument("--model-path", required=True)
    sample_parser.add_argument("--output-path", required=True)
    sample_parser.add_argument("--world-size", type=int, default=8)
    sample_parser.add_argument("--mvbench-task-set", choices=["paper", "official"], default="paper")
    sample_parser.add_argument("--dry-run", action="store_true")

    report_parser = subparsers.add_parser("report", help="compare scorer outputs with paper targets")
    report_parser.add_argument("--metrics", required=True)
    report_parser.add_argument("--benchmark", choices=["all"] + list(BENCHMARKS), default="all")
    report_parser.add_argument("--output")

    merge_parser = subparsers.add_parser("merge", help="merge normalized scorer metric files")
    merge_parser.add_argument("--metrics", nargs="+", required=True)
    merge_parser.add_argument("--output", required=True)

    audit_parser = subparsers.add_parser("audit", help="validate scorer-facing output artifact counts")
    audit_parser.add_argument("--benchmark", choices=list(BENCHMARKS), required=True)
    audit_parser.add_argument("--output-path", required=True)
    audit_parser.add_argument("--mvbench-task-set", choices=["paper", "official"], default="paper")
    audit_parser.add_argument("--output")

    mvbench_prepare = subparsers.add_parser("mvbench-prepare", help="convert all official MVBench tasks for Lance")
    mvbench_prepare.add_argument("--annotation-root", required=True)
    mvbench_prepare.add_argument("--media-root", required=True)
    mvbench_prepare.add_argument("--output-dir", required=True)
    mvbench_prepare.add_argument("--task-set", choices=["paper", "official"], default="paper")
    mvbench_prepare.add_argument("--expected-samples", type=int)
    mvbench_prepare.add_argument("--no-materialize", action="store_true")

    mvbench_score = subparsers.add_parser("mvbench-score", help="score Lance result.json with official option matching")
    mvbench_score.add_argument("--metadata", required=True)
    mvbench_score.add_argument("--results", required=True)
    mvbench_score.add_argument("--run-manifest", required=True)
    mvbench_score.add_argument("--task-set", choices=["paper", "official"], default="paper")
    mvbench_score.add_argument("--allow-partial", action="store_true")
    mvbench_score.add_argument("--output")

    vbench_score = subparsers.add_parser("vbench-score", help="run official VBench with NPU scorer patches")
    vbench_score.add_argument("--videos-path", required=True)
    vbench_score.add_argument("--full-info-path", required=True)
    vbench_score.add_argument("--output-dir", required=True)
    vbench_score.add_argument("--result-name", default="lance_vbench")
    vbench_score.add_argument("--run-manifest", required=True)
    vbench_score.add_argument("--load-checkpoints-locally", action="store_true")
    vbench_score.add_argument("--output")

    geneval_score = subparsers.add_parser("geneval-score", help="aggregate official GenEval results.jsonl")
    geneval_score.add_argument("--results", required=True)
    geneval_score.add_argument("--scorer-root", required=True)
    geneval_score.add_argument("--run-manifest", required=True)
    geneval_score.add_argument("--allow-partial", action="store_true")
    geneval_score.add_argument("--output")

    dpgbench_score = subparsers.add_parser("dpgbench-score", help="normalize official mPLUG result text")
    dpgbench_score.add_argument("--results", required=True)
    dpgbench_score.add_argument("--scorer-root", required=True)
    dpgbench_score.add_argument("--run-manifest", required=True)
    dpgbench_score.add_argument("--allow-partial", action="store_true")
    dpgbench_score.add_argument("--output")

    gedit_score = subparsers.add_parser("gedit-score", help="aggregate official GEdit judge CSV files")
    gedit_score.add_argument("--score-dir", required=True)
    gedit_score.add_argument("--model-name", required=True)
    gedit_score.add_argument("--scorer-root", required=True)
    gedit_score.add_argument("--run-manifest", required=True)
    gedit_score.add_argument("--judge", choices=["gpt-4.1", "qwen2.5-vl-72b"], default="gpt-4.1")
    gedit_score.add_argument("--language", choices=["en", "cn"], default="en")
    gedit_score.add_argument("--file-ext")
    gedit_score.add_argument("--allow-partial", action="store_true")
    gedit_score.add_argument("--output")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "merge":
        result = merge_metric_files(args.metrics)
        _dump(result, args.output)
        return 0
    if args.command == "geneval-score":
        result = score_geneval_results(args.results, strict=not args.allow_partial)
        result = attach_scorer_checkout(result, args.scorer_root, "geneval")
        result = attach_run_manifest(result, args.run_manifest, "geneval")
        _dump(result, args.output)
        return 0
    if args.command == "dpgbench-score":
        result = score_dpgbench_results(args.results, strict=not args.allow_partial)
        result = attach_scorer_checkout(result, args.scorer_root, "dpgbench")
        result = attach_run_manifest(result, args.run_manifest, "dpgbench")
        _dump(result, args.output)
        return 0
    if args.command == "gedit-score":
        result = score_gedit_results(
            args.score_dir,
            args.model_name,
            judge=args.judge,
            language=args.language,
            file_ext=args.file_ext,
            strict=not args.allow_partial,
        )
        result = attach_scorer_checkout(result, args.scorer_root, "gedit")
        result = attach_run_manifest(result, args.run_manifest, "gedit")
        _dump(result, args.output)
        return 0
    if args.command == "mvbench-prepare":
        result = prepare_mvbench(
            args.annotation_root,
            args.media_root,
            args.output_dir,
            expected_samples=args.expected_samples,
            materialize=not args.no_materialize,
            task_set=args.task_set,
        )
        _dump(result)
        return 0
    if args.command == "mvbench-score":
        result = score_mvbench(
            args.metadata,
            args.results,
            task_set=args.task_set,
            strict=not args.allow_partial,
        )
        result = attach_run_manifest(result, args.run_manifest, "mvbench")
        _dump(result, args.output)
        return 0
    if args.command == "vbench-score":
        result = score_vbench(
            args.videos_path,
            args.full_info_path,
            args.output_dir,
            load_checkpoints_locally=args.load_checkpoints_locally,
            result_name=args.result_name,
        )
        if result is not None:
            result = attach_run_manifest(result, args.run_manifest, "vbench")
            _dump(result, args.output)
        return 0

    if args.command == "audit":
        result = audit_outputs(_spec(args.benchmark, args.mvbench_task_set), args.output_path)
        _dump(result, args.output)
        return 0 if result["status"] == "valid" else 2

    names = _names(args.benchmark)
    if args.command == "list":
        _dump(protocol_manifest(names), args.output)
        return 0

    if args.command == "validate":
        if args.dataset_path and len(names) != 1:
            raise SystemExit("--dataset-path requires one --benchmark")
        source = resolve_lance_source(args.lance_source_root)
        if args.model_path and len(names) > 1 and not args.model_variant:
            raise SystemExit("--model-variant is required with --model-path when validating multiple benchmarks")
        validation_variant = args.model_variant or (_model_variant(_spec(names[0], args.mvbench_task_set)) if args.model_path else None)
        source_validation = validate_lance_source(source)
        datasets = {
            name: validate_dataset(_spec(name, args.mvbench_task_set), source, args.dataset_path)
            for name in names
        }
        result = {
            "lance_source_root": str(source),
            "lance_source": source_validation,
            "datasets": datasets,
            "model": validate_model_path(args.model_path, variant=validation_variant) if args.model_path else None,
            "runtime_assets": validate_runtime_assets(
                source, require_vae=any(get_benchmark(name).task != "x2t_video" for name in names)
            ) if args.model_path else None,
        }
        model_valid = result["model"] is None or (
            result["model"]["status"] == "valid"
            and result["runtime_assets"]["status"] == "valid"
        )
        dataset_statuses = {item["status"] for item in datasets.values()}
        if source_validation["status"] != "valid" or "invalid" in dataset_statuses or not model_valid:
            result["status"] = "invalid"
        elif "external-adapter-required" in dataset_statuses:
            result["status"] = "partial"
        else:
            result["status"] = "valid"
        _dump(result, args.output)
        return {"valid": 0, "partial": 3, "invalid": 2}[result["status"]]

    if args.command == "sample":
        spec = _spec(args.benchmark, args.mvbench_task_set)
        source = resolve_lance_source(args.lance_source_root)
        run_manifest = None
        manifest_path = None
        if args.dataset_path:
            dataset = Path(args.dataset_path).resolve()
        elif spec.dataset_path:
            dataset = source / spec.dataset_path
        else:
            raise SystemExit("{} requires --dataset-path".format(spec.name))
        actual_world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        if not args.dry_run and actual_world_size != args.world_size:
            raise SystemExit(
                "WORLD_SIZE={} does not match --world-size={}".format(actual_world_size, args.world_size)
            )
        forwarded = build_sample_arguments(
            spec,
            model_path=Path(args.model_path).resolve(),
            dataset_path=dataset,
            output_path=Path(args.output_path).resolve(),
            world_size=args.world_size,
        )
        if not args.dry_run:
            source_validation = validate_lance_source(source)
            dataset_validation = validate_dataset(spec, source, dataset)
            model_variant = _model_variant(spec)
            model_validation = validate_model_path(args.model_path, variant=model_variant)
            asset_validation = validate_runtime_assets(source, require_vae=spec.task != "x2t_video")
            failures = [
                item for item in (source_validation, dataset_validation, model_validation, asset_validation)
                if item["status"] != "valid"
            ]
            if failures:
                _dump({"status": "invalid", "preflight_failures": failures})
                return 2
            if int(os.environ.get("RANK", "0")) == 0:
                output_path = Path(args.output_path).resolve()
                output_path.mkdir(parents=True, exist_ok=True)
                run_manifest = {
                    "schema_version": 1,
                    "status": "running",
                    "benchmark": spec.to_dict(),
                    "task_set": (
                        "paper-lance-19" if args.mvbench_task_set == "paper"
                        else "official-mvbench-20"
                    ) if spec.name == "mvbench" else None,
                    "lance_source_root": str(source),
                    "lance_source": source_validation,
                    "dataset": dataset_validation,
                    "model": validate_model_path(
                        args.model_path,
                        fingerprint=True,
                        variant=model_variant,
                    ),
                    "runtime_assets": asset_validation,
                    "world_size": args.world_size,
                    "forwarded_arguments": forwarded,
                }
                manifest_path = output_path / "lance_eval_run.json"
                _dump(run_manifest, manifest_path)
        result = run_lance_entrypoint(source, spec.entrypoint, forwarded, dry_run=args.dry_run)
        if args.dry_run:
            result["benchmark"] = spec.to_dict()
            _dump(result)
        elif int(os.environ.get("RANK", "0")) == 0:
            output_audit = audit_outputs(spec, args.output_path)
            run_manifest["output_audit"] = output_audit
            run_manifest["status"] = "completed" if output_audit["status"] == "valid" else "invalid"
            _dump(run_manifest, manifest_path)
            if output_audit["status"] != "valid":
                _dump(output_audit)
                return 2
        return 0

    with Path(args.metrics).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    report = build_alignment_report(payload, names)
    _dump(report, args.output)
    return 0 if report["all_within_tolerance"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
