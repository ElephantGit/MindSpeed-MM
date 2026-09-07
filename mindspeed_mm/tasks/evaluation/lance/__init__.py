"""Reproducible evaluation protocol for Lance."""

from .protocol import BENCHMARKS, BenchmarkSpec, audit_outputs, build_sample_arguments, validate_dataset
from .report import build_alignment_report
from .mvbench import prepare_mvbench, score_mvbench
from .vbench import score_vbench
from .scores import (
    attach_run_manifest,
    attach_scorer_checkout,
    merge_metric_files,
    merge_metric_payloads,
    score_dpgbench_results,
    score_gedit_results,
    score_geneval_results,
)

__all__ = [
    "BENCHMARKS",
    "BenchmarkSpec",
    "build_alignment_report",
    "audit_outputs",
    "build_sample_arguments",
    "prepare_mvbench",
    "score_mvbench",
    "score_vbench",
    "attach_run_manifest",
    "attach_scorer_checkout",
    "score_geneval_results",
    "score_dpgbench_results",
    "score_gedit_results",
    "merge_metric_files",
    "merge_metric_payloads",
    "validate_dataset",
]
