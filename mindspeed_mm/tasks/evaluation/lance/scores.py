"""Normalize official scorer artifacts into the Lance evaluation schema."""

import csv
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .protocol import PINNED_LANCE_REVISION, get_benchmark


GENEVAL_EXPECTED_IMAGES = 2212
GENEVAL_EXPECTED_PROMPTS = 553
DPGBENCH_EXPECTED_GRIDS = 1065
GEDIT_EXPECTED_SAMPLES = 606
GEDIT_GROUPS = (
    "background_change",
    "color_alter",
    "material_alter",
    "motion_change",
    "ps_human",
    "style_change",
    "subject-add",
    "subject-remove",
    "subject-replace",
    "text_change",
    "tone_transfer",
)

PINNED_SCORER_REVISIONS = {
    "geneval": "af4902f24d3ca90ebbb446dd9891a59e0f82725f",
    "dpgbench": "3c228f1dc6c4d3cad0a47493816151a419f14db3",
    "gedit": "5d350cdbeefc8108c8cd9d4134bbb0d33ee05a74",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("Cannot aggregate an empty score list")
    return sum(values) / len(values)


def _as_bool(value: object, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise ValueError("{} must be a JSON boolean or 0/1".format(context))


def score_geneval_results(
    results_path: Union[str, Path],
    strict: bool = True,
) -> Dict[str, object]:
    """Reproduce GenEval's official ``summary_scores.py`` aggregation."""
    path = Path(results_path).expanduser().resolve()
    records: List[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError("GenEval line {} must contain an object".format(line_number))
            missing = {"metadata", "tag", "correct"} - set(value)
            if missing:
                raise ValueError("GenEval line {} is missing {}".format(line_number, sorted(missing)))
            records.append(value)

    task_values: Dict[str, List[float]] = {}
    prompt_values: Dict[str, List[bool]] = {}
    normalized_correct = []
    for index, record in enumerate(records):
        is_correct = _as_bool(record["correct"], "GenEval record {} correct".format(index))
        normalized_correct.append(is_correct)
        task_values.setdefault(str(record["tag"]), []).append(float(is_correct))
        metadata = record["metadata"]
        prompt_key = metadata if isinstance(metadata, str) else json.dumps(metadata, sort_keys=True)
        prompt_values.setdefault(prompt_key, []).append(is_correct)
    if strict and len(records) != GENEVAL_EXPECTED_IMAGES:
        raise ValueError("Expected {} GenEval images, found {}".format(GENEVAL_EXPECTED_IMAGES, len(records)))
    if strict and len(prompt_values) != GENEVAL_EXPECTED_PROMPTS:
        raise ValueError("Expected {} GenEval prompts, found {}".format(GENEVAL_EXPECTED_PROMPTS, len(prompt_values)))
    if strict and len(task_values) != 6:
        raise ValueError("Expected 6 GenEval task groups, found {}".format(len(task_values)))

    task_scores = {name: _mean(values) for name, values in task_values.items()}
    overall = _mean(task_scores.values())
    return {
        "scores": {"geneval": overall},
        "provenance": {
            "geneval": {
                "scorer": "official-geneval",
                "images": len(records),
                "prompts": len(prompt_values),
                "tasks": len(task_scores),
                "aggregation": "macro-task-accuracy",
                "results_sha256": _sha256(path),
            }
        },
        "details": {
            "geneval": {
                "correct_images": sum(normalized_correct),
                "image_accuracy": _mean(normalized_correct),
                "prompt_accuracy": _mean(any(values) for values in prompt_values.values()),
                "per_task": task_scores,
            }
        },
    }


_DPG_SCORE = re.compile(r"^DPG-Bench score:\s*([-+]?\d+(?:\.\d+)?)\s*$")


def score_dpgbench_results(
    results_path: Union[str, Path],
    strict: bool = True,
) -> Dict[str, object]:
    """Parse the result file emitted by ELLA's official mPLUG evaluator."""
    path = Path(results_path).expanduser().resolve()
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary_index = next((index for index, line in enumerate(lines) if line.startswith("Model:")), None)
    sample_lines = lines[:summary_index] if summary_index is not None else []
    matches = [_DPG_SCORE.match(line) for line in lines]
    scores = [float(match.group(1)) for match in matches if match]
    if len(scores) != 1:
        raise ValueError("DPG-Bench result must contain exactly one final score")
    if strict and len(sample_lines) != DPGBENCH_EXPECTED_GRIDS:
        raise ValueError(
            "Expected {} DPG-Bench grid results, found {}".format(
                DPGBENCH_EXPECTED_GRIDS, len(sample_lines)
            )
        )
    return {
        "scores": {"dpgbench": scores[0]},
        "provenance": {
            "dpgbench": {
                "scorer": "official-mplug",
                "grids": len(sample_lines),
                "images_per_prompt": 4,
                "results_sha256": _sha256(path),
            }
        },
        "details": {"dpgbench": {"result_records": len(sample_lines)}},
    }


def _gedit_csv_path(root: Path, model_name: str, group: str, file_ext: str) -> Path:
    return root / "{}_{}_{}_vie_score.csv".format(model_name, group, file_ext)


def score_gedit_results(
    score_dir: Union[str, Path],
    model_name: str,
    judge: str = "gpt-4.1",
    language: str = "en",
    file_ext: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, object]:
    """Reproduce GEdit-Bench ``calculate_statistics.py`` for one language."""
    normalized_judge = judge.lower()
    if normalized_judge not in {"gpt-4.1", "qwen2.5-vl-72b"}:
        raise ValueError("judge must be gpt-4.1 or qwen2.5-vl-72b")
    root = Path(score_dir).expanduser().resolve()
    file_ext = file_ext or language
    per_group = {}
    source_files = []
    sample_count = 0
    for group in GEDIT_GROUPS:
        path = _gedit_csv_path(root, model_name, group, file_ext)
        if not path.is_file():
            raise FileNotFoundError("Missing GEdit score CSV: {}".format(path))
        source_files.append(path)
        semantics = []
        quality = []
        overall = []
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("instruction_language") != language:
                    continue
                semantic_score = float(row["sementics_score"])
                quality_score = float(row["quality_score"])
                semantics.append(semantic_score)
                quality.append(quality_score)
                overall.append(math.sqrt(semantic_score * quality_score))
        if not overall:
            raise ValueError("No {} records in {}".format(language, path))
        sample_count += len(overall)
        per_group[group] = {
            "samples": len(overall),
            "semantics": _mean(semantics),
            "quality": _mean(quality),
            "overall": _mean(overall),
        }
    if strict and sample_count != GEDIT_EXPECTED_SAMPLES:
        raise ValueError("Expected {} GEdit samples, found {}".format(GEDIT_EXPECTED_SAMPLES, sample_count))
    avg_semantics = _mean(value["semantics"] for value in per_group.values())
    avg_quality = _mean(value["quality"] for value in per_group.values())
    avg_overall = _mean(value["overall"] for value in per_group.values())
    combined_digest = hashlib.sha256()
    for path in source_files:
        combined_digest.update(path.name.encode("utf-8"))
        combined_digest.update(bytes.fromhex(_sha256(path)))
    return {
        "scores": {"gedit": avg_overall},
        "provenance": {
            "gedit": {
                "judge": normalized_judge,
                "samples": sample_count,
                "groups": len(per_group),
                "language": language,
                "aggregation": "macro-group-mean-geometric-score",
                "results_sha256": combined_digest.hexdigest(),
            }
        },
        "details": {
            "gedit": {
                "avg_semantics": avg_semantics,
                "avg_quality": avg_quality,
                "avg_overall": avg_overall,
                "per_group": per_group,
            }
        },
    }


def merge_metric_payloads(
    payloads: Sequence[Mapping[str, object]],
    sources: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Merge scorer payloads while rejecting ambiguous duplicate benchmarks."""
    merged: Dict[str, object] = {"schema_version": 1, "scores": {}, "provenance": {}, "details": {}}
    source_names = list(sources) if sources is not None else [str(index) for index in range(len(payloads))]
    if len(source_names) != len(payloads):
        raise ValueError("sources/payloads length mismatch")
    merged["metric_sources"] = source_names
    for source, payload in zip(source_names, payloads):
        for section in ("scores", "provenance"):
            values = payload.get(section, {})
            if not isinstance(values, Mapping):
                raise ValueError("{} in {} must be an object".format(section, source))
            target = merged[section]
            for benchmark, value in values.items():
                if benchmark in target and target[benchmark] != value:
                    raise ValueError("Conflicting {} for {} from {}".format(section, benchmark, source))
                target[benchmark] = value
        if "details" in payload:
            details = payload["details"]
            if not isinstance(details, Mapping):
                raise ValueError("details in {} must be an object".format(source))
            for benchmark, value in details.items():
                if benchmark in merged["details"] and merged["details"][benchmark] != value:
                    raise ValueError("Conflicting details for {} from {}".format(benchmark, source))
                merged["details"][benchmark] = value
    return merged


def merge_metric_files(paths: Sequence[Union[str, Path]]) -> Dict[str, object]:
    resolved = [Path(path).expanduser().resolve() for path in paths]
    payloads = []
    for path in resolved:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, Mapping):
            raise ValueError("Metric file must contain an object: {}".format(path))
        payloads.append(value)
    return merge_metric_payloads(payloads, [str(path) for path in resolved])


def attach_scorer_checkout(
    payload: Mapping[str, object],
    scorer_root: Union[str, Path],
    benchmark: str,
) -> Dict[str, object]:
    """Require an exact, clean checkout for an external official scorer."""
    if benchmark not in PINNED_SCORER_REVISIONS:
        raise ValueError("No pinned Git scorer is defined for {}".format(benchmark))
    root = Path(scorer_root).expanduser().resolve()
    revision_result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = revision_result.stdout.strip() if revision_result.returncode == 0 else None
    expected = PINNED_SCORER_REVISIONS[benchmark]
    if revision != expected:
        raise ValueError(
            "{} scorer revision {} does not match pinned {}".format(benchmark, revision, expected)
        )
    dirty_result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if dirty_result.returncode != 0 or dirty_result.stdout.strip():
        raise ValueError("{} scorer checkout must be clean".format(benchmark))

    result = copy.deepcopy(dict(payload))
    provenance = result.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Scorer provenance must be an object")
    benchmark_provenance = provenance.setdefault(benchmark, {})
    if not isinstance(benchmark_provenance, dict):
        raise ValueError("Benchmark provenance must be an object")
    benchmark_provenance.update(
        {"scorer_revision": revision, "scorer_root": str(root), "scorer_checkout_clean": True}
    )
    return result


def attach_run_manifest(
    payload: Mapping[str, object],
    run_manifest_path: Union[str, Path],
    benchmark: str,
) -> Dict[str, object]:
    """Validate and attach generation provenance to a normalized scorer payload."""
    path = Path(run_manifest_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, Mapping):
        raise ValueError("Lance run manifest must contain an object")
    spec = get_benchmark(benchmark)
    manifest_benchmark = manifest.get("benchmark", {})
    if not isinstance(manifest_benchmark, Mapping) or manifest_benchmark.get("name") != benchmark:
        raise ValueError("Run manifest is not for benchmark {}".format(benchmark))
    if manifest.get("status") != "completed":
        raise ValueError("Run manifest status must be completed")
    source = manifest.get("lance_source", {})
    if (
        not isinstance(source, Mapping)
        or source.get("status") != "valid"
        or source.get("revision") != PINNED_LANCE_REVISION
    ):
        raise ValueError("Run manifest does not contain a valid pinned Lance source")
    dataset = manifest.get("dataset", {})
    if not isinstance(dataset, Mapping) or dataset.get("status") != "valid":
        raise ValueError("Run manifest does not contain a valid dataset")
    if spec.dataset_sha256 and dataset.get("sha256") != spec.dataset_sha256:
        raise ValueError("Run manifest dataset does not match the pinned {} protocol".format(benchmark))
    model = manifest.get("model", {})
    checkpoint_sha256 = model.get("checkpoint_sha256") if isinstance(model, Mapping) else None
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise ValueError("Run manifest is missing the checkpoint SHA-256")
    output_audit = manifest.get("output_audit", {})
    if not isinstance(output_audit, Mapping) or output_audit.get("status") != "valid":
        raise ValueError("Run manifest does not contain a successful output audit")
    manifest_expected = manifest_benchmark.get("expected_outputs")
    allowed_expected = {spec.expected_outputs}
    if benchmark == "mvbench":
        allowed_expected.add(4000)
        task_set = payload.get("provenance", {}).get("mvbench", {}).get("task_set")
        task_set_expected = {"paper-lance-19": 3800, "official-mvbench-20": 4000}.get(task_set)
        if task_set_expected != manifest_expected:
            raise ValueError("MVBench scorer and sampling task sets do not match")
    if manifest_expected not in allowed_expected:
        raise ValueError("Run manifest benchmark output count is not an allowed protocol")
    if output_audit.get("artifacts") != manifest_expected:
        raise ValueError("Run manifest output count does not match the paper protocol")

    result = copy.deepcopy(dict(payload))
    provenance = result.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Scorer provenance must be an object")
    benchmark_provenance = provenance.setdefault(benchmark, {})
    if not isinstance(benchmark_provenance, dict):
        raise ValueError("Benchmark provenance must be an object")
    benchmark_provenance.update(
        {
            "run_manifest_sha256": _sha256(path),
            "checkpoint_sha256": checkpoint_sha256,
            "lance_revision": source.get("revision"),
            "dataset_sha256": dataset.get("sha256"),
        }
    )
    if benchmark == "vbench":
        benchmark_provenance["released_recaption"] = dataset.get("sha256") == spec.dataset_sha256
    return result
