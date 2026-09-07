"""Score normalization and paper-parity reporting for Lance."""

from typing import Dict, Mapping, Optional, Sequence

from .protocol import PINNED_LANCE_REVISION, get_benchmark
from .scores import PINNED_SCORER_REVISIONS


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _score_table(payload: Mapping[str, object]) -> Mapping[str, object]:
    value = payload.get("scores", payload)
    if not isinstance(value, Mapping):
        raise ValueError("metrics JSON must be an object or contain a 'scores' object")
    return value


def _find_score(scores: Mapping[str, object], benchmark: str, metric: str) -> Optional[float]:
    aliases = [benchmark, "{}.{}".format(benchmark, metric), metric]
    for alias in aliases:
        if alias in scores and isinstance(scores[alias], (int, float)):
            return float(scores[alias])
    nested = scores.get(benchmark)
    if isinstance(nested, Mapping) and isinstance(nested.get(metric), (int, float)):
        return float(nested[metric])
    return None


def _comparability_issue(benchmark: str, payload: Mapping[str, object]) -> Optional[str]:
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    benchmark_provenance = provenance.get(benchmark, {})
    if not isinstance(benchmark_provenance, Mapping):
        benchmark_provenance = {}
    common_hashes = ("run_manifest_sha256", "checkpoint_sha256", "dataset_sha256", "results_sha256")
    if any(not _is_sha256(benchmark_provenance.get(key)) for key in common_hashes):
        return "Paper parity requires SHA-256 provenance for the run, checkpoint, dataset, and scorer result."
    if benchmark_provenance.get("lance_revision") != PINNED_LANCE_REVISION:
        return "Paper parity requires the pinned clean Lance source revision."
    if benchmark == "geneval":
        expected = {
            "scorer": "official-geneval",
            "images": 2212,
            "prompts": 553,
            "tasks": 6,
            "aggregation": "macro-task-accuracy",
        }
        if any(benchmark_provenance.get(key) != value for key, value in expected.items()):
            return "GenEval paper parity requires the official 553-prompt/2212-image macro-task protocol."
        if benchmark_provenance.get("scorer_revision") != PINNED_SCORER_REVISIONS["geneval"]:
            return "GenEval paper parity requires the pinned official scorer revision."
    if benchmark == "dpgbench":
        expected = {"scorer": "official-mplug", "grids": 1065, "images_per_prompt": 4}
        if any(benchmark_provenance.get(key) != value for key, value in expected.items()):
            return "DPG-Bench paper parity requires the official 1065-grid mPLUG protocol."
        if benchmark_provenance.get("scorer_revision") != PINNED_SCORER_REVISIONS["dpgbench"]:
            return "DPG-Bench paper parity requires the pinned official scorer revision."
    if benchmark == "gedit":
        expected = {
            "samples": 606,
            "groups": 11,
            "language": "en",
            "aggregation": "macro-group-mean-geometric-score",
        }
        if (
            any(benchmark_provenance.get(key) != value for key, value in expected.items())
            or str(benchmark_provenance.get("judge", "")).lower() != "gpt-4.1"
        ):
            return "GEdit paper parity requires the 606-sample English GPT-4.1 G_O protocol."
        if benchmark_provenance.get("scorer_revision") != PINNED_SCORER_REVISIONS["gedit"]:
            return "GEdit paper parity requires the pinned official evaluator revision."
    if benchmark == "vbench":
        if benchmark_provenance.get("scorer") != "official-vbench":
            return "VBench paper parity requires the official VBench scorer provenance."
        if not benchmark_provenance.get("released_recaption", False):
            return "VBench paper parity requires the released Vbench_recaption.jsonl prompts."
        expected = {"scorer_version": "0.1.2", "prompts": 946, "videos": 6230, "dimensions": 16}
        if any(benchmark_provenance.get(key) != value for key, value in expected.items()):
            return "VBench paper parity requires vbench==0.1.2 and all 16 dimensions over 6230 videos."
    if benchmark == "mvbench":
        expected = {
            "tasks": 19,
            "samples": 3800,
            "task_set": "paper-lance-19",
            "aggregation": "macro-task-accuracy",
            "answer_extraction": "official-leading-option",
        }
        if any(benchmark_provenance.get(key) != value for key, value in expected.items()):
            return (
                "MVBench paper parity requires Lance Table 8's 19-task/3800-sample subset "
                "(without Fine-grained Pose) and macro task accuracy."
            )
        if not _is_sha256(benchmark_provenance.get("metadata_sha256")):
            return "MVBench paper parity requires the frozen converted metadata SHA-256."
    return None


def build_alignment_report(
    payload: Mapping[str, object],
    benchmarks: Optional[Sequence[str]] = None,
    tolerances: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    names = list(benchmarks) if benchmarks is not None else [
        "geneval", "dpgbench", "gedit", "vbench", "mvbench"
    ]
    scores = _score_table(payload)
    results: Dict[str, object] = {}
    complete = True
    comparable = True
    all_within_tolerance = True
    for name in names:
        spec = get_benchmark(name)
        score = _find_score(scores, name, spec.metric)
        tolerance = float(tolerances[name]) if tolerances and name in tolerances else spec.parity_tolerance
        issue = _comparability_issue(name, payload)
        if score is None:
            status = "missing"
            delta = None
            complete = False
            all_within_tolerance = False
        else:
            delta = score - spec.paper_score
            status = "within-tolerance" if abs(delta) <= tolerance else "outside-tolerance"
            if status != "within-tolerance":
                all_within_tolerance = False
        if issue:
            comparable = False
        results[name] = {
            "metric": spec.metric,
            "score": score,
            "paper_score": spec.paper_score,
            "delta": delta,
            "tolerance": tolerance,
            "status": status,
            "paper_comparable": issue is None,
            "comparability_issue": issue,
            "scorer": spec.scorer,
        }
    return {
        "schema_version": 1,
        "complete": complete,
        "paper_comparable": comparable,
        "all_within_tolerance": complete and comparable and all_within_tolerance,
        "results": results,
    }
