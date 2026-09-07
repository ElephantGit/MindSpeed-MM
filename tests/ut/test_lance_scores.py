import csv
import json
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

from mindspeed_mm.tasks.evaluation.lance.protocol import PINNED_LANCE_REVISION
from mindspeed_mm.tasks.evaluation.lance.scores import (
    GEDIT_GROUPS,
    PINNED_SCORER_REVISIONS,
    attach_run_manifest,
    attach_scorer_checkout,
    merge_metric_payloads,
    score_dpgbench_results,
    score_gedit_results,
    score_geneval_results,
)


def test_geneval_normalizer_macro_averages_official_task_groups(tmp_path):
    results = tmp_path / "results.jsonl"
    records = [
        {"metadata": {"id": 0}, "tag": "single_object", "correct": True},
        {"metadata": {"id": 1}, "tag": "two_object", "correct": False},
    ]
    results.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    payload = score_geneval_results(results, strict=False)
    assert payload["scores"]["geneval"] == 0.5
    assert payload["provenance"]["geneval"]["images"] == 2
    assert len(payload["provenance"]["geneval"]["results_sha256"]) == 64


def test_geneval_normalizer_rejects_string_boolean(tmp_path):
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps({"metadata": {"id": 0}, "tag": "single_object", "correct": "false"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="JSON boolean"):
        score_geneval_results(results, strict=False)


def test_dpgbench_normalizer_reads_official_summary(tmp_path):
    results = tmp_path / "mplug.txt"
    results.write_text("sample-0: 1\nModel: lance\nDPG-Bench score: 84.67\n", encoding="utf-8")
    payload = score_dpgbench_results(results, strict=False)
    assert payload["scores"]["dpgbench"] == 84.67
    assert payload["provenance"]["dpgbench"]["grids"] == 1


def test_gedit_normalizer_reproduces_group_macro_geometric_mean(tmp_path):
    for group in GEDIT_GROUPS:
        path = tmp_path / "lance_{}_en_vie_score.csv".format(group)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["instruction_language", "sementics_score", "quality_score"],
            )
            writer.writeheader()
            writer.writerow({"instruction_language": "en", "sementics_score": 9, "quality_score": 4})
    payload = score_gedit_results(tmp_path, "lance", strict=False)
    assert payload["scores"]["gedit"] == 6.0
    assert payload["provenance"]["gedit"]["groups"] == 11


def test_merge_rejects_conflicting_benchmark_scores():
    with pytest.raises(ValueError, match="Conflicting scores"):
        merge_metric_payloads([{"scores": {"geneval": 0.8}}, {"scores": {"geneval": 0.9}}])


def test_attach_scorer_checkout_requires_pinned_clean_revision(monkeypatch, tmp_path):
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout=PINNED_SCORER_REVISIONS["geneval"] + "\n"),
            SimpleNamespace(returncode=0, stdout=""),
        ]
    )
    monkeypatch.setattr(
        "mindspeed_mm.tasks.evaluation.lance.scores.subprocess.run",
        lambda *args, **kwargs: next(responses),
    )
    payload = attach_scorer_checkout(
        {"scores": {"geneval": 0.9}, "provenance": {"geneval": {}}},
        tmp_path,
        "geneval",
    )
    assert payload["provenance"]["geneval"]["scorer_checkout_clean"] is True


def test_attach_run_manifest_binds_checkpoint_dataset_and_sampling(tmp_path):
    manifest = {
        "status": "completed",
        "benchmark": {"name": "mvbench", "expected_outputs": 3800},
        "lance_source": {"status": "valid", "revision": PINNED_LANCE_REVISION},
        "dataset": {"status": "valid", "sha256": "c" * 64},
        "model": {"checkpoint_sha256": "b" * 64},
        "output_audit": {"status": "valid", "artifacts": 3800},
    }
    manifest_path = tmp_path / "lance_eval_run.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    payload = {
        "scores": {"mvbench": 62.0},
        "provenance": {"mvbench": {"task_set": "paper-lance-19"}},
    }
    attached = attach_run_manifest(payload, manifest_path, "mvbench")
    provenance = attached["provenance"]["mvbench"]
    assert provenance["checkpoint_sha256"] == "b" * 64
    assert provenance["dataset_sha256"] == "c" * 64
    assert len(provenance["run_manifest_sha256"]) == 64
