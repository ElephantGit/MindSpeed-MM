import os
import json

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

from mindspeed_mm.tasks.evaluation.lance.mvbench import (
    MVBENCH_PAPER_SCORES,
    MVBENCH_PAPER_TASKS,
    MVBENCH_TASKS,
    _question_and_answer,
    extract_option,
    prepare_mvbench,
    score_mvbench,
)


def test_mvbench_contains_twenty_unique_tasks():
    assert len(MVBENCH_TASKS) == 20
    assert len({task.name for task in MVBENCH_TASKS}) == 20
    assert len({task.annotation for task in MVBENCH_TASKS}) == 20
    assert len(MVBENCH_PAPER_TASKS) == 19
    assert "Fine-grained Pose" not in {task.name for task in MVBENCH_PAPER_TASKS}
    assert sum(MVBENCH_PAPER_SCORES.values()) / len(MVBENCH_PAPER_SCORES) == pytest.approx(62.0, abs=0.01)


def test_mvbench_question_matches_official_option_format():
    question, answer, letter = _question_and_answer(
        {"question": "What happens?", "candidates": ["First", "Second"], "answer": "Second"}
    )
    assert question == "Question: What happens?\nOptions:\n(A) First\n(B) Second\nOnly give the best option."
    assert answer == "(B) Second"
    assert letter == "B"


def test_option_extraction_only_uses_leading_answer():
    assert extract_option("(C) because A is not visible") == "C"
    assert extract_option("Best option:(d)") == "D"
    assert extract_option("C") == "C"
    assert extract_option("The answer is C") is None


def test_mvbench_scorer_reports_overall_and_per_task():
    metadata = [
        {"id": "0", "task": "Action Sequence", "answer_letter": "A"},
        {"id": "1", "task": "Action Sequence", "answer_letter": "B"},
        {"id": "2", "task": "Moving Count", "answer_letter": "C"},
    ]
    results = [{"answer": "(A)"}, {"answer": "(D)"}, {"answer": "Best option: C"}]
    report = score_mvbench(metadata, results, strict=False)
    assert report["scores"]["mvbench"] == 75.0
    assert report["details"]["micro_accuracy"] == 200.0 / 3.0
    assert report["details"]["per_task"]["Action Sequence"]["accuracy"] == 50.0
    assert report["details"]["paper_alignment"]["Action Sequence"]["paper_score"] == 73.9
    assert report["provenance"]["mvbench"]["tasks"] == 2


def test_mvbench_paper_protocol_requires_19_tasks_and_3800_samples():
    metadata = []
    results = []
    for task in MVBENCH_PAPER_TASKS:
        for index in range(200):
            item_id = "{}-{}".format(task.abbreviation, index)
            metadata.append(
                {
                    "id": item_id,
                    "task": task.name,
                    "answer_letter": "A",
                    "question": "Question {}".format(item_id),
                }
            )
            results.append({"question": "Question {}".format(item_id), "answer": "A"})
    report = score_mvbench(metadata, results)
    assert report["scores"]["mvbench"] == 100.0
    assert report["provenance"]["mvbench"] == {
        "tasks": 19,
        "samples": 3800,
        "task_set": "paper-lance-19",
        "aggregation": "macro-task-accuracy",
        "answer_extraction": "official-leading-option",
    }


def test_prepare_mvbench_covers_all_tasks(monkeypatch, tmp_path):
    annotation_root = tmp_path / "json"
    media_root = tmp_path / "media-root"
    output_root = tmp_path / "prepared"
    annotation_root.mkdir()
    for task in MVBENCH_PAPER_TASKS:
        annotation = {
            "video": task.annotation + ("/frames" if task.media_type == "frame" else ".mp4"),
            "question": "Pick one",
            "candidates": ["yes", "no"],
            "answer": "yes",
        }
        if task.bounded:
            annotation.update({"start": 0, "end": 1})
        (annotation_root / task.annotation).write_text(json.dumps([annotation]), encoding="utf-8")
        source = media_root / task.media_prefix / annotation["video"]
        if task.media_type == "frame":
            source.mkdir(parents=True, exist_ok=True)
        else:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.touch()

    def fake_materialize(source, target, media_type, bound, ffmpeg):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        return target

    monkeypatch.setattr("mindspeed_mm.tasks.evaluation.lance.mvbench.shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr("mindspeed_mm.tasks.evaluation.lance.mvbench._materialize_media", fake_materialize)
    result = prepare_mvbench(annotation_root, media_root, output_root, expected_samples=19)
    assert result["samples"] == 19
    assert result["tasks"] == 19
    converted = json.loads((output_root / "mvbench_lance.json").read_text(encoding="utf-8"))
    assert len(converted) == 19
