import os
import importlib.util
from pathlib import Path

import pytest

os.environ.setdefault("NON_MEGATRON", "true")


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "mindspeed_mm/tasks/evaluation/gen_impl/vbench_utils/compute_score.py"
)
SPEC = importlib.util.spec_from_file_location("lance_vbench_compute_score", MODULE_PATH)
COMPUTE_SCORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPUTE_SCORE)


def test_vbench_compute_score_returns_all_aggregates():
    # Normalization maxima map all dimensions to exactly one after weighting.
    result = {name: COMPUTE_SCORE.NORMALIZE_DIC[name]["Max"] for name in COMPUTE_SCORE.TASK_INFO}
    scores = COMPUTE_SCORE.compute_score(result)
    assert scores["quality_score"] == 1.0
    assert scores["semantic_score"] == 1.0
    assert scores["total_score"] == 1.0


def test_vbench_scorer_rejects_incomplete_video_set_before_loading_npu(tmp_path):
    from mindspeed_mm.tasks.evaluation.lance.vbench import score_vbench

    full_info = tmp_path / "full_info.json"
    full_info.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected 6230"):
        score_vbench(tmp_path, full_info, tmp_path / "scores")
