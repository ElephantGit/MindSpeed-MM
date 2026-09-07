import os

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

from mindspeed_mm.models.omni.lance.training_contract import (
    PAPER_OPTIMIZER,
    PAPER_TASK_MIX,
    STAGES,
    LanceStageError,
    training_manifest,
)


def test_paper_stage_hyperparameters_are_frozen():
    assert STAGES["pt"].steps == 350000
    assert STAGES["pt"].timestep_shift == 1.0
    assert STAGES["pt"].ce_weight == 0.25
    assert STAGES["ct"].steps == 80000
    assert STAGES["ct"].max_context == 70000
    assert STAGES["ct"].ce_weight == 0.5
    assert STAGES["sft"].steps == 15000
    assert STAGES["sft"].scheduler == "cosine"
    assert STAGES["rl"].steps == 800
    assert PAPER_TASK_MIX == {
        "video_generation": 64,
        "video_understanding": 16,
        "image_generation": 16,
        "image_understanding": 4,
    }
    assert PAPER_OPTIMIZER == {
        "name": "adamw",
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 1e-15,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "ema_decay": 0.9999,
        "min_learning_rate": 1e-7,
        "cosine_cycles": 5.0,
    }


def test_manifest_keeps_paper_and_strict_random_initialization_distinct():
    paper = training_manifest("pt", "qwen2_5_vl", world_size=8)
    random = training_manifest("pt", "random", world_size=8)
    assert paper["initialization"]["label"] == "paper-pretraining-initialization"
    assert random["initialization"]["label"] == "strict-random-initialization"
    assert paper["global_expected_tokens_per_step"] == 352000
    assert len(paper["checkpoint_required_state"]) == 7


def test_pt_manifest_rejects_lance_checkpoint_as_from_scratch():
    with pytest.raises(LanceStageError, match="from-scratch"):
        training_manifest("pt", "lance_checkpoint", world_size=8)
