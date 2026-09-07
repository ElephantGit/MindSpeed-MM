import json
import os

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

from mindspeed_mm.models.omni.lance.task_mixer import (
    LanceTaskMixer,
    LanceTaskMixerError,
    group_by_token_budget,
)


def test_default_task_mixer_emits_exact_paper_ratio_each_cycle():
    mixer = LanceTaskMixer(global_seed=2025, world_size=8, global_rank=3)
    assert mixer.cycle_length == 25
    cycle = mixer.take(mixer.cycle_length)
    assert cycle.count("video_generation") == 16
    assert cycle.count("video_understanding") == 4
    assert cycle.count("image_generation") == 4
    assert cycle.count("image_understanding") == 1


def test_task_mixer_resume_is_bit_exact_after_json_round_trip():
    original = LanceTaskMixer(global_seed=9, world_size=4, global_rank=2)
    original.take(31)
    state = json.loads(json.dumps(original.state_dict()))
    expected = original.take(100)

    resumed = LanceTaskMixer(global_seed=9, world_size=4, global_rank=2)
    resumed.load_state_dict(state)
    assert resumed.take(100) == expected


def test_task_mixer_rejects_state_from_another_rank():
    source = LanceTaskMixer(global_seed=9, world_size=4, global_rank=1)
    source.take(3)
    target = LanceTaskMixer(global_seed=9, world_size=4, global_rank=2)
    with pytest.raises(LanceTaskMixerError, match="global_rank"):
        target.load_state_dict(source.state_dict())


class Sample:
    def __init__(self, length):
        self.length = length


def test_token_budget_groups_at_expected_target_without_crossing_hard_limit():
    groups = group_by_token_budget(
        [Sample(4), Sample(3), Sample(4), Sample(6), Sample(2)],
        expected_tokens=7,
        max_tokens=10,
    )
    assert [[sample.length for sample in group] for group in groups] == [[4, 3], [4, 6], [2]]
    assert all(sum(sample.length for sample in group) <= 10 for group in groups)


def test_token_budget_rejects_oversized_sample():
    with pytest.raises(LanceTaskMixerError, match="max_sample_tokens"):
        group_by_token_budget(
            [Sample(11)],
            expected_tokens=7,
            max_tokens=10,
            max_sample_tokens=8,
        )
