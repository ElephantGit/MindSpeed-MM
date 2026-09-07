import os

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

torch = pytest.importorskip("torch")

from mindspeed_mm.models.omni.lance.task_mixer import LanceTaskMixer
from mindspeed_mm.models.omni.lance.training_state import (
    LanceTrainingState,
    LanceTrainingStateError,
    build_lance_dcp_state,
)


MANIFEST_SHA = "a" * 64


class StatefulCounter:
    def __init__(self, value=0):
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


def _state(rank=0, scheduler=None, ema_metadata=None, manifest_sha=MANIFEST_SHA):
    generator = torch.Generator().manual_seed(123)
    return LanceTrainingState(
        iteration=10,
        consumed_train_samples=80,
        training_manifest_sha256=manifest_sha,
        dataloader_cursor={"shard": 3, "row": 91},
        mixer=LanceTaskMixer(global_seed=7, world_size=2, global_rank=rank),
        noise_generator=generator,
        scheduler=scheduler,
        ema_metadata=ema_metadata,
    )


def test_resume_restores_task_and_noise_sequences_bit_exactly():
    original = _state()
    original.mixer.take(10)
    torch.randn(5, generator=original.noise_generator)
    saved = original.state_dict()
    expected_tasks = original.mixer.take(20)
    expected_noise = torch.randn(20, generator=original.noise_generator)

    resumed = _state()
    resumed.load_state_dict(saved)
    assert resumed.mixer.take(20) == expected_tasks
    torch.testing.assert_close(
        torch.randn(20, generator=resumed.noise_generator),
        expected_noise,
        rtol=0,
        atol=0,
    )
    assert resumed.iteration == 10
    assert resumed.dataloader_cursor == {"shard": 3, "row": 91}


def test_resume_restores_scheduler_and_ema_and_builds_dcp_mapping():
    scheduler = StatefulCounter(12)
    original = _state(scheduler=scheduler, ema_metadata={"updates": 34, "decay": 0.9999})
    saved = original.state_dict()
    scheduler.value = 0
    original.load_state_dict(saved)
    assert scheduler.value == 12
    assert original.ema_metadata == {"updates": 34, "decay": 0.9999}
    model, optimizer, ema_model = object(), object(), object()
    dcp_state = build_lance_dcp_state(model, original, optimizer, ema_model)
    assert dcp_state["model"] is model
    assert dcp_state["optimizer"] is optimizer
    assert dcp_state["ema_model"] is ema_model
    assert dcp_state["extra_state"]["ema"] == {"updates": 34, "decay": 0.9999}


def test_resume_rejects_manifest_drift_and_partial_state():
    saved = _state().state_dict()
    with pytest.raises(LanceTrainingStateError, match="manifest"):
        _state(manifest_sha="b" * 64).load_state_dict(saved)
    del saved["noise_rng_state"]
    with pytest.raises(LanceTrainingStateError, match="missing"):
        _state().load_state_dict(saved)


def test_training_state_rejects_non_sha_manifest_identity():
    with pytest.raises(LanceTrainingStateError, match="SHA-256"):
        _state(manifest_sha="not-a-sha")
