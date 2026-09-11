from contextlib import contextmanager
from unittest.mock import patch
import sys
import types

from mindspeed_mm.models.omni.lance.ascend_runtime import (
    patch_upstream_lance_fsdp_optimizer_resume,
)


class _FakeOptimizer:
    def __init__(self):
        self.loaded = None

    def load_state_dict(self, state_dict):
        self.loaded = state_dict


def test_upstream_fsdp_optimizer_state_is_converted_before_loading():
    calls = []

    class FakeFSDP:
        @staticmethod
        @contextmanager
        def state_dict_type(model, state_dict_type, optim_state_dict_config=None):
            calls.append(("context", model, state_dict_type, optim_state_dict_config.offload_to_cpu))
            yield

        @staticmethod
        def optim_state_dict_to_load(model, optimizer, state_dict):
            calls.append(("convert", model, optimizer, state_dict))
            return {"converted": state_dict}

    class FakeShardedOptimStateDictConfig:
        def __init__(self, offload_to_cpu):
            self.offload_to_cpu = offload_to_cpu

    class FakeCheckpoint:
        @staticmethod
        def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
            del scheduler, fsdp_config
            optimizer.load_state_dict({"raw_sharded_adam_state": resume_from})
            return "optimizer", "scheduler", 10, "data-status"

    def original_wrapper(model, *_args, **_kwargs):
        return ("fsdp", model)

    train_package = types.ModuleType("train")
    train_package.__path__ = []
    fsdp_utils = types.ModuleType("train.fsdp_utils")
    fsdp_utils.FSDP = FakeFSDP
    fsdp_utils.StateDictType = types.SimpleNamespace(SHARDED_STATE_DICT="sharded")
    fsdp_utils.ShardedOptimStateDictConfig = FakeShardedOptimStateDictConfig
    fsdp_utils.FSDPCheckpoint = FakeCheckpoint
    fsdp_utils.fsdp_wrapper = original_wrapper
    train_package.fsdp_utils = fsdp_utils

    with patch.dict(
        sys.modules,
        {"train": train_package, "train.fsdp_utils": fsdp_utils},
    ):
        result = patch_upstream_lance_fsdp_optimizer_resume()
        wrapped_model = fsdp_utils.fsdp_wrapper("trainable-model")
        optimizer = _FakeOptimizer()
        loaded = fsdp_utils.FSDPCheckpoint.try_load_train_state(
            "/checkpoint/0000009",
            optimizer,
            object(),
            object(),
        )
        repeated = patch_upstream_lance_fsdp_optimizer_resume()

    assert result["status"] == "installed"
    assert repeated["status"] == "already-installed"
    assert loaded == ("optimizer", "scheduler", 10, "data-status")
    assert optimizer.loaded == {
        "converted": {"raw_sharded_adam_state": "/checkpoint/0000009"}
    }
    assert "load_state_dict" not in optimizer.__dict__
    assert calls[0] == ("context", wrapped_model, "sharded", True)
    assert calls[1][0] == "convert"
    assert calls[1][1] == wrapped_model
    assert calls[1][2] is optimizer
