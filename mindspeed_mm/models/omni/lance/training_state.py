"""Strict extra-state contract for resumable Lance FSDP/DCP training."""

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Dict, Mapping, Optional

import torch

from .task_mixer import LanceTaskMixer


class LanceTrainingStateError(ValueError):
    pass


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_KEYS = {
    "schema_version",
    "iteration",
    "consumed_train_samples",
    "training_manifest_sha256",
    "dataloader_cursor",
    "mixture_rng",
    "noise_rng_state",
    "scheduler",
    "ema",
    "torch_rng_state",
    "accelerator_rng_state",
}


def _accelerator_rng_state() -> Optional[Dict[str, object]]:
    npu = getattr(torch, "npu", None)
    if npu is not None and hasattr(npu, "is_available") and npu.is_available():
        return {"backend": "npu", "states": npu.get_rng_state_all()}
    if torch.cuda.is_available():
        return {"backend": "cuda", "states": torch.cuda.get_rng_state_all()}
    return None


def _restore_accelerator_rng_state(state: Optional[Mapping[str, object]]) -> None:
    if state is None:
        return
    backend = state.get("backend")
    states = state.get("states")
    if backend == "npu":
        npu = getattr(torch, "npu", None)
        if npu is None or not npu.is_available():
            raise LanceTrainingStateError("checkpoint contains NPU RNG state but NPU is unavailable")
        npu.set_rng_state_all(states)
    elif backend == "cuda":
        if not torch.cuda.is_available():
            raise LanceTrainingStateError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(states)
    else:
        raise LanceTrainingStateError("checkpoint contains an unknown accelerator RNG backend")


@dataclass
class LanceTrainingState:
    iteration: int
    consumed_train_samples: int
    training_manifest_sha256: str
    dataloader_cursor: Mapping[str, object]
    mixer: LanceTaskMixer
    noise_generator: torch.Generator
    scheduler: Optional[Any] = None
    ema_metadata: Optional[Mapping[str, object]] = None

    def __post_init__(self) -> None:
        if self.iteration < 0 or self.consumed_train_samples < 0:
            raise LanceTrainingStateError("iteration and consumed sample count must be non-negative")
        if not _SHA256.fullmatch(self.training_manifest_sha256):
            raise LanceTrainingStateError("training_manifest_sha256 must be a lowercase SHA-256")
        if not isinstance(self.dataloader_cursor, Mapping):
            raise LanceTrainingStateError("dataloader_cursor must be a mapping")

    def state_dict(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "iteration": self.iteration,
            "consumed_train_samples": self.consumed_train_samples,
            "training_manifest_sha256": self.training_manifest_sha256,
            "dataloader_cursor": deepcopy(dict(self.dataloader_cursor)),
            "mixture_rng": self.mixer.state_dict(),
            "noise_rng_state": self.noise_generator.get_state(),
            "scheduler": deepcopy(self.scheduler.state_dict()) if self.scheduler is not None else None,
            # EMA parameters are saved as a top-level sharded DCP model.  Only
            # small update metadata belongs in per-rank extra state.
            "ema": deepcopy(dict(self.ema_metadata)) if self.ema_metadata is not None else None,
            "torch_rng_state": torch.get_rng_state(),
            "accelerator_rng_state": _accelerator_rng_state(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        missing = sorted(_REQUIRED_KEYS - set(state))
        unexpected = sorted(set(state) - _REQUIRED_KEYS)
        if missing or unexpected:
            raise LanceTrainingStateError(
                "invalid Lance extra-state keys; missing={}, unexpected={}".format(missing, unexpected)
            )
        if state["schema_version"] != 1:
            raise LanceTrainingStateError("unsupported Lance extra-state schema version")
        if state["training_manifest_sha256"] != self.training_manifest_sha256:
            raise LanceTrainingStateError("training manifest SHA-256 changed across resume")
        iteration = state["iteration"]
        consumed = state["consumed_train_samples"]
        cursor = state["dataloader_cursor"]
        if not isinstance(iteration, int) or iteration < 0:
            raise LanceTrainingStateError("checkpoint iteration is invalid")
        if not isinstance(consumed, int) or consumed < 0:
            raise LanceTrainingStateError("checkpoint consumed sample count is invalid")
        if not isinstance(cursor, Mapping):
            raise LanceTrainingStateError("checkpoint dataloader cursor is invalid")
        if self.scheduler is None and state["scheduler"] is not None:
            raise LanceTrainingStateError("checkpoint has scheduler state but no scheduler was provided")

        self.mixer.load_state_dict(state["mixture_rng"])
        try:
            self.noise_generator.set_state(state["noise_rng_state"])
            torch.set_rng_state(state["torch_rng_state"])
        except (TypeError, RuntimeError) as exc:
            raise LanceTrainingStateError("checkpoint contains invalid torch RNG state") from exc
        _restore_accelerator_rng_state(state["accelerator_rng_state"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        ema_metadata = state["ema"]
        if ema_metadata is not None and not isinstance(ema_metadata, Mapping):
            raise LanceTrainingStateError("checkpoint EMA metadata is invalid")
        self.ema_metadata = deepcopy(dict(ema_metadata)) if ema_metadata is not None else None
        self.iteration = iteration
        self.consumed_train_samples = consumed
        self.dataloader_cursor = deepcopy(dict(cursor))


def build_lance_dcp_state(
    model: Any,
    training_state: LanceTrainingState,
    optimizer: Optional[Any] = None,
    ema_model: Optional[Any] = None,
) -> Dict[str, object]:
    """Build the state mapping consumed by MindSpeed-MM DistributedCheckpointer."""

    state: Dict[str, object] = {"model": model, "extra_state": training_state.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer
    if ema_model is not None:
        state["ema_model"] = ema_model
    return state
