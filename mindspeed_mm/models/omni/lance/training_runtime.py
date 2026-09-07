"""Optimizer, scheduler, EMA, and one-step runtime for native Lance."""

from copy import deepcopy
import math
from typing import Dict, Optional, Union

import torch
from torch import nn

from .modeling_lance import LanceNativeModel
from .training_contract import PAPER_OPTIMIZER, STAGES, LanceStage
from .training_lance import LanceLossWeights, LanceTrainingBatch, lance_training_step


class LanceTrainingRuntimeError(ValueError):
    pass


def configure_lance_trainability(
    model: LanceNativeModel,
    *,
    freeze_vit: bool = True,
) -> Dict[str, int]:
    """Freeze the released vision encoder while retaining all Lance heads/experts."""

    frozen = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        if freeze_vit and name.startswith("vit_model."):
            parameter.requires_grad_(False)
        if parameter.requires_grad:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}


def build_lance_optimizer(
    model: nn.Module,
    stage: Union[str, LanceStage],
) -> torch.optim.Optimizer:
    selected = STAGES[stage] if isinstance(stage, str) else stage
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise LanceTrainingRuntimeError("Lance optimizer has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=selected.learning_rate,
        betas=(PAPER_OPTIMIZER["beta1"], PAPER_OPTIMIZER["beta2"]),
        eps=PAPER_OPTIMIZER["epsilon"],
        weight_decay=PAPER_OPTIMIZER["weight_decay"],
    )


class LanceLRScheduler:
    """Dependency-free equivalent of the released HF warmup schedulers."""

    def __init__(self, optimizer: torch.optim.Optimizer, stage: Union[str, LanceStage]) -> None:
        self.optimizer = optimizer
        self.stage = STAGES[stage] if isinstance(stage, str) else stage
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.step_number = 0
        self._set_learning_rates(self.step_number)

    def _factor(self, step: int, base_lr: float) -> float:
        if step < self.stage.warmup_steps:
            return float(step) / float(max(1, self.stage.warmup_steps))
        if self.stage.scheduler == "constant":
            return 1.0
        progress = min(
            1.0,
            float(step - self.stage.warmup_steps)
            / float(max(1, self.stage.steps - self.stage.warmup_steps)),
        )
        cosine = 0.5 * (
            1.0 + math.cos(2.0 * math.pi * PAPER_OPTIMIZER["cosine_cycles"] * progress)
        )
        minimum_factor = min(1.0, PAPER_OPTIMIZER["min_learning_rate"] / base_lr)
        return minimum_factor + (1.0 - minimum_factor) * cosine

    def _set_learning_rates(self, step: int) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * self._factor(step, base_lr)

    def step(self) -> None:
        self.step_number += 1
        self._set_learning_rates(self.step_number)

    def get_last_lr(self):
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "stage": self.stage.name,
            "base_lrs": list(self.base_lrs),
            "step_number": self.step_number,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if state.get("schema_version") != 1 or state.get("stage") != self.stage.name:
            raise LanceTrainingRuntimeError("scheduler state does not match the configured stage")
        base_lrs = list(state.get("base_lrs", []))
        step = state.get("step_number")
        if base_lrs != self.base_lrs or not isinstance(step, int) or not 0 <= step <= self.stage.steps:
            raise LanceTrainingRuntimeError("scheduler state is incompatible")
        self.step_number = step
        self._set_learning_rates(step)


class LanceEMAController:
    """Maintain a separately sharded EMA model and small resumable metadata."""

    def __init__(
        self,
        model: LanceNativeModel,
        *,
        decay: float = PAPER_OPTIMIZER["ema_decay"],
        start_step: int = 0,
    ) -> None:
        if not 0.0 <= decay < 1.0 or start_step < 0:
            raise LanceTrainingRuntimeError("EMA decay/start_step is invalid")
        self.model = deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.decay = float(decay)
        self.start_step = int(start_step)
        self.num_updates = 0

    @torch.no_grad()
    def update(self, source: LanceNativeModel, step: int) -> None:
        if step < self.start_step:
            return
        source_parameters = dict(source.named_parameters())
        effective_decay = 0.0 if self.num_updates == 0 else self.decay
        for name, ema_parameter in self.model.named_parameters():
            source_parameter = source_parameters[name]
            if source_parameter.requires_grad:
                ema_parameter.mul_(effective_decay).add_(
                    source_parameter.detach().to(device=ema_parameter.device, dtype=ema_parameter.dtype),
                    alpha=1.0 - effective_decay,
                )
        self.num_updates += 1

    def metadata_state_dict(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "decay": self.decay,
            "start_step": self.start_step,
            "num_updates": self.num_updates,
        }

    def load_metadata_state_dict(self, state: Dict[str, object]) -> None:
        expected = (1, self.decay, self.start_step)
        actual = (state.get("schema_version"), state.get("decay"), state.get("start_step"))
        updates = state.get("num_updates")
        if actual != expected or not isinstance(updates, int) or updates < 0:
            raise LanceTrainingRuntimeError("EMA metadata is incompatible")
        self.num_updates = updates


class LanceNativeTrainingRuntime:
    """Minimal eager runtime used before wrapping the model with FSDP2."""

    def __init__(
        self,
        model: LanceNativeModel,
        stage: Union[str, LanceStage],
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[LanceLRScheduler] = None,
        ema: Optional[LanceEMAController] = None,
    ) -> None:
        self.model = model
        self.stage = STAGES[stage] if isinstance(stage, str) else stage
        self.optimizer = optimizer or build_lance_optimizer(model, self.stage)
        self.scheduler = scheduler or LanceLRScheduler(self.optimizer, self.stage)
        self.ema = ema
        self.completed_steps = 0

    def step(self, batch: LanceTrainingBatch) -> Dict[str, object]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = lance_training_step(
            self.model,
            batch,
            LanceLossWeights(self.stage.ce_weight, self.stage.mse_weight),
            self.stage.timestep_shift,
        )
        losses["loss"].backward()
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, PAPER_OPTIMIZER["max_grad_norm"])
        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.model, self.completed_steps)
        self.scheduler.step()
        self.completed_steps += 1
        return {
            **losses,
            "grad_norm": grad_norm,
            "learning_rate": self.scheduler.get_last_lr()[0],
            "completed_steps": self.completed_steps,
        }
