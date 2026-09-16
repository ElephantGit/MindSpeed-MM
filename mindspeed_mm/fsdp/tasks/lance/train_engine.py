"""Lance-specific execution on MindSpeed-MM's optimized FSDP2 runtime."""

from contextlib import nullcontext
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.stateful import Stateful

from mindspeed.fsdp.utils.log import print_rank
from mindspeed_mm.fsdp.distributed.fully_shard_parallel import pregather_fsdp_params
from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.optimizer.clip_grad_norm import clip_grad_norm
from mindspeed_mm.fsdp.tools.memory_profiler import memory_profiler
from mindspeed_mm.fsdp.train.train_engine import TrainEngine
from mindspeed_mm.fsdp.utils.device import get_device_type
from mindspeed_mm.fsdp.utils.dtype import get_dtype
from mindspeed_mm.fsdp.utils.utils import get_time
from mindspeed_mm.models.omni.lance.training_lance import LanceTrainingBatch


logger = logging.getLogger(__name__)


def _trainable_parameters(model):
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _accelerator_rng_state():
    device_type = get_device_type()
    module = getattr(torch, device_type, None)
    getter = getattr(module, "get_rng_state", None)
    return getter() if callable(getter) else None


def _restore_accelerator_rng_state(state):
    if state is None:
        return
    module = getattr(torch, get_device_type(), None)
    setter = getattr(module, "set_rng_state", None)
    if callable(setter):
        setter(state)


def _move(value: Any, dtype=None):
    """Move tensors recursively while preserving Lance attention metadata."""

    if isinstance(value, torch.Tensor):
        target_dtype = dtype if torch.is_floating_point(value) else None
        return value.to(
            device=get_device_type(), dtype=target_dtype, non_blocking=True
        )
    if isinstance(value, LanceTrainingBatch):
        for name in value.__dataclass_fields__:
            setattr(value, name, _move(getattr(value, name), dtype))
        return value
    if isinstance(value, dict):
        return {key: _move(item, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, dtype) for item in value)
    return value


class LanceEMAState(Stateful):
    """EMA stored directly as FSDP2-compatible sharded tensors."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in _trainable_parameters(model).items()
        }

    @torch.no_grad()
    def reset(self, model: torch.nn.Module) -> None:
        parameters = _trainable_parameters(model)
        if set(parameters) != set(self.shadow):
            raise RuntimeError("EMA parameter tree changed after FSDP2 initialization")
        for name, value in self.shadow.items():
            value.copy_(parameters[name].detach())
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        parameters = _trainable_parameters(model)
        if set(parameters) != set(self.shadow):
            raise RuntimeError("EMA parameter tree changed during training")
        alpha = 1.0 if self.num_updates == 0 else 1.0 - self.decay
        # Never mix ordinary tensors and DTensors in a foreach dispatch.
        # Keeping homogeneous groups avoids the optimizer failure encountered
        # by the old bridge on checkpoint resume.
        groups = {True: ([], []), False: ([], [])}
        for name, target in self.shadow.items():
            source = parameters[name].detach()
            group = isinstance(target, DTensor)
            groups[group][0].append(target)
            groups[group][1].append(source)
        for targets, sources in groups.values():
            if targets:
                torch._foreach_lerp_(targets, sources, alpha)
        self.num_updates += 1

    def state_dict(self):
        return {
            "parameters": self.shadow,
            "decay": torch.tensor(self.decay, dtype=torch.float64),
            "num_updates": torch.tensor(self.num_updates, dtype=torch.int64),
        }

    def load_state_dict(self, state_dict):
        required = {"parameters", "decay", "num_updates"}
        missing_fields = required - set(state_dict)
        if missing_fields:
            raise RuntimeError(
                "EMA checkpoint is missing fields: {}".format(
                    ", ".join(sorted(missing_fields))
                )
            )
        parameters = state_dict["parameters"]
        expected_names = set(self.shadow)
        loaded_names = set(parameters)
        if loaded_names != expected_names:
            missing = sorted(expected_names - loaded_names)
            unexpected = sorted(loaded_names - expected_names)
            raise RuntimeError(
                "EMA parameter tree differs from the current model; "
                "missing={}, unexpected={}".format(missing, unexpected)
            )
        for name, target in self.shadow.items():
            target.copy_(parameters[name])
        decay = float(state_dict["decay"].item())
        num_updates = int(state_dict["num_updates"].item())
        if not 0.0 <= decay < 1.0:
            raise RuntimeError("EMA checkpoint decay must be in [0, 1)")
        if num_updates < 0:
            raise RuntimeError("EMA checkpoint num_updates must be non-negative")
        self.decay = decay
        self.num_updates = num_updates


class LanceTrainEngine(TrainEngine):
    """Joint CE/MSE, EMA, fail-fast training loop for native Lance."""

    def __init__(self, *args, **kwargs) -> None:
        model = kwargs.get("model") if "model" in kwargs else args[2]
        config = kwargs.get("args") if "args" in kwargs else args[0]
        self.ema_state = LanceEMAState(
            model,
            decay=float(getattr(config.model, "ema_decay", 0.9999)),
        ) if bool(getattr(config.model, "use_ema", True)) else None
        self._loaded_release = False
        self.last_metrics = {}
        raw_trace = getattr(config.training, "trace_file", None)
        self.trace_file = Path(raw_trace).expanduser().resolve() if raw_trace else None
        self._step_batch_paths = []
        super().__init__(*args, **kwargs)
        if self._loaded_release and self.ema_state is not None:
            self.ema_state.reset(self.model)

    def set_loss_func(self, batch_data):
        # Lance computes token-normalized joint CE/MSE inside the model.
        return None

    def train_step(self, train_dataloader_iter):
        accum_steps = self.args.training.gradient_accumulation_steps
        dtype = (
            get_dtype(self.args.parallel.fsdp_plan.param_dtype)
            if self.args.parallel.fsdp_plan.param_dtype else None
        )
        total_loss = None
        self._step_batch_paths = []
        # Keep metric accumulation on the accelerator.  Converting every
        # micro-step scalar to Python would serialize the NPU stream; FP64 is
        # also unnecessary for at most world_size * 50K token counts.
        metric_sums = torch.zeros(
            4, device=get_device_type(), dtype=torch.float32
        )
        for accum_index in range(accum_steps):
            raw_batch = self.get_batch(train_dataloader_iter)
            if isinstance(raw_batch, dict) and "batch_path" in raw_batch:
                self._step_batch_paths.append(str(raw_batch["batch_path"]))
            batch = _move(raw_batch, dtype)
            sync = self.model.no_sync if (
                hasattr(self.model, "no_sync") and accum_index < accum_steps - 1
            ) else nullcontext
            with sync():
                output = self.model(**batch)
                loss = output.loss / accum_steps
                if not torch.isfinite(loss.detach()).all():
                    raise FloatingPointError("native Lance produced a non-finite loss")
                loss.backward()
            total_loss = loss if total_loss is None else total_loss + loss
            if output.ce_loss is not None:
                metric_sums[0] += output.ce_loss.detach().float() / accum_steps
            if output.mse_loss is not None:
                metric_sums[1] += output.mse_loss.detach().float() / accum_steps
            metric_sums[2] += output.ce_tokens
            metric_sums[3] += output.mse_tokens
        averaged = self.average_losses_across_data_parallel_group([total_loss])
        dp_group = get_parallel_state().get_dp_group()
        dist.all_reduce(metric_sums, group=dp_group)
        metric_sums[:2] /= dist.get_world_size(group=dp_group)
        self.last_metrics = dict(zip(
            ("ce", "mse", "ce_tokens", "mse_tokens"),
            metric_sums.cpu().tolist(),
        ))
        return averaged

    def _write_trace(self, loss, grad_norm, learning_rate):
        """Write deterministic continuity evidence only when explicitly enabled."""

        if self.trace_file is None:
            return
        rank = dist.get_rank()
        suffix = self.trace_file.suffix or ".jsonl"
        stem = self.trace_file.name[:-len(self.trace_file.suffix)] if self.trace_file.suffix else self.trace_file.name
        rank_trace = self.trace_file.with_name(
            "{}.rank{:05d}{}".format(stem, rank, suffix)
        )
        rank_trace.parent.mkdir(parents=True, exist_ok=True)
        checksum = torch.zeros(2, device=get_device_type(), dtype=torch.float32)
        parameter_count = torch.zeros(1, device=get_device_type(), dtype=torch.int64)
        maximum = torch.zeros(1, device=get_device_type(), dtype=torch.float32)
        for parameter in _trainable_parameters(self.model).values():
            local = parameter.detach().to_local() if isinstance(parameter, DTensor) else parameter.detach()
            value = local.float()
            checksum[0] += value.sum()
            checksum[1] += value.square().sum()
            parameter_count += value.numel()
            if value.numel():
                maximum = torch.maximum(maximum, value.abs().max().reshape(1))
        dp_group = get_parallel_state().get_dp_group()
        dist.all_reduce(checksum, group=dp_group)
        dist.all_reduce(parameter_count, group=dp_group)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=dp_group)
        record = {
            "rank": rank,
            "iteration": self.iteration,
            "consumed_train_samples": self.consumed_train_samples,
            "batch_paths": self._step_batch_paths,
            "loss": float(loss.detach().item()),
            "grad_norm": None if grad_norm is None else float(grad_norm),
            "learning_rate": float(learning_rate),
            "metrics": self.last_metrics,
            "parameter_checksum": {
                "sum": float(checksum[0].item()),
                "square_sum": float(checksum[1].item()),
                # Each rank contributes its local FSDP2 shards before the DP
                # all-reduce, so this is the global trainable parameter count.
                "global_numel": int(parameter_count.item()),
                "max_abs": float(maximum.item()),
            },
        }
        with rank_trace.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def training_log(self, *args, **kwargs):
        super().training_log(*args, **kwargs)
        if self.last_metrics:
            print_rank(
                logger.info,
                " Lance metrics | ce: {:.6E} | mse: {:.6E} | ce tokens: {:.0f} | mse tokens: {:.0f} |".format(
                    self.last_metrics["ce"], self.last_metrics["mse"],
                    self.last_metrics["ce_tokens"], self.last_metrics["mse_tokens"],
                ),
            )

    def load(self):
        state = {"model": self.model, "extra_state": {}}
        if not self.args.training.no_load_optim:
            state["optimizer"] = self.optimizer
        if self.ema_state is not None:
            state["ema_state"] = self.ema_state
        release = self.checkpointer.load(
            path=self.args.training.load,
            state=state,
            load_rank0_and_broadcast=self.args.training.load_rank0_and_broadcast,
            load_strict=self.args.training.load_strict,
        )
        self._loaded_release = bool(release)
        if release:
            iteration, consumed = 0, 0
        else:
            extra = state["extra_state"]
            iteration = int(extra["iteration"])
            consumed = int(extra["consumed_train_samples"])
            self.lr_scheduler.load_state_dict(extra["lr_scheduler"])
            self.train_dataloader.load_state_dict(extra["train_dataloader"])
            if not self.args.training.no_load_rng and "torch_rng_state" in extra:
                torch.set_rng_state(extra["torch_rng_state"])
                _restore_accelerator_rng_state(extra.get("accelerator_rng_state"))
        dist.barrier()
        return iteration, consumed

    def save(self, iteration, consumed_train_samples):
        extra_state = {
            "iteration": iteration,
            "consumed_train_samples": consumed_train_samples,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "train_dataloader": self.train_dataloader.state_dict(),
        }
        if not self.args.training.no_save_rng:
            extra_state["torch_rng_state"] = torch.get_rng_state()
            accelerator_state = _accelerator_rng_state()
            if accelerator_state is not None:
                extra_state["accelerator_rng_state"] = accelerator_state
        state = {"model": self.model, "extra_state": extra_state}
        if not self.args.training.no_save_optim:
            state["optimizer"] = self.optimizer
        if self.ema_state is not None:
            state["ema_state"] = self.ema_state
        self.checkpointer.save(
            self.args.training.save,
            state=state,
            iteration=iteration,
            save_async=self.args.training.save_async,
        )
        dist.barrier()

    def train(self):
        # Parent loop already supplies FSDP pregather, fused optimizer, profiler,
        # checkpoint intervals, and stateful data iteration.  Only insert EMA at
        # the point immediately following the parameter update.
        from mindspeed_mm.fsdp.data.data_utils.utils import build_iterations

        iterator, _, _ = build_iterations(self.train_dataloader)
        self.model.train()
        current_lr = self.lr_scheduler.get_last_lr()[0]
        stop_after = int(getattr(
            self.args.training, "stop_after_iters", self.args.training.train_iters
        ))
        if not 0 < stop_after <= self.args.training.train_iters:
            raise ValueError("training.stop_after_iters must be in (0, train_iters]")
        saved_iteration = None
        while self.iteration < stop_after:
            memory_profiler.step()
            start = get_time(barrier=True)
            if self.args.parallel.fsdp_plan.pregather:
                pregather_fsdp_params(self.model)
            loss = self.train_step(iterator)
            grad_norm = clip_grad_norm(
                self.model,
                max_norm=self.args.training.clip_grad,
                foreach=self.args.training.clip_grad_foreach,
            )
            self.optimizer.step()
            self.lr_scheduler.step()
            if self.ema_state is not None:
                self.ema_state.update(self.model)
            self.optimizer.zero_grad(set_to_none=True)
            self.profiler.step()
            self.consumed_train_samples += self.args.training.global_batch_size
            self.iteration += 1
            self._write_trace(loss[0], grad_norm, current_lr)
            elapsed = get_time(barrier=True) - start
            if self.iteration % self.args.training.log_interval == 0:
                self.training_log(
                    self.iteration, elapsed, current_lr,
                    self.consumed_train_samples, loss, grad_norm,
                )
            current_lr = self.lr_scheduler.get_last_lr()[0]
            if (
                self.args.training.save
                and self.args.training.save_interval > 0
                and self.iteration % self.args.training.save_interval == 0
            ):
                self.save(self.iteration, self.consumed_train_samples)
                saved_iteration = self.iteration
        self.profiler.stop()
        memory_profiler.stop()
        if self.args.training.save and saved_iteration != self.iteration:
            self.save(self.iteration, self.consumed_train_samples)
