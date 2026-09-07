"""Paper-aligned stage contracts for Lance PT/CT/SFT/RL runs."""

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


class LanceStageError(ValueError):
    pass


PAPER_OPTIMIZER = {
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


@dataclass(frozen=True)
class LanceStage:
    name: str
    steps: int
    learning_rate: float
    scheduler: str
    warmup_steps: int
    expected_tokens_per_rank: int
    max_tokens_per_rank: int
    max_context: int
    timestep_shift: float
    ce_weight: float
    mse_weight: float
    text_dropout: float
    multimodal_dropout: float

    def __post_init__(self) -> None:
        if self.scheduler not in ("constant", "cosine"):
            raise LanceStageError("scheduler must be constant or cosine")
        if min(
            self.steps,
            self.warmup_steps,
            self.expected_tokens_per_rank,
            self.max_tokens_per_rank,
            self.max_context,
        ) <= 0:
            raise LanceStageError("stage counts must be positive")
        if self.warmup_steps >= self.steps:
            raise LanceStageError("warmup_steps must be smaller than total steps")
        if self.expected_tokens_per_rank > self.max_tokens_per_rank:
            raise LanceStageError("expected token budget cannot exceed its hard limit")
        if self.max_context > self.max_tokens_per_rank:
            raise LanceStageError("sample context cannot exceed the packed rank budget")
        if self.learning_rate <= 0 or self.timestep_shift <= 0:
            raise LanceStageError("learning rate and timestep shift must be positive")
        if self.ce_weight < 0 or self.mse_weight < 0 or self.ce_weight + self.mse_weight <= 0:
            raise LanceStageError("loss weights must be non-negative and not both zero")
        if not 0 <= self.text_dropout <= 1 or not 0 <= self.multimodal_dropout <= 1:
            raise LanceStageError("dropout probabilities must be in [0, 1]")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


PAPER_TASK_MIX: Dict[str, int] = {
    "video_generation": 64,
    "video_understanding": 16,
    "image_generation": 16,
    "image_understanding": 4,
}


STAGES: Dict[str, LanceStage] = {
    "pt": LanceStage(
        name="pt",
        steps=350000,
        learning_rate=1e-4,
        scheduler="constant",
        warmup_steps=2500,
        expected_tokens_per_rank=44000,
        max_tokens_per_rank=50000,
        max_context=40000,
        timestep_shift=1.0,
        ce_weight=0.25,
        mse_weight=1.0,
        text_dropout=0.10,
        multimodal_dropout=0.0,
    ),
    "ct": LanceStage(
        name="ct",
        steps=80000,
        learning_rate=1e-4,
        scheduler="constant",
        warmup_steps=2500,
        expected_tokens_per_rank=74000,
        max_tokens_per_rank=80000,
        max_context=70000,
        timestep_shift=4.0,
        ce_weight=0.5,
        mse_weight=1.0,
        text_dropout=0.05,
        multimodal_dropout=0.05,
    ),
    "sft": LanceStage(
        name="sft",
        steps=15000,
        learning_rate=2.5e-5,
        scheduler="cosine",
        warmup_steps=500,
        expected_tokens_per_rank=74000,
        max_tokens_per_rank=80000,
        max_context=70000,
        timestep_shift=4.0,
        ce_weight=0.25,
        mse_weight=1.0,
        text_dropout=0.05,
        multimodal_dropout=0.05,
    ),
    "rl": LanceStage(
        name="rl",
        steps=800,
        learning_rate=2e-6,
        scheduler="constant",
        warmup_steps=50,
        expected_tokens_per_rank=74000,
        max_tokens_per_rank=80000,
        max_context=70000,
        timestep_shift=4.0,
        ce_weight=0.25,
        mse_weight=1.0,
        text_dropout=0.0,
        multimodal_dropout=0.0,
    ),
}


def training_manifest(
    stage_name: str,
    init_mode: str,
    world_size: int,
    seed: int = 2025,
) -> Dict[str, object]:
    if stage_name not in STAGES:
        raise LanceStageError("unknown Lance stage: {}".format(stage_name))
    if init_mode not in ("qwen2_5_vl", "random", "lance_checkpoint"):
        raise LanceStageError("unsupported initialization mode: {}".format(init_mode))
    if world_size <= 0:
        raise LanceStageError("world_size must be positive")
    stage = STAGES[stage_name]
    if stage_name == "pt" and init_mode == "lance_checkpoint":
        raise LanceStageError("PT cannot be labeled from-scratch when initialized from a Lance checkpoint")
    reproduction_label = {
        "qwen2_5_vl": "paper-pretraining-initialization",
        "random": "strict-random-initialization",
        "lance_checkpoint": "resume-or-finetune",
    }[init_mode]
    return {
        "schema_version": 1,
        "stage": stage.to_dict(),
        "initialization": {"mode": init_mode, "label": reproduction_label},
        "world_size": world_size,
        "global_seed": seed,
        "rank_seed_formula": "global_seed * world_size + global_rank",
        "task_mix": dict(PAPER_TASK_MIX),
        "task_mix_total": sum(PAPER_TASK_MIX.values()),
        "optimizer": dict(PAPER_OPTIMIZER),
        "global_expected_tokens_per_step": stage.expected_tokens_per_rank * world_size,
        "global_max_tokens_per_step": stage.max_tokens_per_rank * world_size,
        "frozen_modules": ["vit_model", "vae_model"],
        "checkpoint_required_state": [
            "model",
            "optimizer",
            "scheduler",
            "ema",
            "dataloader_cursor",
            "mixture_rng",
            "noise_rng",
        ],
    }
