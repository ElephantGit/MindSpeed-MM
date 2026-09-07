"""Native Lance flow-matching sampler and model-facing denoiser context."""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch

from .modeling_lance import LanceNativeModel
from .training_lance import shift_timesteps
from .sequence import LancePackedSequence


class LanceSamplingError(ValueError):
    pass


@dataclass
class LanceDenoiseContext:
    """One conditional branch of a packed native Lance generation sample.

    ``vae_indexes`` maps every row in the latent tensor into the packed model
    sequence. ``prediction_indexes`` selects model outputs for the rows named
    by ``prediction_latent_indexes``. The latter permits edit tasks to retain
    clean conditioning latents while updating only their target region.
    """

    token_ids: torch.Tensor
    text_indexes: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: Any
    understanding_indexes: torch.Tensor
    generation_indexes: torch.Tensor
    vae_indexes: torch.Tensor
    latent_position_ids: torch.Tensor
    prediction_indexes: torch.Tensor
    prediction_latent_indexes: torch.Tensor
    vit_indexes: Optional[torch.Tensor] = None
    vit_embeddings: Optional[torch.Tensor] = None

    def validate(self, model: LanceNativeModel, latent_count: int) -> None:
        length = int(self.token_ids.numel())
        if self.token_ids.ndim != 1:
            raise LanceSamplingError("token_ids must be one-dimensional")
        if self.position_ids.shape != (3, length):
            raise LanceSamplingError("position_ids must have shape [3, sequence_length]")
        if isinstance(self.attention_mask, LancePackedSequence):
            if self.attention_mask.length != length:
                raise LanceSamplingError("packed attention metadata length does not match sequence")
        elif self.attention_mask is not None and self.attention_mask.shape != (length, length):
            raise LanceSamplingError("reference attention mask must have shape [length, length]")
        for name, indexes, upper_bound in (
            ("text", self.text_indexes, length),
            ("understanding", self.understanding_indexes, length),
            ("generation", self.generation_indexes, length),
            ("VAE", self.vae_indexes, length),
            ("prediction", self.prediction_indexes, length),
            ("prediction latent", self.prediction_latent_indexes, latent_count),
        ):
            _validate_indexes(name, indexes, upper_bound)
        routes = torch.cat((self.understanding_indexes, self.generation_indexes))
        if routes.numel() != length or torch.unique(routes).numel() != length:
            raise LanceSamplingError("expert indexes must cover every token exactly once")
        if self.vae_indexes.numel() != latent_count:
            raise LanceSamplingError("vae_indexes must map every latent row")
        if self.latent_position_ids.shape != (latent_count,):
            raise LanceSamplingError("latent_position_ids must map every latent row")
        if self.prediction_indexes.numel() != self.prediction_latent_indexes.numel():
            raise LanceSamplingError("prediction sequence and latent indexes must have equal length")
        if (self.vit_indexes is None) != (self.vit_embeddings is None):
            raise LanceSamplingError("ViT indexes and embeddings must be provided together")
        if self.vit_indexes is not None:
            _validate_indexes("ViT", self.vit_indexes, length)
            if self.vit_embeddings.shape != (self.vit_indexes.numel(), model.config.hidden_size):
                raise LanceSamplingError("vit_embeddings shape does not match ViT indexes")


def _validate_indexes(name: str, indexes: torch.Tensor, upper_bound: int) -> None:
    if indexes.ndim != 1 or indexes.dtype != torch.long:
        raise LanceSamplingError("{} indexes must be one-dimensional torch.long".format(name))
    if indexes.numel() and (
        int(indexes.min().item()) < 0
        or int(indexes.max().item()) >= upper_bound
        or torch.unique(indexes).numel() != indexes.numel()
    ):
        raise LanceSamplingError("{} indexes contain duplicates or out-of-range values".format(name))


def lance_sampling_schedule(
    num_steps: int,
    timestep_shift: float,
    *,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return shifted model timesteps and positive Euler step widths."""

    if num_steps <= 0:
        raise LanceSamplingError("num_steps must be positive")
    base = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
    shifted = shift_timesteps(base, timestep_shift)
    return shifted[:-1], shifted[:-1] - shifted[1:]


def lance_cfg_velocity(
    conditional: torch.Tensor,
    text_unconditional: Optional[torch.Tensor],
    *,
    text_scale: float,
    vision_unconditional: Optional[torch.Tensor] = None,
    vision_scale: float = 1.0,
    renorm_min: float = 0.0,
    renorm_type: str = "global",
) -> torch.Tensor:
    """Apply Lance's two- or three-branch classifier-free guidance."""

    if text_scale < 1.0 or vision_scale < 1.0:
        raise LanceSamplingError("CFG scales must be at least 1")
    if not 0.0 <= renorm_min <= 1.0:
        raise LanceSamplingError("renorm_min must be in [0, 1]")
    if text_scale == 1.0 and vision_scale == 1.0:
        return conditional
    if text_unconditional is None:
        raise LanceSamplingError("text-unconditional velocity is required when CFG is active")
    if conditional.shape != text_unconditional.shape:
        raise LanceSamplingError("conditional and unconditional velocities must have equal shape")

    if vision_scale > 1.0:
        if vision_unconditional is None or vision_unconditional.shape != conditional.shape:
            raise LanceSamplingError("vision-unconditional velocity is required for vision CFG")
        guided = (
            vision_unconditional
            + text_scale * (conditional - text_unconditional)
            + vision_scale * (text_unconditional - vision_unconditional)
        )
    else:
        guided = text_unconditional + text_scale * (conditional - text_unconditional)

    normalized_type = renorm_type.lower()
    if normalized_type in ("", "none", "null"):
        return guided
    if normalized_type == "global":
        conditional_norm = torch.linalg.vector_norm(conditional.float())
        guided_norm = torch.linalg.vector_norm(guided.float())
    elif normalized_type == "channel":
        conditional_norm = torch.linalg.vector_norm(conditional.float(), dim=-1, keepdim=True)
        guided_norm = torch.linalg.vector_norm(guided.float(), dim=-1, keepdim=True)
    else:
        raise LanceSamplingError("unsupported CFG renorm type: {}".format(renorm_type))
    scale = (conditional_norm / (guided_norm + 1e-8)).clamp(min=renorm_min, max=1.0)
    return guided * scale.to(guided.dtype)


def predict_native_velocity(
    model: LanceNativeModel,
    context: LanceDenoiseContext,
    latents: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Run one native Lance velocity prediction for a conditional branch."""

    if latents.ndim != 2 or latents.shape[1] != model.config.patch_latent_dim:
        raise LanceSamplingError("latents must have shape [tokens, patch_latent_dim]")
    context.validate(model, latents.shape[0])
    if timestep.ndim != 0 or not 0.0 <= float(timestep.item()) <= 1.0:
        raise LanceSamplingError("timestep must be a scalar in [0, 1]")

    token_embeddings = model.language_model.model.embed_tokens(context.token_ids)
    sequence = token_embeddings.new_zeros((context.token_ids.numel(), model.config.hidden_size))
    sequence[context.text_indexes] = token_embeddings[context.text_indexes]
    if context.vit_indexes is not None:
        sequence[context.vit_indexes] = context.vit_embeddings.to(sequence.dtype)

    latent_timesteps = latents.new_zeros((latents.shape[0],))
    latent_timesteps[context.prediction_latent_indexes] = timestep.to(latent_timesteps.dtype)
    sequence[context.vae_indexes] = (
        model.vae2llm(latents)
        + model.time_embedder(latent_timesteps)
        + model.latent_pos_embed(context.latent_position_ids)
    ).to(sequence.dtype)
    hidden_states = model.forward_language(
        sequence,
        context.position_ids,
        context.attention_mask,
        context.understanding_indexes,
        context.generation_indexes,
    )
    return model.llm2vae(hidden_states[context.prediction_indexes])


VelocityFunction = Callable[[torch.Tensor, torch.Tensor, str], torch.Tensor]


def euler_flow_sample(
    initial_latents: torch.Tensor,
    velocity_function: VelocityFunction,
    *,
    num_steps: int,
    timestep_shift: float,
    update_indexes: Optional[torch.Tensor] = None,
    cfg_interval: Tuple[float, float] = (0.0, 1.0),
    text_scale: float = 1.0,
    vision_scale: float = 1.0,
    renorm_min: float = 0.0,
    renorm_type: str = "global",
) -> torch.Tensor:
    """Integrate Lance velocity from noise (t=1) to data (t=0)."""

    lower, upper = cfg_interval
    if not 0.0 <= lower <= upper <= 1.0:
        raise LanceSamplingError("cfg_interval must satisfy 0 <= lower <= upper <= 1")
    if update_indexes is None:
        update_indexes = torch.arange(initial_latents.shape[0], device=initial_latents.device)
    _validate_indexes("update", update_indexes, initial_latents.shape[0])
    latents = initial_latents.clone()
    timesteps, widths = lance_sampling_schedule(
        num_steps,
        timestep_shift,
        device=initial_latents.device,
    )
    for timestep, width in zip(timesteps, widths):
        conditional = velocity_function(latents, timestep, "conditional")
        expected_shape = latents[update_indexes].shape
        if conditional.shape != expected_shape:
            raise LanceSamplingError("velocity shape must match the selected update rows")
        active = bool((timestep > lower) & (timestep <= upper))
        if active and (text_scale > 1.0 or vision_scale > 1.0):
            text_unconditional = velocity_function(latents, timestep, "text_unconditional")
            vision_unconditional = None
            if vision_scale > 1.0:
                vision_unconditional = velocity_function(latents, timestep, "vision_unconditional")
            velocity = lance_cfg_velocity(
                conditional,
                text_unconditional,
                text_scale=text_scale,
                vision_unconditional=vision_unconditional,
                vision_scale=vision_scale,
                renorm_min=renorm_min,
                renorm_type=renorm_type,
            )
        else:
            velocity = conditional
        latents[update_indexes] = latents[update_indexes] - velocity.to(latents.dtype) * width
    return latents


def sample_native_lance(
    model: LanceNativeModel,
    context: LanceDenoiseContext,
    initial_latents: torch.Tensor,
    *,
    num_steps: int,
    timestep_shift: float,
    text_unconditional_context: Optional[LanceDenoiseContext] = None,
    vision_unconditional_context: Optional[LanceDenoiseContext] = None,
    cfg_interval: Tuple[float, float] = (0.0, 1.0),
    text_scale: float = 1.0,
    vision_scale: float = 1.0,
    renorm_min: float = 0.0,
    renorm_type: str = "global",
) -> torch.Tensor:
    """Sample with native model contexts; VAE decoding is intentionally external."""

    contexts = {
        "conditional": context,
        "text_unconditional": text_unconditional_context,
        "vision_unconditional": vision_unconditional_context,
    }

    def velocity(latents: torch.Tensor, timestep: torch.Tensor, branch: str) -> torch.Tensor:
        branch_context = contexts[branch]
        if branch_context is None:
            raise LanceSamplingError("missing {} context".format(branch.replace("_", "-")))
        return predict_native_velocity(model, branch_context, latents, timestep)

    return euler_flow_sample(
        initial_latents,
        velocity,
        num_steps=num_steps,
        timestep_shift=timestep_shift,
        update_indexes=context.prediction_latent_indexes,
        cfg_interval=cfg_interval,
        text_scale=text_scale,
        vision_scale=vision_scale,
        renorm_min=renorm_min,
        renorm_type=renorm_type,
    )
