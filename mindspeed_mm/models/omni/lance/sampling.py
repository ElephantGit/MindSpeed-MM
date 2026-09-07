"""Native Lance flow-matching sampler and model-facing denoiser context."""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch

from .modeling_lance import (
    KVAttentionBackend,
    LanceKVCache,
    LanceNativeModel,
    reference_kv_sdpa,
)
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


@dataclass(frozen=True)
class LanceCachedDenoiseState:
    """Compiled prefix cache for the common condition + noisy-VAE topology."""

    context: LanceDenoiseContext
    kv_cache: LanceKVCache
    query_position_ids: torch.Tensor
    query_understanding_indexes: torch.Tensor
    query_generation_indexes: torch.Tensor
    query_latent_position_ids: torch.Tensor
    attention_backend: KVAttentionBackend
    is_causal: bool


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


def _condition_attention_contract(
    context: LanceDenoiseContext,
    condition_length: int,
) -> Tuple[Any, bool]:
    """Extract a cacheable prefix and determine the noisy-query mask mode."""

    query_length = context.token_ids.numel() - condition_length
    if isinstance(context.attention_mask, LancePackedSequence):
        packed = context.attention_mask
        if len(packed.documents) != 1:
            raise LanceSamplingError("KV-cache sampling currently requires one packed document")
        segments = packed.documents[0].segments
        if len(segments) < 2 or segments[-1].length != query_length:
            raise LanceSamplingError("KV-cache query must be the final packed segment")
        if any(segment.normalized_attention_mode == "noise" for segment in segments[:-1]):
            raise LanceSamplingError(
                "KV-cache condition cannot include noise segments hidden from the final query"
            )
        condition = LancePackedSequence(
            (
                type(packed.documents[0])(
                    packed.documents[0].sample_id,
                    segments[:-1],
                ),
            )
        )
        if condition.length != condition_length:
            raise LanceSamplingError("KV-cache condition/query boundary must align to a segment")
        return condition, segments[-1].normalized_attention_mode == "causal"

    mask = context.attention_mask
    if mask is None or mask.shape != (context.token_ids.numel(), context.token_ids.numel()):
        raise LanceSamplingError("KV-cache sampling requires dense or packed attention metadata")
    if bool(mask[:condition_length, condition_length:].any()):
        raise LanceSamplingError("condition tokens must not attend to dynamic query tokens")
    if not bool(mask[condition_length:, :condition_length].all()):
        raise LanceSamplingError("dynamic query tokens must attend to the entire condition prefix")
    query_mask = mask[condition_length:, condition_length:].bool()
    full = torch.ones_like(query_mask)
    causal = full.tril()
    if torch.equal(query_mask, full):
        is_causal = False
    elif torch.equal(query_mask, causal):
        is_causal = True
    else:
        raise LanceSamplingError("dynamic query self-attention must be full or causal")
    return mask[:condition_length, :condition_length], is_causal


def compile_native_kv_cache(
    model: LanceNativeModel,
    context: LanceDenoiseContext,
    latents: torch.Tensor,
    *,
    attention_backend: KVAttentionBackend = reference_kv_sdpa,
) -> LanceCachedDenoiseState:
    """Prefill static condition K/V once for diffusion sampling.

    The cacheable native layout is deliberately strict: prediction tokens are
    one contiguous VAE suffix.  Refusing a non-equivalent split prevents silent
    attention changes for uncommon editing templates.
    """

    if latents.ndim != 2 or latents.shape[1] != model.config.patch_latent_dim:
        raise LanceSamplingError("latents must have shape [tokens, patch_latent_dim]")
    context.validate(model, latents.shape[0])
    query_length = int(context.prediction_indexes.numel())
    if query_length == 0:
        raise LanceSamplingError("KV-cache sampling requires prediction tokens")
    condition_length = int(context.token_ids.numel()) - query_length
    expected_query = torch.arange(
        condition_length,
        context.token_ids.numel(),
        dtype=torch.long,
        device=context.prediction_indexes.device,
    )
    if not torch.equal(context.prediction_indexes, expected_query):
        raise LanceSamplingError("prediction tokens must be a contiguous sequence suffix")
    mapped_predictions = context.vae_indexes[context.prediction_latent_indexes]
    if not torch.equal(mapped_predictions, context.prediction_indexes):
        raise LanceSamplingError("prediction tokens must map exactly to prediction latent rows")
    if condition_length <= 0:
        raise LanceSamplingError("KV-cache sampling requires a non-empty condition prefix")

    condition_attention, is_causal = _condition_attention_contract(context, condition_length)
    token_embeddings = model.language_model.model.embed_tokens(context.token_ids)
    sequence = token_embeddings.new_zeros((context.token_ids.numel(), model.config.hidden_size))
    sequence[context.text_indexes] = token_embeddings[context.text_indexes]
    if context.vit_indexes is not None:
        sequence[context.vit_indexes] = context.vit_embeddings.to(sequence.dtype)
    zero_timesteps = latents.new_zeros((latents.shape[0],))
    sequence[context.vae_indexes] = (
        model.vae2llm(latents)
        + model.time_embedder(zero_timesteps)
        + model.latent_pos_embed(context.latent_position_ids)
    ).to(sequence.dtype)

    condition_understanding = context.understanding_indexes[
        context.understanding_indexes < condition_length
    ]
    condition_generation = context.generation_indexes[
        context.generation_indexes < condition_length
    ]
    _, kv_cache = model.build_language_kv_cache(
        sequence[:condition_length],
        context.position_ids[:, :condition_length],
        condition_attention,
        condition_understanding,
        condition_generation,
    )
    query_understanding = (
        context.understanding_indexes[context.understanding_indexes >= condition_length]
        - condition_length
    )
    query_generation = (
        context.generation_indexes[context.generation_indexes >= condition_length]
        - condition_length
    )
    return LanceCachedDenoiseState(
        context=context,
        kv_cache=kv_cache,
        query_position_ids=context.position_ids[:, condition_length:],
        query_understanding_indexes=query_understanding,
        query_generation_indexes=query_generation,
        query_latent_position_ids=context.latent_position_ids[context.prediction_latent_indexes],
        attention_backend=attention_backend,
        is_causal=is_causal,
    )


def predict_cached_native_velocity(
    model: LanceNativeModel,
    state: LanceCachedDenoiseState,
    latents: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Recompute only the dynamic VAE query against a compiled condition."""

    context = state.context
    if latents.ndim != 2 or latents.shape[1] != model.config.patch_latent_dim:
        raise LanceSamplingError("latents must have shape [tokens, patch_latent_dim]")
    if timestep.ndim != 0 or not 0.0 <= float(timestep.item()) <= 1.0:
        raise LanceSamplingError("timestep must be a scalar in [0, 1]")
    query_latents = latents[context.prediction_latent_indexes]
    query_timesteps = timestep.to(query_latents.dtype).expand(query_latents.shape[0])
    query = (
        model.vae2llm(query_latents)
        + model.time_embedder(query_timesteps)
        + model.latent_pos_embed(state.query_latent_position_ids)
    )
    hidden_states = model.forward_language_with_kv_cache(
        query,
        state.query_position_ids,
        state.query_understanding_indexes,
        state.query_generation_indexes,
        state.kv_cache,
        attention_backend=state.attention_backend,
        is_causal=state.is_causal,
    )
    return model.llm2vae(hidden_states)


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


def sample_native_lance_cached(
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
    attention_backend: KVAttentionBackend = reference_kv_sdpa,
) -> torch.Tensor:
    """Euler/CFG sampling with one KV prefill per active guidance branch."""

    contexts = {
        "conditional": context,
        "text_unconditional": text_unconditional_context,
        "vision_unconditional": vision_unconditional_context,
    }
    required = {"conditional"}
    if text_scale > 1.0 or vision_scale > 1.0:
        required.add("text_unconditional")
    if vision_scale > 1.0:
        required.add("vision_unconditional")
    states = {}
    for branch in required:
        branch_context = contexts[branch]
        if branch_context is None:
            raise LanceSamplingError("missing {} context".format(branch.replace("_", "-")))
        states[branch] = compile_native_kv_cache(
            model,
            branch_context,
            initial_latents,
            attention_backend=attention_backend,
        )

    def velocity(latents: torch.Tensor, timestep: torch.Tensor, branch: str) -> torch.Tensor:
        return predict_cached_native_velocity(model, states[branch], latents, timestep)

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
