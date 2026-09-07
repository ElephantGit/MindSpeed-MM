"""Native Lance joint CE/flow-matching training step.

The batch contract accepts pre-encoded ViT features and VAE latents.  This is the
paper training path: ViT and VAE remain frozen while the two MoT experts, bridge
projections, timestep embedder, LM head, and latent head are optimized.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .modeling_lance import LanceNativeModel
from .sequence import LancePackedSequence


class LanceTrainingError(ValueError):
    pass


@dataclass(frozen=True)
class LanceLossWeights:
    ce: float
    mse: float

    def __post_init__(self) -> None:
        if self.ce < 0 or self.mse < 0 or self.ce + self.mse <= 0:
            raise LanceTrainingError("loss weights must be non-negative and not both zero")


@dataclass
class LanceTrainingBatch:
    token_ids: torch.Tensor
    text_indexes: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: Any
    understanding_indexes: torch.Tensor
    generation_indexes: torch.Tensor

    vae_indexes: Optional[torch.Tensor] = None
    clean_latents: Optional[torch.Tensor] = None
    latent_position_ids: Optional[torch.Tensor] = None
    timesteps: Optional[torch.Tensor] = None
    noise: Optional[torch.Tensor] = None

    vit_indexes: Optional[torch.Tensor] = None
    vit_embeddings: Optional[torch.Tensor] = None

    ce_indexes: Optional[torch.Tensor] = None
    ce_labels: Optional[torch.Tensor] = None
    ce_weights: Optional[torch.Tensor] = None
    mse_indexes: Optional[torch.Tensor] = None

    @property
    def sequence_length(self) -> int:
        return int(self.token_ids.numel())

    def validate(self, model: LanceNativeModel) -> None:
        length = self.sequence_length
        hidden = model.config.hidden_size
        patch_dim = model.config.patch_latent_dim
        if self.token_ids.ndim != 1:
            raise LanceTrainingError("token_ids must be one-dimensional")
        if self.position_ids.shape != (3, length):
            raise LanceTrainingError("position_ids must have shape [3, sequence_length]")
        if isinstance(self.attention_mask, LancePackedSequence):
            if self.attention_mask.length != length:
                raise LanceTrainingError("packed attention metadata length does not match sequence")
        elif self.attention_mask is not None and self.attention_mask.shape != (length, length):
            raise LanceTrainingError("reference attention_mask must have shape [sequence_length, sequence_length]")
        for name, indexes in (
            ("text", self.text_indexes),
            ("understanding", self.understanding_indexes),
            ("generation", self.generation_indexes),
        ):
            _validate_indexes(name, indexes, length)
        routes = torch.cat((self.understanding_indexes, self.generation_indexes))
        if routes.numel() != length or torch.unique(routes).numel() != length:
            raise LanceTrainingError("expert indexes must cover every token exactly once")

        vae_fields = (self.vae_indexes, self.clean_latents, self.latent_position_ids, self.timesteps)
        if any(field is not None for field in vae_fields) and not all(field is not None for field in vae_fields):
            raise LanceTrainingError("VAE indexes, latents, positions, and timesteps must be provided together")
        if self.vae_indexes is not None:
            _validate_indexes("VAE", self.vae_indexes, length)
            count = self.vae_indexes.numel()
            if self.clean_latents.shape != (count, patch_dim):
                raise LanceTrainingError("clean_latents shape does not match VAE token count")
            if self.latent_position_ids.shape != (count,) or self.timesteps.shape != (count,):
                raise LanceTrainingError("latent positions/timesteps must match VAE token count")
            if self.noise is not None and self.noise.shape != self.clean_latents.shape:
                raise LanceTrainingError("noise shape must match clean_latents")

        if (self.vit_indexes is None) != (self.vit_embeddings is None):
            raise LanceTrainingError("ViT indexes and embeddings must be provided together")
        if self.vit_indexes is not None:
            _validate_indexes("ViT", self.vit_indexes, length)
            if self.vit_embeddings.shape != (self.vit_indexes.numel(), hidden):
                raise LanceTrainingError("vit_embeddings shape does not match ViT token count")

        ce_fields = (self.ce_indexes, self.ce_labels, self.ce_weights)
        if any(field is not None for field in ce_fields) and not all(field is not None for field in ce_fields):
            raise LanceTrainingError("CE indexes, labels, and weights must be provided together")
        if self.ce_indexes is not None:
            _validate_indexes("CE", self.ce_indexes, length)
            if self.ce_labels.shape != self.ce_indexes.shape or self.ce_weights.shape != self.ce_indexes.shape:
                raise LanceTrainingError("CE labels and weights must match CE indexes")
            if torch.any(self.ce_weights <= 0):
                raise LanceTrainingError("CE weights must be positive")
        if self.mse_indexes is not None:
            _validate_indexes("MSE", self.mse_indexes, length)
            if self.vae_indexes is None:
                raise LanceTrainingError("MSE loss requires VAE latents")
            if not set(self.mse_indexes.tolist()).issubset(set(self.vae_indexes.tolist())):
                raise LanceTrainingError("MSE indexes must be a subset of VAE token indexes")
        if self.ce_indexes is not None and self.mse_indexes is not None:
            if set(self.ce_indexes.tolist()) & set(self.mse_indexes.tolist()):
                raise LanceTrainingError("CE and MSE indexes must not overlap")


def _validate_indexes(name: str, indexes: torch.Tensor, length: int) -> None:
    if indexes.ndim != 1 or indexes.dtype != torch.long:
        raise LanceTrainingError("{} indexes must be a one-dimensional torch.long tensor".format(name))
    if indexes.numel() and (
        int(indexes.min().item()) < 0
        or int(indexes.max().item()) >= length
        or torch.unique(indexes).numel() != indexes.numel()
    ):
        raise LanceTrainingError("{} indexes contain duplicates or out-of-range values".format(name))


def shift_timesteps(timesteps: torch.Tensor, shift: float) -> torch.Tensor:
    if shift <= 0:
        raise LanceTrainingError("timestep shift must be positive")
    if torch.any(timesteps < 0) or torch.any(timesteps > 1):
        raise LanceTrainingError("timesteps must be in [0, 1]")
    return shift * timesteps / (1.0 + (shift - 1.0) * timesteps)


def _distributed_average(local_sum: torch.Tensor, local_weight: torch.Tensor) -> torch.Tensor:
    denominator = local_weight.detach().clone().to(device=local_sum.device, dtype=torch.float32)
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    if float(denominator.item()) <= 0:
        raise LanceTrainingError("global loss denominator must be positive")
    return local_sum * world_size / denominator


def lance_training_step(
    model: LanceNativeModel,
    batch: LanceTrainingBatch,
    loss_weights: LanceLossWeights,
    timestep_shift: float,
) -> Dict[str, Optional[torch.Tensor]]:
    """Run one joint understanding/generation step on a packed sequence."""

    batch.validate(model)
    embedded_tokens = model.language_model.model.embed_tokens(batch.token_ids)
    hidden_inputs = embedded_tokens.new_zeros((batch.sequence_length, model.config.hidden_size))
    hidden_inputs[batch.text_indexes] = embedded_tokens[batch.text_indexes]

    if batch.vit_indexes is not None:
        hidden_inputs[batch.vit_indexes] = batch.vit_embeddings.to(hidden_inputs.dtype)

    velocity_target = None
    shifted_timesteps = None
    if batch.vae_indexes is not None:
        noise = batch.noise if batch.noise is not None else torch.randn_like(batch.clean_latents)
        shifted_timesteps = shift_timesteps(batch.timesteps, timestep_shift)
        noisy_latents = (
            (1.0 - shifted_timesteps.unsqueeze(1)) * batch.clean_latents
            + shifted_timesteps.unsqueeze(1) * noise
        )
        velocity_target = noise - batch.clean_latents
        latent_embedding = (
            model.vae2llm(noisy_latents)
            + model.time_embedder(shifted_timesteps)
            + model.latent_pos_embed(batch.latent_position_ids)
        )
        hidden_inputs[batch.vae_indexes] = latent_embedding.to(hidden_inputs.dtype)

    hidden_states = model.forward_language(
        hidden_inputs,
        batch.position_ids,
        batch.attention_mask,
        batch.understanding_indexes,
        batch.generation_indexes,
    )

    ce_loss = None
    if batch.ce_indexes is not None and batch.ce_indexes.numel():
        logits = model.language_model.lm_head(hidden_states[batch.ce_indexes])
        per_token_ce = F.cross_entropy(logits.float(), batch.ce_labels, reduction="none")
        ce_sum = (per_token_ce * batch.ce_weights.float()).sum()
        ce_loss = _distributed_average(ce_sum, batch.ce_weights.float().sum())

    mse_loss = None
    if batch.mse_indexes is not None and batch.mse_indexes.numel():
        predictions = model.llm2vae(hidden_states[batch.mse_indexes])
        global_targets = predictions.new_zeros((batch.sequence_length, model.config.patch_latent_dim))
        global_targets[batch.vae_indexes] = velocity_target.to(predictions.dtype)
        per_token_mse = (predictions.float() - global_targets[batch.mse_indexes].float()).pow(2).mean(dim=-1)
        mse_loss = _distributed_average(
            per_token_mse.sum(),
            torch.tensor(per_token_mse.numel(), device=per_token_mse.device, dtype=torch.float32),
        )

    total_loss = hidden_states.sum() * 0.0
    if ce_loss is not None:
        total_loss = total_loss + loss_weights.ce * ce_loss
    if mse_loss is not None:
        total_loss = total_loss + loss_weights.mse * mse_loss
    if ce_loss is None and mse_loss is None:
        raise LanceTrainingError("a training batch must select CE or MSE loss tokens")

    return {
        "loss": total_loss,
        "ce_loss": ce_loss,
        "mse_loss": mse_loss,
        "hidden_states": hidden_states,
        "shifted_timesteps": shifted_timesteps,
        "velocity_target": velocity_target,
    }
