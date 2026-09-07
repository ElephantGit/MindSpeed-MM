"""Pre-encoded Lance sample schema and deterministic packed collator."""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import torch

from .native_config import LanceNativeConfig
from .sequence import (
    LanceDocument,
    LanceLossSelection,
    LancePackedSequence,
    LanceSegment,
    LanceSequenceError,
)
from .training_lance import LanceTrainingBatch


class LanceDataError(ValueError):
    pass


def ce_length_weight(length: int, reduction: str = "square") -> float:
    """Match Lance ``len2weight`` for token/sample/square reduction."""

    if length < 0:
        raise LanceDataError("CE span length must be non-negative")
    if length == 0:
        return 0.0
    if reduction == "token":
        return 1.0
    if reduction == "sample":
        return 1.0 / length
    if reduction == "square":
        return 1.0 / math.sqrt(length)
    raise LanceDataError("unsupported CE reduction: {}".format(reduction))


@dataclass
class LancePreparedSample:
    """One tokenizer/ViT/VAE-prepared document before cross-sample packing."""

    sample_id: str
    segments: Tuple[LanceSegment, ...]
    token_ids: torch.Tensor
    text_indexes: torch.Tensor
    position_ids: torch.Tensor

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
    def length(self) -> int:
        return int(self.token_ids.numel())

    @property
    def document(self) -> LanceDocument:
        return LanceDocument(self.sample_id, self.segments)

    def validate(self, config: LanceNativeConfig) -> None:
        if self.token_ids.ndim != 1 or self.token_ids.dtype != torch.long:
            raise LanceDataError("token_ids must be a one-dimensional torch.long tensor")
        if not self.length or sum(segment.length for segment in self.segments) != self.length:
            raise LanceDataError("segment lengths must equal token count")
        if self.position_ids.shape != (3, self.length) or self.position_ids.dtype != torch.long:
            raise LanceDataError("position_ids must be torch.long with shape [3, token_count]")
        if torch.any(self.token_ids < 0) or torch.any(self.token_ids >= config.vocab_size):
            raise LanceDataError("token_ids contain values outside the configured vocabulary")

        text = _optional_indexes("text", self.text_indexes, self.length, required=True)
        vae = _optional_indexes("VAE", self.vae_indexes, self.length)
        vit = _optional_indexes("ViT", self.vit_indexes, self.length)
        input_indexes = torch.cat((text, vae, vit))
        if input_indexes.numel() != self.length or torch.unique(input_indexes).numel() != self.length:
            raise LanceDataError("text, VAE, and ViT indexes must cover every token exactly once")

        vae_fields = (self.clean_latents, self.latent_position_ids, self.timesteps)
        if self.vae_indexes is None:
            if any(field is not None for field in vae_fields + (self.noise,)):
                raise LanceDataError("VAE tensors require vae_indexes")
        elif not all(field is not None for field in vae_fields):
            raise LanceDataError("VAE indexes, latents, positions, and timesteps are required together")
        else:
            count = vae.numel()
            expected_latent_shape = (count, config.patch_latent_dim)
            if self.clean_latents.shape != expected_latent_shape:
                raise LanceDataError("clean_latents shape does not match VAE indexes")
            if self.latent_position_ids.shape != (count,) or self.latent_position_ids.dtype != torch.long:
                raise LanceDataError("latent_position_ids must map each VAE token")
            if self.timesteps.shape != (count,):
                raise LanceDataError("timesteps must map each VAE token")
            if torch.any(self.timesteps < 0) or torch.any(self.timesteps > 1):
                raise LanceDataError("timesteps must be in [0, 1]")
            if torch.any(self.latent_position_ids < 0) or torch.any(
                self.latent_position_ids >= config.latent_position_count
            ):
                raise LanceDataError("latent_position_ids exceed the configured position table")
            if self.noise is not None and self.noise.shape != expected_latent_shape:
                raise LanceDataError("noise shape does not match clean_latents")

        if (self.vit_indexes is None) != (self.vit_embeddings is None):
            raise LanceDataError("ViT indexes and embeddings must be provided together")
        if self.vit_indexes is not None and self.vit_embeddings.shape != (
            vit.numel(),
            config.hidden_size,
        ):
            raise LanceDataError("vit_embeddings shape does not match ViT indexes")

        ce = _optional_indexes("CE", self.ce_indexes, self.length)
        mse = _optional_indexes("MSE", self.mse_indexes, self.length)
        ce_fields = (self.ce_indexes, self.ce_labels, self.ce_weights)
        if any(field is not None for field in ce_fields) and not all(field is not None for field in ce_fields):
            raise LanceDataError("CE indexes, labels, and weights must be provided together")
        if self.ce_indexes is not None:
            if self.ce_labels.shape != ce.shape or self.ce_labels.dtype != torch.long:
                raise LanceDataError("CE labels must be torch.long and match CE indexes")
            if self.ce_weights.shape != ce.shape or torch.any(self.ce_weights <= 0):
                raise LanceDataError("CE weights must be positive and match CE indexes")
            if torch.any(self.ce_labels < 0) or torch.any(self.ce_labels >= config.vocab_size):
                raise LanceDataError("CE labels contain values outside the configured vocabulary")
        if self.mse_indexes is not None:
            if self.vae_indexes is None:
                raise LanceDataError("MSE indexes require VAE inputs")
            membership = torch.isin(mse, vae)
            if not bool(torch.all(membership)):
                raise LanceDataError("MSE indexes must be a subset of VAE indexes")
        try:
            LanceLossSelection(
                tuple(ce.tolist()),
                tuple(self.ce_labels.tolist()) if self.ce_labels is not None else (),
                tuple(float(item) for item in self.ce_weights.tolist()) if self.ce_weights is not None else (),
                tuple(mse.tolist()),
            ).validate(self.length)
        except LanceSequenceError as exc:
            raise LanceDataError(str(exc)) from exc


def _optional_indexes(
    name: str,
    indexes: Optional[torch.Tensor],
    length: int,
    *,
    required: bool = False,
) -> torch.Tensor:
    if indexes is None:
        if required:
            raise LanceDataError("{} indexes are required".format(name))
        return torch.empty(0, dtype=torch.long)
    if indexes.ndim != 1 or indexes.dtype != torch.long:
        raise LanceDataError("{} indexes must be one-dimensional torch.long".format(name))
    if indexes.numel() and (
        int(indexes.min().item()) < 0
        or int(indexes.max().item()) >= length
        or torch.unique(indexes).numel() != indexes.numel()
    ):
        raise LanceDataError("{} indexes contain duplicates or out-of-range values".format(name))
    return indexes


@dataclass(frozen=True)
class LancePackedTrainingData:
    batch: LanceTrainingBatch
    packed_sequence: LancePackedSequence
    sample_ids: Tuple[str, ...]


def pack_preencoded_samples(
    samples: Sequence[LancePreparedSample],
    config: LanceNativeConfig,
    *,
    max_tokens: int,
    attention_backend: str = "ascend",
) -> LancePackedTrainingData:
    """Pack prepared samples without crossing document attention boundaries."""

    if not samples:
        raise LanceDataError("at least one prepared sample is required")
    if max_tokens <= 0:
        raise LanceDataError("max_tokens must be positive")
    if attention_backend not in ("ascend", "reference"):
        raise LanceDataError("attention_backend must be 'ascend' or 'reference'")
    for sample in samples:
        sample.validate(config)
    packed_sequence = LancePackedSequence(tuple(sample.document for sample in samples))
    if packed_sequence.length > max_tokens:
        raise LanceDataError(
            "packed token count {} exceeds max_tokens {}".format(packed_sequence.length, max_tokens)
        )

    offsets = []
    cursor = 0
    for sample in samples:
        offsets.append(cursor)
        cursor += sample.length

    token_ids = torch.cat([sample.token_ids for sample in samples])
    position_ids = torch.cat([sample.position_ids for sample in samples], dim=1)
    text_indexes = _pack_indexes(samples, offsets, "text_indexes")
    routes = packed_sequence.token_expert_indexes()
    understanding_indexes = torch.tensor(routes["understanding"], dtype=torch.long)
    generation_indexes = torch.tensor(routes["generation"], dtype=torch.long)

    vae_samples = [sample for sample in samples if sample.vae_indexes is not None]
    vae_indexes = _pack_indexes(samples, offsets, "vae_indexes") if vae_samples else None
    clean_latents = torch.cat([sample.clean_latents for sample in vae_samples]) if vae_samples else None
    latent_position_ids = (
        torch.cat([sample.latent_position_ids for sample in vae_samples]) if vae_samples else None
    )
    timesteps = torch.cat([sample.timesteps for sample in vae_samples]) if vae_samples else None
    explicit_noise = [sample.noise is not None for sample in vae_samples]
    if explicit_noise and any(explicit_noise) and not all(explicit_noise):
        raise LanceDataError("explicit noise must be provided for every packed VAE sample or none")
    noise = torch.cat([sample.noise for sample in vae_samples]) if explicit_noise and all(explicit_noise) else None

    vit_samples = [sample for sample in samples if sample.vit_indexes is not None]
    vit_indexes = _pack_indexes(samples, offsets, "vit_indexes") if vit_samples else None
    vit_embeddings = torch.cat([sample.vit_embeddings for sample in vit_samples]) if vit_samples else None

    ce_samples = [sample for sample in samples if sample.ce_indexes is not None]
    ce_indexes = _pack_indexes(samples, offsets, "ce_indexes") if ce_samples else None
    ce_labels = torch.cat([sample.ce_labels for sample in ce_samples]) if ce_samples else None
    ce_weights = torch.cat([sample.ce_weights for sample in ce_samples]) if ce_samples else None
    mse_samples = [sample for sample in samples if sample.mse_indexes is not None]
    mse_indexes = _pack_indexes(samples, offsets, "mse_indexes") if mse_samples else None

    if attention_backend == "reference":
        attention_metadata = torch.tensor(packed_sequence.dense_attention_mask(), dtype=torch.bool)
    else:
        attention_metadata = packed_sequence
    batch = LanceTrainingBatch(
        token_ids=token_ids,
        text_indexes=text_indexes,
        position_ids=position_ids,
        attention_mask=attention_metadata,
        understanding_indexes=understanding_indexes,
        generation_indexes=generation_indexes,
        vae_indexes=vae_indexes,
        clean_latents=clean_latents,
        latent_position_ids=latent_position_ids,
        timesteps=timesteps,
        noise=noise,
        vit_indexes=vit_indexes,
        vit_embeddings=vit_embeddings,
        ce_indexes=ce_indexes,
        ce_labels=ce_labels,
        ce_weights=ce_weights,
        mse_indexes=mse_indexes,
    )
    return LancePackedTrainingData(
        batch=batch,
        packed_sequence=packed_sequence,
        sample_ids=tuple(sample.sample_id for sample in samples),
    )


def _pack_indexes(
    samples: Sequence[LancePreparedSample],
    offsets: Sequence[int],
    field: str,
) -> torch.Tensor:
    values = []
    for sample, offset in zip(samples, offsets):
        indexes = getattr(sample, field)
        if indexes is not None:
            values.append(indexes + offset)
    return torch.cat(values) if values else torch.empty(0, dtype=torch.long)
