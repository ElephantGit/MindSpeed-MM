"""Pre-encoded Lance sample schema and deterministic packed collator."""

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence, Tuple

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

    # Attention segmentation and MoT routing are orthogonal in upstream
    # Lance: one full_noise segment can contain understanding boundary tokens
    # around generation VAE tokens.  Explicit routes preserve that layout.
    understanding_indexes: Optional[torch.Tensor] = None
    generation_indexes: Optional[torch.Tensor] = None

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

        if (self.understanding_indexes is None) != (self.generation_indexes is None):
            raise LanceDataError("explicit understanding/generation routes must be provided together")
        if self.understanding_indexes is not None:
            understanding = _optional_indexes(
                "understanding", self.understanding_indexes, self.length, required=True
            )
            generation = _optional_indexes(
                "generation", self.generation_indexes, self.length, required=True
            )
            routes = torch.cat((understanding, generation))
            if routes.numel() != self.length or torch.unique(routes).numel() != self.length:
                raise LanceDataError("explicit expert routes must cover every token exactly once")

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
    explicit_routes = [sample.understanding_indexes is not None for sample in samples]
    if any(explicit_routes) and not all(explicit_routes):
        raise LanceDataError("packed samples must all use explicit expert routes or all use segment routes")
    if all(explicit_routes):
        understanding_indexes = _pack_indexes(samples, offsets, "understanding_indexes")
        generation_indexes = _pack_indexes(samples, offsets, "generation_indexes")
    else:
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


def _as_mapping(batch: Any) -> Mapping[str, Any]:
    if isinstance(batch, Mapping):
        return batch
    if hasattr(batch, "to_dict"):
        value = batch.to_dict()
        if isinstance(value, Mapping):
            return value
    raise LanceDataError("upstream batch must be a mapping or expose to_dict()")


def _split_documents(
    sample_lens: Sequence[int],
    split_lens: Sequence[int],
    attention_modes: Sequence[str],
    vae_indexes: torch.Tensor,
    vit_indexes: torch.Tensor,
    sample_ids: Optional[Sequence[str]],
) -> Tuple[LanceDocument, ...]:
    if len(split_lens) != len(attention_modes):
        raise LanceDataError("split_lens and attn_modes must have equal length")
    documents = []
    if sample_ids is not None and len(sample_ids) != len(sample_lens):
        raise LanceDataError("sample_ids must match sample_lens")
    split_cursor = 0
    token_cursor = 0
    for document_index, sample_length in enumerate(sample_lens):
        consumed = 0
        segments = []
        while consumed < sample_length and split_cursor < len(split_lens):
            split_length = int(split_lens[split_cursor])
            if split_length <= 0 or consumed + split_length > sample_length:
                raise LanceDataError("attention splits do not align to sample lengths")
            start = token_cursor + consumed
            end = start + split_length
            positions = torch.arange(start, end, device=vae_indexes.device)
            has_vae = bool(torch.isin(positions, vae_indexes).any())
            has_vit = bool(torch.isin(positions, vit_indexes).any())
            modality = "vae" if has_vae else "vit" if has_vit else "text"
            # This nominal segment expert is used only by legacy callers;
            # adapt_upstream_lance_batch installs exact per-token routes.
            only_vae = has_vae and not bool((~torch.isin(positions, vae_indexes)).any())
            expert = "generation" if only_vae else "understanding"
            segments.append(
                LanceSegment(
                    split_length,
                    str(attention_modes[split_cursor]),
                    modality,
                    expert,
                )
            )
            consumed += split_length
            split_cursor += 1
        if consumed != sample_length:
            raise LanceDataError("attention splits do not cover every sample")
        sample_id = sample_ids[document_index] if sample_ids is not None else "sample-{}".format(document_index)
        documents.append(LanceDocument(str(sample_id), tuple(segments)))
        token_cursor += sample_length
    if split_cursor != len(split_lens):
        raise LanceDataError("attention splits remain after all samples")
    return tuple(documents)


def _patchify_upstream_latents(
    padded_latents: Any,
    latent_shapes: Sequence[Sequence[int]],
    patch_size: Tuple[int, int, int],
) -> torch.Tensor:
    if padded_latents is None:
        raise LanceDataError("upstream generation batch requires VAE-encoded padded_latent")
    pt, ph, pw = patch_size
    patches = []
    for index, shape in enumerate(latent_shapes):
        if len(shape) != 3:
            raise LanceDataError("patchified_vae_latent_shapes entries must be (t, h, w)")
        t, h, w = (int(value) for value in shape)
        latent = padded_latents[index]
        expected = (t * pt, h * ph, w * pw)
        if latent.ndim != 4 or any(latent.shape[axis] < size for axis, size in enumerate(expected)):
            raise LanceDataError("padded latent is smaller than its declared patchified shape")
        latent = latent[: expected[0], : expected[1], : expected[2]]
        latent = latent.reshape(t, pt, h, ph, w, pw, latent.shape[-1])
        patches.append(latent.permute(0, 2, 4, 1, 3, 5, 6).reshape(t * h * w, -1))
    return torch.cat(patches, dim=0)


def adapt_upstream_lance_batch(
    upstream_batch: Any,
    config: LanceNativeConfig,
    *,
    attention_backend: str = "ascend",
    vit_embeddings: Optional[torch.Tensor] = None,
    sample_ids: Optional[Sequence[str]] = None,
    timesteps_are_logits: bool = True,
) -> LancePackedTrainingData:
    """Convert official Lance ``PackedDataset`` output after VAE/ViT encoding.

    Online VAE pixels must first be encoded into ``padded_latent`` using the
    frozen Wan2.2 VAE.  Online ViT patches may be supplied as already merged
    ``vit_embeddings``; offline batches whose stored tokens already have the
    LLM hidden size are accepted directly.
    """

    raw = _as_mapping(upstream_batch)
    token_ids = raw["packed_text_ids"].long()
    sequence_length = int(raw.get("sequence_length", token_ids.numel()))
    if token_ids.ndim != 1 or token_ids.numel() != sequence_length:
        raise LanceDataError("packed_text_ids must cover sequence_length")
    sample_lens = tuple(int(value) for value in raw["sample_lens"])
    if sum(sample_lens) != sequence_length:
        raise LanceDataError("sample_lens must sum to sequence_length")

    text_indexes = raw["packed_text_indexes"].long()
    vae_indexes = raw.get("packed_vae_token_indexes")
    vae_indexes = vae_indexes.long() if vae_indexes is not None else torch.empty(0, dtype=torch.long)
    vit_indexes = raw.get("packed_vit_token_indexes")
    vit_indexes = vit_indexes.long() if vit_indexes is not None else torch.empty(0, dtype=torch.long)
    documents = _split_documents(
        sample_lens,
        raw["split_lens"],
        raw["attn_modes"],
        vae_indexes,
        vit_indexes,
        sample_ids,
    )
    packed_sequence = LancePackedSequence(documents)
    if attention_backend == "ascend":
        attention_metadata = packed_sequence
    elif attention_backend == "reference":
        attention_metadata = torch.tensor(packed_sequence.dense_attention_mask(), dtype=torch.bool)
    else:
        raise LanceDataError("attention_backend must be 'ascend' or 'reference'")

    position_ids = raw["packed_position_ids"].long()
    if position_ids.ndim == 1:
        position_ids = position_ids.repeat(3, 1)
    elif position_ids.shape == (sequence_length, 3):
        position_ids = position_ids.transpose(0, 1).contiguous()
    if position_ids.shape != (3, sequence_length):
        raise LanceDataError("packed_position_ids must be [L], [L,3], or [3,L]")

    all_indexes = torch.arange(sequence_length, device=vae_indexes.device)
    understanding_indexes = all_indexes[~torch.isin(all_indexes, vae_indexes)]
    generation_indexes = vae_indexes

    clean_latents = None
    latent_position_ids = None
    timesteps = None
    if vae_indexes.numel():
        clean_latents = _patchify_upstream_latents(
            raw.get("padded_latent"),
            raw["patchified_vae_latent_shapes"],
            config.latent_patch_size,
        )
        if clean_latents.shape != (vae_indexes.numel(), config.patch_latent_dim):
            raise LanceDataError("patchified VAE latents do not match packed VAE indexes")
        latent_position_ids = raw["packed_latent_position_ids"].long()
        raw_timesteps = raw["packed_timesteps"].to(dtype=torch.float32)
        timesteps = raw_timesteps.sigmoid() if timesteps_are_logits else raw_timesteps

    if vit_indexes.numel():
        if vit_embeddings is None:
            stored = raw.get("packed_vit_tokens")
            if isinstance(stored, (list, tuple)) and stored:
                stored = torch.cat(stored, dim=0)
            if isinstance(stored, torch.Tensor) and stored.shape == (
                vit_indexes.numel(),
                config.hidden_size,
            ):
                vit_embeddings = stored
            else:
                raise LanceDataError(
                    "raw ViT patches require frozen LanceNativeModel.vit_model encoding before adaptation"
                )

    ce_indexes = raw.get("ce_loss_indexes")
    ce_labels = raw.get("packed_label_ids")
    ce_weights = raw.get("ce_loss_weights")
    mse_indexes = raw.get("mse_loss_indexes")
    batch = LanceTrainingBatch(
        token_ids=token_ids,
        text_indexes=text_indexes,
        position_ids=position_ids,
        attention_mask=attention_metadata,
        understanding_indexes=understanding_indexes,
        generation_indexes=generation_indexes,
        vae_indexes=vae_indexes if vae_indexes.numel() else None,
        clean_latents=clean_latents,
        latent_position_ids=latent_position_ids,
        timesteps=timesteps,
        vit_indexes=vit_indexes if vit_indexes.numel() else None,
        vit_embeddings=vit_embeddings,
        ce_indexes=ce_indexes.long() if ce_indexes is not None else None,
        ce_labels=ce_labels.long() if ce_labels is not None else None,
        ce_weights=ce_weights.float() if ce_weights is not None else None,
        mse_indexes=mse_indexes.long() if mse_indexes is not None else None,
    )
    return LancePackedTrainingData(
        batch=batch,
        packed_sequence=packed_sequence,
        sample_ids=tuple(document.sample_id for document in documents),
    )


@torch.no_grad()
def encode_upstream_vit_embeddings(
    model: Any,
    upstream_batch: Any,
) -> Optional[torch.Tensor]:
    """Encode official online ViT patches while preserving offline features."""

    raw = _as_mapping(upstream_batch)
    indexes = raw.get("packed_vit_token_indexes")
    if indexes is None or not indexes.numel():
        return None
    if not hasattr(model, "vit_model"):
        raise LanceDataError("this Lance variant has no native ViT")
    tokens = raw.get("packed_vit_tokens")
    modes = raw.get("vit_data_mode")
    grids = raw.get("vit_video_grid_thw")
    if not isinstance(tokens, (list, tuple)) or not tokens or len(tokens) != len(modes or []):
        raise LanceDataError("packed_vit_tokens and vit_data_mode must describe every ViT item")
    if grids is None or len(grids) != len(tokens):
        raise LanceDataError("vit_video_grid_thw must describe every ViT item")
    embeddings = []
    for item, mode, grid in zip(tokens, modes, grids):
        if mode == "online":
            grid_tensor = torch.as_tensor(grid, dtype=torch.long, device=item.device).reshape(1, 3)
            embeddings.append(model.vit_model(item, grid_tensor))
        elif mode == "offline":
            if item.ndim != 2 or item.shape[1] != model.config.hidden_size:
                raise LanceDataError("offline ViT features must already use the LLM hidden size")
            embeddings.append(item)
        else:
            raise LanceDataError("vit_data_mode must contain only 'online' or 'offline'")
    result = torch.cat(embeddings, dim=0)
    if result.shape != (indexes.numel(), model.config.hidden_size):
        raise LanceDataError("encoded ViT feature count does not match packed ViT indexes")
    return result


@torch.no_grad()
def encode_upstream_vae_latents(
    upstream_batch: Any,
    vae_encoder: Any,
) -> Optional[Sequence[torch.Tensor]]:
    """Resolve mixed online pixels/offline latents from official PackedDataset."""

    raw = _as_mapping(upstream_batch)
    indexes = raw.get("packed_vae_token_indexes")
    if indexes is None or not indexes.numel():
        return None
    values = raw.get("padded_videos")
    modes = raw.get("vae_data_mode")
    if not isinstance(values, (list, tuple)) or len(values) != len(modes or []):
        raise LanceDataError("padded_videos and vae_data_mode must describe every VAE item")
    encode = vae_encoder.vae_encode if hasattr(vae_encoder, "vae_encode") else vae_encoder
    if not callable(encode):
        raise LanceDataError("VAE encoder must be callable or expose vae_encode")
    latents = []
    for value, mode in zip(values, modes):
        if mode == "online":
            encoded = encode([value])
            if not isinstance(encoded, (list, tuple)) or len(encoded) != 1:
                raise LanceDataError("VAE encoder must return one latent per input")
            latents.append(encoded[0])
        elif mode == "offline":
            latents.append(value)
        else:
            raise LanceDataError("vae_data_mode must contain only 'online' or 'offline'")
    return latents


def prepare_upstream_lance_batch(
    upstream_batch: Any,
    model: Any,
    *,
    vae_encoder: Any = None,
    attention_backend: str = "ascend",
    sample_ids: Optional[Sequence[str]] = None,
    timesteps_are_logits: bool = True,
) -> LancePackedTrainingData:
    """End-to-end adapter from official PackedDataset output to native training."""

    raw = dict(_as_mapping(upstream_batch))
    vae_indexes = raw.get("packed_vae_token_indexes")
    if vae_indexes is not None and vae_indexes.numel() and raw.get("padded_latent") is None:
        if vae_encoder is None:
            raise LanceDataError("online VAE inputs require a frozen VAE encoder")
        raw["padded_latent"] = encode_upstream_vae_latents(raw, vae_encoder)
    vit_embeddings = encode_upstream_vit_embeddings(model, raw)
    return adapt_upstream_lance_batch(
        raw,
        model.config,
        attention_backend=attention_backend,
        vit_embeddings=vit_embeddings,
        sample_ids=sample_ids,
        timesteps_are_logits=timesteps_are_logits,
    )
