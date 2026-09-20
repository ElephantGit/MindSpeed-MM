"""Native construction of Lance training samples from frozen media features.

Raw pretraining samples are assembled from independent modality segments, as
in BAGEL and Lance's template-free data path.  Every text segment is delimited
by ``<|im_start|> ... <|im_end|>`` and every visual segment by
``<|vision_start|> ... <|vision_end|>``.  Understanding CE starts at the
target text segment's ``<|im_start|>`` position, which predicts its first text
token; the final text token predicts ``<|im_end|>``.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
from einops import rearrange

from .data import LancePreparedSample, ce_length_weight
from .native_config import LanceNativeConfig
from .sequence import LanceSegment, flatten_latent_position_ids


@dataclass(frozen=True)
class LanceSpecialTokens:
    im_start: int
    im_end: int
    vision_start: int
    vision_end: int
    video_pad: int
    image_pad: int

    @classmethod
    def from_tokenizer(cls, tokenizer):
        names = (
            "<|im_start|>", "<|im_end|>", "<|vision_start|>",
            "<|vision_end|>", "<|video_pad|>", "<|image_pad|>",
        )
        values = [int(tokenizer.convert_tokens_to_ids(name)) for name in names]
        if any(value < 0 for value in values) or len(set(values)) != len(values):
            raise ValueError("Qwen tokenizer does not expose the required Lance special tokens")
        return cls(*values)


def prepare_lance_tokenizer(tokenizer):
    """Apply the same four-token extension used by Lance training."""

    known = []
    for value in tokenizer.special_tokens_map.values():
        known.extend(value if isinstance(value, list) else (value,))
    required = (
        "<|im_start|>", "<|im_end|>",
        "<|vision_start|>", "<|vision_end|>",
    )
    missing = [token for token in required if token not in known]
    if missing:
        tokenizer.add_tokens(missing)
    LanceSpecialTokens.from_tokenizer(tokenizer)
    return tokenizer


@dataclass(frozen=True)
class LanceEncodedVisual:
    modality: str
    vae_latent: Optional[torch.Tensor] = None
    vae_log_variance: Optional[torch.Tensor] = None
    vit_embedding: Optional[torch.Tensor] = None
    # Raw Qwen ViT patch grid before the 2x2 spatial merger.  It is retained
    # only long enough to build the LLM's multimodal RoPE positions.
    vit_grid_thw: Optional[tuple] = None

    def __post_init__(self):
        if self.modality not in ("image", "video"):
            raise ValueError("encoded visual modality must be image or video")
        if self.vae_latent is None and self.vit_embedding is None:
            raise ValueError("encoded visual requires a VAE latent or ViT embedding")
        if self.vae_latent is not None and self.vae_latent.ndim != 4:
            raise ValueError("VAE latent must use THWC layout")
        if self.vae_log_variance is not None:
            if self.vae_latent is None or self.vae_log_variance.shape != self.vae_latent.shape:
                raise ValueError("VAE log variance must match the THWC latent mean")
        if self.vit_embedding is not None and self.vit_embedding.ndim != 2:
            raise ValueError("ViT embedding must have shape [tokens, hidden]")
        if self.vit_grid_thw is not None:
            if self.vit_embedding is None:
                raise ValueError("vit_grid_thw requires a ViT embedding")
            if len(self.vit_grid_thw) != 3 or any(int(item) <= 0 for item in self.vit_grid_thw):
                raise ValueError("vit_grid_thw must contain three positive integers")


def _visual_mrope_grid(visual, encoder_kind, config, expected_tokens):
    """Return the post-merger T/H/W grid consumed by Qwen multimodal RoPE."""

    if encoder_kind == "vit":
        if visual.vit_grid_thw is None:
            # Synthetic/unit-test features do not originate from a pixel grid.
            # A flat grid preserves the exact token count without weakening the
            # production path, which always records vit_grid_thw.
            grid = (1, 1, int(expected_tokens))
        else:
            temporal, height, width = (int(item) for item in visual.vit_grid_thw)
            merge = config.vit_spatial_merge_size
            if height % merge or width % merge:
                raise ValueError("ViT grid does not align to spatial_merge_size")
            grid = (temporal, height // merge, width // merge)
    elif encoder_kind == "vae":
        temporal, height, width, _ = visual.vae_latent.shape
        patch_t, patch_h, patch_w = config.latent_patch_size
        grid = (temporal // patch_t, height // patch_h, width // patch_w)
    else:
        raise ValueError("visual encoder kind must be vit or vae")
    if grid[0] * grid[1] * grid[2] != int(expected_tokens):
        raise ValueError("visual MRoPE grid does not match its payload token count")
    return grid


def _qwen_mrope_positions(token_count, spans, visuals, config):
    """Build Qwen2.5-VL MRoPE plus Lance's 1000-step MaPE shifts.

    A span is ``(vision_start, payload_start, vision_end)``.  Qwen treats the
    boundary tokens as ordinary text positions and the payload as a T/H/W
    grid.  Lance then moves semantic ViT conditions to temporal band 1000 and
    gives an edit target the same positions as its clean VAE condition.
    """

    positions = torch.empty((3, token_count), dtype=torch.long)
    cursor = 0
    next_position = 0
    roles = []
    for (start, payload, end), (visual, encoder_kind, target, _) in zip(spans, visuals):
        text_length = payload - cursor
        if text_length <= 0:
            raise ValueError("visual payload must follow a vision-start boundary")
        text_positions = torch.arange(
            next_position, next_position + text_length, dtype=torch.long
        )
        positions[:, cursor:payload] = text_positions.unsqueeze(0).expand(3, -1)

        grid_t, grid_h, grid_w = _visual_mrope_grid(
            visual, encoder_kind, config, end - payload
        )
        temporal = torch.arange(grid_t, dtype=torch.long).view(-1, 1)
        temporal = temporal.expand(-1, grid_h * grid_w).reshape(-1) * 2
        height = torch.arange(grid_h, dtype=torch.long).view(1, -1, 1)
        height = height.expand(grid_t, -1, grid_w).reshape(-1)
        width = torch.arange(grid_w, dtype=torch.long).view(1, 1, -1)
        width = width.expand(grid_t, grid_h, -1).reshape(-1)
        visual_positions = torch.stack((temporal, height, width))
        visual_positions += next_position + text_length
        positions[:, payload:end] = visual_positions
        next_position = int(visual_positions.max().item()) + 1
        cursor = end
        roles.append({
            "kind": encoder_kind,
            "target": bool(target),
            "range": (start, end + 1),
        })

    if cursor < token_count:
        tail = torch.arange(
            next_position, next_position + token_count - cursor, dtype=torch.long
        )
        positions[:, cursor:] = tail.unsqueeze(0).expand(3, -1)

    vit_roles = [role for role in roles if role["kind"] == "vit"]
    if vit_roles:
        first = vit_roles[0]["range"][0]
        shift = 1000 - int(positions[0, first].item())
        for role in vit_roles:
            start, end = role["range"]
            positions[0, start:end] += shift

    conditions = [
        role for role in roles if role["kind"] == "vae" and not role["target"]
    ]
    targets = [
        role for role in roles if role["kind"] == "vae" and role["target"]
    ]
    if len(conditions) == 1 and len(targets) == 1:
        condition_start, condition_end = conditions[0]["range"]
        target_start, target_end = targets[0]["range"]
        if condition_end - condition_start == target_end - target_start:
            positions[:, target_start:target_end] = positions[:, condition_start:condition_end]
    return positions


def patchify_qwen_video(
    video: torch.Tensor,
    spatial_patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
) -> torch.Tensor:
    """Patchify CTHW pixels in the exact Qwen2.5-VL merger order."""

    temporal, channels, height, width = rearrange(video, "c t h w -> t c h w").shape
    patch = int(spatial_patch_size)
    temporal_patch = int(temporal_patch_size)
    merge = int(merge_size)
    if temporal % temporal_patch or height % (patch * merge) or width % (patch * merge):
        raise ValueError("ViT media dimensions do not align to temporal/spatial patch merging")
    grid_t, grid_h, grid_w = temporal // temporal_patch, height // patch, width // patch
    value = rearrange(video, "c t h w -> t c h w").reshape(
        grid_t, temporal_patch, channels,
        grid_h // merge, merge, patch,
        grid_w // merge, merge, patch,
    )
    value = value.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return value.reshape(grid_t * grid_h * grid_w, channels * temporal_patch * patch * patch)


class _SampleAssembler:
    def __init__(self, sample_id, tokenizer, config, tokens):
        self.sample_id = str(sample_id)
        self.tokenizer = tokenizer
        self.config = config
        self.tokens = tokens
        self.token_ids: List[int] = []
        self.text_indexes: List[int] = []
        self.segments: List[LanceSegment] = []
        self.vae_indexes: List[int] = []
        self.vae_latents: List[torch.Tensor] = []
        self.vae_log_variances: List[torch.Tensor] = []
        self.latent_positions: List[torch.Tensor] = []
        self.timesteps: List[torch.Tensor] = []
        self.vit_indexes: List[int] = []
        self.vit_embeddings: List[torch.Tensor] = []
        self.ce_indexes: List[int] = []
        self.ce_labels: List[int] = []
        self.ce_weights: List[float] = []
        self.mse_indexes: List[int] = []
        self.visual_spans = []
        self.visual_specs = []
        self.position_ids: Optional[torch.Tensor] = None

    def _add_vit_payload(self, visual: LanceEncodedVisual, payload: int) -> None:
        embedding = visual.vit_embedding
        if embedding is None:
            raise ValueError("ViT condition is missing its embedding")
        # Frozen Qwen ViT merger output is vit_out_hidden_size wide; when the
        # LLM hidden size differs the trainable connector projects it at
        # training time.  Both widths are therefore valid in packed data.
        allowed_widths = (self.config.hidden_size, self.config.vit_out_hidden_size)
        if embedding.shape[1] not in allowed_widths:
            raise ValueError(
                "ViT output width {} does not match the Lance ViT contract {}".format(
                    embedding.shape[1], allowed_widths
                )
            )
        self.vit_indexes.extend(range(payload, payload + embedding.shape[0]))
        self.vit_embeddings.append(embedding.to(device="cpu", dtype=torch.bfloat16))

    def _add_vae_payload(
        self, visual: LanceEncodedVisual, payload: int, *, target: bool,
        condition_frames=(),
    ) -> None:
        latent = visual.vae_latent
        if latent is None:
            raise ValueError("VAE visual is missing its latent")
        temporal, height, width, channels = latent.shape
        if channels != self.config.latent_channels:
            raise ValueError("VAE latent channel count does not match Lance")
        patch_t, patch_h, patch_w = self.config.latent_patch_size
        if temporal % patch_t or height % patch_h or width % patch_w:
            raise ValueError("VAE latent dimensions do not align to latent_patch_size")
        grid_t, grid_h, grid_w = temporal // patch_t, height // patch_h, width // patch_w
        if grid_h > self.config.max_latent_size or grid_w > self.config.max_latent_size:
            raise ValueError("VAE latent grid exceeds Lance positional table")
        flattened = latent.reshape(
            grid_t, patch_t, grid_h, patch_h, grid_w, patch_w, channels
        ).permute(0, 2, 4, 1, 3, 5, 6).reshape(
            grid_t * grid_h * grid_w, -1
        ).to(device="cpu", dtype=torch.bfloat16)
        indexes = list(range(payload, payload + flattened.shape[0]))
        condition_frames = tuple(int(index) for index in condition_frames)
        if condition_frames and not target:
            raise ValueError("condition_frames are valid only within a target VAE span")
        if any(index < 0 or index >= grid_t for index in condition_frames):
            raise ValueError("condition frame index exceeds the VAE temporal grid")
        tokens_per_frame = grid_h * grid_w
        condition_offsets = {
            index * tokens_per_frame + offset
            for index in condition_frames
            for offset in range(tokens_per_frame)
        }
        has_posterior = visual.vae_log_variance is not None
        if self.vae_latents and has_posterior != bool(self.vae_log_variances):
            raise ValueError("cannot mix posterior and fixed VAE latents in one sample")
        self.vae_indexes.extend(indexes)
        self.vae_latents.append(flattened)
        if visual.vae_log_variance is not None:
            log_variance = visual.vae_log_variance.reshape(
                grid_t, patch_t, grid_h, patch_h, grid_w, patch_w, channels
            ).permute(0, 2, 4, 1, 3, 5, 6).reshape(
                grid_t * grid_h * grid_w, -1
            ).to(device="cpu", dtype=torch.bfloat16)
            self.vae_log_variances.append(log_variance)
        self.latent_positions.append(
            torch.tensor(
                flatten_latent_position_ids(
                    grid_t, grid_h, grid_w, self.config.max_latent_size
                ),
                dtype=torch.long,
            )
        )
        # A positive sentinel marks target tokens.  First/last-frame video
        # conditioning remains clean (t=0) inside the same noise-attention span.
        timestep = torch.zeros(len(indexes))
        if target:
            target_offsets = [
                offset for offset in range(len(indexes))
                if offset not in condition_offsets
            ]
            timestep[target_offsets] = 0.5
            self.mse_indexes.extend(indexes[offset] for offset in target_offsets)
        self.timesteps.append(timestep)

    def add_text(self, text: str, *, target: bool = False) -> None:
        """Append one independently delimited raw text segment."""

        text_ids = list(self.tokenizer.encode(str(text), add_special_tokens=False))
        start = len(self.token_ids)
        segment_ids = [self.tokens.im_start] + text_ids + [self.tokens.im_end]
        self.token_ids.extend(segment_ids)
        self.text_indexes.extend(range(start, start + len(segment_ids)))
        self.segments.append(
            LanceSegment(len(segment_ids), "causal", "text", "understanding")
        )
        if target:
            labels = text_ids + [self.tokens.im_end]
            self.ce_indexes.extend(range(start, start + len(labels)))
            self.ce_labels.extend(labels)
            self.ce_weights.extend(
                [ce_length_weight(len(labels), "square")] * len(labels)
            )

    def add_visual(
        self,
        visual: LanceEncodedVisual,
        encoder_kind: str,
        *,
        target: bool = False,
        condition_frames=(),
    ) -> None:
        """Append one independently delimited ViT or VAE visual segment."""

        if encoder_kind == "vit":
            count = 0 if visual.vit_embedding is None else int(visual.vit_embedding.shape[0])
        elif encoder_kind == "vae":
            latent = visual.vae_latent
            if latent is None:
                count = 0
            else:
                patch_t, patch_h, patch_w = self.config.latent_patch_size
                count = (
                    int(latent.shape[0]) // patch_t
                    * (int(latent.shape[1]) // patch_h)
                    * (int(latent.shape[2]) // patch_w)
                )
        else:
            raise ValueError("visual encoder kind must be vit or vae")
        if count <= 0:
            raise ValueError("encoded visual span must contain at least one token")

        start = len(self.token_ids)
        payload = start + 1
        end = payload + count
        self.token_ids.extend(
            [self.tokens.vision_start]
            + [self.tokens.video_pad] * count
            + [self.tokens.vision_end]
        )
        self.text_indexes.extend((start, end))
        if encoder_kind == "vit":
            if target:
                raise ValueError("ViT spans cannot be flow-matching targets")
            self._add_vit_payload(visual, payload)
            mode, modality, expert = "full", "vit", "understanding"
        else:
            self._add_vae_payload(
                visual, payload, target=target, condition_frames=condition_frames
            )
            mode, modality, expert = (
                "noise" if target else "full_noise"
            ), "vae", "generation"
        self.segments.append(
            LanceSegment(count + 2, mode, modality, expert)
        )
        self.visual_spans.append((start, payload, end))
        self.visual_specs.append(
            (visual, encoder_kind, target, tuple(condition_frames))
        )

    def build(self) -> LancePreparedSample:
        length = len(self.token_ids)
        if not length:
            raise ValueError("one Lance prepared sample must contain at least one segment")
        self.position_ids = _qwen_mrope_positions(
            length, self.visual_spans, self.visual_specs, self.config
        )
        vae_indexes = torch.tensor(self.vae_indexes, dtype=torch.long) if self.vae_indexes else None
        vit_indexes = torch.tensor(self.vit_indexes, dtype=torch.long) if self.vit_indexes else None
        ce_indexes = torch.tensor(self.ce_indexes, dtype=torch.long) if self.ce_indexes else None
        mse_indexes = torch.tensor(self.mse_indexes, dtype=torch.long) if self.mse_indexes else None
        all_indexes = torch.arange(length, dtype=torch.long)
        generation = vae_indexes if vae_indexes is not None else torch.empty(0, dtype=torch.long)
        understanding = all_indexes[~torch.isin(all_indexes, generation)]
        return LancePreparedSample(
            sample_id=self.sample_id,
            segments=tuple(self.segments),
            token_ids=torch.tensor(self.token_ids, dtype=torch.long),
            text_indexes=torch.tensor(self.text_indexes, dtype=torch.long),
            position_ids=self.position_ids,
            understanding_indexes=understanding,
            generation_indexes=generation,
            vae_indexes=vae_indexes,
            clean_latents=torch.cat(self.vae_latents) if self.vae_latents else None,
            latent_log_variance=(
                torch.cat(self.vae_log_variances)
                if self.vae_log_variances else None
            ),
            latent_position_ids=torch.cat(self.latent_positions) if self.latent_positions else None,
            timesteps=torch.cat(self.timesteps) if self.timesteps else None,
            vit_indexes=vit_indexes,
            vit_embeddings=torch.cat(self.vit_embeddings) if self.vit_embeddings else None,
            ce_indexes=ce_indexes,
            ce_labels=torch.tensor(self.ce_labels, dtype=torch.long) if self.ce_labels else None,
            ce_weights=torch.tensor(self.ce_weights, dtype=torch.float32) if self.ce_weights else None,
            mse_indexes=mse_indexes,
        )


def build_generation_sample(
    sample_id, caption, target, tokenizer, config, *,
    condition_frames=(),
):
    assembler = _SampleAssembler(sample_id, tokenizer, config, LanceSpecialTokens.from_tokenizer(tokenizer))
    # BAGEL-style CFG dropout removes the complete caption segment.  It does
    # not leave an empty im_start/im_end pair in the unconditional branch.
    if caption is not None:
        assembler.add_text(str(caption), target=False)
    assembler.add_visual(
        target, "vae", target=True, condition_frames=condition_frames
    )
    return assembler.build()


def build_understanding_sample(
    sample_id, prompt, answer, condition, tokenizer, config
):
    assembler = _SampleAssembler(sample_id, tokenizer, config, LanceSpecialTokens.from_tokenizer(tokenizer))
    assembler.add_visual(condition, "vit", target=False)
    if prompt is not None and str(prompt).strip():
        assembler.add_text(str(prompt), target=False)
    assembler.add_text(str(answer), target=True)
    return assembler.build()


def build_edit_sample(
    sample_id, instruction, condition, target, tokenizer, config
):
    if condition.modality != target.modality:
        raise ValueError("edit condition and target modalities must match")
    assembler = _SampleAssembler(sample_id, tokenizer, config, LanceSpecialTokens.from_tokenizer(tokenizer))
    # Keep the existing edit ordering, but delimit every modality segment
    # independently: semantic and clean-latent conditions, instruction text,
    # then the noisy VAE target.
    assembler.add_visual(condition, "vit", target=False)
    assembler.add_visual(condition, "vae", target=False)
    assembler.add_text(str(instruction), target=False)
    assembler.add_visual(target, "vae", target=True)
    return assembler.build()
