"""Native construction of Lance training samples from frozen media features.

The prompt renderer and visual-span expansion below are a dependency-free port
of Lance's Apache-2.0 ``system_prompt_render.py`` semantics.  In particular,
the official training path uses Qwen's video-pad token for both image and video
spans and applies CE only after the ``assistant\n`` marker.
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


_CAPTION_SYSTEM_PROMPTS = (
    "Generate a detailed and accurate description of the {vision}, including all the key moments and visual details.",
    "Write an in-depth depiction of the {vision}, covering all its aspects.",
    "Write an exhaustive depiction of the given {vision}, capturing its essence and key moments.",
    "Describe the key features of the input {vision}, including color, shape, size, texture, objects, background.",
)


def lance_system_prompt(prompt_type: str, vision_type: str, choice: int = 0) -> str:
    """Return one of the system prompts used by the official PT datasets."""

    if prompt_type == "caption":
        candidates = _CAPTION_SYSTEM_PROMPTS
    elif prompt_type in ("t2v", "i2v"):
        candidates = (
            "Describe the {vision} by detailing the color, quantity, visible text, shape, size, texture, "
            "spatial relationships and motion/camera movements of the objects and background:",
        )
    elif prompt_type == "t2i":
        candidates = (
            "Describe the {vision} by detailing the color, quantity, text, shape, size, texture, "
            "spatial relationships of the objects and background:",
        )
    elif "edit" in prompt_type:
        candidates = (
            "Describe the key features of the input {vision} (color, shape, size, texture, objects, "
            "background), then explain how the user’s text instruction should alter or modify the "
            "{vision}. Generate a new {vision} that meets the user’s requirements while maintaining "
            "consistency with the original input where appropriate.",
        )
    else:
        raise ValueError("unsupported Lance system-prompt type: {}".format(prompt_type))
    return candidates[int(choice) % len(candidates)].format(vision=vision_type)


def _render_chat(
    system_prompt: str,
    user_content: str,
    assistant_content: str,
    *,
    close_assistant: bool = True,
) -> str:
    """Render the exact two-turn Qwen template used by Lance PT."""

    assistant_end = "<|im_end|>" if close_assistant else ""
    return (
        "<|im_start|>system\n{}<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n{}{}"
    ).format(system_prompt, user_content, assistant_content, assistant_end)


def _visual_placeholder(modality: str) -> str:
    # The official renderer intentionally uses video_pad for image as well.
    if modality not in ("image", "video"):
        raise ValueError("visual placeholder modality must be image or video")
    return "<|vision_start|><|video_pad|><|vision_end|>"


def _find_subsequence(values: List[int], needle: List[int], *, reverse: bool = False) -> int:
    if not needle:
        raise ValueError("cannot search for an empty token subsequence")
    candidates = range(len(values) - len(needle), -1, -1) if reverse else range(
        len(values) - len(needle) + 1
    )
    for start in candidates:
        if values[start:start + len(needle)] == needle:
            return start
    raise ValueError("Qwen assistant marker was not found in rendered prompt")


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
        self.position_ids: Optional[torch.Tensor] = None

    def _add_vit_payload(self, visual: LanceEncodedVisual, payload: int) -> None:
        embedding = visual.vit_embedding
        if embedding is None:
            raise ValueError("ViT condition is missing its embedding")
        if embedding.shape[1] != self.config.hidden_size:
            raise ValueError("ViT output width does not match Lance hidden size")
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

    def add_chat(
        self,
        system_prompt: str,
        user_content: str,
        assistant_content: str,
        visuals,
        *,
        assistant_ce: bool,
        close_assistant: bool = True,
    ) -> None:
        """Tokenize one official Lance chat template and install media spans.

        ``visuals`` is ordered by placeholder occurrence and contains
        ``(visual, encoder_kind, is_target)`` tuples.  Encoder kind is ``vit``
        or ``vae``; only target VAE spans select flow-matching loss.
        """

        if self.token_ids:
            raise ValueError("one Lance prepared sample must contain exactly one rendered chat")
        placeholder = _visual_placeholder("video")
        rendered = _render_chat(
            system_prompt,
            user_content,
            assistant_content,
            close_assistant=close_assistant,
        )
        if rendered.count(placeholder) != len(visuals):
            raise ValueError("rendered visual placeholders do not match encoded visual inputs")
        normalized_visuals = []
        for item in visuals:
            if len(item) == 3:
                visual, encoder_kind, target = item
                condition_frames = ()
            elif len(item) == 4:
                visual, encoder_kind, target, condition_frames = item
            else:
                raise ValueError("visual entries must contain 3 or 4 fields")
            normalized_visuals.append(
                (visual, encoder_kind, target, condition_frames)
            )
        for visual, encoder_kind, _, _ in normalized_visuals:
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
            expanded = (
                "<|vision_start|>" + "<|video_pad|>" * count + "<|vision_end|>"
            )
            rendered = rendered.replace(placeholder, expanded, 1)

        ids = list(self.tokenizer.encode(rendered.strip(), add_special_tokens=False))
        self.token_ids.extend(ids)
        spans = []
        cursor = 0
        while cursor < len(ids):
            try:
                start = ids.index(self.tokens.vision_start, cursor)
            except ValueError:
                break
            end = start + 1
            while end < len(ids) and ids[end] == self.tokens.video_pad:
                end += 1
            if end == start + 1 or end >= len(ids) or ids[end] != self.tokens.vision_end:
                raise ValueError("rendered prompt contains a malformed visual token span")
            spans.append((start, start + 1, end))
            cursor = end + 1
        if len(spans) != len(visuals):
            raise ValueError("tokenized visual spans do not match encoded visual inputs")
        self.position_ids = _qwen_mrope_positions(
            len(ids), spans, normalized_visuals, self.config
        )

        cursor = 0
        for (start, payload, end), (visual, encoder_kind, target, condition_frames) in zip(
            spans, normalized_visuals
        ):
            if start > cursor:
                length = start - cursor
                self.text_indexes.extend(range(cursor, start))
                self.segments.append(LanceSegment(length, "causal", "text", "understanding"))
            length = end - start + 1
            self.text_indexes.extend((start, end))
            if encoder_kind == "vit":
                if target:
                    raise ValueError("ViT spans cannot be flow-matching targets")
                self._add_vit_payload(visual, payload)
                mode, modality, expert = "full", "vit", "understanding"
            else:
                self._add_vae_payload(
                    visual, payload, target=target,
                    condition_frames=condition_frames,
                )
                mode, modality, expert = (
                    "noise" if target else "full_noise"
                ), "vae", "generation"
            self.segments.append(LanceSegment(length, mode, modality, expert))
            cursor = end + 1
        if cursor < len(ids):
            length = len(ids) - cursor
            self.text_indexes.extend(range(cursor, len(ids)))
            self.segments.append(LanceSegment(length, "causal", "text", "understanding"))

        if assistant_ce:
            marker = list(self.tokenizer.encode(
                "<|im_start|>assistant\n", add_special_tokens=False
            ))
            target_start = _find_subsequence(ids, marker, reverse=True) + len(marker)
            labels = ids[target_start:]
            if not labels:
                raise ValueError("understanding template has no assistant target tokens")
            self.ce_indexes.extend(range(target_start - 1, len(ids) - 1))
            self.ce_labels.extend(labels)
            self.ce_weights.extend(
                [ce_length_weight(len(labels), "square")] * len(labels)
            )

    def build(self) -> LancePreparedSample:
        length = len(self.token_ids)
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
    sample_id, caption, target, tokenizer, config, *, system_prompt=None,
    condition_frames=(),
):
    assembler = _SampleAssembler(sample_id, tokenizer, config, LanceSpecialTokens.from_tokenizer(tokenizer))
    system_prompt = system_prompt or lance_system_prompt(
        "t2i" if target.modality == "image" else "t2v", target.modality
    )
    assembler.add_chat(
        system_prompt,
        "" if caption is None else str(caption),
        _visual_placeholder(target.modality),
        ((target, "vae", True, condition_frames),),
        assistant_ce=False,
    )
    return assembler.build()


def build_understanding_sample(
    sample_id, prompt, answer, condition, tokenizer, config, *, system_prompt=None
):
    assembler = _SampleAssembler(sample_id, tokenizer, config, LanceSpecialTokens.from_tokenizer(tokenizer))
    assembler.add_chat(
        system_prompt or lance_system_prompt("caption", condition.modality),
        _visual_placeholder(condition.modality) + ("" if not prompt else str(prompt)),
        str(answer),
        ((condition, "vit", False),),
        assistant_ce=True,
    )
    return assembler.build()


def build_understanding_prompt_sample(
    sample_id, prompt, condition, tokenizer, config, *, system_prompt=None
):
    """Build an I2T/V2T inference prefix ending at ``assistant\n``."""

    assembler = _SampleAssembler(
        sample_id,
        tokenizer,
        config,
        LanceSpecialTokens.from_tokenizer(tokenizer),
    )
    assembler.add_chat(
        system_prompt or lance_system_prompt("caption", condition.modality),
        _visual_placeholder(condition.modality) + ("" if not prompt else str(prompt)),
        "",
        ((condition, "vit", False),),
        assistant_ce=False,
        close_assistant=False,
    )
    return assembler.build()


def build_edit_sample(
    sample_id, instruction, condition, target, tokenizer, config, *, system_prompt=None
):
    if condition.modality != target.modality:
        raise ValueError("edit condition and target modalities must match")
    assembler = _SampleAssembler(sample_id, tokenizer, config, LanceSpecialTokens.from_tokenizer(tokenizer))
    # Official text_template_user is rotated once before rendering, producing
    # VIT condition, VAE condition, then the textual edit instruction.
    visual = _visual_placeholder(condition.modality)
    assembler.add_chat(
        system_prompt or lance_system_prompt("edit", target.modality),
        visual + visual + str(instruction),
        _visual_placeholder(target.modality),
        (
            (condition, "vit", False),
            (condition, "vae", False),
            (target, "vae", True),
        ),
        assistant_ce=False,
    )
    return assembler.build()
