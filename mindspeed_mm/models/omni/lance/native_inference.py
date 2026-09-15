"""End-to-end helpers for native Lance text-to-image/video inference.

This module deliberately uses only MindSpeed-MM's native Lance model, prompt
builder, sampler, and Wan2.2 VAE.  It does not import or execute the standalone
Lance checkout.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple, Union

import torch

from .data import LancePreparedSample
from .modeling_lance import LanceNativeModel
from .native_config import LanceNativeConfig
from .sampling import LanceDenoiseContext
from .sequence import LancePackedSequence


class LanceNativeInferenceError(RuntimeError):
    """Raised when a native inference artifact or geometry is incompatible."""


@dataclass(frozen=True)
class LanceGenerationGeometry:
    """Decoded and latent geometry for one T2I/T2V request."""

    modality: str
    frames: int
    height: int
    width: int
    latent_frames: int
    latent_height: int
    latent_width: int

    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        return self.latent_frames, self.latent_height, self.latent_width


def resolve_native_dcp(path: Union[str, Path]) -> Path:
    """Resolve either a DCP iteration directory or a checkpoint root."""

    root = Path(path).expanduser().resolve()
    if (root / ".metadata").is_file():
        return root
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise LanceNativeInferenceError(
            "native Lance checkpoint must be an iteration directory containing "
            ".metadata or a root containing latest_checkpointed_iteration.txt: {}".format(root)
        )
    value = tracker.read_text(encoding="utf-8").strip()
    if value == "release":
        iteration = root / "release"
    else:
        try:
            iteration = root / "iter_{:07d}".format(int(value))
        except ValueError as exc:
            raise LanceNativeInferenceError(
                "invalid native Lance checkpoint tracker value: {}".format(value)
            ) from exc
    if not (iteration / ".metadata").is_file():
        raise LanceNativeInferenceError(
            "resolved native Lance DCP is incomplete: {}".format(iteration)
        )
    return iteration


def generation_geometry(
    task: str,
    frames: int,
    height: int,
    width: int,
    config: LanceNativeConfig,
) -> LanceGenerationGeometry:
    """Validate output geometry and derive the unpatched VAE latent shape."""

    if task not in ("t2i", "t2v"):
        raise LanceNativeInferenceError("native generation task must be t2i or t2v")
    if task == "t2i" and frames != 1:
        raise LanceNativeInferenceError("t2i requires exactly one frame")
    if min(frames, height, width) <= 0:
        raise LanceNativeInferenceError("frames, height, and width must be positive")
    if frames > config.max_num_frames:
        raise LanceNativeInferenceError(
            "requested {} frames exceeds model maximum {}".format(
                frames, config.max_num_frames
            )
        )
    temporal = config.latent_temporal_downsample
    # Wan2.2 is causal in time and reconstructs exactly 4k+1 frames.  Requiring
    # that contract prevents a nominal frame count from silently producing a
    # shorter decoded video.
    if (frames - 1) % temporal:
        raise LanceNativeInferenceError(
            "Wan2.2/Lance T2V frames must satisfy frames = {}k + 1; got {}".format(
                temporal, frames
            )
        )
    patch_t, patch_h, patch_w = config.latent_patch_size
    spatial_downsample = 16
    height_alignment = spatial_downsample * patch_h
    width_alignment = spatial_downsample * patch_w
    if height % height_alignment or width % width_alignment:
        raise LanceNativeInferenceError(
            "height/width must be divisible by {}/{} for latent_patch_size={}; "
            "got {}x{}".format(
                height_alignment,
                width_alignment,
                tuple(config.latent_patch_size),
                height,
                width,
            )
        )
    latent_frames = (frames - 1) // temporal + 1
    latent_height = height // spatial_downsample
    latent_width = width // spatial_downsample
    if latent_frames % patch_t or latent_height % patch_h or latent_width % patch_w:
        raise LanceNativeInferenceError(
            "derived VAE latent does not align to latent_patch_size"
        )
    token_grid = (
        latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
    )
    if token_grid[1] > config.max_latent_size or token_grid[2] > config.max_latent_size:
        raise LanceNativeInferenceError(
            "derived latent token grid {} exceeds max_latent_size {}".format(
                token_grid, config.max_latent_size
            )
        )
    return LanceGenerationGeometry(
        modality="image" if task == "t2i" else "video",
        frames=frames,
        height=height,
        width=width,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
    )


def prepared_sample_to_denoise_context(
    sample: LancePreparedSample,
    device: Union[str, torch.device],
) -> LanceDenoiseContext:
    """Convert one native generation sample into the sampler-facing context."""

    required = {
        "vae_indexes": sample.vae_indexes,
        "clean_latents": sample.clean_latents,
        "latent_position_ids": sample.latent_position_ids,
        "mse_indexes": sample.mse_indexes,
        "understanding_indexes": sample.understanding_indexes,
        "generation_indexes": sample.generation_indexes,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise LanceNativeInferenceError(
            "generation sample is missing fields: {}".format(", ".join(missing))
        )
    vae_indexes = sample.vae_indexes.long()
    prediction_indexes = sample.mse_indexes.long()
    prediction_latent_indexes = torch.searchsorted(vae_indexes, prediction_indexes)
    if (
        prediction_latent_indexes.numel() != prediction_indexes.numel()
        or torch.any(prediction_latent_indexes >= vae_indexes.numel())
        or not torch.equal(vae_indexes[prediction_latent_indexes], prediction_indexes)
    ):
        raise LanceNativeInferenceError(
            "MSE prediction indexes are not a subset of the VAE token indexes"
        )

    target = torch.device(device)

    def move(value):
        return None if value is None else value.to(target, non_blocking=True)

    return LanceDenoiseContext(
        token_ids=move(sample.token_ids),
        text_indexes=move(sample.text_indexes),
        position_ids=move(sample.position_ids),
        attention_mask=LancePackedSequence((sample.document,)),
        understanding_indexes=move(sample.understanding_indexes),
        generation_indexes=move(sample.generation_indexes),
        vae_indexes=move(vae_indexes),
        latent_position_ids=move(sample.latent_position_ids),
        prediction_indexes=move(prediction_indexes),
        prediction_latent_indexes=move(prediction_latent_indexes),
        vit_indexes=move(sample.vit_indexes),
        vit_embeddings=move(sample.vit_embeddings),
    )


def unpatchify_lance_latents(
    patchified: torch.Tensor,
    geometry: LanceGenerationGeometry,
    config: LanceNativeConfig,
) -> torch.Tensor:
    """Invert Lance's token-major latent patching and return THWC latents."""

    patch_t, patch_h, patch_w = config.latent_patch_size
    grid_t = geometry.latent_frames // patch_t
    grid_h = geometry.latent_height // patch_h
    grid_w = geometry.latent_width // patch_w
    expected = (
        grid_t * grid_h * grid_w,
        patch_t * patch_h * patch_w * config.latent_channels,
    )
    if tuple(patchified.shape) != expected:
        raise LanceNativeInferenceError(
            "patchified latent has shape {}, expected {}".format(
                tuple(patchified.shape), expected
            )
        )
    return (
        patchified.reshape(
            grid_t,
            grid_h,
            grid_w,
            patch_t,
            patch_h,
            patch_w,
            config.latent_channels,
        )
        .permute(0, 3, 1, 4, 2, 5, 6)
        .reshape(
            geometry.latent_frames,
            geometry.latent_height,
            geometry.latent_width,
            config.latent_channels,
        )
        .contiguous()
    )


def _metadata_tensor_shape(metadata) -> Tuple[int, ...]:
    size = getattr(metadata, "size", None)
    if size is None:
        raise LanceNativeInferenceError("DCP model entry is not tensor metadata")
    return tuple(int(value) for value in size)


def _validate_dcp_model_metadata(
    entries: Mapping[str, object],
    model_state: Mapping[str, torch.Tensor],
) -> None:
    prefix = "model."
    actual = {
        name[len(prefix) :]: metadata
        for name, metadata in entries.items()
        if name.startswith(prefix)
    }
    missing = sorted(set(model_state) - set(actual))
    mismatched = sorted(
        name
        for name in set(model_state) & set(actual)
        if tuple(model_state[name].shape) != _metadata_tensor_shape(actual[name])
    )
    if missing or mismatched:
        raise LanceNativeInferenceError(
            "DCP/model contract mismatch: missing={}, shape_mismatches={}".format(
                missing[:8], mismatched[:8]
            )
        )


def load_native_generation_dcp(
    model: LanceNativeModel,
    checkpoint: Union[str, Path],
    *,
    use_ema: bool = True,
) -> Dict[str, object]:
    """Load native training DCP weights directly into an allocated model.

    Generation does not need the frozen ViT or the PT-only connector.  The
    loader therefore requests the native MoT/bridge/head subset and safely
    ignores those extra DCP entries.  EMA parameters overwrite their matching
    model tensors without allocating a second model tree.
    """

    try:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
        from torch.distributed.checkpoint.state_dict_loader import load_state_dict
    except ImportError as exc:
        raise LanceNativeInferenceError(
            "native Lance DCP loading requires torch.distributed.checkpoint"
        ) from exc

    directory = resolve_native_dcp(checkpoint)
    reader = FileSystemReader(str(directory))
    entries = reader.read_metadata().state_dict_metadata
    model_state = model.state_dict()
    _validate_dcp_model_metadata(entries, model_state)

    def load(payload) -> None:
        planner = DefaultLoadPlanner(allow_partial_load=False)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            dcp.load(payload, storage_reader=reader, planner=planner)
        else:
            # The CLI is intentionally single-process.  PyTorch 2.7's public
            # load_state_dict API makes that explicit instead of requiring a
            # one-rank process group merely to restore an unsharded model.
            load_state_dict(
                payload,
                storage_reader=reader,
                planner=planner,
                no_dist=True,
            )

    load({"model": model_state})
    model.load_state_dict(model_state, strict=True)

    ema_count = 0
    if use_ema:
        prefix = "ema_state.parameters."
        ema_names = {
            name[len(prefix) :]
            for name in entries
            if name.startswith(prefix)
        }
        required_ema = set(model_state) - {"latent_pos_embed.pos_embed"}
        missing_ema = sorted(required_ema - ema_names)
        if missing_ema:
            raise LanceNativeInferenceError(
                "DCP EMA is incomplete for native generation: {}".format(
                    missing_ema[:8]
                )
            )
        ema_parameters = {
            name: model_state[name]
            for name in sorted(required_ema)
        }
        load({"ema_state": {"parameters": ema_parameters}})
        model.load_state_dict(ema_parameters, strict=False)
        ema_count = len(ema_parameters)

    return {
        "checkpoint": str(directory),
        "weights": "ema" if use_ema else "model",
        "model_tensor_count": len(model_state),
        "ema_tensor_count": ema_count,
    }
