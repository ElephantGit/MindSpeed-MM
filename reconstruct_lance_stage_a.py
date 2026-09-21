#!/usr/bin/env python3
"""Reconstruct the deterministic Lance Stage-A T2I training point.

This diagnostic deliberately follows the Stage-A training contract instead of
the normal t=1 -> 0 sampler:

    x_t = (1 - t) * x_0 + t * noise
    v_target = noise - x_0
    x_0_hat = x_t - t * v_pred

The packed training batch supplies the exact text/attention/position inputs.
Noise is regenerated on CPU in FP32 and then cast to the stored latent dtype,
matching NativeLancePreencodedDataset.
"""

import argparse
from dataclasses import fields, replace
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping

os.environ.setdefault("NON_MEGATRON", "true")

import torch
import torch.nn.functional as F

from inference_lance_native import (
    _build_model,
    _build_vae,
    _decode_latents,
    _empty_accelerator_cache,
    _resolve_qwen_config,
    _save_media,
    _set_device,
)
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.native_inference import (
    LanceNativeInferenceError,
    generation_geometry,
    load_native_generation_dcp,
    resolve_native_dcp,
    unpatchify_lance_latents,
)
from mindspeed_mm.models.omni.lance.sequence import flatten_latent_position_ids
from mindspeed_mm.models.omni.lance.training_lance import (
    LanceLossWeights,
    LanceTrainingBatch,
    lance_training_step,
    shift_timesteps,
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Stage-A DCP root or iteration")
    parser.add_argument("--packed-batch", required=True, help="Stage-A batch-XXXXXXXX.pt")
    parser.add_argument("--qwen-path", required=True, help="Qwen tokenizer/config directory")
    parser.add_argument("--vae-path", required=True, help="Wan2.2_VAE.pth")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--noise-seed", type=int, default=2025)
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=0,
        help="index of this file in the sorted packed dataset; Stage A batch-00000000 is 0",
    )
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--timestep-shift", type=float, default=1.0)
    parser.add_argument("--variant", choices=("image", "video"), default="video")
    parser.add_argument(
        "--latent-patch-size", nargs=3, type=int, default=(1, 2, 2),
        metavar=("T", "H", "W"),
    )
    parser.add_argument("--max-latent-size", type=int, default=64)
    parser.add_argument("--max-num-frames", type=int, default=121)
    parser.add_argument(
        "--ema-weights",
        action="store_true",
        help="use EMA instead of ordinary model weights (Stage A normally has no EMA)",
    )
    return parser.parse_args(argv)


def _load_training_batch(path: Path) -> LanceTrainingBatch:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, LanceTrainingBatch):
        return value
    if isinstance(value, Mapping):
        return LanceTrainingBatch(**dict(value))
    raise LanceNativeInferenceError(
        "{} does not contain a LanceTrainingBatch".format(path)
    )


def _fixed_training_noise(clean_latents, seed: int, dataset_index: int):
    if dataset_index < 0:
        raise LanceNativeInferenceError("dataset-index must be non-negative")
    generator = torch.Generator(device="cpu").manual_seed(seed + dataset_index)
    return torch.randn(
        clean_latents.shape,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(dtype=clean_latents.dtype)


def _move_batch(
    batch: LanceTrainingBatch,
    device: torch.device,
    float_dtype: torch.dtype,
) -> LanceTrainingBatch:
    values = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        values[field.name] = (
            value.to(
                device=device,
                dtype=float_dtype if torch.is_floating_point(value) else None,
                non_blocking=True,
            )
            if isinstance(value, torch.Tensor)
            else value
        )
    return replace(batch, **values)


def _selected_latent_indexes(batch: LanceTrainingBatch) -> torch.Tensor:
    if batch.vae_indexes is None or batch.mse_indexes is None:
        raise LanceNativeInferenceError("Stage-A batch requires VAE and MSE indexes")
    if not batch.mse_indexes.numel():
        raise LanceNativeInferenceError("Stage-A batch contains no MSE target tokens")
    indexes = torch.searchsorted(batch.vae_indexes, batch.mse_indexes)
    if (
        indexes.numel() != batch.mse_indexes.numel()
        or torch.any(indexes >= batch.vae_indexes.numel())
        or not torch.equal(batch.vae_indexes[indexes], batch.mse_indexes)
    ):
        raise LanceNativeInferenceError("MSE indexes are not a subset of VAE indexes")
    return indexes


def _infer_t2i_geometry(position_ids, config: LanceNativeConfig):
    positions = position_ids.detach().long().cpu()
    spatial = config.max_latent_size * config.max_latent_size
    temporal_ids = positions // spatial
    height_ids = (positions % spatial) // config.max_latent_size
    width_ids = positions % config.max_latent_size
    grid_t = int(torch.unique(temporal_ids).numel())
    grid_h = int(torch.unique(height_ids).numel())
    grid_w = int(torch.unique(width_ids).numel())
    expected = torch.tensor(
        flatten_latent_position_ids(grid_t, grid_h, grid_w, config.max_latent_size),
        dtype=torch.long,
    )
    if not torch.equal(positions, expected):
        raise LanceNativeInferenceError(
            "selected Stage-A latent positions are not one complete ordered T2I grid"
        )
    patch_t, patch_h, patch_w = config.latent_patch_size
    latent_frames = grid_t * patch_t
    if latent_frames != 1:
        raise LanceNativeInferenceError(
            "Stage-A single-image reconstruction expected one latent frame, got {}".format(
                latent_frames
            )
        )
    height = grid_h * patch_h * 16
    width = grid_w * patch_w * 16
    return generation_geometry("t2i", 1, height, width, config)


def _tensor_metrics(prediction, target):
    prediction = prediction.detach().float()
    target = target.detach().float()
    difference = prediction - target
    mse = float(difference.square().mean().item())
    rmse = math.sqrt(mse)
    target_rms = math.sqrt(float(target.square().mean().item()))
    cosine = float(
        F.cosine_similarity(prediction.flatten(), target.flatten(), dim=0).item()
    )
    return {
        "mse": mse,
        "rmse": rmse,
        "relative_rmse": rmse / max(target_rms, 1.0e-12),
        "cosine_similarity": cosine,
        "max_abs_error": float(difference.abs().max().item()),
    }


def _image_metrics(prediction, target):
    prediction = torch.from_numpy(prediction).float().div(255.0)
    target = torch.from_numpy(target).float().div(255.0)
    mse = float((prediction - target).square().mean().item())
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "psnr_db": float("inf") if mse == 0 else -10.0 * math.log10(mse),
    }


def main(argv=None) -> int:
    args = parse_arguments(argv)
    if not 0.0 <= args.timestep <= 1.0:
        raise LanceNativeInferenceError("timestep must be in [0, 1]")
    if args.timestep_shift <= 0:
        raise LanceNativeInferenceError("timestep-shift must be positive")

    packed_batch = Path(args.packed_batch).expanduser().resolve()
    qwen_path = Path(args.qwen_path).expanduser().resolve()
    vae_path = Path(args.vae_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not packed_batch.is_file():
        raise LanceNativeInferenceError("packed batch not found: {}".format(packed_batch))
    if not qwen_path.is_dir():
        raise LanceNativeInferenceError("Qwen directory not found: {}".format(qwen_path))
    if not vae_path.is_file():
        raise LanceNativeInferenceError("Wan2.2 VAE checkpoint not found: {}".format(vae_path))
    checkpoint = resolve_native_dcp(args.checkpoint)
    planned_outputs = (
        output_dir / "training_latent_target.png",
        output_dir / "stage_a_single_step_reconstruction.png",
        output_dir / "stage_a_single_step_reconstruction.json",
    )
    existing = [str(path) for path in planned_outputs if path.exists()]
    if existing:
        raise LanceNativeInferenceError(
            "refusing to overwrite Stage-A reconstruction outputs: {}".format(existing)
        )

    config = LanceNativeConfig.from_llm_config(
        _resolve_qwen_config(qwen_path), variant=args.variant
    ).with_overrides(
        latent_patch_size=tuple(args.latent_patch_size),
        max_latent_size=args.max_latent_size,
        max_num_frames=args.max_num_frames,
    )
    batch = _load_training_batch(packed_batch)
    if batch.clean_latents is None or batch.timesteps is None:
        raise LanceNativeInferenceError("packed batch does not contain VAE training latents")
    if batch.vit_indexes is not None and batch.vit_indexes.numel():
        raise LanceNativeInferenceError("Stage-A T2I reconstruction does not accept ViT inputs")

    latent_indexes = _selected_latent_indexes(batch)
    expected_indexes = torch.arange(batch.clean_latents.shape[0], dtype=torch.long)
    if not torch.equal(latent_indexes.cpu(), expected_indexes):
        raise LanceNativeInferenceError(
            "Stage-A diagnostic requires one pure T2I sample whose complete latent is supervised"
        )
    stored_timesteps = batch.timesteps[latent_indexes].float()
    if not torch.allclose(
        stored_timesteps,
        torch.full_like(stored_timesteps, args.timestep),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise LanceNativeInferenceError(
            "packed target timestep range [{:.8f}, {:.8f}] does not match requested {}".format(
                float(stored_timesteps.min().item()),
                float(stored_timesteps.max().item()),
                args.timestep,
            )
        )
    geometry = _infer_t2i_geometry(batch.latent_position_ids[latent_indexes], config)
    fixed_noise = _fixed_training_noise(
        batch.clean_latents, args.noise_seed, args.dataset_index
    )
    batch = replace(
        batch,
        latent_log_variance=None,
        noise=fixed_noise,
        resample_timesteps=False,
    )

    device = _set_device(args.device)
    print("Building native Lance model on {}...".format(device), flush=True)
    model = _build_model(config, device)
    load_report = load_native_generation_dcp(
        model, checkpoint, use_ema=args.ema_weights
    )
    # The Lance train engine casts every floating batch tensor (including the
    # timestep) to the FSDP parameter dtype before forward.  Match that detail,
    # rather than merely moving the stored CPU tensors to the accelerator.
    device_batch = _move_batch(batch, device, torch.bfloat16)
    device_latent_indexes = _selected_latent_indexes(device_batch)

    print(
        "Reconstructing exact Stage-A point (t={}, noise_seed={}, dataset_index={})...".format(
            args.timestep, args.noise_seed, args.dataset_index
        ),
        flush=True,
    )
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        step = lance_training_step(
            model,
            device_batch,
            LanceLossWeights(ce=0.0, mse=1.0),
            args.timestep_shift,
        )
        velocity_prediction = model.llm2vae(
            step["hidden_states"][device_batch.mse_indexes]
        )
        velocity_target = step["velocity_target"][device_latent_indexes]
        shifted_timesteps = shift_timesteps(
            device_batch.timesteps[device_latent_indexes], args.timestep_shift
        )
        clean_target = device_batch.clean_latents[device_latent_indexes]
        selected_noise = device_batch.noise[device_latent_indexes]
        noisy_latents = (
            (1.0 - shifted_timesteps.unsqueeze(1)) * clean_target
            + shifted_timesteps.unsqueeze(1) * selected_noise
        )
        reconstructed = noisy_latents - shifted_timesteps.unsqueeze(1) * velocity_prediction

    training_step_mse = float(step["mse_loss"].detach().float().item())
    shifted_timestep_value = float(shifted_timesteps[0].float().cpu().item())
    velocity_metrics = _tensor_metrics(velocity_prediction, velocity_target)
    latent_metrics = _tensor_metrics(reconstructed, clean_target)
    clean_target = clean_target.cpu()
    reconstructed = reconstructed.cpu()
    noise_stats = {
        "mean": float(fixed_noise.float().mean().item()),
        "std": float(fixed_noise.float().std().item()),
        "rms": math.sqrt(float(fixed_noise.float().square().mean().item())),
    }
    del (
        step,
        velocity_prediction,
        velocity_target,
        noisy_latents,
        selected_noise,
        shifted_timesteps,
        device_batch,
        model,
    )
    _empty_accelerator_cache(device)

    target_latent = unpatchify_lance_latents(clean_target, geometry, config)
    reconstructed_latent = unpatchify_lance_latents(reconstructed, geometry, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    vae = _build_vae(vae_path, device)
    try:
        print("Decoding training latent target...", flush=True)
        target_image = _decode_latents(target_latent, geometry, vae, device)
        print("Decoding one-step reconstruction...", flush=True)
        reconstructed_image = _decode_latents(reconstructed_latent, geometry, vae, device)
    finally:
        vae.close()
        _empty_accelerator_cache(device)

    target_path, reconstructed_path, manifest_path = planned_outputs
    _save_media(target_image, target_path, "t2i", fps=1)
    _save_media(reconstructed_image, reconstructed_path, "t2i", fps=1)
    report = {
        "schema_version": 1,
        "status": "completed",
        "mode": "lance-stage-a-exact-single-step-reconstruction",
        "checkpoint": str(checkpoint),
        "weights": "ema" if args.ema_weights else "model",
        "packed_batch": str(packed_batch),
        "dataset_index": args.dataset_index,
        "noise": {
            "seed": args.noise_seed,
            "effective_seed": args.noise_seed + args.dataset_index,
            "generator": "cpu-fp32-then-cast-to-packed-latent-dtype",
            "stored_dtype": str(fixed_noise.dtype),
            **noise_stats,
        },
        "timestep": {
            "raw": args.timestep,
            "shift": args.timestep_shift,
            "shifted": shifted_timestep_value,
        },
        "geometry": dict(geometry.__dict__),
        "sequence_length": batch.sequence_length,
        "latent_token_count": int(clean_target.shape[0]),
        "velocity_metrics": velocity_metrics,
        "training_step_mse": training_step_mse,
        "latent_x0_metrics": latent_metrics,
        "decoded_image_metrics": _image_metrics(reconstructed_image, target_image),
        "load_report": load_report,
        "outputs": {
            "training_latent_target": str(target_path),
            "single_step_reconstruction": str(reconstructed_path),
        },
    }
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print("Completed Stage-A reconstruction: {}".format(manifest_path), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LanceNativeInferenceError, ValueError, RuntimeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)
