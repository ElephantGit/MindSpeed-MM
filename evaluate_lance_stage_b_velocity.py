#!/usr/bin/env python3
"""Measure Lance velocity error across the Stage-B denoising trajectory."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import statistics
import sys

os.environ.setdefault("NON_MEGATRON", "true")

import torch

from inference_lance_native import _build_model, _resolve_qwen_config, _set_device
from reconstruct_lance_stage_a import (
    _load_training_batch,
    _move_batch,
    _selected_latent_indexes,
    _tensor_metrics,
)
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.native_inference import (
    LanceNativeInferenceError,
    load_native_generation_dcp,
    resolve_native_dcp,
)
from mindspeed_mm.models.omni.lance.training_lance import (
    LanceLossWeights,
    lance_training_step,
    shift_timesteps,
)


DEFAULT_TIMESTEPS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--packed-batch", required=True)
    parser.add_argument("--qwen-path", required=True)
    parser.add_argument("--output", required=True, help="destination JSON report")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--timesteps", nargs="+", type=float, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--noise-seeds", nargs="+", type=int, default=(2025, 2026, 2027, 2028))
    parser.add_argument(
        "--posterior-mode",
        choices=("mean", "sample", "both"),
        default="both",
    )
    parser.add_argument("--timestep-shift", type=float, default=1.0)
    parser.add_argument("--variant", choices=("image", "video"), default="video")
    parser.add_argument("--latent-patch-size", nargs=3, type=int, default=(1, 2, 2))
    parser.add_argument("--max-latent-size", type=int, default=64)
    parser.add_argument("--max-num-frames", type=int, default=121)
    parser.add_argument("--ema-weights", action="store_true")
    return parser.parse_args(argv)


def _cpu_normal_like(value, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        value.shape,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(value.dtype)


def _posterior_latents(batch, mode, seed):
    mean = batch.clean_latents
    if mode == "mean":
        return mean
    if batch.latent_log_variance is None:
        raise LanceNativeInferenceError(
            "posterior sample evaluation requires latent_log_variance in the packed batch"
        )
    posterior_noise = _cpu_normal_like(mean, seed + 1_000_000)
    sampled = mean.float() + torch.exp(
        0.5 * batch.latent_log_variance.float()
    ) * posterior_noise.float()
    return sampled.to(mean.dtype)


def _mean_metrics(rows):
    names = (
        "mse",
        "rmse",
        "relative_rmse",
        "cosine_similarity",
        "max_abs_error",
    )
    return {name: statistics.fmean(row[name] for row in rows) for name in names}


def main(argv=None):
    args = parse_arguments(argv)
    if any(not 0.0 <= value <= 1.0 for value in args.timesteps):
        raise LanceNativeInferenceError("all timesteps must be in [0, 1]")
    if not args.noise_seeds:
        raise LanceNativeInferenceError("provide at least one noise seed")
    if args.timestep_shift <= 0:
        raise LanceNativeInferenceError("timestep-shift must be positive")

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise LanceNativeInferenceError("refusing to overwrite output: {}".format(output))
    packed_path = Path(args.packed_batch).expanduser().resolve()
    qwen_path = Path(args.qwen_path).expanduser().resolve()
    if not packed_path.is_file():
        raise LanceNativeInferenceError("packed batch not found: {}".format(packed_path))
    if not qwen_path.is_dir():
        raise LanceNativeInferenceError("Qwen directory not found: {}".format(qwen_path))

    checkpoint = resolve_native_dcp(args.checkpoint)
    config = LanceNativeConfig.from_llm_config(
        _resolve_qwen_config(qwen_path), variant=args.variant
    ).with_overrides(
        latent_patch_size=tuple(args.latent_patch_size),
        max_latent_size=args.max_latent_size,
        max_num_frames=args.max_num_frames,
    )
    batch = _load_training_batch(packed_path)
    if batch.clean_latents is None or batch.timesteps is None:
        raise LanceNativeInferenceError("packed batch has no VAE training target")
    latent_indexes = _selected_latent_indexes(batch)
    expected = torch.arange(batch.clean_latents.shape[0], dtype=torch.long)
    if not torch.equal(latent_indexes.cpu(), expected):
        raise LanceNativeInferenceError(
            "velocity sweep requires one pure T2I batch with every latent supervised"
        )

    modes = ("mean", "sample") if args.posterior_mode == "both" else (args.posterior_mode,)
    device = _set_device(args.device)
    model = _build_model(config, device)
    load_report = load_native_generation_dcp(
        model, checkpoint, use_ema=args.ema_weights
    )

    records = []
    for mode in modes:
        for timestep in args.timesteps:
            for seed in args.noise_seeds:
                clean_latents = _posterior_latents(batch, mode, seed)
                noise = _cpu_normal_like(clean_latents, seed)
                timesteps = batch.timesteps.clone()
                timesteps[latent_indexes] = timestep
                trial = replace(
                    batch,
                    clean_latents=clean_latents,
                    latent_log_variance=None,
                    timesteps=timesteps,
                    noise=noise,
                    resample_timesteps=False,
                )
                device_batch = _move_batch(trial, device, torch.bfloat16)
                device_indexes = _selected_latent_indexes(device_batch)
                with torch.no_grad(), torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16
                ):
                    step = lance_training_step(
                        model,
                        device_batch,
                        LanceLossWeights(ce=0.0, mse=1.0),
                        args.timestep_shift,
                    )
                    prediction = model.llm2vae(
                        step["hidden_states"][device_batch.mse_indexes]
                    )
                    target = step["velocity_target"][device_indexes]
                    shifted = shift_timesteps(
                        device_batch.timesteps[device_indexes], args.timestep_shift
                    )
                    noisy = (
                        (1.0 - shifted.unsqueeze(1))
                        * device_batch.clean_latents[device_indexes]
                        + shifted.unsqueeze(1) * device_batch.noise[device_indexes]
                    )
                    reconstructed = noisy - shifted.unsqueeze(1) * prediction
                record = {
                    "posterior_mode": mode,
                    "raw_timestep": timestep,
                    "shifted_timestep": float(shifted[0].float().cpu().item()),
                    "noise_seed": seed,
                    "velocity": _tensor_metrics(prediction, target),
                    "single_step_x0": _tensor_metrics(
                        reconstructed,
                        device_batch.clean_latents[device_indexes],
                    ),
                    "training_step_mse": float(step["mse_loss"].float().cpu().item()),
                }
                records.append(record)
                print(
                    "mode={} t={:.3f} seed={} velocity_mse={:.6g} cosine={:.6f}".format(
                        mode,
                        timestep,
                        seed,
                        record["velocity"]["mse"],
                        record["velocity"]["cosine_similarity"],
                    ),
                    flush=True,
                )
                del step, prediction, target, shifted, noisy, reconstructed, device_batch

    aggregates = []
    for mode in modes:
        for timestep in args.timesteps:
            selected = [
                row for row in records
                if row["posterior_mode"] == mode and row["raw_timestep"] == timestep
            ]
            aggregates.append(
                {
                    "posterior_mode": mode,
                    "raw_timestep": timestep,
                    "sample_count": len(selected),
                    "velocity": _mean_metrics([row["velocity"] for row in selected]),
                    "single_step_x0": _mean_metrics(
                        [row["single_step_x0"] for row in selected]
                    ),
                    "training_step_mse": statistics.fmean(
                        row["training_step_mse"] for row in selected
                    ),
                }
            )

    report = {
        "schema_version": 1,
        "mode": "lance-stage-b-velocity-sweep",
        "checkpoint": str(checkpoint),
        "weights": "ema" if args.ema_weights else "model",
        "packed_batch": str(packed_path),
        "timesteps": list(args.timesteps),
        "noise_seeds": list(args.noise_seeds),
        "posterior_modes": list(modes),
        "timestep_shift": args.timestep_shift,
        "load_report": load_report,
        "aggregates": aggregates,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Wrote velocity sweep: {}".format(output), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LanceNativeInferenceError, ValueError, RuntimeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)
