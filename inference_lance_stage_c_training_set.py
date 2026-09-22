#!/usr/bin/env python3
"""Generate and score every example in a Lance Stage-C training manifest."""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys

os.environ.setdefault("NON_MEGATRON", "true")

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from inference_lance_native import (
    _build_model,
    _build_vae,
    _decode_latents,
    _empty_accelerator_cache,
    _resolve_qwen_config,
    _save_media,
    _seed_accelerator,
    _set_device,
)
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.native_inference import (
    LanceNativeInferenceError,
    generation_geometry,
    load_native_generation_dcp,
    prepared_sample_to_denoise_context,
    resolve_native_dcp,
    unpatchify_lance_latents,
)
from mindspeed_mm.models.omni.lance.preprocessing import (
    LanceEncodedVisual,
    build_generation_sample,
    prepare_lance_tokenizer,
)
from mindspeed_mm.models.omni.lance.sampling import sample_native_lance


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--qwen-path", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--timestep-shift", type=float, default=1.0)
    parser.add_argument(
        "--timestep-schedule",
        choices=("linear", "sigmoid_normal"),
        default="linear",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--variant", choices=("image", "video"), default="video")
    parser.add_argument("--latent-patch-size", nargs=3, type=int, default=(1, 2, 2))
    parser.add_argument("--max-latent-size", type=int, default=64)
    parser.add_argument("--max-num-frames", type=int, default=121)
    parser.add_argument("--ema-weights", action="store_true")
    return parser.parse_args(argv)


def _load_manifest(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LanceNativeInferenceError(
            "cannot read Stage-C manifest {}: {}".format(path, exc)
        ) from exc
    examples = payload.get("examples") if isinstance(payload, dict) else None
    if not isinstance(examples, list) or not examples:
        raise LanceNativeInferenceError("Stage-C manifest contains no examples")
    required = {"index", "sample_id", "prompt", "height", "width", "training_target"}
    for position, entry in enumerate(examples):
        if not isinstance(entry, dict) or required - set(entry):
            raise LanceNativeInferenceError(
                "manifest example {} is missing required fields".format(position)
            )
        if int(entry["index"]) != position:
            raise LanceNativeInferenceError(
                "manifest example indexes must be contiguous from zero"
            )
        if not str(entry["prompt"]).strip():
            raise LanceNativeInferenceError(
                "manifest example {} has an empty prompt".format(position)
            )
        if not Path(entry["training_target"]).is_file():
            raise LanceNativeInferenceError(
                "training target not found: {}".format(entry["training_target"])
            )
    return payload, examples


def _atomic_save_tensor(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _image_metrics(prediction, target_path):
    with Image.open(target_path) as image:
        target = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    prediction = prediction.astype(np.float32) / 255.0
    if prediction.shape != target.shape:
        raise LanceNativeInferenceError(
            "decoded shape {} differs from target {} for {}".format(
                prediction.shape, target.shape, target_path
            )
        )
    difference = prediction - target
    mse = float(np.mean(np.square(difference), dtype=np.float64))
    correlation = float(np.corrcoef(prediction.ravel(), target.ravel())[0, 1])
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "psnr_db": float("inf") if mse == 0 else -10.0 * math.log10(mse),
        "correlation": correlation,
    }


def _aggregate(records):
    names = ("mse", "rmse", "psnr_db", "correlation")
    average = {
        name: statistics.fmean(record["metrics"][name] for record in records)
        for name in names
    }
    worst = {
        "mse": max(record["metrics"]["mse"] for record in records),
        "rmse": max(record["metrics"]["rmse"] for record in records),
        "psnr_db": min(record["metrics"]["psnr_db"] for record in records),
        "correlation": min(record["metrics"]["correlation"] for record in records),
    }
    return average, worst


def _contact_sheet(records, destination):
    thumb = 192
    label_height = 20
    pair_width = thumb * 2
    tile_height = thumb + label_height
    columns = 4
    rows = math.ceil(len(records) / columns)
    canvas = Image.new("RGB", (columns * pair_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(canvas)
    for position, record in enumerate(records):
        column = position % columns
        row = position // columns
        left = column * pair_width
        top = row * tile_height
        with Image.open(record["training_target"]) as image:
            target = ImageOps.pad(image.convert("RGB"), (thumb, thumb), color="white")
        with Image.open(record["generated"]) as image:
            generated = ImageOps.pad(image.convert("RGB"), (thumb, thumb), color="white")
        canvas.paste(target, (left, top))
        canvas.paste(generated, (left + thumb, top))
        draw.text(
            (left + 4, top + thumb + 3),
            "{:02d} target | generated".format(record["index"]),
            fill="black",
        )
    canvas.save(destination)


def main(argv=None):
    args = parse_arguments(argv)
    if args.num_steps <= 0 or args.timestep_shift <= 0:
        raise LanceNativeInferenceError("num-steps and timestep-shift must be positive")
    manifest_path = Path(args.manifest).expanduser().resolve()
    qwen_path = Path(args.qwen_path).expanduser().resolve()
    vae_path = Path(args.vae_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not manifest_path.is_file():
        raise LanceNativeInferenceError("manifest not found: {}".format(manifest_path))
    if not qwen_path.is_dir():
        raise LanceNativeInferenceError("Qwen directory not found: {}".format(qwen_path))
    if not vae_path.is_file():
        raise LanceNativeInferenceError("VAE checkpoint not found: {}".format(vae_path))
    final_report = output_dir / "stage_c_training_set_inference.json"
    if final_report.exists():
        raise LanceNativeInferenceError(
            "completed Stage-C report already exists: {}".format(final_report)
        )
    manifest, examples = _load_manifest(manifest_path)
    checkpoint = resolve_native_dcp(args.checkpoint)
    config = LanceNativeConfig.from_llm_config(
        _resolve_qwen_config(qwen_path), variant=args.variant
    ).with_overrides(
        latent_patch_size=tuple(args.latent_patch_size),
        max_latent_size=args.max_latent_size,
        max_num_frames=args.max_num_frames,
    )
    geometries = [
        generation_geometry(
            "t2i", 1, int(entry["height"]), int(entry["width"]), config
        )
        for entry in examples
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = output_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    latent_paths = [latent_dir / "{:06d}.pt".format(index) for index in range(len(examples))]
    sampling_state_path = output_dir / "sampling_state.json"
    expected_sampling_state = {
        "checkpoint": str(checkpoint),
        "weights": "ema" if args.ema_weights else "model",
        "input_manifest": str(manifest_path),
        "sample_count": len(examples),
        "num_steps": args.num_steps,
        "timestep_shift": args.timestep_shift,
        "timestep_schedule": args.timestep_schedule,
        "base_seed": args.seed,
    }
    if sampling_state_path.is_file():
        sampling_state = json.loads(sampling_state_path.read_text(encoding="utf-8"))
        actual_contract = {
            key: sampling_state.get(key) for key in expected_sampling_state
        }
        if actual_contract != expected_sampling_state:
            raise LanceNativeInferenceError(
                "partial output was produced with a different sampling contract: {}".format(
                    sampling_state_path
                )
            )
    else:
        if any(path.is_file() for path in latent_paths):
            raise LanceNativeInferenceError(
                "latent cache exists without sampling_state.json under {}".format(output_dir)
            )
        sampling_state = dict(expected_sampling_state)
        sampling_state["load_report"] = None
        sampling_state_path.write_text(
            json.dumps(sampling_state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    missing_latents = [index for index, path in enumerate(latent_paths) if not path.is_file()]

    load_report = sampling_state.get("load_report")
    device = _set_device(args.device)
    if missing_latents:
        from transformers import AutoTokenizer

        tokenizer = prepare_lance_tokenizer(
            AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=False)
        )
        if len(tokenizer) > config.vocab_size:
            raise LanceNativeInferenceError(
                "tokenizer size {} exceeds model vocabulary {}".format(
                    len(tokenizer), config.vocab_size
                )
            )
        print("Building native Lance model on {}...".format(device), flush=True)
        model = _build_model(config, device)
        load_report = load_native_generation_dcp(
            model, checkpoint, use_ema=args.ema_weights
        )
        sampling_state["load_report"] = load_report
        sampling_state_path.write_text(
            json.dumps(sampling_state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for index in missing_latents:
            entry = examples[index]
            geometry = geometries[index]
            placeholder = LanceEncodedVisual(
                geometry.modality,
                vae_latent=torch.zeros(
                    *geometry.latent_shape,
                    config.latent_channels,
                    dtype=torch.bfloat16,
                ),
            )
            sample = build_generation_sample(
                "stage-c-training-{:06d}".format(index),
                str(entry["prompt"]),
                placeholder,
                tokenizer,
                config,
            )
            sample.validate(config)
            context = prepared_sample_to_denoise_context(sample, device)
            sample_seed = args.seed + index
            _seed_accelerator(sample_seed, device)
            initial = torch.randn(
                sample.clean_latents.shape,
                device=device,
                dtype=torch.bfloat16,
            )
            print(
                "Sampling training example {}/{} (seed={}, {}x{})...".format(
                    index + 1,
                    len(examples),
                    sample_seed,
                    geometry.width,
                    geometry.height,
                ),
                flush=True,
            )
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16
            ):
                result = sample_native_lance(
                    model,
                    context,
                    initial,
                    num_steps=args.num_steps,
                    timestep_shift=args.timestep_shift,
                    timestep_schedule=args.timestep_schedule,
                    text_scale=1.0,
                    renorm_type="none",
                )
            _atomic_save_tensor(result.cpu(), latent_paths[index])
            del result, initial, context, sample, placeholder
        del model
        _empty_accelerator_cache(device)
    else:
        print("All sampled latents already exist; skipping model loading.", flush=True)

    print("Building VAE decoder on {}...".format(device), flush=True)
    vae = _build_vae(vae_path, device)
    records = []
    try:
        for index, (entry, geometry, latent_path) in enumerate(
            zip(examples, geometries, latent_paths)
        ):
            sample_dir = output_dir / "{:06d}".format(index)
            sample_dir.mkdir(parents=True, exist_ok=True)
            generated_path = sample_dir / "generated.png"
            if generated_path.is_file():
                with Image.open(generated_path) as image:
                    generated = np.asarray(image.convert("RGB"), dtype=np.uint8)
            else:
                print(
                    "Decoding training example {}/{}...".format(index + 1, len(examples)),
                    flush=True,
                )
                patchified = torch.load(latent_path, map_location="cpu", weights_only=True)
                latent = unpatchify_lance_latents(patchified, geometry, config)
                frames = _decode_latents(latent, geometry, vae, device)
                _save_media(frames, generated_path, "t2i", fps=1)
                generated = frames[0]
            metrics = _image_metrics(generated, Path(entry["training_target"]))
            record = {
                "index": index,
                "sample_id": entry["sample_id"],
                "prompt": entry["prompt"],
                "seed": args.seed + index,
                "geometry": dict(geometry.__dict__),
                "training_target": entry["training_target"],
                "generated": str(generated_path),
                "latent": str(latent_path),
                "metrics": metrics,
            }
            (sample_dir / "metrics.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            records.append(record)
    finally:
        vae.close()
        _empty_accelerator_cache(device)

    average, worst = _aggregate(records)
    contact_sheet = output_dir / "training_set_comparison.png"
    _contact_sheet(records, contact_sheet)
    report = {
        "schema_version": 1,
        "status": "completed",
        "mode": "lance-stage-c-training-set-inference",
        "checkpoint": str(checkpoint),
        "weights": "ema" if args.ema_weights else "model",
        "input_manifest": str(manifest_path),
        "dataset_root": manifest.get("dataset_root"),
        "sample_count": len(records),
        "sampling": {
            "num_steps": args.num_steps,
            "timestep_shift": args.timestep_shift,
            "timestep_schedule": args.timestep_schedule,
            "base_seed": args.seed,
            "cfg_text_scale": 1.0,
        },
        "average": average,
        "worst": worst,
        "load_report": load_report,
        "contact_sheet": str(contact_sheet),
        "predictions": records,
    }
    final_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"average": average, "worst": worst}, indent=2), flush=True)
    print("Completed Stage-C training-set inference: {}".format(final_report), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LanceNativeInferenceError, ValueError, RuntimeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)
