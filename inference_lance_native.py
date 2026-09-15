#!/usr/bin/env python3
"""Native MindSpeed-MM Lance T2I/T2V inference from a training DCP."""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import List

os.environ.setdefault("NON_MEGATRON", "true")

import torch

from mindspeed_mm.models.omni.lance.modeling_lance import LanceNativeModel
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.native_inference import (
    LanceNativeInferenceError,
    generation_geometry,
    load_native_generation_dcp,
    prepared_sample_to_denoise_context,
    resolve_native_dcp,
    unpatchify_lance_latents,
)
from mindspeed_mm.models.omni.lance.npu_attention import AscendBlockAttentionBackend
from mindspeed_mm.models.omni.lance.preprocessing import (
    LanceEncodedVisual,
    build_generation_sample,
    prepare_lance_tokenizer,
)
from mindspeed_mm.models.omni.lance.sampling import sample_native_lance


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="DCP root or iter_XXXXXXX directory")
    parser.add_argument("--qwen-path", required=True, help="Qwen2.5-VL tokenizer/config directory")
    parser.add_argument("--vae-path", required=True, help="Wan2.2_VAE.pth")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", choices=("t2i", "t2v"), required=True)
    parser.add_argument("--prompt", action="append", default=[], help="prompt; may be repeated")
    parser.add_argument("--prompt-file", help="JSON object/list containing prompts")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--variant", choices=("image", "video"), default="video")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--timestep-shift", type=float, default=3.5)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-start", type=float, default=0.4)
    parser.add_argument("--cfg-end", type=float, default=1.0)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", choices=("global", "channel", "none"), default="global")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--latent-patch-size", nargs=3, type=int, default=(1, 2, 2),
        metavar=("T", "H", "W"),
    )
    parser.add_argument("--max-latent-size", type=int, default=64)
    parser.add_argument("--max-num-frames", type=int, default=121)
    parser.add_argument(
        "--model-weights",
        action="store_true",
        help="use ordinary model weights instead of EMA",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _read_prompts(arguments) -> List[str]:
    prompts = [str(value).strip() for value in arguments.prompt if str(value).strip()]
    if arguments.prompt_file:
        source = Path(arguments.prompt_file).expanduser().resolve()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LanceNativeInferenceError(
                "cannot read prompt JSON {}: {}".format(source, exc)
            ) from exc
        if isinstance(payload, dict):
            values = list(payload.values())
        elif isinstance(payload, list):
            values = payload
        else:
            raise LanceNativeInferenceError("prompt JSON must be an object or list")
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise LanceNativeInferenceError("every prompt-file value must be a non-empty string")
        prompts.extend(value.strip() for value in values)
    if not prompts:
        raise LanceNativeInferenceError("provide at least one --prompt or --prompt-file")
    return prompts


def _resolve_qwen_config(qwen_path: Path) -> Path:
    candidates = (qwen_path / "config.json", qwen_path / "llm_config.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise LanceNativeInferenceError(
        "Qwen path contains neither config.json nor llm_config.json: {}".format(qwen_path)
    )


def _set_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "npu":
        raise LanceNativeInferenceError(
            "production native Lance inference currently requires an Ascend NPU device"
        )
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise LanceNativeInferenceError("native Ascend inference requires torch_npu") from exc
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise LanceNativeInferenceError("no Ascend NPU is available")
    torch.npu.set_device(device)
    return device


def _empty_accelerator_cache(device: torch.device) -> None:
    module = getattr(torch, device.type, None)
    empty = getattr(module, "empty_cache", None)
    if callable(empty):
        empty()


def _build_model(config, device):
    attention = AscendBlockAttentionBackend()
    # Construct directly on the target device so non-persistent RoPE buffers,
    # which are intentionally absent from DCP, retain their initialized values.
    # T2I/T2V do not need the frozen vision encoder.
    model = LanceNativeModel(
        config,
        attention_backend=attention,
        include_vit_model=False,
        device=device,
        dtype=torch.bfloat16,
    )
    return model.eval()


def _seed_accelerator(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    module = getattr(torch, device.type, None)
    manual_seed = getattr(module, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(seed)


def _build_vae(vae_path, device):
    from mindspeed_mm.models.omni.lance.wan_vae import LanceWanVAE

    return LanceWanVAE(
        vae_path,
        device=device,
        dtype=torch.bfloat16,
        sample_posterior=False,
        encoder_only=False,
    )


def _decode_latents(latents, geometry, vae, device):
    value = latents.permute(3, 0, 1, 2).unsqueeze(0).to(
        device=device, dtype=torch.bfloat16
    )
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        decoded = vae.vae.decode(value)
    decoded = decoded[0].permute(1, 2, 3, 0).float().cpu()
    expected = (geometry.frames, geometry.height, geometry.width, 3)
    if tuple(decoded.shape) != expected:
        raise LanceNativeInferenceError(
            "Wan2.2 decoded shape {}, expected {}".format(tuple(decoded.shape), expected)
        )
    return decoded.add(1.0).mul(127.5).round().clamp(0, 255).to(torch.uint8).numpy()


def _save_media(frames, destination: Path, task: str, fps: int) -> None:
    import imageio.v2 as imageio

    if destination.exists():
        raise LanceNativeInferenceError("refusing to overwrite output: {}".format(destination))
    if task == "t2i":
        imageio.imwrite(destination, frames[0], format="png")
    else:
        imageio.mimsave(destination, frames, fps=fps, format="mp4", quality=6)


def main(argv=None) -> int:
    args = parse_arguments(argv)
    prompts = _read_prompts(args)
    if args.num_steps <= 0 or args.timestep_shift <= 0 or args.cfg_text_scale < 1:
        raise LanceNativeInferenceError(
            "num-steps/timestep-shift must be positive and cfg-text-scale must be >= 1"
        )
    if not 0 <= args.cfg_start <= args.cfg_end <= 1:
        raise LanceNativeInferenceError("CFG interval must satisfy 0 <= start <= end <= 1")
    if args.fps <= 0:
        raise LanceNativeInferenceError("fps must be positive")
    if args.task == "t2v" and args.variant != "video":
        raise LanceNativeInferenceError("t2v requires --variant video")

    qwen_path = Path(args.qwen_path).expanduser().resolve()
    vae_path = Path(args.vae_path).expanduser().resolve()
    if not qwen_path.is_dir():
        raise LanceNativeInferenceError("Qwen directory not found: {}".format(qwen_path))
    if not vae_path.is_file():
        raise LanceNativeInferenceError("Wan2.2 VAE checkpoint not found: {}".format(vae_path))
    qwen_config = _resolve_qwen_config(qwen_path)
    checkpoint = resolve_native_dcp(args.checkpoint)

    config = LanceNativeConfig.from_llm_config(qwen_config, variant=args.variant).with_overrides(
        latent_patch_size=tuple(args.latent_patch_size),
        max_latent_size=args.max_latent_size,
        max_num_frames=args.max_num_frames,
    )
    default_height, default_width, default_frames = (
        (768, 768, 1) if args.task == "t2i" else (480, 864, 49)
    )
    geometry = generation_geometry(
        args.task,
        default_frames if args.num_frames is None else args.num_frames,
        default_height if args.height is None else args.height,
        default_width if args.width is None else args.width,
        config,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    summary = {
        "schema_version": 1,
        "status": "dry-run" if args.dry_run else "running",
        "mode": "native-mindspeed-mm-lance-generation",
        "task": args.task,
        "checkpoint": str(checkpoint),
        "weights": "model" if args.model_weights else "ema",
        "qwen_path": str(qwen_path),
        "vae_path": str(vae_path),
        "output_dir": str(output_dir),
        "prompt_count": len(prompts),
        "geometry": dict(geometry.__dict__),
        "sampling": {
            "num_steps": args.num_steps,
            "timestep_shift": args.timestep_shift,
            "cfg_text_scale": args.cfg_text_scale,
            "cfg_interval": [args.cfg_start, args.cfg_end],
            "cfg_renorm_min": args.cfg_renorm_min,
            "cfg_renorm_type": args.cfg_renorm_type,
            "seed": args.seed,
            "kv_cache": False,
        },
        "config": config.to_dict(),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    suffix = ".png" if args.task == "t2i" else ".mp4"
    planned_outputs = [
        output_dir / ("{:06d}{}".format(index, suffix))
        for index in range(len(prompts))
    ]
    planned_outputs.append(output_dir / "lance_native_inference.json")
    existing = [str(path) for path in planned_outputs if path.exists()]
    if existing:
        raise LanceNativeInferenceError(
            "refusing to overwrite existing native inference outputs: {}".format(
                existing[:8]
            )
        )

    device = _set_device(args.device)
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
        model,
        checkpoint,
        use_ema=not args.model_weights,
    )
    summary["load_report"] = load_report
    print(
        "Loaded {} weights directly from {}".format(
            load_report["weights"], load_report["checkpoint"]
        ),
        flush=True,
    )

    placeholder = LanceEncodedVisual(
        geometry.modality,
        vae_latent=torch.zeros(
            *geometry.latent_shape,
            config.latent_channels,
            dtype=torch.bfloat16,
        ),
    )
    sampled_latents = []
    for index, prompt in enumerate(prompts):
        conditional_sample = build_generation_sample(
            "conditional-{:06d}".format(index), prompt, placeholder, tokenizer, config
        )
        unconditional_sample = build_generation_sample(
            "unconditional-{:06d}".format(index), None, placeholder, tokenizer, config
        )
        conditional_sample.validate(config)
        unconditional_sample.validate(config)
        conditional = prepared_sample_to_denoise_context(conditional_sample, device)
        unconditional = prepared_sample_to_denoise_context(unconditional_sample, device)
        sample_seed = args.seed + index
        _seed_accelerator(sample_seed, device)
        initial = torch.randn(
            conditional_sample.clean_latents.shape,
            device=device,
            dtype=torch.bfloat16,
        )
        print(
            "Sampling {}/{} (seed={}, tokens={})...".format(
                index + 1, len(prompts), sample_seed, conditional.token_ids.numel()
            ),
            flush=True,
        )
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            result = sample_native_lance(
                model,
                conditional,
                initial,
                num_steps=args.num_steps,
                timestep_shift=args.timestep_shift,
                text_unconditional_context=unconditional,
                cfg_interval=(args.cfg_start, args.cfg_end),
                text_scale=args.cfg_text_scale,
                renorm_min=args.cfg_renorm_min,
                renorm_type=args.cfg_renorm_type,
            )
        sampled_latents.append(
            unpatchify_lance_latents(result.cpu(), geometry, config)
        )
        del result, initial, conditional, unconditional

    # Decode after releasing the MoT model, so a single NPU does not need to
    # keep the complete Lance model and Wan2.2 VAE resident simultaneously.
    del model
    _empty_accelerator_cache(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    vae = _build_vae(vae_path, device)
    try:
        for index, latent in enumerate(sampled_latents):
            print("Decoding {}/{}...".format(index + 1, len(sampled_latents)), flush=True)
            frames = _decode_latents(latent, geometry, vae, device)
            destination = output_dir / ("{:06d}{}".format(index, suffix))
            _save_media(frames, destination, args.task, args.fps)
            outputs.append({
                "path": str(destination),
                "prompt": prompts[index],
                "seed": args.seed + index,
            })
    finally:
        vae.close()
        _empty_accelerator_cache(device)

    summary["status"] = "completed"
    summary["outputs"] = outputs
    manifest = output_dir / "lance_native_inference.json"
    manifest.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Completed native Lance inference: {}".format(manifest), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LanceNativeInferenceError, ValueError, RuntimeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)
