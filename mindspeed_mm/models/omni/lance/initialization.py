"""Initialization and release-checkpoint loading for native Lance."""

from pathlib import Path
from typing import Any, Dict, Mapping, Union

import torch

from .modeling_lance import LanceNativeModel
from .checkpoint import audit_checkpoint_metadata, read_safetensors_header, sha256_file


class LanceInitializationError(ValueError):
    pass


def load_native_lance_checkpoint(
    model: LanceNativeModel,
    checkpoint: Union[str, Path],
    *,
    fingerprint: bool = False,
) -> Dict[str, Any]:
    """Stream an official safetensors checkpoint into an allocated model.

    Only one source tensor is materialized on CPU at a time.  A complete
    payload/name/shape/BF16 audit runs before the first parameter is mutated.
    """

    path = Path(checkpoint).expanduser().resolve()
    if path.is_dir():
        path = path / "model.safetensors"
    header = read_safetensors_header(path)
    audit = audit_checkpoint_metadata(header, model.config)
    if not audit["valid"]:
        raise LanceInitializationError(
            "checkpoint failed the native Lance {} contract".format(model.config.variant)
        )
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise LanceInitializationError("native checkpoint loading requires safetensors") from exc

    targets = dict(model.named_parameters())
    expected_names = set(header.tensors)
    if set(targets) != expected_names:
        raise LanceInitializationError("allocated model parameter tree does not match the checkpoint")
    if any(parameter.is_meta for parameter in targets.values()):
        raise LanceInitializationError("streaming load requires materialized model parameters")

    loaded_bytes = 0
    with torch.no_grad(), safe_open(str(path), framework="pt", device="cpu") as source:
        if set(source.keys()) != expected_names:
            raise LanceInitializationError("safetensors keys changed after header audit")
        for name in sorted(expected_names):
            value = source.get_tensor(name)
            target = targets[name]
            if value.shape != target.shape or value.dtype != torch.bfloat16:
                raise LanceInitializationError("checkpoint tensor changed after header audit: {}".format(name))
            target.copy_(value.to(device=target.device, dtype=target.dtype))
            loaded_bytes += value.numel() * value.element_size()
            del value
    return {
        "status": "loaded",
        "variant": model.config.variant,
        "checkpoint": str(path),
        "sha256": sha256_file(path) if fingerprint else None,
        "tensor_count": len(expected_names),
        "tensor_bytes": loaded_bytes,
        "streaming": True,
        "audit": audit,
    }


def copy_understanding_to_generation(model: LanceNativeModel) -> Dict[str, Any]:
    """Initialize every generation-expert tensor from its understanding twin."""

    parameters = dict(model.named_parameters())
    copied = []
    missing = []
    with torch.no_grad():
        for name, target in parameters.items():
            if "_moe_gen" not in name:
                continue
            source_name = name.replace("_moe_gen", "")
            source = parameters.get(source_name)
            if source is None:
                missing.append({"target": name, "source": source_name})
                continue
            if source.shape != target.shape:
                raise LanceInitializationError(
                    "expert initialization shape mismatch: {} <- {}".format(name, source_name)
                )
            target.copy_(source)
            copied.append({"target": name, "source": source_name})
    if missing:
        raise LanceInitializationError(
            "{} generation tensors have no understanding twin".format(len(missing))
        )
    return {"policy": "copy-understanding-to-generation", "copied": copied, "count": len(copied)}


def qwen_vl_target_name(source_name: str) -> str:
    """Match the released Lance VLM initialization rename rule."""

    if "visual" in source_name:
        return source_name.replace("visual", "vit_model", 1)
    return "language_model." + source_name


def initialize_from_qwen_vl_state_dict(
    model: LanceNativeModel,
    qwen_state_dict: Mapping[str, torch.Tensor],
    copy_generation_expert: bool = True,
) -> Dict[str, Any]:
    """Load matching Qwen2.5-VL tensors, then clone the generation expert.

    Bridge layers, timestep embedding, latent position table, and QK norms that
    are absent from the source VLM remain at their native initialization.
    """

    targets = dict(model.named_parameters())
    loaded = []
    unexpected = []
    shape_mismatches = []
    with torch.no_grad():
        for source_name, source in qwen_state_dict.items():
            target_name = qwen_vl_target_name(source_name)
            target = targets.get(target_name)
            if target is None:
                unexpected.append(source_name)
                continue
            if tuple(target.shape) != tuple(source.shape):
                shape_mismatches.append(
                    {
                        "source": source_name,
                        "target": target_name,
                        "source_shape": list(source.shape),
                        "target_shape": list(target.shape),
                    }
                )
                continue
            target.copy_(source.to(device=target.device, dtype=target.dtype))
            loaded.append({"source": source_name, "target": target_name})
    if shape_mismatches:
        raise LanceInitializationError(
            "Qwen2.5-VL initialization has {} shape mismatches".format(len(shape_mismatches))
        )
    expert_report = copy_understanding_to_generation(model) if copy_generation_expert else None
    return {
        "policy": "qwen2.5-vl",
        "loaded": loaded,
        "loaded_count": len(loaded),
        "unexpected": sorted(unexpected),
        "unexpected_count": len(unexpected),
        "generation_expert": expert_report,
    }


def initialize_random(model: LanceNativeModel, seed: int) -> Dict[str, Any]:
    """Reinitialize trainable matrix/vector parameters for a strict random run."""

    torch.manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif module.__class__.__name__ == "LanceRMSNorm":
                torch.nn.init.ones_(module.weight)
    return {
        "policy": "strict-random",
        "seed": seed,
        "generation_expert_copied": False,
        "note": "Frozen ViT/VAE initialization is external and not randomized by this function.",
    }
