"""Initialization policies for native Lance training."""

from typing import Any, Dict, Mapping

import torch

from .modeling_lance import LanceNativeModel


class LanceInitializationError(ValueError):
    pass


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

