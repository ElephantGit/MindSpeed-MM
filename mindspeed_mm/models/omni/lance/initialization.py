"""Initialization and release-checkpoint loading for native Lance."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

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


def _safetensor_files(path: Union[str, Path]) -> Tuple[Path, ...]:
    """Resolve one Hugging Face safetensors file or a sharded directory."""

    root = Path(path).expanduser().resolve()
    if root.is_file():
        return (root,)
    if not root.is_dir():
        raise LanceInitializationError("initialization path does not exist: {}".format(root))
    index_files = sorted(root.glob("*.safetensors.index.json"))
    if index_files:
        try:
            index = json.loads(index_files[0].read_text(encoding="utf-8"))
            names = sorted(set(index["weight_map"].values()))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise LanceInitializationError(
                "invalid safetensors index {}: {}".format(index_files[0], exc)
            ) from exc
        files = tuple(root / name for name in names)
    else:
        files = tuple(sorted(root.glob("*.safetensors")))
    missing = [str(item) for item in files if not item.is_file()]
    if missing or not files:
        detail = ", ".join(missing) if missing else "no .safetensors files"
        raise LanceInitializationError("cannot resolve weights under {}: {}".format(root, detail))
    return files


def _qwen_uses_tied_word_embeddings(path: Union[str, Path]) -> bool:
    """Return whether the source checkpoint intentionally omits a tied LM head."""

    root = Path(path).expanduser().resolve()
    config_path = (root if root.is_dir() else root.parent) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    text_config = config.get("text_config")
    if isinstance(text_config, Mapping) and "tie_word_embeddings" in text_config:
        return text_config["tie_word_embeddings"] is True
    return config.get("tie_word_embeddings") is True


def _stream_safetensors(files: Iterable[Path]):
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise LanceInitializationError("Qwen initialization requires safetensors") from exc
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as source:
            for name in sorted(source.keys()):
                yield path, name, source.get_tensor(name)


def _vit_target_name(source_name: str) -> str:
    if source_name.startswith("vit_model."):
        return source_name
    if "visual" in source_name:
        return source_name.replace("visual", "vit_model", 1)
    return "vit_model." + source_name


def load_native_vit_checkpoint(vit_model, checkpoint: Union[str, Path]) -> Dict[str, Any]:
    """Stream the extracted Qwen2.5-VL ViT artifact into a native ViT."""

    targets = dict(vit_model.named_parameters())
    if any(parameter.is_meta for parameter in targets.values()):
        raise LanceInitializationError("ViT streaming load requires materialized parameters")
    loaded = set()
    mismatched = []
    unexpected = []
    with torch.no_grad():
        for path, source_name, value in _stream_safetensors(_safetensor_files(checkpoint)):
            normalized = source_name
            for prefix in ("visual.", "vit_model."):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
                    break
            target = targets.get(normalized)
            if target is None:
                unexpected.append(source_name)
                continue
            if tuple(target.shape) != tuple(value.shape):
                mismatched.append(
                    {"source": source_name, "target": normalized, "source_shape": list(value.shape),
                     "target_shape": list(target.shape)}
                )
                continue
            target.copy_(value.to(device=target.device, dtype=target.dtype))
            loaded.add(normalized)
            del value
    missing = sorted(set(targets) - loaded)
    if missing or mismatched:
        raise LanceInitializationError(
            "ViT checkpoint is incompatible: missing={}, shape_mismatches={}".format(
                len(missing), len(mismatched)
            )
        )
    return {
        "status": "loaded",
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "loaded_count": len(loaded),
        "unexpected": sorted(unexpected),
        "streaming": True,
    }


def initialize_from_qwen_vl_files(
    model: LanceNativeModel,
    qwen_path: Union[str, Path],
    *,
    vit_path: Optional[Union[str, Path]] = None,
    copy_generation_expert: bool = True,
    require_complete: bool = True,
) -> Dict[str, Any]:
    """Stream Qwen2.5-VL and optional extracted ViT weights into Lance.

    This is the native replacement for constructing an upstream Transformers
    Lance model and then handing it to an external trainer.  At most one source
    tensor is materialized at a time.  The separate ``vit_path`` is useful for
    the released ``Qwen2.5-VL-ViT/vit.safetensors`` artifact and deliberately
    takes precedence over any visual tensors in the full VLM directory.
    """

    if any(parameter.is_meta for parameter in model.parameters()):
        raise LanceInitializationError("Qwen streaming load requires materialized parameters")
    targets = dict(model.named_parameters())
    loaded: Dict[str, Dict[str, Any]] = {}
    shape_mismatches = []
    unexpected = []

    sources = [("qwen", _safetensor_files(qwen_path), qwen_vl_target_name)]
    if vit_path is not None:
        sources.append(("vit", _safetensor_files(vit_path), _vit_target_name))

    with torch.no_grad():
        for source_kind, files, rename in sources:
            for path, source_name, value in _stream_safetensors(files):
                target_name = rename(source_name)
                target = targets.get(target_name)
                if target is None:
                    unexpected.append(
                        {"source_kind": source_kind, "source": source_name, "file": str(path)}
                    )
                    continue
                if tuple(target.shape) != tuple(value.shape):
                    shape_mismatches.append(
                        {
                            "source_kind": source_kind,
                            "source": source_name,
                            "target": target_name,
                            "source_shape": list(value.shape),
                            "target_shape": list(target.shape),
                        }
                    )
                    continue
                target.copy_(value.to(device=target.device, dtype=target.dtype))
                loaded[target_name] = {
                    "source_kind": source_kind,
                    "source": source_name,
                    "file": str(path),
                }
                del value

        # Hugging Face does not serialize lm_head.weight for Qwen checkpoints
        # whose config declares tied word embeddings.  Lance keeps the output
        # head as an independent parameter, so materialize the tied source
        # value into it before enforcing the complete-initialization contract.
        embed_name = "language_model.model.embed_tokens.weight"
        lm_head_name = "language_model.lm_head.weight"
        tied_lm_head_source = None
        if (
            lm_head_name not in loaded
            and embed_name in loaded
            and _qwen_uses_tied_word_embeddings(qwen_path)
        ):
            embed = targets[embed_name]
            lm_head = targets[lm_head_name]
            if tuple(embed.shape) != tuple(lm_head.shape):
                raise LanceInitializationError(
                    "tied Qwen embedding and Lance LM head shapes do not match: {} != {}".format(
                        tuple(embed.shape), tuple(lm_head.shape)
                    )
                )
            lm_head.copy_(embed)
            tied_lm_head_source = {
                "source_kind": "qwen-tied-embedding",
                "source": loaded[embed_name]["source"],
                "file": loaded[embed_name]["file"],
            }
            loaded[lm_head_name] = tied_lm_head_source

    if shape_mismatches:
        raise LanceInitializationError(
            "Qwen2.5-VL initialization has {} shape mismatches; first: {}".format(
                len(shape_mismatches), shape_mismatches[0]
            )
        )

    # These parameters are Lance additions and intentionally retain their
    # seeded native initialization.  All ordinary understanding-side Qwen and
    # frozen ViT parameters must be sourced from the requested checkpoints.
    native_only_fragments = (
        "_moe_gen", ".q_norm.", ".k_norm.", "vae2llm.", "llm2vae.",
        "time_embedder.", "latent_pos_embed.", "connector.",
    )
    required = {
        name for name in targets
        if not any(fragment in name for fragment in native_only_fragments)
    }
    missing_required = sorted(required - set(loaded))
    if require_complete and missing_required:
        raise LanceInitializationError(
            "Qwen initialization did not provide {} required tensors; first: {}".format(
                len(missing_required), missing_required[0]
            )
        )
    expert_report = copy_understanding_to_generation(model) if copy_generation_expert else None
    return {
        "policy": "qwen2.5-vl-streaming",
        "qwen_path": str(Path(qwen_path).expanduser().resolve()),
        "vit_path": str(Path(vit_path).expanduser().resolve()) if vit_path is not None else None,
        "loaded": loaded,
        "loaded_count": len(loaded),
        "required_count": len(required),
        "missing_required": missing_required,
        "unexpected": unexpected,
        "unexpected_count": len(unexpected),
        "generation_expert": expert_report,
        "tied_lm_head_source": tied_lm_head_source,
        "streaming": True,
    }


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
