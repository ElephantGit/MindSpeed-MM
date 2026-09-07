"""Safetensors metadata audit and conversion planning for Lance checkpoints."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .native_config import LanceNativeConfig


class LanceCheckpointError(ValueError):
    """Raised when a checkpoint violates the native Lance contract."""


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


OFFICIAL_CHECKPOINTS: Dict[str, Dict[str, Any]] = {
    "image": {
        "repository": "bytedance-research/Lance",
        "subdirectory": "Lance_3B",
        "filename": "model.safetensors",
        "tensor_count": 1021,
        "elements": 6185205808,
        "tensor_bytes": 12370411616,
        "sha256": "a2cfed3992486699aa550c1ea9b3519bd19dde475a0992daf2249f2486b268a3",
    },
    "video": {
        "repository": "bytedance-research/Lance",
        "subdirectory": "Lance_3B_Video",
        "filename": "model.safetensors",
        "tensor_count": 1411,
        "elements": 7105548336,
        "tensor_bytes": 14211096672,
        "sha256": "7f0550e1d1511b29a4740a67c1e18e176302a4ecb3177c8a5850ff5fe6447c25",
    },
}


def _numel(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0:
            raise LanceCheckpointError("invalid tensor shape: {}".format(list(shape)))
        result *= dimension
    return result


@dataclass(frozen=True)
class SafetensorsHeader:
    tensors: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]
    header_length: int
    file_size: Optional[int] = None

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    @property
    def elements(self) -> int:
        return sum(_numel(spec["shape"]) for spec in self.tensors.values())

    @property
    def tensor_bytes(self) -> int:
        return sum(int(spec["data_offsets"][1]) - int(spec["data_offsets"][0]) for spec in self.tensors.values())


def read_safetensors_header(
    path: Union[str, Path],
    max_header_bytes: int = 64 * 1024 * 1024,
) -> SafetensorsHeader:
    """Read only the JSON header of a safetensors file.

    This intentionally does not import torch or safetensors and does not map the
    multi-gigabyte tensor payload into memory.
    """

    checkpoint = Path(path).expanduser().resolve()
    try:
        with checkpoint.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise LanceCheckpointError("safetensors file is shorter than its length prefix")
            header_length = struct.unpack("<Q", length_bytes)[0]
            if header_length <= 1 or header_length > max_header_bytes:
                raise LanceCheckpointError("unsafe safetensors header length: {}".format(header_length))
            raw_header = stream.read(header_length)
            if len(raw_header) != header_length:
                raise LanceCheckpointError("truncated safetensors JSON header")
    except OSError as exc:
        raise LanceCheckpointError("could not read checkpoint {}: {}".format(checkpoint, exc)) from exc

    try:
        decoded = json.loads(raw_header.decode("utf-8").rstrip(" \t\r\n\0"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise LanceCheckpointError("invalid safetensors JSON header: {}".format(exc)) from exc
    if not isinstance(decoded, dict):
        raise LanceCheckpointError("safetensors header must be a JSON object")
    metadata = decoded.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise LanceCheckpointError("safetensors __metadata__ must be an object")
    tensors: Dict[str, Dict[str, Any]] = {}
    for name, spec in decoded.items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            raise LanceCheckpointError("invalid safetensors tensor entry")
        if set(("dtype", "shape", "data_offsets")) - set(spec):
            raise LanceCheckpointError("tensor {} has an incomplete header".format(name))
        shape = spec["shape"]
        offsets = spec["data_offsets"]
        if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise LanceCheckpointError("tensor {} has an invalid shape or offset".format(name))
        _numel(shape)
        if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in offsets):
            raise LanceCheckpointError("tensor {} has invalid data offsets".format(name))
        if offsets[1] < offsets[0]:
            raise LanceCheckpointError("tensor {} has descending data offsets".format(name))
        tensors[name] = dict(spec)
    return SafetensorsHeader(
        tensors=tensors,
        metadata=metadata,
        header_length=header_length,
        file_size=checkpoint.stat().st_size,
    )


def expected_state_shapes(config: LanceNativeConfig) -> Dict[str, Tuple[int, ...]]:
    """Return the exact upstream state-dict contract for a Lance release."""

    shapes: Dict[str, Tuple[int, ...]] = {}
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    kv_dim = config.kv_dim
    head_dim = config.head_dim
    for layer_index in range(config.num_hidden_layers):
        prefix = "language_model.model.layers.{}.".format(layer_index)
        for norm in (
            "input_layernorm",
            "input_layernorm_moe_gen",
            "post_attention_layernorm",
            "post_attention_layernorm_moe_gen",
        ):
            shapes[prefix + norm + ".weight"] = (hidden,)
        for mlp in ("mlp", "mlp_moe_gen"):
            shapes[prefix + mlp + ".gate_proj.weight"] = (intermediate, hidden)
            shapes[prefix + mlp + ".up_proj.weight"] = (intermediate, hidden)
            shapes[prefix + mlp + ".down_proj.weight"] = (hidden, intermediate)
        for suffix in ("", "_moe_gen"):
            attention = prefix + "self_attn."
            shapes[attention + "q_proj" + suffix + ".weight"] = (hidden, hidden)
            shapes[attention + "q_proj" + suffix + ".bias"] = (hidden,)
            shapes[attention + "k_proj" + suffix + ".weight"] = (kv_dim, hidden)
            shapes[attention + "k_proj" + suffix + ".bias"] = (kv_dim,)
            shapes[attention + "v_proj" + suffix + ".weight"] = (kv_dim, hidden)
            shapes[attention + "v_proj" + suffix + ".bias"] = (kv_dim,)
            shapes[attention + "o_proj" + suffix + ".weight"] = (hidden, hidden)
            shapes[attention + "q_norm" + suffix + ".weight"] = (head_dim,)
            shapes[attention + "k_norm" + suffix + ".weight"] = (head_dim,)

    shapes.update(
        {
            "language_model.lm_head.weight": (config.vocab_size, hidden),
            "language_model.model.embed_tokens.weight": (config.vocab_size, hidden),
            "language_model.model.norm.weight": (hidden,),
            "language_model.model.norm_moe_gen.weight": (hidden,),
            "latent_pos_embed.pos_embed": (config.latent_position_count, hidden),
            "llm2vae.weight": (config.patch_latent_dim, hidden),
            "llm2vae.bias": (config.patch_latent_dim,),
            "vae2llm.weight": (hidden, config.patch_latent_dim),
            "vae2llm.bias": (hidden,),
            "time_embedder.mlp.0.weight": (hidden, 256),
            "time_embedder.mlp.0.bias": (hidden,),
            "time_embedder.mlp.2.weight": (hidden, hidden),
            "time_embedder.mlp.2.bias": (hidden,),
        }
    )

    if config.has_vit:
        for layer_index in range(config.vit_depth):
            prefix = "vit_model.blocks.{}.".format(layer_index)
            shapes[prefix + "attn.proj.bias"] = (config.vit_hidden_size,)
            shapes[prefix + "attn.proj.weight"] = (config.vit_hidden_size, config.vit_hidden_size)
            shapes[prefix + "attn.qkv.bias"] = (3 * config.vit_hidden_size,)
            shapes[prefix + "attn.qkv.weight"] = (3 * config.vit_hidden_size, config.vit_hidden_size)
            shapes[prefix + "mlp.down_proj.bias"] = (config.vit_hidden_size,)
            shapes[prefix + "mlp.down_proj.weight"] = (config.vit_hidden_size, config.vit_intermediate_size)
            shapes[prefix + "mlp.gate_proj.bias"] = (config.vit_intermediate_size,)
            shapes[prefix + "mlp.gate_proj.weight"] = (config.vit_intermediate_size, config.vit_hidden_size)
            shapes[prefix + "mlp.up_proj.bias"] = (config.vit_intermediate_size,)
            shapes[prefix + "mlp.up_proj.weight"] = (config.vit_intermediate_size, config.vit_hidden_size)
            shapes[prefix + "norm1.weight"] = (config.vit_hidden_size,)
            shapes[prefix + "norm2.weight"] = (config.vit_hidden_size,)
        patch_kernel = (config.vit_temporal_patch_size, config.vit_patch_size, config.vit_patch_size)
        merged = config.vit_hidden_size * config.vit_spatial_merge_size ** 2
        shapes.update(
            {
                "vit_model.patch_embed.proj.weight": (
                    config.vit_hidden_size,
                    config.vit_in_channels,
                ) + patch_kernel,
                "vit_model.merger.ln_q.weight": (config.vit_hidden_size,),
                "vit_model.merger.mlp.0.weight": (merged, merged),
                "vit_model.merger.mlp.0.bias": (merged,),
                "vit_model.merger.mlp.2.weight": (config.vit_out_hidden_size, merged),
                "vit_model.merger.mlp.2.bias": (config.vit_out_hidden_size,),
            }
        )
    return shapes


def contract_totals(shapes: Mapping[str, Sequence[int]], dtype: str = "BF16") -> Dict[str, int]:
    if dtype not in DTYPE_BYTES:
        raise LanceCheckpointError("unsupported dtype: {}".format(dtype))
    elements = sum(_numel(shape) for shape in shapes.values())
    return {"tensor_count": len(shapes), "elements": elements, "tensor_bytes": elements * DTYPE_BYTES[dtype]}


def audit_checkpoint_metadata(
    header: SafetensorsHeader,
    config: LanceNativeConfig,
    allow_rebuild_position_embedding: bool = False,
    metadata_only: bool = False,
) -> Dict[str, Any]:
    expected = expected_state_shapes(config)
    actual_names = set(header.tensors)
    expected_names = set(expected)
    rebuildable = {"latent_pos_embed.pos_embed"} if allow_rebuild_position_embedding else set()
    missing = sorted((expected_names - actual_names) - rebuildable)
    rebuild = sorted((expected_names - actual_names) & rebuildable)
    unexpected = sorted(actual_names - expected_names)
    shape_mismatches: List[Dict[str, Any]] = []
    dtype_mismatches: List[Dict[str, Any]] = []
    offset_mismatches: List[Dict[str, Any]] = []
    intervals: List[Tuple[int, int, str]] = []
    for name in sorted(actual_names & expected_names):
        spec = header.tensors[name]
        actual_shape = tuple(spec["shape"])
        if actual_shape != expected[name]:
            shape_mismatches.append(
                {"name": name, "expected": list(expected[name]), "actual": list(actual_shape)}
            )
        dtype = spec["dtype"]
        if dtype != "BF16":
            dtype_mismatches.append({"name": name, "expected": "BF16", "actual": dtype})
        start, end = spec["data_offsets"]
        intervals.append((start, end, name))
        if dtype in DTYPE_BYTES:
            expected_bytes = _numel(actual_shape) * DTYPE_BYTES[dtype]
            if end - start != expected_bytes:
                offset_mismatches.append(
                    {"name": name, "expected_bytes": expected_bytes, "actual_bytes": end - start}
                )
        else:
            dtype_mismatches.append({"name": name, "expected": "known safetensors dtype", "actual": dtype})
    intervals.sort()
    if intervals and intervals[0][0] != 0:
        offset_mismatches.append(
            {"name": intervals[0][2], "error": "payload must start at offset zero"}
        )
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            offset_mismatches.append(
                {"name": current[2], "error": "overlap", "overlaps": previous[2]}
            )
        elif current[0] > previous[1]:
            offset_mismatches.append(
                {"name": current[2], "error": "gap", "previous": previous[2]}
            )

    expected_file_size = None
    file_size_mismatch = None
    if intervals:
        expected_file_size = 8 + header.header_length + max(interval[1] for interval in intervals)
    if not metadata_only and header.file_size is not None and expected_file_size != header.file_size:
        file_size_mismatch = {
            "expected": expected_file_size,
            "actual": header.file_size,
            "error": "safetensors payload is truncated or has trailing bytes",
        }

    valid = not any(
        (missing, unexpected, shape_mismatches, dtype_mismatches, offset_mismatches, file_size_mismatch)
    )
    return {
        "status": "valid" if valid else "invalid",
        "variant": config.variant,
        "valid": valid,
        "allow_rebuild_position_embedding": allow_rebuild_position_embedding,
        "metadata_only": metadata_only,
        "rebuild": rebuild,
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "offset_mismatches": offset_mismatches,
        "file_size": header.file_size,
        "expected_file_size": expected_file_size,
        "file_size_mismatch": file_size_mismatch,
        "actual": {
            "tensor_count": header.tensor_count,
            "elements": header.elements,
            "tensor_bytes": header.tensor_bytes,
        },
        "expected": contract_totals(expected),
        "official": dict(OFFICIAL_CHECKPOINTS[config.variant]),
    }


def build_checkpoint_conversion_plan(
    header: SafetensorsHeader,
    config: LanceNativeConfig,
    allow_rebuild_position_embedding: bool = False,
    metadata_only: bool = False,
) -> Dict[str, Any]:
    audit = audit_checkpoint_metadata(
        header,
        config,
        allow_rebuild_position_embedding,
        metadata_only=metadata_only,
    )
    if not audit["valid"]:
        raise LanceCheckpointError("checkpoint metadata does not satisfy the Lance {} contract".format(config.variant))
    groups: Dict[str, List[str]] = {"language": [], "vision": [], "bridge": [], "rebuild": []}
    for name in sorted(header.tensors):
        if name.startswith("language_model."):
            groups["language"].append(name)
        elif name.startswith("vit_model."):
            groups["vision"].append(name)
        else:
            groups["bridge"].append(name)
    groups["rebuild"].extend(audit["rebuild"])
    return {
        "format_version": 1,
        "source_format": "safetensors",
        "target_format": "torch-distributed-checkpoint",
        "variant": config.variant,
        "mapping": "identity",
        "lossless": True,
        "audit": audit,
        "groups": groups,
        "notes": [
            "Native Lance parameter names intentionally match the released state dict.",
            "latent_pos_embed.pos_embed is a deterministic 3D sin/cos table and may be rebuilt.",
            "TP/CP/DP sharding is applied by the target distributed checkpoint writer, not by renaming tensors.",
        ],
    }


def sha256_file(path: Union[str, Path], chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
