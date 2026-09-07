import json
import os
from pathlib import Path
import struct

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

from mindspeed_mm.models.omni.lance.checkpoint import (
    LanceCheckpointError,
    OFFICIAL_CHECKPOINTS,
    SafetensorsHeader,
    audit_checkpoint_metadata,
    build_checkpoint_conversion_plan,
    contract_totals,
    expected_state_shapes,
    read_safetensors_header,
)
from mindspeed_mm.models.omni.lance.native_config import LanceConfigError, LanceNativeConfig


def _header_from_shapes(shapes, omit=(), overrides=None):
    overrides = overrides or {}
    tensors = {}
    offset = 0
    for name, shape in shapes.items():
        if name in omit:
            continue
        actual_shape = overrides.get(name, shape)
        size = 2
        for dimension in actual_shape:
            size *= dimension
        tensors[name] = {"dtype": "BF16", "shape": list(actual_shape), "data_offsets": [offset, offset + size]}
        offset += size
    return SafetensorsHeader(tensors=tensors, metadata={"format": "pt"}, header_length=1)


@pytest.mark.parametrize(
    "variant,tensor_count,elements,tensor_bytes,latent_positions",
    [
        ("image", 1021, 6185205808, 12370411616, 4096),
        ("video", 1411, 7105548336, 14211096672, 126976),
    ],
)
def test_released_checkpoint_contract_totals(variant, tensor_count, elements, tensor_bytes, latent_positions):
    config = LanceNativeConfig.for_variant(variant)
    shapes = expected_state_shapes(config)
    assert config.head_dim == 128
    assert config.kv_dim == 256
    assert config.patch_latent_dim == 48
    assert config.latent_position_count == latent_positions
    assert contract_totals(shapes) == {
        "tensor_count": tensor_count,
        "elements": elements,
        "tensor_bytes": tensor_bytes,
    }
    assert OFFICIAL_CHECKPOINTS[variant]["elements"] == elements


def test_language_layer_has_two_complete_experts():
    shapes = expected_state_shapes(LanceNativeConfig.for_variant("image"))
    layer = [name for name in shapes if name.startswith("language_model.model.layers.0.")]
    assert len(layer) == 28
    assert shapes["language_model.model.layers.0.self_attn.k_proj.weight"] == (256, 2048)
    assert shapes["language_model.model.layers.0.self_attn.k_proj_moe_gen.weight"] == (256, 2048)
    assert shapes["language_model.model.layers.0.self_attn.q_norm_moe_gen.weight"] == (128,)


def test_video_vit_contract_has_390_tensors():
    image_names = set(expected_state_shapes(LanceNativeConfig.for_variant("image")))
    video_shapes = expected_state_shapes(LanceNativeConfig.for_variant("video"))
    assert len(set(video_shapes) - image_names) == 390
    assert video_shapes["vit_model.patch_embed.proj.weight"] == (1280, 3, 2, 14, 14)
    assert video_shapes["vit_model.merger.mlp.0.weight"] == (5120, 5120)
    assert video_shapes["vit_model.merger.mlp.2.weight"] == (2048, 5120)


def test_config_loads_hugging_face_fields_and_validates_mrope():
    raw = {
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
    }
    config = LanceNativeConfig.from_llm_config(raw, variant="video")
    assert config.max_num_frames == 121
    assert config.mrope_section == (16, 24, 24)
    with pytest.raises(LanceConfigError, match="mrope_section"):
        LanceNativeConfig.from_llm_config(
            {"rope_scaling": {"mrope_section": [16, 16, 16]}}, variant="image"
        )


def test_read_safetensors_header_without_loading_payload(tmp_path):
    payload = b"\x00\x01\x02\x03"
    document = {
        "__metadata__": {"format": "pt"},
        "weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
    }
    raw_header = json.dumps(document, separators=(",", ":")).encode("utf-8")
    checkpoint = tmp_path / "tiny.safetensors"
    checkpoint.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)
    header = read_safetensors_header(checkpoint)
    assert header.tensor_count == 1
    assert header.elements == 2
    assert header.tensor_bytes == 4
    assert header.metadata == {"format": "pt"}
    assert header.file_size == 8 + len(raw_header) + len(payload)


def test_header_rejects_truncation_and_unsafe_length(tmp_path):
    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(struct.pack("<Q", 100) + b"{}")
    with pytest.raises(LanceCheckpointError, match="truncated"):
        read_safetensors_header(truncated)

    unsafe = tmp_path / "unsafe.safetensors"
    unsafe.write_bytes(struct.pack("<Q", 1024))
    with pytest.raises(LanceCheckpointError, match="unsafe"):
        read_safetensors_header(unsafe, max_header_bytes=100)


def test_checkpoint_audit_accepts_exact_contract_and_identity_plan():
    config = LanceNativeConfig.for_variant("image")
    header = _header_from_shapes(expected_state_shapes(config))
    audit = audit_checkpoint_metadata(header, config)
    assert audit["valid"]
    plan = build_checkpoint_conversion_plan(header, config)
    assert plan["mapping"] == "identity"
    assert plan["lossless"]
    assert len(plan["groups"]["language"]) == 1012
    assert len(plan["groups"]["bridge"]) == 9


def test_checkpoint_audit_reports_shape_dtype_offset_and_unexpected():
    config = LanceNativeConfig.for_variant("image")
    shapes = expected_state_shapes(config)
    key = "llm2vae.weight"
    header = _header_from_shapes(shapes, overrides={key: (47, 2048)})
    header.tensors[key]["dtype"] = "F16"
    header.tensors[key]["data_offsets"][1] += 2
    header.tensors["unexpected.weight"] = {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
    audit = audit_checkpoint_metadata(header, config)
    assert not audit["valid"]
    assert audit["unexpected"] == ["unexpected.weight"]
    assert audit["shape_mismatches"][0]["name"] == key
    assert audit["dtype_mismatches"][0]["name"] == key
    assert any(item["name"] == key for item in audit["offset_mismatches"])
    with pytest.raises(LanceCheckpointError):
        build_checkpoint_conversion_plan(header, config)


def test_only_deterministic_position_table_can_be_rebuilt():
    config = LanceNativeConfig.for_variant("video")
    shapes = expected_state_shapes(config)
    position = "latent_pos_embed.pos_embed"
    header = _header_from_shapes(shapes, omit=(position,))
    strict = audit_checkpoint_metadata(header, config)
    assert strict["missing"] == [position]
    assert not strict["valid"]
    relaxed = audit_checkpoint_metadata(header, config, allow_rebuild_position_embedding=True)
    assert relaxed["valid"]
    assert relaxed["rebuild"] == [position]


def test_file_backed_audit_rejects_missing_tensor_payload(tmp_path):
    config = LanceNativeConfig.for_variant("image")
    shapes = expected_state_shapes(config)
    complete_header = _header_from_shapes(shapes)
    document = dict(complete_header.tensors)
    raw_header = json.dumps(document, separators=(",", ":")).encode("utf-8")
    checkpoint = tmp_path / "header-only.safetensors"
    checkpoint.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header)
    header = read_safetensors_header(checkpoint)
    strict = audit_checkpoint_metadata(header, config)
    assert not strict["valid"]
    assert strict["file_size_mismatch"]["actual"] == checkpoint.stat().st_size
    assert audit_checkpoint_metadata(header, config, metadata_only=True)["valid"]
