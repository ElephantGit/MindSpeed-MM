import os

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from mindspeed_mm.models.omni.lance.dcp import (
    LanceDCPError,
    convert_safetensors_to_dcp,
    verify_dcp_metadata,
)
from mindspeed_mm.models.omni.lance.modeling_lance import LanceNativeModel
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig


def _tiny_config():
    return LanceNativeConfig(
        vocab_size=41,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=10000.0,
        mrope_section=(1, 1, 2),
        latent_channels=4,
        max_latent_size=2,
        max_num_frames=1,
        vit_depth=1,
        vit_hidden_size=8,
        vit_intermediate_size=16,
        vit_num_heads=2,
        vit_patch_size=2,
        vit_temporal_patch_size=2,
        vit_out_hidden_size=32,
    )


def test_safetensors_to_dcp_metadata_round_trip(tmp_path):
    config = _tiny_config()
    model = LanceNativeModel(config, dtype=torch.bfloat16)
    source = tmp_path / "model.safetensors"
    safetensors_torch.save_file(model.state_dict(), str(source), metadata={"format": "pt"})
    output = tmp_path / "dcp"

    result = convert_safetensors_to_dcp(source, output, config)
    assert result["status"] == "completed"
    assert result["mapping"] == "identity"
    assert result["verification"]["valid"]
    assert (output / "release" / ".metadata").is_file()
    assert (output / "latest_checkpointed_iteration.txt").read_text() == "release"

    verification = verify_dcp_metadata(output / "release", config)
    assert verification["valid"]
    assert verification["tensor_count"] == len(model.state_dict())
    assert not verification["missing"]
    assert not verification["unexpected"]


def test_dcp_conversion_refuses_to_overwrite_existing_output(tmp_path):
    config = _tiny_config()
    model = LanceNativeModel(config, dtype=torch.bfloat16)
    source = tmp_path / "model.safetensors"
    safetensors_torch.save_file(model.state_dict(), str(source))
    output = tmp_path / "dcp"
    output.mkdir()
    (output / "user-data").write_text("preserve", encoding="utf-8")
    with pytest.raises(LanceDCPError, match="refusing to overwrite"):
        convert_safetensors_to_dcp(source, output, config, fingerprint=False)
    assert (output / "user-data").read_text(encoding="utf-8") == "preserve"
