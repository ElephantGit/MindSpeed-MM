import os

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

torch = pytest.importorskip("torch")

from mindspeed_mm.models.omni.lance.initialization import (
    copy_understanding_to_generation,
    initialize_from_qwen_vl_state_dict,
    initialize_random,
    qwen_vl_target_name,
)
from mindspeed_mm.models.omni.lance.modeling_lance import LanceNativeModel
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig


def _config():
    return LanceNativeConfig(
        vocab_size=31,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=10000.0,
        mrope_section=(1, 1, 2),
        latent_channels=8,
        max_latent_size=2,
        vit_depth=1,
        vit_hidden_size=8,
        vit_intermediate_size=16,
        vit_num_heads=2,
        vit_patch_size=2,
        vit_temporal_patch_size=2,
        vit_out_hidden_size=32,
    )


def test_all_generation_parameters_copy_from_understanding_twins():
    model = LanceNativeModel(_config())
    with torch.no_grad():
        model.language_model.model.layers[0].self_attn.q_proj.weight.fill_(0.125)
        model.language_model.model.layers[0].self_attn.q_proj_moe_gen.weight.zero_()
    report = copy_understanding_to_generation(model)
    assert report["count"] == 15
    torch.testing.assert_close(
        model.language_model.model.layers[0].self_attn.q_proj_moe_gen.weight,
        model.language_model.model.layers[0].self_attn.q_proj.weight,
    )


def test_qwen_vl_loading_uses_released_rename_and_then_initializes_generation():
    model = LanceNativeModel(_config())
    source = {
        "model.layers.0.self_attn.q_proj.weight": torch.full((32, 32), 0.75),
        "model.embed_tokens.weight": torch.full((31, 32), 0.25),
        "lm_head.weight": torch.full((31, 32), 0.5),
    }
    report = initialize_from_qwen_vl_state_dict(model, source)
    assert report["loaded_count"] == 3
    assert qwen_vl_target_name("visual.blocks.0.norm1.weight") == "vit_model.blocks.0.norm1.weight"
    layer = model.language_model.model.layers[0]
    torch.testing.assert_close(layer.self_attn.q_proj.weight, source["model.layers.0.self_attn.q_proj.weight"])
    torch.testing.assert_close(layer.self_attn.q_proj_moe_gen.weight, layer.self_attn.q_proj.weight)
    torch.testing.assert_close(model.language_model.model.embed_tokens.weight, source["model.embed_tokens.weight"])


def test_strict_random_mode_is_seeded_and_does_not_copy_experts():
    first = LanceNativeModel(_config())
    second = LanceNativeModel(_config())
    first_report = initialize_random(first, 17)
    initialize_random(second, 17)
    assert first_report["generation_expert_copied"] is False
    torch.testing.assert_close(
        first.language_model.model.layers[0].self_attn.q_proj.weight,
        second.language_model.model.layers[0].self_attn.q_proj.weight,
    )
    assert not torch.equal(
        first.language_model.model.layers[0].self_attn.q_proj.weight,
        first.language_model.model.layers[0].self_attn.q_proj_moe_gen.weight,
    )

