import os

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

torch = pytest.importorskip("torch")

from mindspeed_mm.models.omni.lance.checkpoint import expected_state_shapes
from mindspeed_mm.models.omni.lance.modeling_lance import (
    LanceNativeModel,
    lance_3d_sincos_position_embedding,
    reference_sdpa,
)
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.sequence import (
    LanceDocument,
    LancePackedSequence,
    LanceSegment,
)


def _tiny_config(variant="image"):
    return LanceNativeConfig(
        variant=variant,
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rope_theta=10000.0,
        mrope_section=(1, 1, 2),
        latent_channels=8,
        latent_patch_size=(1, 1, 1),
        max_latent_size=4,
        max_num_frames=5 if variant == "video" else 1,
        vit_depth=2,
        vit_hidden_size=16,
        vit_intermediate_size=32,
        vit_num_heads=4,
        vit_in_channels=3,
        vit_patch_size=2,
        vit_temporal_patch_size=2,
        vit_spatial_merge_size=2,
        vit_out_hidden_size=32,
    )


@pytest.mark.parametrize("variant", ["image", "video"])
def test_native_parameter_tree_exactly_matches_checkpoint_contract(variant):
    config = _tiny_config(variant)
    model = LanceNativeModel(config, device="meta", dtype=torch.bfloat16)
    actual = {name: tuple(parameter.shape) for name, parameter in model.state_dict().items()}
    assert actual == expected_state_shapes(config)


@pytest.mark.parametrize("variant", ["image", "video"])
def test_released_full_size_model_can_be_built_on_meta_device(variant):
    config = LanceNativeConfig.for_variant(variant)
    model = LanceNativeModel(config, device="meta", dtype=torch.bfloat16)
    actual = {name: tuple(parameter.shape) for name, parameter in model.state_dict().items()}
    assert actual == expected_state_shapes(config)


def test_tiny_mot_forward_backward_reaches_both_experts():
    torch.manual_seed(7)
    config = _tiny_config()
    model = LanceNativeModel(config)
    packed = LancePackedSequence(
        (
            LanceDocument(
                "sample",
                (
                    LanceSegment(3, "causal", "text", "understanding"),
                    LanceSegment(3, "noise", "vae", "generation"),
                ),
            ),
        )
    )
    mask = torch.tensor(packed.dense_attention_mask(), dtype=torch.bool)
    routes = packed.token_expert_indexes()
    understanding = torch.tensor(routes["understanding"], dtype=torch.long)
    generation = torch.tensor(routes["generation"], dtype=torch.long)
    hidden_states = torch.randn(packed.length, config.hidden_size, requires_grad=True)
    position_ids = torch.arange(packed.length).repeat(3, 1)
    output = model.forward_language(
        hidden_states,
        position_ids,
        mask,
        understanding,
        generation,
    )
    assert output.shape == hidden_states.shape
    target = torch.randn_like(output)
    (output * target).sum().backward()

    layer = model.language_model.model.layers[0]
    assert layer.self_attn.q_proj.weight.grad.abs().sum().item() > 0
    assert layer.self_attn.q_proj_moe_gen.weight.grad.abs().sum().item() > 0
    assert layer.mlp.gate_proj.weight.grad.abs().sum().item() > 0
    assert layer.mlp_moe_gen.gate_proj.weight.grad.abs().sum().item() > 0
    assert hidden_states.grad.abs().sum().item() > 0


def test_reference_attention_supports_non_equal_gqa_heads():
    torch.manual_seed(11)
    query = torch.randn(5, 4, 8, requires_grad=True)
    key = torch.randn(5, 2, 8, requires_grad=True)
    value = torch.randn(5, 2, 8, requires_grad=True)
    mask = torch.ones(5, 5, dtype=torch.bool).tril()
    output = reference_sdpa(query, key, value, mask)
    assert output.shape == query.shape
    output.sum().backward()
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None


def test_invalid_expert_routing_is_rejected_before_projection():
    config = _tiny_config()
    model = LanceNativeModel(config)
    hidden_states = torch.randn(2, config.hidden_size)
    position_ids = torch.arange(2).repeat(3, 1)
    mask = torch.ones(2, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="cover"):
        model.forward_language(
            hidden_states,
            position_ids,
            mask,
            torch.tensor([0]),
            torch.tensor([], dtype=torch.long),
        )


def test_latent_position_table_matches_released_numpy_formula():
    np = pytest.importorskip("numpy")
    hidden_size, frames, height, width = 32, 2, 3, 4

    grid = np.stack(
        np.meshgrid(
            np.arange(frames, dtype=np.float32),
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        ),
        axis=0,
    )
    axis_size = hidden_size // 3
    axis_size = axis_size if axis_size % 2 == 0 else axis_size - 1
    dimensions = (axis_size, axis_size, hidden_size - 2 * axis_size)
    expected_axes = []
    for dimension, coordinate in zip(dimensions, grid):
        omega = np.arange(dimension // 2, dtype=np.float64)
        omega /= dimension / 2.0
        omega = 1.0 / 10000**omega
        angles = np.einsum("m,d->md", coordinate.reshape(-1), omega)
        expected_axes.append(np.concatenate((np.sin(angles), np.cos(angles)), axis=1))
    expected = np.concatenate(expected_axes, axis=1)

    actual = lance_3d_sincos_position_embedding(hidden_size, frames, height, width)
    # ``torch.from_numpy`` is unavailable in the isolated torch 2.2 / NumPy 2
    # compatibility runtime used by CPU CI, so cross the boundary as a list.
    torch.testing.assert_close(actual, torch.tensor(expected.tolist()), rtol=1e-6, atol=1e-6)


def test_native_model_rebuilds_frozen_latent_position_table():
    config = _tiny_config("video")
    model = LanceNativeModel(config)
    expected = lance_3d_sincos_position_embedding(
        config.hidden_size,
        config.max_latent_frames,
        config.max_latent_size,
        config.max_latent_size,
    )
    assert model.latent_pos_embed.pos_embed.requires_grad is False
    torch.testing.assert_close(model.latent_pos_embed.pos_embed, expected)
