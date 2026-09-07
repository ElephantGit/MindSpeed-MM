import os

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

torch = pytest.importorskip("torch")

from mindspeed_mm.models.omni.lance.modeling_lance import (
    LanceVisionModel,
    reference_vision_sdpa,
)
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig


def _config():
    return LanceNativeConfig(
        variant="video",
        vocab_size=41,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=10000.0,
        mrope_section=(1, 1, 2),
        latent_channels=8,
        max_latent_size=2,
        max_num_frames=5,
        vit_depth=2,
        vit_hidden_size=16,
        vit_intermediate_size=32,
        vit_num_heads=4,
        vit_in_channels=3,
        vit_patch_size=2,
        vit_temporal_patch_size=2,
        vit_spatial_merge_size=2,
        vit_window_size=8,
        vit_fullatt_block_indexes=(1,),
        vit_out_hidden_size=32,
    )


def test_window_index_matches_released_merger_group_order():
    model = LanceVisionModel(_config())
    indexes, cumulative = model.get_window_index(torch.tensor([[1, 8, 8]]))
    assert indexes.tolist() == [
        0, 1, 4, 5,
        2, 3, 6, 7,
        8, 9, 12, 13,
        10, 11, 14, 15,
    ]
    assert list(dict.fromkeys(cumulative)) == [0, 16, 32, 48, 64]


def test_visual_rotary_shape_and_patch_order_are_deterministic():
    model = LanceVisionModel(_config())
    grid = torch.tensor([[1, 4, 4]])
    first = model.rotary_position_embedding(grid)
    second = model.rotary_position_embedding(grid)
    assert first.shape == (16, 2)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first[0], torch.zeros(2))


def test_vision_forward_uses_window_then_released_full_attention_blocks():
    calls = []

    def recording_backend(query, key, value, cumulative_lengths):
        calls.append(cumulative_lengths.tolist())
        return reference_vision_sdpa(query, key, value, cumulative_lengths)

    model = LanceVisionModel(_config(), attention_backend=recording_backend)
    patch_dimension = 3 * 2 * 2 * 2
    patches = torch.randn(64, patch_dimension, requires_grad=True)
    output = model(patches, torch.tensor([[1, 8, 8]]))
    assert output.shape == (16, 32)
    assert calls == [[0, 16, 32, 48, 64], [0, 64]]
    output.square().mean().backward()
    assert patches.grad is not None
    assert model.patch_embed.proj.weight.grad is not None


def test_reference_vision_attention_isolates_cumulative_sequences():
    torch.manual_seed(19)
    query = torch.randn(4, 2, 4)
    key = torch.randn(4, 2, 4)
    value = torch.randn(4, 2, 4)
    cumulative = torch.tensor([0, 2, 4], dtype=torch.int32)
    baseline = reference_vision_sdpa(query, key, value, cumulative)
    changed_key = key.clone()
    changed_value = value.clone()
    changed_key[:2] += 100
    changed_value[:2] += 100
    changed = reference_vision_sdpa(query, changed_key, changed_value, cumulative)
    torch.testing.assert_close(changed[2:], baseline[2:], rtol=0, atol=0)
