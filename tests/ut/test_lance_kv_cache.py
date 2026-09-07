import os

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

torch = pytest.importorskip("torch")

from mindspeed_mm.models.omni.lance.modeling_lance import (
    LanceKVCache,
    LanceLayerKVCache,
    LanceNativeModel,
)
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.sequence import LanceDocument, LancePackedSequence, LanceSegment


def _tiny_config():
    return LanceNativeConfig(
        variant="image",
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
        max_num_frames=1,
        vit_depth=1,
        vit_hidden_size=16,
        vit_intermediate_size=32,
        vit_num_heads=4,
        vit_in_channels=3,
        vit_out_hidden_size=32,
    )


def _routes(packed):
    routes = packed.token_expert_indexes()
    return (
        torch.tensor(routes["understanding"], dtype=torch.long),
        torch.tensor(routes["generation"], dtype=torch.long),
    )


@pytest.mark.parametrize("query_mode,is_causal", [("noise", False), ("causal", True)])
def test_cached_query_matches_full_sequence_oracle(query_mode, is_causal):
    torch.manual_seed(41)
    model = LanceNativeModel(_tiny_config()).eval()
    condition_segments = (
        LanceSegment(3, "causal", "text", "understanding"),
        LanceSegment(2, "full_noise", "clean_vae", "generation"),
    )
    query_segment = LanceSegment(4, query_mode, "vae", "generation")
    condition = LancePackedSequence((LanceDocument("sample", condition_segments),))
    full = LancePackedSequence(
        (LanceDocument("sample", condition_segments + (query_segment,)),)
    )
    hidden = torch.randn(full.length, model.config.hidden_size)
    position_ids = torch.arange(full.length).repeat(3, 1)
    full_understanding, full_generation = _routes(full)

    expected = model.forward_language(
        hidden,
        position_ids,
        torch.tensor(full.dense_attention_mask(), dtype=torch.bool),
        full_understanding,
        full_generation,
    )

    condition_understanding, condition_generation = _routes(condition)
    encoded_condition, cache = model.build_language_kv_cache(
        hidden[: condition.length],
        position_ids[:, : condition.length],
        torch.tensor(condition.dense_attention_mask(), dtype=torch.bool),
        condition_understanding,
        condition_generation,
    )
    query_understanding = torch.empty(0, dtype=torch.long)
    query_generation = torch.arange(full.length - condition.length)
    actual = model.forward_language_with_kv_cache(
        hidden[condition.length :],
        position_ids[:, condition.length :],
        query_understanding,
        query_generation,
        cache,
        is_causal=is_causal,
    )

    torch.testing.assert_close(
        encoded_condition,
        expected[: condition.length],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        actual,
        expected[condition.length :],
        rtol=1e-5,
        atol=1e-6,
    )
    assert cache.condition_length == condition.length
    assert len(cache.layers) == model.config.num_hidden_layers


def test_cache_reuse_does_not_reproject_condition_tokens():
    torch.manual_seed(43)
    model = LanceNativeModel(_tiny_config()).eval()
    hidden = torch.randn(5, model.config.hidden_size)
    positions = torch.arange(5).repeat(3, 1)
    understanding = torch.arange(3)
    no_generation = torch.empty(0, dtype=torch.long)
    cache_calls = [0]

    def count_projection(_module, inputs, _output):
        cache_calls[0] += inputs[0].shape[0]

    hooks = [
        layer.self_attn.k_proj.register_forward_hook(count_projection)
        for layer in model.language_model.model.layers
    ]
    try:
        _, cache = model.build_language_kv_cache(
            hidden[:3],
            positions[:, :3],
            torch.ones(3, 3, dtype=torch.bool).tril(),
            understanding,
            no_generation,
        )
        assert cache_calls[0] == 3 * model.config.num_hidden_layers
        for _ in range(3):
            model.forward_language_with_kv_cache(
                hidden[3:],
                positions[:, 3:],
                torch.empty(0, dtype=torch.long),
                torch.arange(2),
                cache,
            )
        assert cache_calls[0] == 3 * model.config.num_hidden_layers
    finally:
        for hook in hooks:
            hook.remove()


def test_invalid_cache_shape_is_rejected():
    model = LanceNativeModel(_tiny_config()).eval()
    bad_layer = LanceLayerKVCache(
        key=torch.zeros(2, 1, model.config.head_dim),
        value=torch.zeros(2, 1, model.config.head_dim),
    )
    cache = LanceKVCache(
        layers=(bad_layer,) * model.config.num_hidden_layers,
        condition_length=2,
    )
    with pytest.raises(ValueError, match="shape"):
        model.forward_language_with_kv_cache(
            torch.randn(1, model.config.hidden_size),
            torch.zeros(3, 1, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.tensor([0]),
            cache,
        )
