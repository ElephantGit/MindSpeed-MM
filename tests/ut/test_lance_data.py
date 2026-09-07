import os

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

torch = pytest.importorskip("torch")

from mindspeed_mm.models.omni.lance.data import (
    LanceDataError,
    LancePreparedSample,
    adapt_upstream_lance_batch,
    ce_length_weight,
    encode_upstream_vae_latents,
    encode_upstream_vit_embeddings,
    pack_preencoded_samples,
)
from mindspeed_mm.models.omni.lance.modeling_lance import LanceNativeModel
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.sequence import LancePackedSequence, LanceSegment
from mindspeed_mm.models.omni.lance.training_lance import LanceLossWeights, lance_training_step


def _config():
    return LanceNativeConfig(
        variant="image",
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
        max_num_frames=1,
        vit_depth=1,
        vit_hidden_size=16,
        vit_intermediate_size=32,
        vit_num_heads=4,
        vit_out_hidden_size=32,
    )


def _samples():
    understanding = LancePreparedSample(
        sample_id="understanding",
        segments=(LanceSegment(2, "causal", "text", "understanding"),),
        token_ids=torch.tensor([1, 2]),
        text_indexes=torch.tensor([0, 1]),
        position_ids=torch.arange(2).repeat(3, 1),
        ce_indexes=torch.tensor([0]),
        ce_labels=torch.tensor([2]),
        ce_weights=torch.tensor([ce_length_weight(1)]),
    )
    generation = LancePreparedSample(
        sample_id="generation",
        segments=(
            LanceSegment(1, "causal", "text", "understanding"),
            LanceSegment(2, "noise", "vae", "generation"),
        ),
        token_ids=torch.tensor([3, 0, 0]),
        text_indexes=torch.tensor([0]),
        position_ids=torch.arange(3).repeat(3, 1),
        vae_indexes=torch.tensor([1, 2]),
        clean_latents=torch.randn(2, 8),
        latent_position_ids=torch.tensor([0, 1]),
        timesteps=torch.tensor([0.25, 0.25]),
        noise=torch.randn(2, 8),
        mse_indexes=torch.tensor([1, 2]),
    )
    return understanding, generation


def test_preencoded_collator_offsets_indexes_and_isolates_documents():
    packed = pack_preencoded_samples(_samples(), _config(), max_tokens=8, attention_backend="reference")
    batch = packed.batch
    assert packed.sample_ids == ("understanding", "generation")
    assert batch.text_indexes.tolist() == [0, 1, 2]
    assert batch.vae_indexes.tolist() == [3, 4]
    assert batch.ce_indexes.tolist() == [0]
    assert batch.mse_indexes.tolist() == [3, 4]
    assert batch.understanding_indexes.tolist() == [0, 1, 2]
    assert batch.generation_indexes.tolist() == [3, 4]
    assert not bool(batch.attention_mask[2, :2].any())
    assert not bool(batch.attention_mask[:2, 2:].any())


def test_ascend_collator_returns_dynamic_schedule_without_dense_mask():
    packed = pack_preencoded_samples(_samples(), _config(), max_tokens=8, attention_backend="ascend")
    assert isinstance(packed.batch.attention_mask, LancePackedSequence)
    assert packed.batch.attention_mask is packed.packed_sequence


def test_packed_batch_runs_joint_forward_backward():
    config = _config()
    packed = pack_preencoded_samples(_samples(), config, max_tokens=8, attention_backend="reference")
    model = LanceNativeModel(config)
    output = lance_training_step(model, packed.batch, LanceLossWeights(0.25, 1.0), 1.0)
    output["loss"].backward()
    assert output["ce_loss"] is not None
    assert output["mse_loss"] is not None
    assert model.vae2llm.weight.grad is not None


def test_collator_rejects_token_overflow_and_incomplete_input_coverage():
    samples = _samples()
    with pytest.raises(LanceDataError, match="exceeds"):
        pack_preencoded_samples(samples, _config(), max_tokens=4)
    samples[0].text_indexes = torch.tensor([0])
    with pytest.raises(LanceDataError, match="cover"):
        pack_preencoded_samples(samples, _config(), max_tokens=8)


def test_ce_length_weight_matches_released_modes():
    assert ce_length_weight(4, "token") == 1.0
    assert ce_length_weight(4, "sample") == 0.25
    assert ce_length_weight(4, "square") == 0.5


def test_explicit_routes_preserve_mixed_experts_inside_one_attention_segment():
    sample = LancePreparedSample(
        sample_id="mixed-full-noise",
        segments=(LanceSegment(4, "full_noise", "mixed", "understanding"),),
        token_ids=torch.tensor([1, 0, 0, 2]),
        text_indexes=torch.tensor([0, 3]),
        position_ids=torch.arange(4).repeat(3, 1),
        understanding_indexes=torch.tensor([0, 3]),
        generation_indexes=torch.tensor([1, 2]),
        vae_indexes=torch.tensor([1, 2]),
        clean_latents=torch.randn(2, 8),
        latent_position_ids=torch.tensor([0, 1]),
        timesteps=torch.tensor([0.4, 0.4]),
        mse_indexes=torch.tensor([1, 2]),
    )
    packed = pack_preencoded_samples((sample,), _config(), max_tokens=4)
    assert packed.batch.understanding_indexes.tolist() == [0, 3]
    assert packed.batch.generation_indexes.tolist() == [1, 2]


def test_official_packed_dataset_batch_adapts_without_losing_boundary_routes():
    config = _config()
    upstream = {
        "sequence_length": 6,
        "sample_lens": [6],
        "split_lens": [2, 4],
        "attn_modes": ["causal", "full_noise"],
        "packed_text_ids": torch.tensor([1, 2, 3, 0, 0, 4]),
        "packed_text_indexes": torch.tensor([0, 1, 2, 5]),
        "packed_position_ids": torch.arange(6),
        "packed_vae_token_indexes": torch.tensor([3, 4]),
        "padded_latent": torch.randn(1, 1, 2, 1, 8),
        "patchified_vae_latent_shapes": [(1, 2, 1)],
        "packed_latent_position_ids": torch.tensor([0, 1]),
        "packed_timesteps": torch.tensor([0.2, 0.2]),
        "mse_loss_indexes": torch.tensor([3, 4]),
        "ce_loss_indexes": torch.tensor([0]),
        "packed_label_ids": torch.tensor([2]),
        "ce_loss_weights": torch.tensor([1.0]),
    }
    adapted = adapt_upstream_lance_batch(
        upstream,
        config,
        attention_backend="reference",
        sample_ids=("official-row",),
    )
    batch = adapted.batch
    assert adapted.sample_ids == ("official-row",)
    assert batch.understanding_indexes.tolist() == [0, 1, 2, 5]
    assert batch.generation_indexes.tolist() == [3, 4]
    assert batch.clean_latents.shape == (2, 8)
    torch.testing.assert_close(batch.timesteps, torch.tensor([0.2, 0.2]).sigmoid())
    model = LanceNativeModel(config)
    batch.validate(model)
    losses = lance_training_step(model, batch, LanceLossWeights(0.25, 1.0), 1.0)
    losses["loss"].backward()
    assert losses["ce_loss"] is not None and losses["mse_loss"] is not None


def test_upstream_adapter_requires_encoded_online_vit_features():
    config = _config()
    upstream = {
        "sequence_length": 2,
        "sample_lens": [2],
        "split_lens": [2],
        "attn_modes": ["full"],
        "packed_text_ids": torch.tensor([0, 0]),
        "packed_text_indexes": torch.empty(0, dtype=torch.long),
        "packed_position_ids": torch.arange(2),
        "packed_vit_token_indexes": torch.tensor([0, 1]),
        "packed_vit_tokens": [torch.randn(8, 12)],
    }
    with pytest.raises(LanceDataError, match="ViT patches"):
        adapt_upstream_lance_batch(upstream, config)


def test_mixed_online_offline_vae_encoding_preserves_item_order():
    class FakeVAE:
        def vae_encode(self, values):
            return [values[0].permute(1, 2, 3, 0) + 10.0]

    online = torch.zeros(8, 1, 1, 1)
    offline = torch.ones(1, 1, 1, 8)
    raw = {
        "packed_vae_token_indexes": torch.tensor([0, 1]),
        "padded_videos": [online, offline],
        "vae_data_mode": ["online", "offline"],
    }
    result = encode_upstream_vae_latents(raw, FakeVAE())
    torch.testing.assert_close(result[0], torch.full((1, 1, 1, 8), 10.0))
    torch.testing.assert_close(result[1], offline)


def test_online_native_vit_encoding_matches_direct_model_call():
    config = LanceNativeConfig(
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
        max_num_frames=1,
        vit_depth=1,
        vit_hidden_size=16,
        vit_intermediate_size=32,
        vit_num_heads=4,
        vit_patch_size=2,
        vit_temporal_patch_size=2,
        vit_spatial_merge_size=2,
        vit_window_size=8,
        vit_fullatt_block_indexes=(0,),
        vit_out_hidden_size=32,
    )
    torch.manual_seed(59)
    model = LanceNativeModel(config).eval()
    raw_patches = torch.randn(4, 24)
    raw = {
        "packed_vit_token_indexes": torch.tensor([0]),
        "packed_vit_tokens": [raw_patches],
        "vit_data_mode": ["online"],
        "vit_video_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    actual = encode_upstream_vit_embeddings(model, raw)
    expected = model.vit_model(raw_patches, torch.tensor([[1, 2, 2]]))
    torch.testing.assert_close(actual, expected)
