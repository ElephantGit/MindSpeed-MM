import os

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

torch = pytest.importorskip("torch")

from mindspeed_mm.models.omni.lance.data import (
    LanceDataError,
    LancePreparedSample,
    ce_length_weight,
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
