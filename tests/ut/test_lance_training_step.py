import os

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

torch = pytest.importorskip("torch")

from mindspeed_mm.models.omni.lance.modeling_lance import LanceNativeModel
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.sequence import (
    LanceDocument,
    LancePackedSequence,
    LanceSegment,
)
from mindspeed_mm.models.omni.lance.training_lance import (
    LanceLossWeights,
    LanceTrainingBatch,
    LanceTrainingError,
    lance_training_step,
    shift_timesteps,
)


def _config():
    return LanceNativeConfig(
        vocab_size=53,
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
        max_num_frames=1,
        vit_depth=1,
        vit_hidden_size=8,
        vit_intermediate_size=16,
        vit_num_heads=2,
        vit_patch_size=2,
        vit_temporal_patch_size=2,
        vit_out_hidden_size=32,
    )


def _batch(config):
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
    routes = packed.token_expert_indexes()
    return LanceTrainingBatch(
        token_ids=torch.tensor([1, 2, 3, 0, 0, 0], dtype=torch.long),
        text_indexes=torch.tensor([0, 1, 2], dtype=torch.long),
        position_ids=torch.arange(6).repeat(3, 1),
        attention_mask=torch.tensor(packed.dense_attention_mask(), dtype=torch.bool),
        understanding_indexes=torch.tensor(routes["understanding"], dtype=torch.long),
        generation_indexes=torch.tensor(routes["generation"], dtype=torch.long),
        vae_indexes=torch.tensor([3, 4, 5], dtype=torch.long),
        clean_latents=torch.randn(3, config.patch_latent_dim),
        latent_position_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        timesteps=torch.tensor([0.2, 0.5, 0.8]),
        noise=torch.randn(3, config.patch_latent_dim),
        ce_indexes=torch.tensor([0, 1], dtype=torch.long),
        ce_labels=torch.tensor([2, 3], dtype=torch.long),
        ce_weights=torch.tensor([1.0, 0.5]),
        mse_indexes=torch.tensor([3, 4, 5], dtype=torch.long),
    )


def test_joint_training_step_runs_backward_through_language_and_bridge_heads():
    torch.manual_seed(101)
    config = _config()
    model = LanceNativeModel(config)
    batch = _batch(config)
    result = lance_training_step(
        model,
        batch,
        LanceLossWeights(ce=0.25, mse=1.0),
        timestep_shift=4.0,
    )
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["ce_loss"])
    assert torch.isfinite(result["mse_loss"])
    torch.testing.assert_close(result["velocity_target"], batch.noise - batch.clean_latents)
    torch.testing.assert_close(
        result["shifted_timesteps"],
        torch.tensor([0.5, 0.8, 16.0 / 17.0]),
    )
    result["loss"].backward()
    assert model.vae2llm.weight.grad.abs().sum().item() > 0
    assert model.llm2vae.weight.grad.abs().sum().item() > 0
    assert model.language_model.lm_head.weight.grad.abs().sum().item() > 0
    assert model.language_model.model.layers[0].self_attn.q_proj.weight.grad.abs().sum().item() > 0
    assert model.language_model.model.layers[0].self_attn.q_proj_moe_gen.weight.grad.abs().sum().item() > 0


def test_batch_rejects_mse_index_outside_vae_tokens():
    config = _config()
    model = LanceNativeModel(config)
    batch = _batch(config)
    batch.mse_indexes = torch.tensor([2], dtype=torch.long)
    with pytest.raises(LanceTrainingError, match="subset"):
        batch.validate(model)


def test_timestep_validation_rejects_out_of_range_values():
    with pytest.raises(LanceTrainingError, match=r"\[0, 1\]"):
        shift_timesteps(torch.tensor([-0.1, 0.5]), 1.0)

