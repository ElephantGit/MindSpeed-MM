import os

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

torch = pytest.importorskip("torch")

from mindspeed_mm.fsdp.models.lance.modeling_lance import (
    LanceFSDPModel,
    LanceVisionConnector,
)
from mindspeed_mm.fsdp.tasks.lance.train_engine import LanceEMAState
from mindspeed_mm.models.omni.lance.data import LancePreparedSample, pack_preencoded_samples
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.sequence import LanceSegment


def _config():
    return LanceNativeConfig(
        variant="video",
        vocab_size=41,
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
        max_num_frames=5,
        vit_depth=1,
        vit_hidden_size=16,
        vit_intermediate_size=32,
        vit_num_heads=4,
        vit_patch_size=2,
        vit_temporal_patch_size=2,
        vit_out_hidden_size=32,
    )


def _batch(config):
    sample = LancePreparedSample(
        sample_id="joint",
        segments=(
            LanceSegment(4, "causal", "text", "understanding"),
            LanceSegment(4, "noise", "vae", "generation"),
        ),
        token_ids=torch.tensor([1, 2, 3, 4, 0, 0, 0, 0]),
        text_indexes=torch.tensor([0, 1, 2, 3]),
        position_ids=torch.arange(8).repeat(3, 1),
        vae_indexes=torch.tensor([4, 5, 6, 7]),
        clean_latents=torch.randn(4, config.patch_latent_dim),
        latent_position_ids=torch.tensor([0, 1, 2, 3]),
        timesteps=torch.full((4,), 0.5),
        noise=torch.randn(4, config.patch_latent_dim),
        ce_indexes=torch.tensor([0, 1, 2]),
        ce_labels=torch.tensor([2, 3, 4]),
        ce_weights=torch.ones(3),
        mse_indexes=torch.tensor([4, 5, 6, 7]),
    )
    return pack_preencoded_samples(
        (sample,), config, max_tokens=8, attention_backend="reference"
    ).batch


def _understanding_only_batch(config):
    sample = LancePreparedSample(
        sample_id="understanding-only",
        segments=(LanceSegment(4, "causal", "text", "understanding"),),
        token_ids=torch.tensor([1, 2, 3, 4]),
        text_indexes=torch.tensor([0, 1, 2, 3]),
        position_ids=torch.arange(4).repeat(3, 1),
        ce_indexes=torch.tensor([0, 1, 2]),
        ce_labels=torch.tensor([2, 3, 4]),
        ce_weights=torch.ones(3),
    )
    return pack_preencoded_samples(
        (sample,), config, max_tokens=4, attention_backend="reference"
    ).batch


def test_native_fsdp_model_executes_joint_forward_and_backward():
    config = _config()
    model = LanceFSDPModel(config, use_vit_connector=True, validate_batches=True)
    output = model(_batch(config))
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert output.ce_tokens == 3
    assert output.mse_tokens == 4
    assert model.vae2llm.weight.grad is not None


def test_connector_is_trainable_while_vit_is_frozen():
    config = _config()
    model = LanceFSDPModel(config, use_vit_connector=True)
    assert isinstance(model.connector, LanceVisionConnector)
    model.vit_model.requires_grad_(False)
    assert all(not parameter.requires_grad for parameter in model.vit_model.parameters())
    assert all(parameter.requires_grad for parameter in model.connector.parameters())


def test_preencoded_training_model_omits_frozen_vit():
    model = LanceFSDPModel(
        _config(), use_vit_connector=True, include_vit_model=False
    )
    assert not hasattr(model, "vit_model")
    assert isinstance(model.connector, LanceVisionConnector)
    assert not any(name.startswith("vit_model.") for name in model.state_dict())


def test_modality_absent_heads_and_bridges_stay_in_backward_graph():
    config = _config()
    model = LanceFSDPModel(
        config, use_vit_connector=True, include_vit_model=False,
        validate_batches=True,
    )
    output = model(_understanding_only_batch(config))
    output.loss.backward()
    assert model.connector.fc1.weight.grad is not None
    assert model.vae2llm.weight.grad is not None
    assert model.time_embedder.mlp[0].weight.grad is not None
    assert model.llm2vae.weight.grad is not None
    assert model.language_model.model.layers[0].mlp_moe_gen.gate_proj.weight.grad is not None
    assert model.language_model.model.layers[0].input_layernorm_moe_gen.weight.grad is not None

    model.zero_grad(set_to_none=True)
    generation_only = _batch(config)
    generation_only.ce_indexes = None
    generation_only.ce_labels = None
    generation_only.ce_weights = None
    model(generation_only).loss.backward()
    assert model.language_model.lm_head.weight.grad is not None
    assert model.language_model.model.layers[0].mlp.gate_proj.weight.grad is not None
    assert model.language_model.model.layers[0].input_layernorm.weight.grad is not None


def test_sharded_ema_restore_rejects_incomplete_state():
    model = LanceFSDPModel(
        _config(), use_vit_connector=True, include_vit_model=False
    )
    ema = LanceEMAState(model)
    state = ema.state_dict()
    first_name = next(iter(state["parameters"]))
    incomplete = {
        "parameters": {
            name: value
            for name, value in state["parameters"].items()
            if name != first_name
        },
        "decay": state["decay"],
        "num_updates": state["num_updates"],
    }
    with pytest.raises(RuntimeError, match="parameter tree differs"):
        ema.load_state_dict(incomplete)

    with pytest.raises(RuntimeError, match="missing fields"):
        ema.load_state_dict({"parameters": state["parameters"]})
