import os

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

torch = pytest.importorskip("torch")

from mindspeed_mm.models.omni.lance.data import LancePreparedSample, pack_preencoded_samples
from mindspeed_mm.models.omni.lance.modeling_lance import LanceNativeModel
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.sequence import LanceSegment
from mindspeed_mm.models.omni.lance.training_contract import LanceStage
from mindspeed_mm.models.omni.lance.training_runtime import (
    LanceEMAController,
    LanceLRScheduler,
    LanceNativeTrainingRuntime,
    build_lance_optimizer,
    configure_lance_trainability,
)


def _config(variant="image"):
    return LanceNativeConfig(
        variant=variant,
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
        max_num_frames=5 if variant == "video" else 1,
        vit_depth=1,
        vit_hidden_size=16,
        vit_intermediate_size=32,
        vit_num_heads=4,
        vit_in_channels=3,
        vit_patch_size=2,
        vit_temporal_patch_size=2,
        vit_spatial_merge_size=2,
        vit_out_hidden_size=32,
    )


def _stage(scheduler="constant", steps=10, warmup=2):
    return LanceStage(
        name="tiny-{}".format(scheduler),
        steps=steps,
        learning_rate=0.1,
        scheduler=scheduler,
        warmup_steps=warmup,
        expected_tokens_per_rank=3,
        max_tokens_per_rank=5,
        max_context=5,
        timestep_shift=1.0,
        ce_weight=0.25,
        mse_weight=1.0,
        text_dropout=0.0,
        multimodal_dropout=0.0,
    )


def _training_batch(config):
    sample = LancePreparedSample(
        sample_id="joint",
        segments=(
            LanceSegment(2, "causal", "text", "understanding"),
            LanceSegment(2, "noise", "vae", "generation"),
        ),
        token_ids=torch.tensor([1, 2, 0, 0]),
        text_indexes=torch.tensor([0, 1]),
        position_ids=torch.arange(4).repeat(3, 1),
        vae_indexes=torch.tensor([2, 3]),
        clean_latents=torch.randn(2, 8),
        latent_position_ids=torch.tensor([0, 1]),
        timesteps=torch.tensor([0.25, 0.25]),
        noise=torch.randn(2, 8),
        ce_indexes=torch.tensor([0]),
        ce_labels=torch.tensor([2]),
        ce_weights=torch.ones(1),
        mse_indexes=torch.tensor([2, 3]),
    )
    return pack_preencoded_samples(
        [sample], config, max_tokens=4, attention_backend="reference"
    ).batch


def test_optimizer_matches_release_and_excludes_frozen_vit():
    model = LanceNativeModel(_config("video"))
    counts = configure_lance_trainability(model)
    assert counts["frozen_parameters"] > model.latent_pos_embed.pos_embed.numel()
    assert all(not parameter.requires_grad for parameter in model.vit_model.parameters())
    optimizer = build_lance_optimizer(model, _stage())
    group = optimizer.param_groups[0]
    assert group["betas"] == (0.9, 0.95)
    assert group["eps"] == 1e-15
    assert group["weight_decay"] == 0.0
    optimized = {id(parameter) for parameter in group["params"]}
    assert not any(id(parameter) in optimized for parameter in model.vit_model.parameters())


def test_constant_scheduler_reproduces_zero_to_linear_warmup():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = LanceLRScheduler(optimizer, _stage())
    assert scheduler.get_last_lr() == [0.0]
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([0.05])
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([0.1])


def test_cosine_scheduler_preserves_released_five_cycle_contract():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = LanceLRScheduler(optimizer, _stage("cosine", steps=12, warmup=2))
    for _ in range(3):
        scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([1e-7])
    for _ in range(9):
        scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([0.1])


def test_ema_first_update_copies_then_applies_decay():
    model = LanceNativeModel(_config())
    ema = LanceEMAController(model, decay=0.5)
    name, parameter = next(
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    ema_parameter = dict(ema.model.named_parameters())[name]
    with torch.no_grad():
        parameter.add_(1.0)
    ema.update(model, step=0)
    torch.testing.assert_close(ema_parameter, parameter)
    previous = ema_parameter.clone()
    with torch.no_grad():
        parameter.add_(1.0)
    ema.update(model, step=1)
    torch.testing.assert_close(ema_parameter, previous * 0.5 + parameter * 0.5)
    assert ema.metadata_state_dict()["num_updates"] == 2


def test_eager_runtime_executes_optimizer_scheduler_and_ema_step():
    torch.manual_seed(23)
    config = _config()
    model = LanceNativeModel(config)
    stage = _stage()
    ema = LanceEMAController(model)
    runtime = LanceNativeTrainingRuntime(model, stage, ema=ema)
    batch = _training_batch(config)
    before = model.vae2llm.weight.detach().clone()
    runtime.step(batch)
    result = runtime.step(batch)
    assert result["completed_steps"] == 2
    assert torch.isfinite(result["grad_norm"])
    assert result["learning_rate"] == pytest.approx(stage.learning_rate)
    assert not torch.equal(before, model.vae2llm.weight)
    assert ema.num_updates == 2
