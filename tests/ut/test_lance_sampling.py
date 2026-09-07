import os

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

torch = pytest.importorskip("torch")

from mindspeed_mm.models.omni.lance.modeling_lance import LanceNativeModel
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.sampling import (
    LanceDenoiseContext,
    euler_flow_sample,
    lance_cfg_velocity,
    lance_sampling_schedule,
    sample_native_lance,
)


def _tiny_config():
    return LanceNativeConfig(
        variant="image",
        vocab_size=31,
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


def test_sampling_schedule_matches_released_shift_and_integrates_unit_interval():
    timesteps, widths = lance_sampling_schedule(4, 3.0)
    base = torch.linspace(1.0, 0.0, 5)
    expected = 3.0 * base / (1.0 + 2.0 * base)
    torch.testing.assert_close(timesteps, expected[:-1])
    torch.testing.assert_close(widths, expected[:-1] - expected[1:])
    torch.testing.assert_close(widths.sum(), torch.tensor(1.0))


def test_euler_sampler_uses_noise_to_data_velocity_direction_and_subset_updates():
    initial = torch.ones(3, 2)

    def constant_velocity(latents, timestep, branch):
        assert branch == "conditional"
        return torch.full((2, 2), 2.0)

    sampled = euler_flow_sample(
        initial,
        constant_velocity,
        num_steps=4,
        timestep_shift=3.0,
        update_indexes=torch.tensor([0, 2]),
    )
    torch.testing.assert_close(sampled[[0, 2]], torch.full((2, 2), -1.0))
    torch.testing.assert_close(sampled[1], initial[1])


def test_cfg_matches_released_text_and_vision_formula():
    conditional = torch.tensor([[3.0, 5.0]])
    text_unconditional = torch.tensor([[2.0, 2.0]])
    vision_unconditional = torch.tensor([[1.0, 1.0]])
    actual = lance_cfg_velocity(
        conditional,
        text_unconditional,
        text_scale=4.0,
        vision_unconditional=vision_unconditional,
        vision_scale=2.0,
        renorm_type="none",
    )
    expected = vision_unconditional + 4.0 * (conditional - text_unconditional) + 2.0 * (
        text_unconditional - vision_unconditional
    )
    torch.testing.assert_close(actual, expected)


def test_cfg_interval_only_invokes_unconditional_branch_when_active():
    calls = []

    def velocity(latents, timestep, branch):
        calls.append((round(float(timestep), 6), branch))
        return torch.ones_like(latents)

    euler_flow_sample(
        torch.zeros(1, 2),
        velocity,
        num_steps=4,
        timestep_shift=1.0,
        cfg_interval=(0.25, 0.75),
        text_scale=2.0,
        renorm_type="none",
    )
    assert [branch for _, branch in calls].count("conditional") == 4
    assert [branch for _, branch in calls].count("text_unconditional") == 2


def test_tiny_native_model_runs_deterministic_end_to_end_sampling():
    torch.manual_seed(17)
    model = LanceNativeModel(_tiny_config()).eval()
    mask = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, True],
            [True, True, True, True],
        ]
    )
    context = LanceDenoiseContext(
        token_ids=torch.tensor([1, 2, 0, 0]),
        text_indexes=torch.tensor([0, 1]),
        position_ids=torch.arange(4).repeat(3, 1),
        attention_mask=mask,
        understanding_indexes=torch.tensor([0, 1]),
        generation_indexes=torch.tensor([2, 3]),
        vae_indexes=torch.tensor([2, 3]),
        latent_position_ids=torch.tensor([0, 1]),
        prediction_indexes=torch.tensor([2, 3]),
        prediction_latent_indexes=torch.tensor([0, 1]),
    )
    initial = torch.randn(2, model.config.patch_latent_dim)
    with torch.no_grad():
        first = sample_native_lance(
            model,
            context,
            initial,
            num_steps=3,
            timestep_shift=3.0,
        )
        second = sample_native_lance(
            model,
            context,
            initial,
            num_steps=3,
            timestep_shift=3.0,
        )
    assert first.shape == initial.shape
    assert not torch.equal(first, initial)
    torch.testing.assert_close(first, second)
