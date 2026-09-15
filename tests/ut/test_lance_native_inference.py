import os

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

torch = pytest.importorskip("torch")

from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.modeling_lance import LanceNativeModel
from mindspeed_mm.models.omni.lance.native_inference import (
    LanceNativeInferenceError,
    generation_geometry,
    load_native_generation_dcp,
    prepared_sample_to_denoise_context,
    resolve_native_dcp,
    unpatchify_lance_latents,
)
from mindspeed_mm.models.omni.lance.preprocessing import (
    LanceEncodedVisual,
    build_generation_sample,
)
from mindspeed_mm.models.omni.lance.sequence import LancePackedSequence


class _Tokenizer:
    mapping = {
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<|vision_start|>": 3,
        "<|vision_end|>": 4,
        "<|image_pad|>": 5,
        "<|video_pad|>": 6,
    }

    def convert_tokens_to_ids(self, value):
        return self.mapping[value]

    def encode(self, value, add_special_tokens=False):
        assert not add_special_tokens
        result = []
        cursor = 0
        special = sorted(self.mapping, key=len, reverse=True)
        while cursor < len(value):
            token = next((item for item in special if value.startswith(item, cursor)), None)
            if token is None:
                result.append(7 + ord(value[cursor]) % 20)
                cursor += 1
            else:
                result.append(self.mapping[token])
                cursor += len(token)
        return result


def _config():
    return LanceNativeConfig(
        variant="video",
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
        mrope_section=(1, 1, 2),
        latent_channels=2,
        latent_patch_size=(1, 2, 2),
        max_latent_size=8,
        max_num_frames=9,
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


def test_resolve_native_dcp_accepts_iteration_or_tracker_root(tmp_path):
    iteration = tmp_path / "iter_0001500"
    iteration.mkdir()
    (iteration / ".metadata").write_bytes(b"metadata")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("1500")
    assert resolve_native_dcp(iteration) == iteration.resolve()
    assert resolve_native_dcp(tmp_path) == iteration.resolve()


def test_generation_model_can_skip_video_vit_allocation():
    model = LanceNativeModel(_config(), include_vit_model=False)
    assert not hasattr(model, "vit_model")
    assert model.language_model.model.rotary_emb.inv_freq.device.type == "cpu"


def test_native_generation_loader_restores_model_and_overlays_ema(tmp_path):
    dcp = pytest.importorskip("torch.distributed.checkpoint")
    config = _config()
    source = LanceNativeModel(config, include_vit_model=False)
    model_state = {
        name: torch.full_like(value, 1.0)
        for name, value in source.state_dict().items()
    }
    ema_parameters = {
        name: torch.full_like(value, 2.0)
        for name, value in model_state.items()
        if name != "latent_pos_embed.pos_embed"
    }
    checkpoint = tmp_path / "iter_0001500"
    dcp.save(
        {
            "model": model_state,
            "ema_state": {
                "parameters": ema_parameters,
                "decay": torch.tensor(0.9999),
                "num_updates": torch.tensor(1500),
            },
        },
        checkpoint_id=checkpoint,
    )
    target = LanceNativeModel(config, include_vit_model=False)
    report = load_native_generation_dcp(target, checkpoint, use_ema=True)
    assert report["weights"] == "ema"
    for name, value in target.state_dict().items():
        expected = 1.0 if name == "latent_pos_embed.pos_embed" else 2.0
        torch.testing.assert_close(value, torch.full_like(value, expected))


def test_generation_geometry_enforces_causal_temporal_and_patch_alignment():
    config = _config()
    geometry = generation_geometry("t2v", 5, 64, 96, config)
    assert geometry.latent_shape == (2, 4, 6)
    with pytest.raises(LanceNativeInferenceError, match="4k"):
        generation_geometry("t2v", 6, 64, 96, config)
    with pytest.raises(LanceNativeInferenceError, match="divisible"):
        generation_geometry("t2v", 5, 64, 80, config)


def test_unpatchify_inverts_native_lance_patch_order():
    config = _config()
    geometry = generation_geometry("t2v", 5, 64, 96, config)
    source = torch.arange(2 * 4 * 6 * 2).reshape(2, 4, 6, 2)
    patchified = (
        source.reshape(2, 1, 2, 2, 3, 2, 2)
        .permute(0, 2, 4, 1, 3, 5, 6)
        .reshape(2 * 2 * 3, -1)
    )
    actual = unpatchify_lance_latents(patchified, geometry, config)
    torch.testing.assert_close(actual, source)


def test_generation_sample_converts_to_sampler_context_without_dense_mask():
    config = _config()
    geometry = generation_geometry("t2i", 1, 64, 64, config)
    visual = LanceEncodedVisual(
        "image",
        vae_latent=torch.zeros(*geometry.latent_shape, config.latent_channels),
    )
    sample = build_generation_sample(
        "native-t2i", "a small red panda", visual, _Tokenizer(), config
    )
    sample.validate(config)
    context = prepared_sample_to_denoise_context(sample, "cpu")
    assert isinstance(context.attention_mask, LancePackedSequence)
    assert torch.equal(
        context.vae_indexes[context.prediction_latent_indexes],
        context.prediction_indexes,
    )
    assert context.prediction_indexes.numel() == sample.clean_latents.shape[0]
