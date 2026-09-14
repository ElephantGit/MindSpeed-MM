import os

import pytest

os.environ.setdefault("NON_MEGATRON", "true")

torch = pytest.importorskip("torch")

from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.preprocessing import (
    LanceEncodedVisual,
    build_edit_sample,
    build_generation_sample,
    build_understanding_sample,
    patchify_qwen_video,
)
from scripts.prepare_lance_native_data import _sample_video_indices


class _Tokenizer:
    special_tokens_map = {
        "additional_special_tokens": [
            "<|im_start|>", "<|im_end|>", "<|vision_start|>",
            "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
        ]
    }
    mapping = {
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<|vision_start|>": 3,
        "<|vision_end|>": 4,
        "<|image_pad|>": 5,
        "<|video_pad|>": 26,
    }

    def convert_tokens_to_ids(self, value):
        return self.mapping[value]

    def encode(self, value, add_special_tokens=False):
        assert add_special_tokens is False
        result = []
        cursor = 0
        special = sorted(self.mapping, key=len, reverse=True)
        while cursor < len(value):
            token = next((item for item in special if value.startswith(item, cursor)), None)
            if token is None:
                result.append(27 + (ord(value[cursor]) % 20))
                cursor += 1
            else:
                result.append(self.mapping[token])
                cursor += len(token)
        return result


def _config():
    return LanceNativeConfig(
        variant="video", vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, rope_theta=10000.0, mrope_section=(1, 1, 2),
        latent_channels=8, max_latent_size=4, max_num_frames=5,
        vit_depth=1, vit_hidden_size=16, vit_intermediate_size=32,
        vit_num_heads=4, vit_patch_size=2, vit_temporal_patch_size=2,
        vit_spatial_merge_size=2, vit_window_size=8,
        vit_fullatt_block_indexes=(0,), vit_out_hidden_size=32,
    )


def _visual(vit=True, vae=True, posterior=False):
    latent = torch.randn(1, 2, 2, 8) if vae else None
    return LanceEncodedVisual(
        "image",
        vae_latent=latent,
        vae_log_variance=torch.randn_like(latent) if posterior else None,
        vit_embedding=torch.randn(3, 32) if vit else None,
    )


def test_native_generation_and_understanding_samples_validate():
    config = _config()
    tokenizer = _Tokenizer()
    generation = build_generation_sample(
        "t2i", "caption", _visual(vit=False), tokenizer, config
    )
    understanding = build_understanding_sample(
        "i2t", "prompt", "answer", _visual(vae=False), tokenizer, config
    )
    generation.validate(config)
    understanding.validate(config)
    assert generation.token_ids[0].item() == tokenizer.mapping["<|im_start|>"]
    assert generation.token_ids.tolist().count(tokenizer.mapping["<|video_pad|>"]) == 4
    assert tokenizer.mapping["<|image_pad|>"] not in generation.token_ids.tolist()
    assert generation.mse_indexes.numel() == 4
    assert generation.ce_indexes is None
    assert understanding.ce_indexes.numel() > 0
    assert understanding.ce_labels[-1].item() == tokenizer.mapping["<|im_end|>"]
    assert understanding.mse_indexes is None
    # Lance shifts semantic ViT conditions into temporal MaPE band 1000.
    assert understanding.position_ids[0, understanding.vit_indexes[0]].item() >= 1000
    assert torch.unique(generation.position_ids[:, generation.vae_indexes], dim=1).shape[1] > 1


def test_native_edit_routes_all_vae_tokens_to_generation_expert():
    config = _config()
    sample = build_edit_sample(
        "i2i", "instruction", _visual(), _visual(vit=False), _Tokenizer(), config
    )
    sample.validate(config)
    assert set(sample.vae_indexes.tolist()) == set(sample.generation_indexes.tolist())
    assert sample.vae_indexes.numel() == 8
    assert sample.mse_indexes.numel() == 4
    torch.testing.assert_close(
        sample.position_ids[:, sample.vae_indexes[:4]],
        sample.position_ids[:, sample.vae_indexes[4:]],
    )


def test_native_generation_preserves_vae_posterior_for_runtime_sampling():
    sample = build_generation_sample(
        "posterior", "caption", _visual(vit=False, posterior=True),
        _Tokenizer(), _config(),
    )
    assert sample.clean_latents.shape == sample.latent_log_variance.shape
    assert sample.latent_log_variance.dtype == torch.bfloat16


def test_first_frame_video_condition_is_clean_and_excluded_from_mse():
    visual = LanceEncodedVisual(
        "video", vae_latent=torch.randn(2, 2, 2, 8)
    )
    sample = build_generation_sample(
        "ff2v", "caption", visual, _Tokenizer(), _config(),
        condition_frames=(0,),
    )
    assert sample.vae_indexes.numel() == 8
    assert sample.mse_indexes.numel() == 4
    assert torch.equal(sample.mse_indexes, sample.vae_indexes[4:])
    assert torch.equal(sample.timesteps[:4], torch.zeros(4))
    assert torch.all(sample.timesteps[4:] > 0)


def test_qwen_patchify_shape_and_order_contract():
    video = torch.arange(3 * 2 * 4 * 4).reshape(3, 2, 4, 4).float()
    patches = patchify_qwen_video(video, spatial_patch_size=2, temporal_patch_size=2)
    assert patches.shape == (4, 24)


def test_video_sampling_matches_lance_multi_clip_kn_plus_one_contract():
    # Six seconds at the 12 FPS target becomes 72, then the official kN+1
    # adjustment selects 69 frames across the *complete* source video.
    indexes = _sample_video_indices(300, 30, sample_fps=12, max_duration=6, temporal=4)
    assert len(indexes) == 69
    assert indexes[0] == 0
    assert indexes[-1] == 299

    # Upstream deliberately repeats source frames when target FPS is higher;
    # capping the requested count to ``total`` would change VAE temporal shape.
    repeated = _sample_video_indices(5, 1, sample_fps=12, max_duration=6, temporal=4)
    assert len(repeated) == 57
    assert len(set(repeated)) < len(repeated)
