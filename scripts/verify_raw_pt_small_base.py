"""CPU verification of the raw-PT preprocessing against real Qwen3-0.6B assets."""
import os, sys
sys.path.insert(0, "/mnt/models/CODE/MindSpeed-MM")
os.environ.setdefault("NON_MEGATRON", "true")

import torch
from transformers import AutoTokenizer

from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.preprocessing import (
    LanceEncodedVisual,
    build_generation_sample,
    build_understanding_sample,
    prepare_lance_tokenizer,
)

QWEN = "/mnt/models/MODELS/Qwen3-0.6B"

config = LanceNativeConfig.from_llm_config(QWEN + "/config.json", variant="video").with_overrides(
    latent_patch_size=(1, 2, 2), max_latent_size=64, max_num_frames=121,
)
print("config: hidden=%d layers=%d heads=%d kv=%d head_dim=%d kv_dim=%d vit_out=%d tie=%s" % (
    config.hidden_size, config.num_hidden_layers, config.num_attention_heads,
    config.num_key_value_heads, config.head_dim, config.kv_dim,
    config.vit_out_hidden_size, config.tie_word_embeddings,
))
assert config.head_dim == 128 and config.hidden_size == 1024, "head_dim must come from config.json"
assert config.vit_out_hidden_size != config.hidden_size, "small-base mismatch expected"

tokenizer = prepare_lance_tokenizer(AutoTokenizer.from_pretrained(QWEN, trust_remote_code=False))
im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
print("tokenizer len=%d im_start=%d im_end=%d" % (len(tokenizer), im_start, im_end))

# --- t2i: caption -> VAE latents (768x768-ish image -> latent 48x24x24 -> 1x12x12 grid... use small)
caption = "A close-up of a textured, cream-colored upholstered chair backrest."
latent = torch.randn(1, 4, 4, 48)  # THWC, patch (1,2,2) -> 1x2x2 grid = 4 tokens
target = LanceEncodedVisual("image", vae_latent=latent, vae_log_variance=torch.randn_like(latent))
t2i = build_generation_sample("row:0:0", caption, target, tokenizer, config)
t2i.validate(config)

ids = t2i.token_ids.tolist()
text = tokenizer.decode(ids)
print("t2i decoded head:", repr(text[:90]))
print("t2i decoded tail:", repr(text[-40:]))
assert "<|im_start|>system" not in text and "assistant" not in text, "no chat template allowed"
assert ids[0] == im_start, "document opens with <|im_start|>"
assert ids[-1] == tokenizer.convert_tokens_to_ids("<|vision_end|>"), "T2I ends with its visual segment"
assert ids.index(im_end) < ids.index(tokenizer.convert_tokens_to_ids("<|vision_start|>")), "caption must close before target visual"
assert ids.count(tokenizer.convert_tokens_to_ids("<|video_pad|>")) == 4
assert t2i.mse_indexes.numel() == 4 and t2i.ce_indexes is None
segs = [(s.length, s.attention_mode, s.modality, s.expert) for s in t2i.segments]
print("t2i segments:", segs)
assert [s[1] for s in segs] == ["causal", "noise"], "caption | image"

# --- unconditional dropout branch: caption=None
t2i_null = build_generation_sample("row:0:0", None, target, tokenizer, config)
t2i_null.validate(config)
assert t2i_null.token_ids[0].item() == tokenizer.convert_tokens_to_ids("<|vision_start|>")
assert t2i_null.token_ids[-1].item() == tokenizer.convert_tokens_to_ids("<|vision_end|>")
assert im_start not in t2i_null.token_ids.tolist(), "CFG dropout removes the entire text segment"
print("t2i dropout doc ok:", repr(tokenizer.decode(t2i_null.token_ids.tolist())[:50]))

# --- i2t: ViT features (2048-wide!) -> caption with CE
# 6 post-merger tokens <=> pre-merger grid (1, 4, 6) with spatial_merge_size 2
vit = LanceEncodedVisual("image", vit_embedding=torch.randn(6, 2048), vit_grid_thw=(1, 4, 6))
answer = "A chair backrest in cream color."
i2t = build_understanding_sample("row:0:0:i2t", "", answer, vit, tokenizer, config)
i2t.validate(config)

ids = i2t.token_ids.tolist()
text = tokenizer.decode(ids)
print("i2t decoded:", repr(text[:60]), "...", repr(text[-50:]))
assert "<|im_start|>system" not in text
assert ids[0] == tokenizer.convert_tokens_to_ids("<|vision_start|>")
assert ids[-1] == im_end
assert i2t.vit_embeddings.shape == (6, 2048), "packed ViT width must stay at vit_out (2048)"
expected_labels = tokenizer.encode(answer, add_special_tokens=False) + [im_end]
assert i2t.ce_labels.tolist() == expected_labels, "CE = answer + EOS"
assert i2t.ce_indexes.numel() == len(expected_labels)
# first caption token is predicted at its independent text segment's im_start
vid_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
last_ve = len(ids) - 1 - ids[::-1].index(vid_end)
assert i2t.ce_indexes[0].item() == last_ve + 1
assert ids[last_ve + 1] == im_start, "answer must open with im_start"
assert i2t.mse_indexes is None
segs = [(s.length, s.attention_mode, s.modality, s.expert) for s in i2t.segments]
print("i2t segments:", segs)
assert segs[0][1] == "full" and segs[0][2] == "vit" and segs[0][3] == "understanding"
assert segs[1][1] == "causal" and segs[1][3] == "understanding"
# MaPE: ViT temporal band >= 1000
assert i2t.position_ids[0, i2t.vit_indexes[0]].item() >= 1000

# --- i2t with a non-empty question: visual, question text, answer text
i2t_q = build_understanding_sample("row:0:1:i2t", "What is shown?", answer, vit, tokenizer, config)
ids_q = i2t_q.token_ids.tolist()
vision_end_pos = ids_q.index(vid_end)
assert "What is shown?" in tokenizer.decode(ids_q[vision_end_pos + 1:])
assert ids_q[vision_end_pos + 1] == im_start
assert i2t_q.ce_labels.tolist() == expected_labels, "CE must stay answer-only"

print("ALL-RAW-PT-CHECKS-PASSED")
