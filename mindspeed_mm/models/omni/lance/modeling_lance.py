"""Native, checkpoint-compatible Lance model building blocks.

This module intentionally depends only on PyTorch.  It registers parameters
under the exact names used by the released Lance checkpoints and provides a
reference SDPA implementation for small-model correctness tests.  The long
sequence Ascend backend is injected separately; the reference path must not be
used for 40K/70K production sequences because its boolean mask is dense.

The architecture follows the Apache-2.0 licensed Lance and Qwen2.5-VL releases.
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .native_config import LanceNativeConfig


AttentionBackend = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    torch.Tensor,
]

KVAttentionBackend = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, bool],
    torch.Tensor,
]


@dataclass(frozen=True)
class LanceLayerKVCache:
    """Post-RoPE key/value tensors for one decoder layer."""

    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class LanceKVCache:
    """Static condition cache consumed by every diffusion denoising step."""

    layers: Tuple[LanceLayerKVCache, ...]
    condition_length: int


class LanceRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


class LanceMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias, **factory)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias, **factory)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias, **factory)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class LanceMultimodalRotaryEmbedding(nn.Module):
    def __init__(self, config: LanceNativeConfig, device=None) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32, device=device) / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.sections = tuple(config.mrope_section)

    def forward(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 3:
            if position_ids.shape[1] != 1:
                raise ValueError("native Lance reference RoPE currently expects an unbatched packed sequence")
            position_ids = position_ids[:, 0]
        if position_ids.ndim != 2 or position_ids.shape[0] != 3:
            raise ValueError("position_ids must have shape [3, sequence_length]")
        frequencies = position_ids.float().unsqueeze(-1) * self.inv_freq.float().view(1, 1, -1)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype), embedding.sin().to(dtype)

    def apply(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        section_sizes = self.sections + self.sections
        cos_sections = cos.split(section_sizes, dim=-1)
        sin_sections = sin.split(section_sizes, dim=-1)
        selected_cos = torch.cat(
            [section[index % 3] for index, section in enumerate(cos_sections)], dim=-1
        ).unsqueeze(1)
        selected_sin = torch.cat(
            [section[index % 3] for index, section in enumerate(sin_sections)], dim=-1
        ).unsqueeze(1)
        return (
            query * selected_cos + rotate_half(query) * selected_sin,
            key * selected_cos + rotate_half(key) * selected_sin,
        )


def reference_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Small-sequence reference attention with GQA expansion.

    Inputs use Lance's packed TND convention ``[tokens, heads, head_dim]`` and
    the mask uses ``True == visible`` semantics.
    """

    if attention_mask is None:
        raise ValueError("reference attention requires a dense boolean mask")
    if attention_mask.ndim != 2 or attention_mask.shape != (query.shape[0], key.shape[0]):
        raise ValueError("reference attention mask must have shape [query_tokens, key_tokens]")
    if query.shape[1] % key.shape[1]:
        raise ValueError("query head count must be divisible by KV head count")
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    output = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        attn_mask=attention_mask.bool().view(1, 1, query.shape[0], key.shape[0]),
        dropout_p=0.0,
        is_causal=False,
    )
    return output.squeeze(0).transpose(0, 1)


def reference_kv_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    """Reference q_len != kv_len attention with bottom-right causality."""

    query_length, key_length = query.shape[0], key.shape[0]
    if key_length < query_length:
        raise ValueError("KV-cache attention requires key length >= query length")
    if is_causal:
        row = torch.arange(query_length, device=query.device).unsqueeze(1)
        column = torch.arange(key_length, device=query.device).unsqueeze(0)
        attention_mask = column <= row + (key_length - query_length)
    else:
        attention_mask = torch.ones(
            query_length,
            key_length,
            dtype=torch.bool,
            device=query.device,
        )
    return reference_sdpa(query, key, value, attention_mask)


def _validate_routes(length: int, understanding: torch.Tensor, generation: torch.Tensor) -> None:
    if understanding.ndim != 1 or generation.ndim != 1:
        raise ValueError("expert indexes must be one-dimensional")
    combined = torch.cat((understanding, generation))
    if combined.numel() != length:
        raise ValueError("expert routes must cover every token exactly once")
    if combined.numel() and (
        int(combined.min().item()) < 0
        or int(combined.max().item()) >= length
        or int(torch.unique(combined).numel()) != length
    ):
        raise ValueError("expert routes contain duplicates or out-of-range indexes")


class LanceMoTAttention(nn.Module):
    def __init__(
        self,
        config: LanceNativeConfig,
        layer_index: int,
        attention_backend: AttentionBackend = reference_sdpa,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_backend = attention_backend
        factory = {"device": device, "dtype": dtype}

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True, **factory)
        self.k_proj = nn.Linear(self.hidden_size, config.kv_dim, bias=True, **factory)
        self.v_proj = nn.Linear(self.hidden_size, config.kv_dim, bias=True, **factory)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, **factory)
        self.q_norm = LanceRMSNorm(self.head_dim, config.rms_norm_eps, **factory)
        self.k_norm = LanceRMSNorm(self.head_dim, config.rms_norm_eps, **factory)

        self.q_proj_moe_gen = nn.Linear(self.hidden_size, self.hidden_size, bias=True, **factory)
        self.k_proj_moe_gen = nn.Linear(self.hidden_size, config.kv_dim, bias=True, **factory)
        self.v_proj_moe_gen = nn.Linear(self.hidden_size, config.kv_dim, bias=True, **factory)
        self.o_proj_moe_gen = nn.Linear(self.hidden_size, self.hidden_size, bias=False, **factory)
        self.q_norm_moe_gen = LanceRMSNorm(self.head_dim, config.rms_norm_eps, **factory)
        self.k_norm_moe_gen = LanceRMSNorm(self.head_dim, config.rms_norm_eps, **factory)

    @staticmethod
    def _route_projection(
        hidden_states: torch.Tensor,
        understanding_indexes: torch.Tensor,
        generation_indexes: torch.Tensor,
        understanding_projection: nn.Module,
        generation_projection: nn.Module,
        output_size: int,
    ) -> torch.Tensor:
        output = hidden_states.new_empty((hidden_states.shape[0], output_size))
        if understanding_indexes.numel():
            output[understanding_indexes] = understanding_projection(hidden_states[understanding_indexes])
        if generation_indexes.numel():
            output[generation_indexes] = generation_projection(hidden_states[generation_indexes])
        return output

    @staticmethod
    def _route_norm(
        hidden_states: torch.Tensor,
        understanding_indexes: torch.Tensor,
        generation_indexes: torch.Tensor,
        understanding_norm: nn.Module,
        generation_norm: nn.Module,
    ) -> torch.Tensor:
        output = torch.empty_like(hidden_states)
        if understanding_indexes.numel():
            output[understanding_indexes] = understanding_norm(hidden_states[understanding_indexes])
        if generation_indexes.numel():
            output[generation_indexes] = generation_norm(hidden_states[generation_indexes])
        return output

    def project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        rotary_embedding: LanceMultimodalRotaryEmbedding,
        understanding_indexes: torch.Tensor,
        generation_indexes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_routes(hidden_states.shape[0], understanding_indexes, generation_indexes)
        query = self._route_projection(
            hidden_states,
            understanding_indexes,
            generation_indexes,
            self.q_proj,
            self.q_proj_moe_gen,
            self.hidden_size,
        ).view(-1, self.num_heads, self.head_dim)
        key = self._route_projection(
            hidden_states,
            understanding_indexes,
            generation_indexes,
            self.k_proj,
            self.k_proj_moe_gen,
            self.num_key_value_heads * self.head_dim,
        ).view(-1, self.num_key_value_heads, self.head_dim)
        value = self._route_projection(
            hidden_states,
            understanding_indexes,
            generation_indexes,
            self.v_proj,
            self.v_proj_moe_gen,
            self.num_key_value_heads * self.head_dim,
        ).view(-1, self.num_key_value_heads, self.head_dim)

        query = self._route_norm(
            query,
            understanding_indexes,
            generation_indexes,
            self.q_norm,
            self.q_norm_moe_gen,
        )
        key = self._route_norm(
            key,
            understanding_indexes,
            generation_indexes,
            self.k_norm,
            self.k_norm_moe_gen,
        )
        query, key = rotary_embedding.apply(query, key, *position_embeddings)
        return query, key, value

    def project_output(
        self,
        attended: torch.Tensor,
        understanding_indexes: torch.Tensor,
        generation_indexes: torch.Tensor,
    ) -> torch.Tensor:
        attended = attended.reshape(-1, self.hidden_size)
        return self._route_projection(
            attended,
            understanding_indexes,
            generation_indexes,
            self.o_proj,
            self.o_proj_moe_gen,
            self.hidden_size,
        )

    def forward_and_cache(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        rotary_embedding: LanceMultimodalRotaryEmbedding,
        understanding_indexes: torch.Tensor,
        generation_indexes: torch.Tensor,
    ) -> Tuple[torch.Tensor, LanceLayerKVCache]:
        query, key, value = self.project_qkv(
            hidden_states,
            position_embeddings,
            rotary_embedding,
            understanding_indexes,
            generation_indexes,
        )
        attended = self.attention_backend(query, key, value, attention_mask)
        output = self.project_output(attended, understanding_indexes, generation_indexes)
        return output, LanceLayerKVCache(key=key, value=value)

    def forward_with_kv_cache(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        rotary_embedding: LanceMultimodalRotaryEmbedding,
        understanding_indexes: torch.Tensor,
        generation_indexes: torch.Tensor,
        layer_cache: LanceLayerKVCache,
        attention_backend: KVAttentionBackend,
        is_causal: bool,
    ) -> torch.Tensor:
        query, key, value = self.project_qkv(
            hidden_states,
            position_embeddings,
            rotary_embedding,
            understanding_indexes,
            generation_indexes,
        )
        merged_key = torch.cat((layer_cache.key, key), dim=0)
        merged_value = torch.cat((layer_cache.value, value), dim=0)
        attended = attention_backend(query, merged_key, merged_value, is_causal)
        return self.project_output(attended, understanding_indexes, generation_indexes)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        rotary_embedding: LanceMultimodalRotaryEmbedding,
        understanding_indexes: torch.Tensor,
        generation_indexes: torch.Tensor,
    ) -> torch.Tensor:
        output, _ = self.forward_and_cache(
            hidden_states,
            attention_mask,
            position_embeddings,
            rotary_embedding,
            understanding_indexes,
            generation_indexes,
        )
        return output


class LanceMoTDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LanceNativeConfig,
        layer_index: int,
        attention_backend: AttentionBackend = reference_sdpa,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.self_attn = LanceMoTAttention(
            config,
            layer_index,
            attention_backend=attention_backend,
            **factory,
        )
        self.mlp = LanceMLP(config.hidden_size, config.intermediate_size, **factory)
        self.mlp_moe_gen = LanceMLP(config.hidden_size, config.intermediate_size, **factory)
        self.input_layernorm = LanceRMSNorm(config.hidden_size, config.rms_norm_eps, **factory)
        self.input_layernorm_moe_gen = LanceRMSNorm(config.hidden_size, config.rms_norm_eps, **factory)
        self.post_attention_layernorm = LanceRMSNorm(config.hidden_size, config.rms_norm_eps, **factory)
        self.post_attention_layernorm_moe_gen = LanceRMSNorm(config.hidden_size, config.rms_norm_eps, **factory)

    @staticmethod
    def _route(
        hidden_states: torch.Tensor,
        understanding_indexes: torch.Tensor,
        generation_indexes: torch.Tensor,
        understanding_module: nn.Module,
        generation_module: nn.Module,
    ) -> torch.Tensor:
        output = torch.empty_like(hidden_states)
        if understanding_indexes.numel():
            output[understanding_indexes] = understanding_module(hidden_states[understanding_indexes])
        if generation_indexes.numel():
            output[generation_indexes] = generation_module(hidden_states[generation_indexes])
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        rotary_embedding: LanceMultimodalRotaryEmbedding,
        understanding_indexes: torch.Tensor,
        generation_indexes: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self._route(
            hidden_states,
            understanding_indexes,
            generation_indexes,
            self.input_layernorm,
            self.input_layernorm_moe_gen,
        )
        hidden_states = hidden_states + self.self_attn(
            normalized,
            attention_mask,
            position_embeddings,
            rotary_embedding,
            understanding_indexes,
            generation_indexes,
        )
        post_attention = self._route(
            hidden_states,
            understanding_indexes,
            generation_indexes,
            self.post_attention_layernorm,
            self.post_attention_layernorm_moe_gen,
        )
        feed_forward = self._route(
            post_attention,
            understanding_indexes,
            generation_indexes,
            self.mlp,
            self.mlp_moe_gen,
        )
        return hidden_states + feed_forward

    def forward_and_cache(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        rotary_embedding: LanceMultimodalRotaryEmbedding,
        understanding_indexes: torch.Tensor,
        generation_indexes: torch.Tensor,
    ) -> Tuple[torch.Tensor, LanceLayerKVCache]:
        normalized = self._route(
            hidden_states,
            understanding_indexes,
            generation_indexes,
            self.input_layernorm,
            self.input_layernorm_moe_gen,
        )
        attention_output, layer_cache = self.self_attn.forward_and_cache(
            normalized,
            attention_mask,
            position_embeddings,
            rotary_embedding,
            understanding_indexes,
            generation_indexes,
        )
        hidden_states = hidden_states + attention_output
        post_attention = self._route(
            hidden_states,
            understanding_indexes,
            generation_indexes,
            self.post_attention_layernorm,
            self.post_attention_layernorm_moe_gen,
        )
        feed_forward = self._route(
            post_attention,
            understanding_indexes,
            generation_indexes,
            self.mlp,
            self.mlp_moe_gen,
        )
        return hidden_states + feed_forward, layer_cache

    def forward_with_kv_cache(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        rotary_embedding: LanceMultimodalRotaryEmbedding,
        understanding_indexes: torch.Tensor,
        generation_indexes: torch.Tensor,
        layer_cache: LanceLayerKVCache,
        attention_backend: KVAttentionBackend,
        is_causal: bool,
    ) -> torch.Tensor:
        normalized = self._route(
            hidden_states,
            understanding_indexes,
            generation_indexes,
            self.input_layernorm,
            self.input_layernorm_moe_gen,
        )
        hidden_states = hidden_states + self.self_attn.forward_with_kv_cache(
            normalized,
            position_embeddings,
            rotary_embedding,
            understanding_indexes,
            generation_indexes,
            layer_cache,
            attention_backend,
            is_causal,
        )
        post_attention = self._route(
            hidden_states,
            understanding_indexes,
            generation_indexes,
            self.post_attention_layernorm,
            self.post_attention_layernorm_moe_gen,
        )
        feed_forward = self._route(
            post_attention,
            understanding_indexes,
            generation_indexes,
            self.mlp,
            self.mlp_moe_gen,
        )
        return hidden_states + feed_forward


class LanceDecoderModel(nn.Module):
    def __init__(
        self,
        config: LanceNativeConfig,
        attention_backend: AttentionBackend = reference_sdpa,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, **factory)
        self.layers = nn.ModuleList(
            [
                LanceMoTDecoderLayer(
                    config,
                    layer_index,
                    attention_backend=attention_backend,
                    **factory,
                )
                for layer_index in range(config.num_hidden_layers)
            ]
        )
        self.norm = LanceRMSNorm(config.hidden_size, config.rms_norm_eps, **factory)
        self.norm_moe_gen = LanceRMSNorm(config.hidden_size, config.rms_norm_eps, **factory)
        self.rotary_emb = LanceMultimodalRotaryEmbedding(config, device=device)
        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        understanding_indexes: torch.Tensor,
        generation_indexes: torch.Tensor,
    ) -> torch.Tensor:
        _validate_routes(hidden_states.shape[0], understanding_indexes, generation_indexes)
        position_embeddings = self.rotary_emb(position_ids, hidden_states.dtype)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and hidden_states.requires_grad:
                def layer_forward(states, current_layer=layer):
                    return current_layer(
                        states,
                        attention_mask,
                        position_embeddings,
                        self.rotary_emb,
                        understanding_indexes,
                        generation_indexes,
                    )

                hidden_states = checkpoint(layer_forward, hidden_states, use_reentrant=False)
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                    self.rotary_emb,
                    understanding_indexes,
                    generation_indexes,
                )
        return LanceMoTDecoderLayer._route(
            hidden_states,
            understanding_indexes,
            generation_indexes,
            self.norm,
            self.norm_moe_gen,
        )

    def build_kv_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        understanding_indexes: torch.Tensor,
        generation_indexes: torch.Tensor,
    ) -> Tuple[torch.Tensor, LanceKVCache]:
        """Encode a static condition once and retain every layer's K/V."""

        _validate_routes(hidden_states.shape[0], understanding_indexes, generation_indexes)
        position_embeddings = self.rotary_emb(position_ids, hidden_states.dtype)
        layer_caches = []
        for layer in self.layers:
            hidden_states, layer_cache = layer.forward_and_cache(
                hidden_states,
                attention_mask,
                position_embeddings,
                self.rotary_emb,
                understanding_indexes,
                generation_indexes,
            )
            layer_caches.append(layer_cache)
        normalized = LanceMoTDecoderLayer._route(
            hidden_states,
            understanding_indexes,
            generation_indexes,
            self.norm,
            self.norm_moe_gen,
        )
        return normalized, LanceKVCache(tuple(layer_caches), hidden_states.shape[0])

    def forward_with_kv_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        understanding_indexes: torch.Tensor,
        generation_indexes: torch.Tensor,
        kv_cache: LanceKVCache,
        attention_backend: KVAttentionBackend = reference_kv_sdpa,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Decode dynamic query tokens against a static condition cache."""

        _validate_routes(hidden_states.shape[0], understanding_indexes, generation_indexes)
        if len(kv_cache.layers) != len(self.layers):
            raise ValueError("KV-cache layer count does not match the decoder")
        position_embeddings = self.rotary_emb(position_ids, hidden_states.dtype)
        for layer, layer_cache in zip(self.layers, kv_cache.layers):
            expected = (
                kv_cache.condition_length,
                layer.self_attn.num_key_value_heads,
                layer.self_attn.head_dim,
            )
            if layer_cache.key.shape != expected or layer_cache.value.shape != expected:
                raise ValueError("KV-cache tensor shape does not match the decoder")
            hidden_states = layer.forward_with_kv_cache(
                hidden_states,
                position_embeddings,
                self.rotary_emb,
                understanding_indexes,
                generation_indexes,
                layer_cache,
                attention_backend,
                is_causal,
            )
        return LanceMoTDecoderLayer._route(
            hidden_states,
            understanding_indexes,
            generation_indexes,
            self.norm,
            self.norm_moe_gen,
        )


class LanceForCausalLM(nn.Module):
    def __init__(
        self,
        config: LanceNativeConfig,
        attention_backend: AttentionBackend = reference_sdpa,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.model = LanceDecoderModel(config, attention_backend, device=device, dtype=dtype)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )


class LanceTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 256, device=None, dtype=None) -> None:
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size, bias=True, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=dtype),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.frequency_size // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
        )
        arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((arguments.cos(), arguments.sin()), dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


def lance_3d_sincos_position_embedding(
    hidden_size: int,
    frames: int,
    height: int,
    width: int,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Rebuild Lance's frozen 3D latent position table.

    This is a torch translation of the released ``get_3d_sincos_pos_embed``
    implementation.  Dimension allocation is deliberately asymmetric when the
    hidden size is not divisible by three: time and height receive the largest
    even value below ``hidden_size // 3`` and width receives the remainder.
    """

    if hidden_size <= 0 or hidden_size % 2:
        raise ValueError("hidden_size must be a positive even integer")
    if min(frames, height, width) <= 0:
        raise ValueError("frames, height, and width must be positive")

    axis_size = hidden_size // 3
    axis_size -= axis_size % 2
    dimensions = (axis_size, axis_size, hidden_size - 2 * axis_size)
    if any(size <= 0 or size % 2 for size in dimensions):
        raise ValueError("hidden_size cannot be split into three positive even dimensions")

    # Upstream creates the coordinate grid as float32 and the frequencies as
    # float64.  Keeping that promotion order makes checkpoint regeneration
    # numerically equivalent before casting to the model dtype.
    coordinates = torch.meshgrid(
        torch.arange(frames, dtype=torch.float32, device=device),
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    embeddings = []
    for coordinate, dimension in zip(coordinates, dimensions):
        omega = torch.arange(dimension // 2, dtype=torch.float64, device=device)
        omega = 1.0 / (10000.0 ** (omega / (dimension / 2.0)))
        angles = coordinate.reshape(-1).to(torch.float64).unsqueeze(1) * omega.unsqueeze(0)
        embeddings.append(torch.cat((angles.sin(), angles.cos()), dim=1))
    return torch.cat(embeddings, dim=1).to(dtype=dtype)


class LancePositionEmbedding3D(nn.Module):
    def __init__(self, config: LanceNativeConfig, device=None, dtype=None) -> None:
        super().__init__()
        parameter_dtype = dtype or torch.get_default_dtype()
        self.pos_embed = nn.Parameter(
            lance_3d_sincos_position_embedding(
                config.hidden_size,
                config.max_latent_frames,
                config.max_latent_size,
                config.max_latent_size,
                device=device,
                dtype=parameter_dtype,
            ),
            requires_grad=False,
        )

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pos_embed[position_ids]


VisionAttentionBackend = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


def reference_vision_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cumulative_lengths: torch.Tensor,
) -> torch.Tensor:
    """Full attention inside each Qwen-VL window/frame, without cross-item KV."""

    if cumulative_lengths.ndim != 1 or cumulative_lengths.numel() < 2:
        raise ValueError("vision cumulative lengths must be one-dimensional with at least two entries")
    boundaries = cumulative_lengths.tolist()
    if boundaries[0] != 0 or boundaries[-1] != query.shape[0]:
        raise ValueError("vision cumulative lengths must cover every token")
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        output = F.scaled_dot_product_attention(
            query[start:end].transpose(0, 1).unsqueeze(0),
            key[start:end].transpose(0, 1).unsqueeze(0),
            value[start:end].transpose(0, 1).unsqueeze(0),
            dropout_p=0.0,
            is_causal=False,
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
    if not outputs:
        raise ValueError("vision cumulative lengths contain no non-empty sequence")
    return torch.cat(outputs, dim=0)


def apply_vision_rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    query_dtype, key_dtype = query.dtype, key.dtype
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    query_float = query.float()
    key_float = key.float()
    return (
        (query_float * cos + rotate_half(query_float) * sin).to(query_dtype),
        (key_float * cos + rotate_half(key_float) * sin).to(key_dtype),
    )


class LanceVisionAttention(nn.Module):
    def __init__(
        self,
        config: LanceNativeConfig,
        attention_backend: VisionAttentionBackend = reference_vision_sdpa,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.num_heads = config.vit_num_heads
        self.head_dim = config.vit_hidden_size // config.vit_num_heads
        self.attention_backend = attention_backend
        self.qkv = nn.Linear(config.vit_hidden_size, 3 * config.vit_hidden_size, bias=True, **factory)
        self.proj = nn.Linear(config.vit_hidden_size, config.vit_hidden_size, bias=True, **factory)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cumulative_lengths: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).view(length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=1)
        query, key = apply_vision_rotary_embedding(
            query,
            key,
            position_embeddings[0],
            position_embeddings[1],
        )
        output = self.attention_backend(query, key, value, cumulative_lengths)
        return self.proj(output.reshape(length, -1))


class LanceVisionBlock(nn.Module):
    def __init__(
        self,
        config: LanceNativeConfig,
        attention_backend: VisionAttentionBackend = reference_vision_sdpa,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.norm1 = LanceRMSNorm(config.vit_hidden_size, 1e-6, **factory)
        self.norm2 = LanceRMSNorm(config.vit_hidden_size, 1e-6, **factory)
        self.attn = LanceVisionAttention(config, attention_backend=attention_backend, **factory)
        self.mlp = LanceMLP(
            config.vit_hidden_size,
            config.vit_intermediate_size,
            bias=True,
            **factory,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cumulative_lengths: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cumulative_lengths,
            position_embeddings,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class LanceVisionMerger(nn.Module):
    def __init__(self, config: LanceNativeConfig, device=None, dtype=None) -> None:
        super().__init__()
        merged = config.vit_hidden_size * config.vit_spatial_merge_size ** 2
        self.ln_q = LanceRMSNorm(config.vit_hidden_size, 1e-6, device=device, dtype=dtype)
        self.mlp = nn.Sequential(
            nn.Linear(merged, merged, bias=True, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(merged, config.vit_out_hidden_size, bias=True, device=device, dtype=dtype),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.ln_q(hidden_states)
        return self.mlp(normalized.reshape(-1, self.mlp[0].in_features))


class LanceVisionModel(nn.Module):
    """Checkpoint-compatible Qwen2.5-VL ViT with released window semantics."""

    def __init__(
        self,
        config: LanceNativeConfig,
        attention_backend: VisionAttentionBackend = reference_vision_sdpa,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.vit_spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size ** 2
        self.patch_size = config.vit_patch_size
        self.window_size = config.vit_window_size
        self.fullatt_block_indexes = frozenset(config.vit_fullatt_block_indexes)
        self.patch_embed = LanceVisionPatchEmbed(config, device=device, dtype=dtype)
        rotary_dim = (config.vit_hidden_size // config.vit_num_heads) // 2
        inv_freq = 1.0 / (
            10000.0
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
        )
        self.register_buffer("visual_inv_freq", inv_freq, persistent=False)
        self.blocks = nn.ModuleList(
            [
                LanceVisionBlock(
                    config,
                    attention_backend=attention_backend,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(config.vit_depth)
            ]
        )
        self.merger = LanceVisionMerger(config, device=device, dtype=dtype)

    def rotary_frequencies(self, sequence_length: int) -> torch.Tensor:
        positions = torch.arange(
            sequence_length,
            dtype=self.visual_inv_freq.dtype,
            device=self.visual_inv_freq.device,
        )
        return torch.outer(positions, self.visual_inv_freq)

    def rotary_position_embedding(self, grid_thw: torch.Tensor) -> torch.Tensor:
        position_ids = []
        for temporal, height, width in grid_thw.tolist():
            if height % self.spatial_merge_size or width % self.spatial_merge_size:
                raise ValueError("ViT grid height/width must be divisible by spatial_merge_size")
            height_ids = torch.arange(height, device=grid_thw.device).unsqueeze(1).expand(-1, width)
            height_ids = height_ids.reshape(
                height // self.spatial_merge_size,
                self.spatial_merge_size,
                width // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            width_ids = torch.arange(width, device=grid_thw.device).unsqueeze(0).expand(height, -1)
            width_ids = width_ids.reshape(
                height // self.spatial_merge_size,
                self.spatial_merge_size,
                width // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            position_ids.append(torch.stack((height_ids, width_ids), dim=-1).repeat(temporal, 1))
        positions = torch.cat(position_ids, dim=0)
        maximum_grid = int(grid_thw[:, 1:].max().item())
        return self.rotary_frequencies(maximum_grid)[positions].flatten(1)

    def get_window_index(self, grid_thw: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        window_indexes = []
        cumulative_lengths = [0]
        index_offset = 0
        merger_window = self.window_size // self.spatial_merge_size // self.patch_size
        for temporal, height, width in grid_thw.tolist():
            llm_height = height // self.spatial_merge_size
            llm_width = width // self.spatial_merge_size
            index = torch.arange(
                temporal * llm_height * llm_width,
                device=grid_thw.device,
            ).reshape(temporal, llm_height, llm_width)
            # Keep the released padding rule exactly, including a full padding
            # window when an axis is already divisible.
            pad_height = merger_window - llm_height % merger_window
            pad_width = merger_window - llm_width % merger_window
            windows_height = (llm_height + pad_height) // merger_window
            windows_width = (llm_width + pad_width) // merger_window
            padded = F.pad(index, (0, pad_width, 0, pad_height), "constant", -100)
            padded = padded.reshape(
                temporal,
                windows_height,
                merger_window,
                windows_width,
                merger_window,
            ).permute(0, 1, 3, 2, 4).reshape(
                temporal,
                windows_height * windows_width,
                merger_window,
                merger_window,
            )
            sequence_lengths = (padded != -100).sum((2, 3)).reshape(-1)
            flat = padded.reshape(-1)
            window_indexes.append(flat[flat != -100] + index_offset)
            local_cumulative = sequence_lengths.cumsum(0) * self.spatial_merge_unit
            cumulative_lengths.extend(
                (local_cumulative + cumulative_lengths[-1]).tolist()
            )
            index_offset += temporal * llm_height * llm_width
        return torch.cat(window_indexes), tuple(cumulative_lengths)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError("grid_thw must have shape [items, 3]")
        if torch.any(grid_thw <= 0):
            raise ValueError("grid_thw values must be positive")
        expected_patches = int(grid_thw.prod(dim=1).sum().item())
        if hidden_states.shape[0] != expected_patches:
            raise ValueError("raw ViT patch count does not match grid_thw")
        hidden_states = self.patch_embed(hidden_states)
        rotary = self.rotary_position_embedding(grid_thw)
        window_index, window_lengths = self.get_window_index(grid_thw)
        window_lengths_tensor = torch.tensor(
            window_lengths,
            dtype=torch.int32,
            device=hidden_states.device,
        ).unique_consecutive()

        sequence_length = hidden_states.shape[0]
        if sequence_length % self.spatial_merge_unit:
            raise ValueError("ViT patch count must align to spatial_merge_unit")
        hidden_states = hidden_states.reshape(
            sequence_length // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )[window_index].reshape(sequence_length, -1)
        rotary = rotary.reshape(
            sequence_length // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )[window_index].reshape(sequence_length, -1)
        doubled = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (doubled.cos(), doubled.sin())

        full_lengths = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0],
        ).cumsum(dim=0, dtype=torch.int32)
        full_lengths = F.pad(full_lengths, (1, 0), value=0)
        for layer_index, block in enumerate(self.blocks):
            cumulative_lengths = (
                full_lengths if layer_index in self.fullatt_block_indexes else window_lengths_tensor
            )
            hidden_states = block(hidden_states, cumulative_lengths, position_embeddings)
        merged = self.merger(hidden_states)
        return merged[torch.argsort(window_index)]


class LanceVisionPatchEmbed(nn.Module):
    def __init__(self, config: LanceNativeConfig, device=None, dtype=None) -> None:
        super().__init__()
        self.in_channels = config.vit_in_channels
        self.temporal_patch_size = config.vit_temporal_patch_size
        self.patch_size = config.vit_patch_size
        self.hidden_size = config.vit_hidden_size
        self.proj = nn.Conv3d(
            config.vit_in_channels,
            config.vit_hidden_size,
            kernel_size=(
                config.vit_temporal_patch_size,
                config.vit_patch_size,
                config.vit_patch_size,
            ),
            stride=(
                config.vit_temporal_patch_size,
                config.vit_patch_size,
                config.vit_patch_size,
            ),
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        patch_dimension = (
            self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        )
        if hidden_states.ndim != 2 or hidden_states.shape[1] != patch_dimension:
            raise ValueError(
                "raw ViT patches must have shape [tokens, {}]".format(patch_dimension)
            )
        reshaped = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(reshaped.to(self.proj.weight.dtype)).reshape(-1, self.hidden_size)


class LanceNativeModel(nn.Module):
    """Native Lance parameter tree plus a reference language forward."""

    def __init__(
        self,
        config: LanceNativeConfig,
        attention_backend: AttentionBackend = reference_sdpa,
        vision_attention_backend: VisionAttentionBackend = reference_vision_sdpa,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.language_model = LanceForCausalLM(
            config,
            attention_backend=attention_backend,
            device=device,
            dtype=dtype,
        )
        self.latent_pos_embed = LancePositionEmbedding3D(config, device=device, dtype=dtype)
        self.llm2vae = nn.Linear(
            config.hidden_size,
            config.patch_latent_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.vae2llm = nn.Linear(
            config.patch_latent_dim,
            config.hidden_size,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.time_embedder = LanceTimestepEmbedder(config.hidden_size, device=device, dtype=dtype)
        if config.has_vit:
            self.vit_model = LanceVisionModel(
                config,
                attention_backend=vision_attention_backend,
                device=device,
                dtype=dtype,
            )

    def forward_language(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        understanding_indexes: torch.Tensor,
        generation_indexes: torch.Tensor,
    ) -> torch.Tensor:
        return self.language_model.model(
            hidden_states,
            position_ids,
            attention_mask,
            understanding_indexes,
            generation_indexes,
        )

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.language_model.model.set_gradient_checkpointing(enabled)

    def build_language_kv_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        understanding_indexes: torch.Tensor,
        generation_indexes: torch.Tensor,
    ) -> Tuple[torch.Tensor, LanceKVCache]:
        return self.language_model.model.build_kv_cache(
            hidden_states,
            position_ids,
            attention_mask,
            understanding_indexes,
            generation_indexes,
        )

    def forward_language_with_kv_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        understanding_indexes: torch.Tensor,
        generation_indexes: torch.Tensor,
        kv_cache: LanceKVCache,
        attention_backend: KVAttentionBackend = reference_kv_sdpa,
        is_causal: bool = False,
    ) -> torch.Tensor:
        return self.language_model.model.forward_with_kv_cache(
            hidden_states,
            position_ids,
            understanding_indexes,
            generation_indexes,
            kv_cache,
            attention_backend,
            is_causal,
        )

    def compute_heads(
        self,
        hidden_states: torch.Tensor,
        ce_indexes: Optional[torch.Tensor] = None,
        mse_indexes: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        return {
            "logits": self.language_model.lm_head(hidden_states[ce_indexes]) if ce_indexes is not None else None,
            "velocity": self.llm2vae(hidden_states[mse_indexes]) if mse_indexes is not None else None,
        }
