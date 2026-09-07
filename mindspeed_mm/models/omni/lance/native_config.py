"""Dependency-free configuration contract for the native Lance port.

The values in this module describe the released Lance 3B checkpoints.  Keeping
the contract independent from torch/transformers lets checkpoint inspection and
CI validation run on machines that do not have an Ascend software stack.
"""

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union


class LanceConfigError(ValueError):
    """Raised when a configuration cannot describe a supported Lance model."""


@dataclass(frozen=True)
class LanceNativeConfig:
    """Shape-bearing configuration of the released Lance 3B architecture."""

    variant: str = "image"
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 128000
    rope_theta: float = 1000000.0
    mrope_section: Tuple[int, int, int] = (16, 24, 24)
    qkv_bias: bool = True
    tie_word_embeddings: bool = False

    latent_channels: int = 48
    latent_patch_size: Tuple[int, int, int] = (1, 1, 1)
    max_latent_size: int = 64
    max_num_frames: int = 1
    latent_temporal_downsample: int = 4

    vit_depth: int = 32
    vit_hidden_size: int = 1280
    vit_intermediate_size: int = 3420
    vit_num_heads: int = 16
    vit_in_channels: int = 3
    vit_patch_size: int = 14
    vit_temporal_patch_size: int = 2
    vit_spatial_merge_size: int = 2
    vit_window_size: int = 112
    vit_fullatt_block_indexes: Tuple[int, ...] = (7, 15, 23, 31)
    vit_out_hidden_size: int = 2048

    def __post_init__(self) -> None:
        if self.variant not in ("image", "video"):
            raise LanceConfigError("variant must be 'image' or 'video'")
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "latent_channels": self.latent_channels,
            "max_latent_size": self.max_latent_size,
            "max_num_frames": self.max_num_frames,
            "latent_temporal_downsample": self.latent_temporal_downsample,
            "vit_depth": self.vit_depth,
            "vit_hidden_size": self.vit_hidden_size,
            "vit_intermediate_size": self.vit_intermediate_size,
            "vit_num_heads": self.vit_num_heads,
            "vit_window_size": self.vit_window_size,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise LanceConfigError("configuration values must be positive: {}".format(", ".join(invalid)))
        if self.hidden_size % self.num_attention_heads:
            raise LanceConfigError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise LanceConfigError("num_attention_heads must be divisible by num_key_value_heads")
        if sum(self.mrope_section) != self.head_dim // 2:
            raise LanceConfigError("mrope_section must sum to half of the attention head dimension")
        if len(self.latent_patch_size) != 3 or any(item <= 0 for item in self.latent_patch_size):
            raise LanceConfigError("latent_patch_size must contain three positive integers")
        if self.vit_out_hidden_size != self.hidden_size:
            raise LanceConfigError("released Lance checkpoints require ViT output size == LLM hidden size")
        if self.vit_hidden_size % self.vit_num_heads:
            raise LanceConfigError("vit_hidden_size must be divisible by vit_num_heads")
        if self.vit_window_size % (self.vit_spatial_merge_size * self.vit_patch_size):
            raise LanceConfigError("vit_window_size must align to merged ViT patches")
        if len(set(self.vit_fullatt_block_indexes)) != len(self.vit_fullatt_block_indexes) or any(
            index < 0 for index in self.vit_fullatt_block_indexes
        ):
            raise LanceConfigError("ViT full-attention block indexes must be unique and non-negative")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def patch_latent_dim(self) -> int:
        result = self.latent_channels
        for size in self.latent_patch_size:
            result *= size
        return result

    @property
    def max_latent_frames(self) -> int:
        # This is the exact formula used by upstream Lance.__init__.
        return self.max_num_frames // self.latent_temporal_downsample + 1

    @property
    def latent_position_count(self) -> int:
        return self.max_latent_frames * self.max_latent_size * self.max_latent_size

    @property
    def has_vit(self) -> bool:
        return self.variant == "video"

    def to_dict(self) -> Dict[str, Any]:
        result = dict(self.__dict__)
        result["mrope_section"] = list(self.mrope_section)
        result["latent_patch_size"] = list(self.latent_patch_size)
        result["vit_fullatt_block_indexes"] = list(self.vit_fullatt_block_indexes)
        result.update(
            {
                "head_dim": self.head_dim,
                "kv_dim": self.kv_dim,
                "patch_latent_dim": self.patch_latent_dim,
                "max_latent_frames": self.max_latent_frames,
                "latent_position_count": self.latent_position_count,
                "has_vit": self.has_vit,
            }
        )
        return result

    @classmethod
    def for_variant(cls, variant: str) -> "LanceNativeConfig":
        if variant == "image":
            return cls(variant="image", max_num_frames=1)
        if variant == "video":
            # Released Lance_3B_Video uses 121 decoded frames -> 31 VAE frames.
            return cls(variant="video", max_num_frames=121)
        raise LanceConfigError("variant must be 'image' or 'video'")

    @classmethod
    def from_llm_config(
        cls,
        source: Union[str, Path, Mapping[str, Any]],
        variant: str = "image",
    ) -> "LanceNativeConfig":
        """Load shape-bearing LLM fields from a Hugging Face config JSON.

        Lance releases a standalone ``llm_config.json``.  Nested ``text_config``
        is accepted as well so this validator can also consume a Qwen2.5-VL
        style config without importing transformers.
        """

        if isinstance(source, Mapping):
            raw = dict(source)
        else:
            path = Path(source).expanduser().resolve()
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise LanceConfigError("could not read LLM config {}: {}".format(path, exc)) from exc
        if isinstance(raw.get("text_config"), Mapping):
            raw = dict(raw["text_config"])

        aliases = {
            "vocab_size": "vocab_size",
            "hidden_size": "hidden_size",
            "intermediate_size": "intermediate_size",
            "num_hidden_layers": "num_hidden_layers",
            "num_attention_heads": "num_attention_heads",
            "num_key_value_heads": "num_key_value_heads",
            "rms_norm_eps": "rms_norm_eps",
            "max_position_embeddings": "max_position_embeddings",
            "rope_theta": "rope_theta",
            "tie_word_embeddings": "tie_word_embeddings",
        }
        updates = {target: raw[key] for key, target in aliases.items() if key in raw}
        rope_scaling = raw.get("rope_scaling")
        if isinstance(rope_scaling, Mapping) and "mrope_section" in rope_scaling:
            updates["mrope_section"] = tuple(int(item) for item in rope_scaling["mrope_section"])
        return replace(cls.for_variant(variant), **updates)
