"""FSDP2-facing Lance model.

This module is deliberately a normal MindSpeed-MM model plugin.  It does not
import the standalone Lance repository and does not patch a foreign training
entrypoint at runtime.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import torch
from torch.distributed.fsdp import fully_shard as torch_fully_shard

from mindspeed_mm.fsdp.models.base_model import BaseModel, WeightInitMixin
from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.distributed.fully_shard_parallel import get_mixprecision_policy
from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE
from mindspeed_mm.fsdp.utils.register import model_register
from mindspeed_mm.models.omni.lance.initialization import load_native_lance_checkpoint
from mindspeed_mm.models.omni.lance.initialization import initialize_random
from mindspeed_mm.models.omni.lance.modeling_lance import (
    LanceNativeModel,
    LanceVisionConnector,
    reference_sdpa,
    reference_vision_sdpa,
)
from mindspeed_mm.models.omni.lance.native_config import LanceNativeConfig
from mindspeed_mm.models.omni.lance.npu_attention import (
    AscendBlockAttentionBackend,
    AscendVisionAttentionBackend,
)
from mindspeed_mm.models.omni.lance.training_lance import (
    LanceLossWeights,
    LanceTrainingBatch,
    lance_training_step,
)


@dataclass
class LanceModelOutput:
    """Output contract consumed by the native FSDP2 training engine."""

    loss: torch.Tensor
    ce_loss: Optional[torch.Tensor]
    mse_loss: Optional[torch.Tensor]
    ce_tokens: int
    mse_tokens: int


def _getattr(config, name, default=None):
    value = getattr(config, name, default)
    return default if value is None else value


class LanceFSDPModel(LanceNativeModel, BaseModel, WeightInitMixin):
    """Native Lance architecture registered with MindSpeed-MM's ModelHub."""

    def __init__(
        self,
        native_config: LanceNativeConfig,
        *,
        ce_weight: float = 0.25,
        mse_weight: float = 1.0,
        timestep_shift: float = 1.0,
        attention_backend=None,
        vision_attention_backend=None,
        validate_batches: bool = False,
        use_vit_connector: bool = False,
        include_vit_model: bool = True,
        effective_vocab_size: Optional[int] = None,
        device=None,
        dtype=None,
    ) -> None:
        if attention_backend is None:
            attention_backend = AscendBlockAttentionBackend() if IS_NPU_AVAILABLE else reference_sdpa
        if vision_attention_backend is None:
            vision_attention_backend = (
                AscendVisionAttentionBackend() if IS_NPU_AVAILABLE else reference_vision_sdpa
            )
        super().__init__(
            native_config,
            attention_backend=attention_backend,
            vision_attention_backend=vision_attention_backend,
            include_vit_model=include_vit_model,
            use_vit_connector=use_vit_connector,
            device=device,
            dtype=dtype,
        )
        self.loss_weights = LanceLossWeights(float(ce_weight), float(mse_weight))
        self.timestep_shift = float(timestep_shift)
        self.validate_batches = bool(validate_batches)
        self.effective_vocab_size = int(
            native_config.vocab_size
            if effective_vocab_size is None else effective_vocab_size
        )
        if not 0 < self.effective_vocab_size <= native_config.vocab_size:
            raise ValueError("effective_vocab_size must be in (0, vocab_size]")

    @staticmethod
    def _native_config(model_args) -> LanceNativeConfig:
        variant = str(_getattr(model_args, "variant", "video"))
        llm_config = _getattr(model_args, "llm_config", None)
        config = (
            LanceNativeConfig.from_llm_config(llm_config, variant=variant)
            if llm_config
            else LanceNativeConfig.for_variant(variant)
        )
        overrides = _getattr(model_args, "native_config", None)
        if overrides:
            if hasattr(overrides, "to_dict"):
                overrides = overrides.to_dict()
            elif not isinstance(overrides, dict):
                overrides = vars(overrides)
            derived = {
                "head_dim", "kv_dim", "patch_latent_dim", "max_latent_frames",
                "latent_position_count", "has_vit",
            }
            config = config.with_overrides(**{
                key: value for key, value in overrides.items() if key not in derived
            })
        return config

    @classmethod
    def _from_config(cls, model_args) -> "LanceFSDPModel":
        model = cls(
            cls._native_config(model_args),
            ce_weight=float(_getattr(model_args, "ce_weight", 0.25)),
            mse_weight=float(_getattr(model_args, "mse_weight", 1.0)),
            timestep_shift=float(_getattr(model_args, "timestep_shift", 1.0)),
            validate_batches=bool(_getattr(model_args, "validate_batches", False)),
            use_vit_connector=bool(_getattr(model_args, "use_vit_connector", False)),
            include_vit_model=bool(_getattr(model_args, "include_vit_model", True)),
            effective_vocab_size=_getattr(model_args, "effective_vocab_size", None),
        )
        model.set_gradient_checkpointing(bool(_getattr(model_args, "gradient_checkpointing", False)))
        model._freeze_configured_modules(model_args)
        return model

    @classmethod
    def from_pretrained(cls, model_args) -> "LanceFSDPModel":
        model = cls._from_config(model_args)
        checkpoint = _getattr(model_args, "model_name_or_path", None)
        if not checkpoint:
            if str(_getattr(model_args, "init_mode", "")) == "random":
                initialize_random(model, int(_getattr(model_args, "init_seed", 2025)))
                return model
            raise ValueError(
                "model.model_name_or_path is required unless model.init_mode=random"
            )
        checkpoint = Path(checkpoint).expanduser()
        load_native_lance_checkpoint(model, checkpoint)
        return model

    def _freeze_configured_modules(self, model_args) -> None:
        freeze = tuple(str(item) for item in _getattr(model_args, "freeze", ()))
        for name, parameter in self.named_parameters():
            if any(name == prefix or name.startswith(prefix + ".") for prefix in freeze):
                parameter.requires_grad_(False)
        # The released PT recipe always treats the deterministic table and ViT
        # as frozen.  An explicit freeze list may repeat these entries.
        self.latent_pos_embed.pos_embed.requires_grad_(False)
        if hasattr(self, "vit_model"):
            self.vit_model.requires_grad_(False)

    def fully_shard(self, fsdp_plan):
        """Install FSDP2 units and prefetch in Lance's real execution order."""

        config = {
            "mesh": get_parallel_state().get_fsdp_device_mesh(),
            "reshard_after_forward": fsdp_plan.reshard_after_forward,
            "mp_policy": get_mixprecision_policy(fsdp_plan),
        }
        decoder = self.language_model.model
        # Child units must be wrapped before their containing decoder/root.
        leaf_units = []
        if self.connector is not None:
            leaf_units.append(self.connector)
        leaf_units.extend((
            decoder.embed_tokens,
            self.vae2llm,
            self.time_embedder,
            self.latent_pos_embed,
            *decoder.layers,
            self.language_model.lm_head,
            self.llm2vae,
        ))
        for module in leaf_units:
            torch_fully_shard(module, **config)
        torch_fully_shard(decoder, **config)
        torch_fully_shard(self, **config)

        # Module registration order is not Lance execution order: the connector
        # and latent bridges are invoked before the decoder even though they are
        # registered after it.  Explicit ordering makes forward/backward
        # prefetch overlap useful instead of gathering unrelated parameters.
        execution_order = []
        if self.connector is not None:
            execution_order.append(self.connector)
        execution_order.extend((
            decoder.embed_tokens,
            self.vae2llm,
            self.time_embedder,
            self.latent_pos_embed,
            decoder,
            *decoder.layers,
            self.language_model.lm_head,
            self.llm2vae,
        ))

        def install_prefetch(modules, count, method):
            if count <= 0:
                return
            for index, module in enumerate(modules):
                targets = modules[index + 1:index + 1 + count]
                if targets:
                    getattr(module, method)(targets)

        install_prefetch(
            execution_order, fsdp_plan.num_to_forward_prefetch,
            "set_modules_to_forward_prefetch",
        )
        install_prefetch(
            list(reversed(execution_order)), fsdp_plan.num_to_backward_prefetch,
            "set_modules_to_backward_prefetch",
        )
        return True

    def forward(self, lance_batch: LanceTrainingBatch, **unused) -> LanceModelOutput:
        del unused
        connector_guard = None
        if self.connector is not None:
            # Every rank must enter an individually wrapped FSDP2 module in the
            # same order even when data-parallel ranks carry different tasks.
            # A zero-length input keeps that collective order without adding
            # synthetic tokens or changing the loss.
            connector_input = lance_batch.vit_embeddings
            if connector_input is None:
                connector_input = torch.empty(
                    (0, self.config.hidden_size),
                    device=lance_batch.token_ids.device,
                    dtype=torch.float32,
                )
            connected = self.connector(connector_input)
            connector_guard = connected.sum() * 0.0
            if lance_batch.vit_embeddings is not None:
                # Keep data-loader batches immutable from the caller's perspective;
                # the replacement shares all non-ViT tensors and packed metadata.
                lance_batch = replace(lance_batch, vit_embeddings=connected)
        result = lance_training_step(
            self,
            lance_batch,
            self.loss_weights,
            self.timestep_shift,
            validate=self.validate_batches,
            data_parallel_group=(
                get_parallel_state().get_dp_group()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else None
            ),
        )
        if connector_guard is not None:
            result["loss"] = result["loss"] + connector_guard
        ce_tokens = 0 if lance_batch.ce_indexes is None else int(lance_batch.ce_indexes.numel())
        mse_tokens = 0 if lance_batch.mse_indexes is None else int(lance_batch.mse_indexes.numel())
        return LanceModelOutput(
            loss=result["loss"],
            ce_loss=result["ce_loss"],
            mse_loss=result["mse_loss"],
            ce_tokens=ce_tokens,
            mse_tokens=mse_tokens,
        )


model_register.register("lance")(LanceFSDPModel)
