"""Megatron/MindSpeed-MM FSDP2 model adapter."""
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.transformers.base_model import FSDP2Mixin
from .runtime import build_model


class MindSpeedTuna2(MultiModalModule, FSDP2Mixin):
    def __init__(self, model_config):
        super().__init__(core_transformer_config_from_args(get_args()))
        self.model = build_model(model_config)

    def forward(self, batch):
        return self.model(batch)

    def fully_shard(self, process_group, fsdp2_config_path, **kwargs):
        # FSDP2Mixin operates on this root and the paths in fsdp2.yaml.
        return FSDP2Mixin.fully_shard(self, process_group, fsdp2_config_path, **kwargs)
