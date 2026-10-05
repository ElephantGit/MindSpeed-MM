# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tuna training infrastructure: AutoUnit, trainer, callbacks, FSDP utils."""

from __future__ import annotations

from mindspeed_mm.models.tuna2.upstream.training.callbacks import LocalCheckpointCallback, SaveImageCallback
from mindspeed_mm.models.tuna2.upstream.training.fsdp_utils import (
    build_activation_checkpoint_params,
    constant_lr_with_warmup,
    create_fsdp_strategy,
)
from mindspeed_mm.models.tuna2.upstream.training.pred_unit import TunaPredUnit
from mindspeed_mm.models.tuna2.upstream.training.trainer import build_default_callbacks, build_tb_logger, train
from mindspeed_mm.models.tuna2.upstream.training.unit import TunaUnit


__all__ = [
    "LocalCheckpointCallback",
    "SaveImageCallback",
    "TunaPredUnit",
    "TunaUnit",
    "build_activation_checkpoint_params",
    "build_default_callbacks",
    "build_tb_logger",
    "constant_lr_with_warmup",
    "create_fsdp_strategy",
    "train",
]
