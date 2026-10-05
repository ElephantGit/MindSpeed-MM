"""Train the official Tuna-2 architecture with MindSpeed-MM FSDP2."""
import os

os.environ['USE_TF'] = 'FALSE'
import mindspeed.megatron_adaptor
import torch
from omegaconf import OmegaConf
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args
from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.training import pretrain
from mindspeed_mm.models.tuna2.native import MindSpeedTuna2
from mindspeed_mm.models.tuna2.runtime import build_loaders, MixedIterator, move_batch

_MODEL = None


def tuna_args(parser):
    parser = mm_extra_args_provider(parser)
    parser.add_argument('--tuna-config', required=True)
    return parser


def model_provider(*args, **kwargs):
    global _MODEL
    cfg = OmegaConf.load(get_args().tuna_config)
    args = get_args()
    if args.init_model_with_meta_device:
        raise ValueError("Tuna-2 requires loaded CPU weights; meta initialization is unsupported")
    if any(int(stream.get("batch_size", 1)) != args.micro_batch_size for stream in cfg.data.streams.values()):
        raise ValueError("Native training requires every stream batch_size == micro_batch_size")
    _MODEL = MindSpeedTuna2(cfg.model)
    return _MODEL


def get_batch(iterator):
    device = next(_MODEL.parameters()).device
    return move_batch(next(iterator), device)


def loss_func(output):
    group = mpu.get_data_parallel_group()
    metrics = {}
    for key in ('loss', 'loss_ntp', 'loss_flow'):
        value = output[key].detach().float().clone()
        torch.distributed.all_reduce(value, group=group)
        metrics[key] = value / torch.distributed.get_world_size(group)
    return output['loss'].view(1), metrics


def forward_step(iterator, model):
    return model(get_batch(iterator)), loss_func


def train_valid_test_datasets_provider(*args):
    cfg = OmegaConf.load(get_args().tuna_config)
    group = mpu.get_data_parallel_group()
    loaders = build_loaders(cfg, _MODEL.model, process_group=group)
    iterator = MixedIterator(loaders, cfg.data.sampling_weights,
                             next(_MODEL.parameters()).device,
                             seed=get_args().seed, process_group=group)
    return iterator, None, None


if __name__ == '__main__':
    from mindspeed_mm.patchs import torch_dcp_patch
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder, forward_step, extra_args_provider=tuna_args,
             args_defaults={'dataloader_type': 'external', 'vision_pretraining': False,
                            'curr_forward_iteration': 0})
