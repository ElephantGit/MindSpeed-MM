"""Shared model, data and device adapters for native and standalone training."""
import importlib
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler

from .upstream.models.tuna_2_pixel import Tuna2PixelModel


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_batch(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_batch(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_batch(v, device) for v in value)
    return value


def build_model(config):
    kwargs = OmegaConf.to_container(config, resolve=True)
    kwargs.pop('_target_', None)
    if kwargs.get('attention_backend', 'sdpa') != 'sdpa':
        raise ValueError('MindSpeed-MM Tuna-2 uses SDPA; flexattention is CUDA-specific.')
    return Tuna2PixelModel(**kwargs)


def collate(samples):
    # Pixel-only Tuna-2 does not consume the dataset's SigLIP/reconstruction
    # fallback. Retaining it silently replaces caption embeddings with images.
    from torch.utils.data import default_collate
    return default_collate([{k: v for k, v in s.items() if k != 'images_clip'} for s in samples])


def build_loaders(config, model, process_group=None):
    loaders = {}
    distributed = torch.distributed.is_initialized()
    rank = torch.distributed.get_rank(process_group) if distributed else 0
    world = torch.distributed.get_world_size(process_group) if distributed else 1
    for name, stream in config.data.streams.items():
        kw = OmegaConf.to_container(stream, resolve=True)
        target = kw.pop('_target_')
        module, cls = target.rsplit('.', 1)
        bs = int(kw.pop('batch_size', config.training.get('batch_size', 1)))
        nw = int(kw.pop('num_workers', config.training.get('num_workers', 0)))
        dataset = getattr(importlib.import_module(module), cls)(
            tokenizer=model.text_tokenizer, tuna_token_ids=model.tuna_token_ids, **kw)
        if len(dataset) < bs:
            raise ValueError(f'{name}: dataset size {len(dataset)} smaller than batch size {bs}')
        sampler = DistributedSampler(dataset, world, rank, seed=int(config.get('seed', 42))) if distributed else None
        loaders[name] = DataLoader(dataset, batch_size=bs, num_workers=nw,
                                   shuffle=sampler is None, sampler=sampler,
                                   drop_last=False, collate_fn=collate)
    return loaders


class MixedIterator:
    """Synchronized weighted stream choice; cycle mode covers all smoke tasks."""
    def __init__(self, loaders, weights, device, seed=42, cycle=False, process_group=None):
        self.loaders, self.names = loaders, list(loaders)
        self.weights = [float(weights[n]) for n in self.names]
        if any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
            raise ValueError('Sampling weights must be nonnegative with a positive sum')
        self.rng = random.Random(seed)
        self.device, self.cycle, self.group = device, cycle, process_group
        self.iterators = {n: iter(dl) for n, dl in loaders.items()}
        self.epochs = {n: 0 for n in loaders}
        self.step = 0
        self.last_stream = None

    def __iter__(self):
        return self

    def __next__(self):
        index = self.step % len(self.names) if self.cycle else self.rng.choices(range(len(self.names)), self.weights)[0]
        if torch.distributed.is_initialized():
            index_tensor = torch.tensor(index, dtype=torch.long, device=self.device)
            src = torch.distributed.get_global_rank(self.group, 0) if self.group is not None else 0
            torch.distributed.broadcast(index_tensor, src=src, group=self.group)
            index = index_tensor.item()
        name = self.names[index]
        try:
            batch = next(self.iterators[name])
        except StopIteration:
            self.epochs[name] += 1
            sampler = self.loaders[name].sampler
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(self.epochs[name])
            self.iterators[name] = iter(self.loaders[name])
            batch = next(self.iterators[name])
        self.step += 1
        self.last_stream = name
        return batch


def device_for(name):
    if name == 'npu':
        import torch_npu  # Register the Ascend backend, without CUDA monkeypatches.
        device = torch.device('npu', int(os.getenv('LOCAL_RANK', 0)))
        torch.npu.set_device(device)
        return device
    if name == 'cuda':
        device = torch.device('cuda', int(os.getenv('LOCAL_RANK', 0)))
        torch.cuda.set_device(device)
        return device
    return torch.device('cpu')
