"""Portable training runner for Tuna-2, including CPU smoke and NPU DDP."""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

from .runtime import build_model, build_loaders, MixedIterator, move_batch, device_for, seed_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda', 'npu'], default='cpu')
    p.add_argument('--resume')
    p.add_argument('--max-steps', type=int)
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    device = device_for(args.device)
    distributed = int(os.getenv('WORLD_SIZE', '1')) > 1
    if distributed:
        torch.distributed.init_process_group({'cpu': 'gloo', 'cuda': 'nccl', 'npu': 'hccl'}[args.device])
    rank = torch.distributed.get_rank() if distributed else 0
    seed_all(int(cfg.get('seed', 42)))
    torch.set_num_threads(2)
    model = build_model(cfg.model).to(device)
    model.train()
    loaders = build_loaders(cfg, model)
    iterator = MixedIterator(loaders, cfg.data.sampling_weights, device,
                             seed=int(cfg.get('seed', 42)), cycle=cfg.training.get('cycle_streams', False))
    module = DistributedDataParallel(model, device_ids=None if device.type == 'cpu' else [device.index],
                                     find_unused_parameters=True) if distributed else model
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=float(cfg.training.learning_rate), betas=(.9, .95),
                                  weight_decay=float(cfg.training.weight_decay), eps=1e-8)
    warmup = int(cfg.training.get('warmup_steps', 0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1., (s + 1) / max(1, warmup)))
    decay = cfg.training.get('ema_decay')
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if decay is not None else None
    start = 0
    if args.resume:
        ckpt = torch.load(args.resume.format(rank=rank), map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        ema, start = ckpt['ema'], ckpt['step']
        # Rebuild loader offsets using the initial seed, then restore model RNG.
        for _ in range(ckpt['batches_consumed']):
            next(iterator)
        random.setstate(ckpt['python_rng'])
        np.random.set_state(ckpt['numpy_rng'])
        torch.set_rng_state(ckpt['torch_rng'].cpu())
        if device.type != 'cpu' and 'device_rng' in ckpt:
            getattr(torch, device.type).set_rng_state(ckpt['device_rng'].cpu(), device)
    accum = int(cfg.training.get('gradient_accumulation_steps', 1))
    steps = args.max_steps or int(cfg.training.get('max_steps', cfg.training.get('max_steps_per_epoch', 8)))
    if accum < 1 or steps <= start:
        raise ValueError('Need positive accumulation and max_steps greater than checkpoint step')
    output = Path(cfg.training.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    precision = cfg.training.get('mixed_precision')
    if precision not in (None, 'bf16'):
        raise ValueError('Supported precision: fp32 (null) or bf16')
    before = {k: v.detach().cpu().clone() for k, v in model.named_parameters()}
    records = []
    for step in range(start, steps):
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        for _ in range(accum):
            batch = move_batch(next(iterator), device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
                result = module(batch)
                loss = result['loss'] / accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss for {iterator.last_stream}')
            loss.backward()
            for key in ('loss', 'loss_ntp', 'loss_flow'):
                losses[key] = losses.get(key, 0.) + result[key].detach().float().item() / accum
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.is_floating_point():
                        ema[k].lerp_(v.detach(), 1. - float(decay))
                    else:
                        ema[k].copy_(v)
        row = dict(step=step + 1, stream=iterator.last_stream, grad_norm=float(norm), **losses)
        records.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)
    changed = [k for k, v in model.named_parameters() if not torch.equal(before[k], v.detach().cpu())]
    if not changed:
        raise AssertionError('No parameter updated')
    ckpt = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                ema=ema, step=steps, batches_consumed=iterator.step, config=OmegaConf.to_container(cfg),
                python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state())
    if device.type != 'cpu':
        ckpt['device_rng'] = getattr(torch, device.type).get_rng_state(device)
    # Every rank retains its own RNG and dataloader position for restart.
    checkpoint_path = output / (f'last-rank{rank}.pt' if distributed else 'last.pt')
    torch.save(ckpt, checkpoint_path)
    if rank == 0:
        (output / 'metrics.json').write_text(json.dumps(dict(device=str(device), steps=steps,
              world_size=int(os.getenv('WORLD_SIZE', '1')), parameters=sum(p.numel() for p in model.parameters()),
              updated_parameters=len(changed), records=records), indent=2) + '\n')
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
