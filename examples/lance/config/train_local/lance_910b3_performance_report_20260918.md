# Lance Qwen3-0.6B performance probe (8 x Ascend 910B3)

Date: 2026-09-18

All successful rows use the same initialization, the same 5,292 prepared
T2I/I2T task documents, BF16 parameters, FP32 reductions, FSDP2 size 8,
`micro_batch_size=1`, `gradient_accumulation_steps=1`, decoder gradient
checkpointing, EMA, and `reshard_after_forward=true`.  Step 1 is excluded.
Ragged final packed files are excluded where explicitly noted.

| pack setting | aggregate tokens/step | stable tokens/s | estimated dense TFLOPS (8 NPU) | result |
|---|---:|---:|---:|---|
| 8K (actual 7.0-7.6K/rank) | ~58K | 22.4K | 79.0 | fixed FSDP/optimizer cost dominates |
| 16K (actual 14.0-14.6K/rank) | ~114K | **37.3K** | **131.5** | best measured point (10-step confirmation) |
| 24K (actual 22.0-22.6K/rank) | ~178K | 31.6K | 111.3 | slower than 16K |
| 32K (actual 30.0-30.6K/rank) | ~243K | 29.1K | 102.4 | slower than 24K |
| 40K (actual 38.0-38.5K/rank) | ~306K | 26.5K | 93.2 | excludes one ragged step |
| 50K (actual 44.0-44.6K/rank) | ~354K | 25.2K | 88.7 | excludes one ragged step |
| 74K (actual 72.0-72.6K/rank) | ~578K | 17.3K | 60.8 | highest token pack, poor throughput |

The TFLOPS estimate is a transparent model-FLOP lower bound, not a hardware
counter.  Qwen3-0.6B here has hidden size 1024, intermediate size 3072, 28
layers, 16 query heads, and explicit head dimension 128.  Per token:

```
forward_dense = 28 * 2 * (6 * 1024^2 + 3 * 1024 * 3072)
              = 0.8808 GFLOP/token
train_with_full_decoder_recompute ~= 4 * forward_dense
                                  = 3.523 GFLOP/token
estimated_TFLOPS = measured_tokens_per_second * 3.523e9 / 1e12
```

This excludes attention-score matmuls, bridges, LM head, communication, and
optimizer work, so it must not be labeled as exact MFU.  `npu-smi` on the 16K
confirmation showed about 17% HBM usage on rank 0; AICore samples reached
89-99% during matrix-compute phases and fell near zero during communication or
optimizer phases.  The coarse sampler therefore explains phase behavior but is
not a substitute for a profiler-derived MFU.

Additional probes:

- Disabling decoder checkpointing OOMed on the first forward at both 50K and
  32K.  Active allocation reached roughly 57.6-59.4 GiB on a 64GB device.
- `reshard_after_forward=false` raised sampled HBM usage (about 29% versus 26%
  at 50K) but reduced token throughput.
- Disabling EMA changed stable step throughput only within short-run noise; it
  mainly reduced checkpoint size/time.  It is useful for short-run evaluation
  semantics, not a demonstrated compute optimization.

Evidence is retained under `/mnt/models/outputs/lance-bs-probe-*`.  The final
confirmation is `lance-bs-probe-16k-10step.log` plus its eight rank traces.

