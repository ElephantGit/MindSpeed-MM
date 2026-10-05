#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${TUNA_CONFIG:?Set TUNA_CONFIG to a YAML with local Qwen weights and four JSONL streams}"
: "${SAVE_PATH:=outputs/tuna2-native}"
unset NON_MEGATRON
torchrun --nproc_per_node "${NPUS_PER_NODE:-8}" pretrain_tuna2.py \
  --tuna-config "$TUNA_CONFIG" --mm-model examples/tuna2/model.json --use-torch-fsdp2 \
  --fsdp2-config-path examples/tuna2/fsdp2.yaml \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 --num-layers 28 --hidden-size 3584 \
  --num-attention-heads 28 --seq-length 4096 --max-position-embeddings 4096 \
  --micro-batch-size 1 --global-batch-size "${NPUS_PER_NODE:-8}" \
  --train-iters "${TRAIN_ITERS:-8}" --lr 2e-5 --lr-decay-style constant \
  --lr-warmup-iters 2 --weight-decay .01 --adam-beta1 .9 --adam-beta2 .95 \
  --clip-grad 1 --bf16 --vocab-size 152064 --tokenizer-type NullTokenizer --distributed-backend hccl \
  --ckpt-format torch_dcp --save "$SAVE_PATH" --save-interval 4 \
  --log-interval 1 --eval-iters 0 --untie-embeddings-and-output-weights "$@"
