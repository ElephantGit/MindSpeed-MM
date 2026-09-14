#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6000}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/examples/lance/config/fsdp2_pt_preencoded.yaml}"
LANCE_MODEL_ROOT="${LANCE_MODEL_ROOT:-/mnt/qs/models/bytedance-research/Lance}"
QWEN_PATH="${QWEN_PATH:-/mnt/qs/models/Qwen/Qwen2.5-VL-3B-Instruct}"
VIT_PATH="${VIT_PATH:-${LANCE_MODEL_ROOT}/Qwen2.5-VL-ViT}"
VAE_PATH="${VAE_PATH:-${LANCE_MODEL_ROOT}/Wan2.2_VAE.pth}"
LANCE_EXAMPLE_DATA="${LANCE_EXAMPLE_DATA:-/mnt/qs/datasets/bytedance-research/Lance_example_dataset}"

# These paths contain only MindSpeed-MM artifacts.  No Lance source checkout is
# discovered, mounted, imported, or executed by this launcher.
export LANCE_INIT_DCP="${LANCE_INIT_DCP:-${REPO_ROOT}/checkpoints/lance-qwen-init-dcp}"
export LANCE_LOAD_DCP="${LANCE_LOAD_DCP:-${LANCE_INIT_DCP}}"
export LANCE_PREENCODED_DATA="${LANCE_PREENCODED_DATA:-${REPO_ROOT}/datasets/lance-preencoded}"
export LANCE_OUTPUT_DIR="${LANCE_OUTPUT_DIR:-${REPO_ROOT}/outputs/lance-native-pt}"
export LANCE_SYNTHETIC_OUTPUT="${LANCE_SYNTHETIC_OUTPUT:-${REPO_ROOT}/outputs/lance-native-synthetic}"
export LANCE_LLM_CONFIG="${LANCE_LLM_CONFIG:-${QWEN_PATH}/config.json}"
export LANCE_TRAIN_ITERS="${LANCE_TRAIN_ITERS:-350000}"
export LANCE_WARMUP_RATIO="${LANCE_WARMUP_RATIO:-0.007142857142857143}"
export LANCE_SAVE_INTERVAL="${LANCE_SAVE_INTERVAL:-2000}"
# One pre-packed item can hold 50K multimodal tokens and tens of MiB of latent
# tensors.  Multiprocess prefetch multiplies that footprint across eight ranks
# and can exhaust a container's /dev/shm.  The safe default keeps loading in the
# rank process; operators with a large shared-memory mount can opt into 1--2.
export LANCE_NUM_WORKERS="${LANCE_NUM_WORKERS:-0}"
export LANCE_LATENT_PATCH_T="${LANCE_LATENT_PATCH_T:-1}"
# Lance's PT configuration uses 2x2 spatial latent patches to keep the six
# example tasks, especially V2V, below the 40K per-sample context ceiling.
export LANCE_LATENT_PATCH_H="${LANCE_LATENT_PATCH_H:-2}"
export LANCE_LATENT_PATCH_W="${LANCE_LATENT_PATCH_W:-2}"
export LANCE_MAX_LATENT_SIZE="${LANCE_MAX_LATENT_SIZE:-64}"
export LANCE_MAX_NUM_FRAMES="${LANCE_MAX_NUM_FRAMES:-121}"
export LANCE_STOP_AFTER_ITERS="${LANCE_STOP_AFTER_ITERS:-${LANCE_TRAIN_ITERS}}"
export LANCE_TRACE_FILE="${LANCE_TRACE_FILE:-}"

if [[ "${NATIVE_SMOKE_TEST:-0}" == "1" ]]; then
    CONFIG_FILE="${REPO_ROOT}/examples/lance/config/fsdp2_synthetic_smoke.yaml"
else
    if [[ ! -f "${LANCE_LLM_CONFIG}" ]]; then
        echo "Missing Qwen model config: ${LANCE_LLM_CONFIG}" >&2
        exit 1
    fi
    if [[ -z "${LANCE_EFFECTIVE_VOCAB_SIZE:-}" ]]; then
        LANCE_EFFECTIVE_VOCAB_SIZE="$(python -c 'from transformers import AutoTokenizer; import sys; t=AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=False); known=sum(([v] if isinstance(v,str) else list(v) for v in t.special_tokens_map.values()), []); t.add_tokens([v for v in ("<|im_start|>","<|im_end|>","<|vision_start|>","<|vision_end|>") if v not in known]); print(len(t))' "${QWEN_PATH}")"
    fi
    export LANCE_EFFECTIVE_VOCAB_SIZE
    if [[ ! -d "${LANCE_LOAD_DCP}" ]]; then
        echo "Missing native Lance load DCP: ${LANCE_LOAD_DCP}" >&2
        echo "Create it with: bash scripts/prepare_lance_native_pt.sh init" >&2
        exit 1
    fi
    if [[ ! -e "${LANCE_PREENCODED_DATA}" ]]; then
        echo "Missing native Lance pre-encoded data: ${LANCE_PREENCODED_DATA}" >&2
        echo "Create it with: bash scripts/prepare_lance_native_pt.sh encode && bash scripts/prepare_lance_native_pt.sh pack" >&2
        exit 1
    fi
    python "${REPO_ROOT}/scripts/check_lance_native_artifacts.py" \
        --load "${LANCE_LOAD_DCP}" \
        --data "${LANCE_PREENCODED_DATA}" \
        --llm-config "${LANCE_LLM_CONFIG}" \
        --variant video \
        --latent-patch-size \
            "${LANCE_LATENT_PATCH_T}" "${LANCE_LATENT_PATCH_H}" "${LANCE_LATENT_PATCH_W}" \
        --max-latent-size "${LANCE_MAX_LATENT_SIZE}" \
        --max-num-frames "${LANCE_MAX_NUM_FRAMES}" \
        --effective-vocab-size "${LANCE_EFFECTIVE_VOCAB_SIZE}" \
        --world-size "${NPROC_PER_NODE}"
fi

export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE="${MULTI_STREAM_MEMORY_REUSE:-2}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"
export ACLNN_CACHE_LIMIT="${ACLNN_CACHE_LIMIT:-100000}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

echo "Native Lance config: ${CONFIG_FILE}"
echo "Load DCP: ${LANCE_LOAD_DCP}"
echo "Pre-encoded data: ${LANCE_PREENCODED_DATA}"
echo "Output: ${LANCE_OUTPUT_DIR}"
echo "Training iterations: ${LANCE_TRAIN_ITERS}"
echo "Latent geometry: ${LANCE_LATENT_PATCH_T} ${LANCE_LATENT_PATCH_H} ${LANCE_LATENT_PATCH_W}; max frames: ${LANCE_MAX_NUM_FRAMES}"
if [[ -n "${LANCE_TRACE_FILE}" ]]; then
    echo "Continuity trace: ${LANCE_TRACE_FILE}"
fi

torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${REPO_ROOT}/mindspeed_mm/fsdp/tasks/lance/trainer.py" \
    "${CONFIG_FILE}"
