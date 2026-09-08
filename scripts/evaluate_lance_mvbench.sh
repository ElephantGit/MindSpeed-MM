#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,5,6,7}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

# Runtime configuration. Override these variables before running if necessary.
NUM_NPUS="${NUM_NPUS:-6}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29502}"
LANCE_SOURCE_ROOT="${LANCE_SOURCE_ROOT:-/mnt/qs/Lance}"
MODEL_PATH="${MODEL_PATH:-/mnt/qs/models/bytedance-research/Lance/Lance_3B_Video}"

# Set these two paths to the local MVBench annotation and media directories.
MVBENCH_ANNOTATION_ROOT="${MVBENCH_ANNOTATION_ROOT:-/mnt/qs/evaluation/MVBench/json}"
MVBENCH_MEDIA_ROOT="${MVBENCH_MEDIA_ROOT:-/mnt/qs/evaluation/MVBench/video}"

# The paper task set contains 19 tasks and 3,800 samples, matching Lance's paper.
TASK_SET="${TASK_SET:-paper}"
PREPARED_DIR="${PREPARED_DIR:-${REPO_ROOT}/results/lance-mvbench-prepared}"
SAMPLE_DIR="${SAMPLE_DIR:-${REPO_ROOT}/results/lance-mvbench}"
SCORE_DIR="${SCORE_DIR:-${REPO_ROOT}/results/lance-mvbench-score}"

# Individual phases can be disabled when resuming an interrupted evaluation.
RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_SAMPLE="${RUN_SAMPLE:-1}"
RUN_SCORE="${RUN_SCORE:-1}"
RUN_REPORT="${RUN_REPORT:-1}"

DATASET_PATH="${PREPARED_DIR}/mvbench_lance.json"
METADATA_PATH="${PREPARED_DIR}/mvbench_metadata.json"
RESULT_PATH="${SAMPLE_DIR}/result.json"
RUN_MANIFEST="${SAMPLE_DIR}/lance_eval_run.json"
METRICS_PATH="${SCORE_DIR}/metrics.json"
ALIGNMENT_PATH="${SCORE_DIR}/alignment.json"

mkdir -p "${PREPARED_DIR}" "${SAMPLE_DIR}" "${SCORE_DIR}"

if [[ "${RUN_PREPARE}" == "1" ]]; then
    command -v ffmpeg >/dev/null 2>&1 || {
        echo "ERROR: ffmpeg is required by mvbench-prepare." >&2
        exit 1
    }
    [[ -d "${MVBENCH_ANNOTATION_ROOT}" ]] || {
        echo "ERROR: MVBench annotation directory not found: ${MVBENCH_ANNOTATION_ROOT}" >&2
        exit 1
    }
    [[ -d "${MVBENCH_MEDIA_ROOT}" ]] || {
        echo "ERROR: MVBench media directory not found: ${MVBENCH_MEDIA_ROOT}" >&2
        exit 1
    }

    echo "[mvbench] Preparing ${TASK_SET} task set"
    python evaluate_lance.py mvbench-prepare \
        --annotation-root "${MVBENCH_ANNOTATION_ROOT}" \
        --media-root "${MVBENCH_MEDIA_ROOT}" \
        --output-dir "${PREPARED_DIR}" \
        --task-set "${TASK_SET}"
fi

[[ -f "${DATASET_PATH}" ]] || {
    echo "ERROR: prepared dataset not found: ${DATASET_PATH}" >&2
    exit 1
}
[[ -f "${METADATA_PATH}" ]] || {
    echo "ERROR: prepared metadata not found: ${METADATA_PATH}" >&2
    exit 1
}

echo "[mvbench] Validating dataset and Lance runtime assets"
python evaluate_lance.py validate \
    --benchmark mvbench \
    --dataset-path "${DATASET_PATH}" \
    --lance-source-root "${LANCE_SOURCE_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --mvbench-task-set "${TASK_SET}"

if [[ "${RUN_SAMPLE}" == "1" ]]; then
    echo "[mvbench] Sampling with ${NUM_NPUS} NPU processes"
    torchrun \
        --nproc_per_node "${NUM_NPUS}" \
        --master_addr "${MASTER_ADDR}" \
        --master_port "${MASTER_PORT}" \
        evaluate_lance.py sample \
        --benchmark mvbench \
        --lance-source-root "${LANCE_SOURCE_ROOT}" \
        --model-path "${MODEL_PATH}" \
        --dataset-path "${DATASET_PATH}" \
        --output-path "${SAMPLE_DIR}" \
        --world-size "${NUM_NPUS}" \
        --mvbench-task-set "${TASK_SET}"
fi

[[ -f "${RESULT_PATH}" ]] || {
    echo "ERROR: MVBench result file not found: ${RESULT_PATH}" >&2
    exit 1
}
[[ -f "${RUN_MANIFEST}" ]] || {
    echo "ERROR: run manifest not found: ${RUN_MANIFEST}" >&2
    exit 1
}

echo "[mvbench] Auditing generated answers"
python evaluate_lance.py audit \
    --benchmark mvbench \
    --output-path "${SAMPLE_DIR}" \
    --mvbench-task-set "${TASK_SET}"

if [[ "${RUN_SCORE}" == "1" ]]; then
    echo "[mvbench] Scoring answers"
    python evaluate_lance.py mvbench-score \
        --metadata "${METADATA_PATH}" \
        --results "${RESULT_PATH}" \
        --run-manifest "${RUN_MANIFEST}" \
        --task-set "${TASK_SET}" \
        --output "${METRICS_PATH}"
fi

if [[ "${RUN_REPORT}" == "1" ]]; then
    [[ -f "${METRICS_PATH}" ]] || {
        echo "ERROR: metrics file not found: ${METRICS_PATH}" >&2
        exit 1
    }

    echo "[mvbench] Comparing with the Lance paper score"
    python evaluate_lance.py report \
        --benchmark mvbench \
        --metrics "${METRICS_PATH}" \
        --output "${ALIGNMENT_PATH}"
fi

echo "[mvbench] Done"
echo "  metrics:   ${METRICS_PATH}"
echo "  alignment: ${ALIGNMENT_PATH}"
