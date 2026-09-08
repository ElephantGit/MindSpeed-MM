#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,5,6,7}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

# Lance sampling configuration.
NUM_NPUS="${NUM_NPUS:-6}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29503}"
LANCE_SOURCE_ROOT="${LANCE_SOURCE_ROOT:-/mnt/qs/Lance}"
MODEL_PATH="${MODEL_PATH:-/mnt/qs/models/bytedance-research/Lance/Lance_3B}"
DATASET_PATH="${DATASET_PATH:-${LANCE_SOURCE_ROOT}/benchmarks/image_gen/DPG/DPG.jsonl}"

# Official ELLA/DPG-Bench mPLUG scorer configuration.
DPGBENCH_SCORER_ROOT="${DPGBENCH_SCORER_ROOT:-/mnt/qs/evaluation/ELLA}"
DPGBENCH_SCORER_REVISION="3c228f1dc6c4d3cad0a47493816151a419f14db3"
SCORER_PROCESSES="${SCORER_PROCESSES:-${NUM_NPUS}}"
SCORER_PORT="${SCORER_PORT:-29504}"
DPGBENCH_SCORER_ENV="${DPGBENCH_SCORER_ENV:-}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/mnt/qs/models/modelscope}"

SAMPLE_DIR="${SAMPLE_DIR:-${REPO_ROOT}/results/lance-dpgbench}"
SCORE_DIR="${SCORE_DIR:-${REPO_ROOT}/results/lance-dpgbench-score}"
RUN_MANIFEST="${SAMPLE_DIR}/lance_eval_run.json"
OFFICIAL_RESULTS="${SCORE_DIR}/dpgbench_results.txt"
METRICS_PATH="${SCORE_DIR}/metrics.json"
ALIGNMENT_PATH="${SCORE_DIR}/alignment.json"

# Disable completed phases by setting the corresponding variable to 0.
RUN_SAMPLE="${RUN_SAMPLE:-1}"
RUN_OFFICIAL_SCORE="${RUN_OFFICIAL_SCORE:-1}"
RUN_NORMALIZE="${RUN_NORMALIZE:-1}"
RUN_REPORT="${RUN_REPORT:-1}"

mkdir -p "${SAMPLE_DIR}" "${SCORE_DIR}" "${MODELSCOPE_CACHE}"

echo "[dpgbench] Validating dataset and Lance runtime assets"
python evaluate_lance.py validate \
    --benchmark dpgbench \
    --dataset-path "${DATASET_PATH}" \
    --lance-source-root "${LANCE_SOURCE_ROOT}" \
    --model-path "${MODEL_PATH}"

if [[ "${RUN_SAMPLE}" == "1" ]]; then
    echo "[dpgbench] Generating 1,065 2x2 image grids with ${NUM_NPUS} NPU processes"
    torchrun \
        --nproc_per_node "${NUM_NPUS}" \
        --master_addr "${MASTER_ADDR}" \
        --master_port "${MASTER_PORT}" \
        evaluate_lance.py sample \
        --benchmark dpgbench \
        --lance-source-root "${LANCE_SOURCE_ROOT}" \
        --model-path "${MODEL_PATH}" \
        --dataset-path "${DATASET_PATH}" \
        --output-path "${SAMPLE_DIR}" \
        --world-size "${NUM_NPUS}"
fi

[[ -f "${RUN_MANIFEST}" ]] || {
    echo "ERROR: run manifest not found: ${RUN_MANIFEST}" >&2
    exit 1
}

echo "[dpgbench] Auditing generated image grids"
python evaluate_lance.py audit \
    --benchmark dpgbench \
    --output-path "${SAMPLE_DIR}"

if [[ "${RUN_OFFICIAL_SCORE}" == "1" ]]; then
    [[ -d "${DPGBENCH_SCORER_ROOT}/.git" ]] || {
        echo "ERROR: official ELLA checkout not found: ${DPGBENCH_SCORER_ROOT}" >&2
        exit 1
    }
    [[ -f "${DPGBENCH_SCORER_ROOT}/dpg_bench/compute_dpg_bench.py" ]] || {
        echo "ERROR: official DPG-Bench evaluator is missing under ${DPGBENCH_SCORER_ROOT}" >&2
        exit 1
    }
    [[ -f "${DPGBENCH_SCORER_ROOT}/dpg_bench/dpg_bench.csv" ]] || {
        echo "ERROR: official DPG-Bench CSV is missing under ${DPGBENCH_SCORER_ROOT}" >&2
        exit 1
    }

    ACTUAL_SCORER_REVISION="$(git -C "${DPGBENCH_SCORER_ROOT}" rev-parse HEAD)"
    if [[ "${ACTUAL_SCORER_REVISION}" != "${DPGBENCH_SCORER_REVISION}" ]]; then
        echo "ERROR: DPG-Bench scorer revision must be ${DPGBENCH_SCORER_REVISION}" >&2
        echo "       current revision is ${ACTUAL_SCORER_REVISION}" >&2
        exit 1
    fi
    if [[ -n "$(git -C "${DPGBENCH_SCORER_ROOT}" status --porcelain)" ]]; then
        echo "ERROR: DPG-Bench scorer checkout must be clean: ${DPGBENCH_SCORER_ROOT}" >&2
        exit 1
    fi

    if [[ -n "${DPGBENCH_SCORER_ENV}" ]]; then
        command -v conda >/dev/null 2>&1 || {
            echo "ERROR: conda is required for DPGBENCH_SCORER_ENV=${DPGBENCH_SCORER_ENV}" >&2
            exit 1
        }
        SCORER_LAUNCH=(conda run --no-capture-output -n "${DPGBENCH_SCORER_ENV}" accelerate)
    else
        command -v accelerate >/dev/null 2>&1 || {
            echo "ERROR: accelerate is required by the official DPG-Bench evaluator." >&2
            exit 1
        }
        SCORER_LAUNCH=(accelerate)
    fi

    echo "[dpgbench] Running the official mPLUG evaluator"
    (
        cd "${DPGBENCH_SCORER_ROOT}"
        "${SCORER_LAUNCH[@]}" launch \
            --num_machines 1 \
            --num_processes "${SCORER_PROCESSES}" \
            --multi_gpu \
            --mixed_precision fp16 \
            --main_process_port "${SCORER_PORT}" \
            dpg_bench/compute_dpg_bench.py \
            --image-root-path "${SAMPLE_DIR}" \
            --resolution 768 \
            --pic-num 4 \
            --vqa-model mplug \
            --csv "${DPGBENCH_SCORER_ROOT}/dpg_bench/dpg_bench.csv" \
            --res-path "${OFFICIAL_RESULTS}"
    )
fi

if [[ "${RUN_NORMALIZE}" == "1" ]]; then
    [[ -f "${OFFICIAL_RESULTS}" ]] || {
        echo "ERROR: official scorer result not found: ${OFFICIAL_RESULTS}" >&2
        exit 1
    }

    echo "[dpgbench] Normalizing the official scorer result"
    python evaluate_lance.py dpgbench-score \
        --results "${OFFICIAL_RESULTS}" \
        --scorer-root "${DPGBENCH_SCORER_ROOT}" \
        --run-manifest "${RUN_MANIFEST}" \
        --output "${METRICS_PATH}"
fi

if [[ "${RUN_REPORT}" == "1" ]]; then
    [[ -f "${METRICS_PATH}" ]] || {
        echo "ERROR: normalized metrics not found: ${METRICS_PATH}" >&2
        exit 1
    }

    echo "[dpgbench] Comparing with the Lance paper score"
    python evaluate_lance.py report \
        --benchmark dpgbench \
        --metrics "${METRICS_PATH}" \
        --output "${ALIGNMENT_PATH}"
fi

echo "[dpgbench] Done"
echo "  official:  ${OFFICIAL_RESULTS}"
echo "  metrics:   ${METRICS_PATH}"
echo "  alignment: ${ALIGNMENT_PATH}"
