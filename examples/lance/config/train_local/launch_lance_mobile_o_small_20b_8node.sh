#!/usr/bin/env bash
# Eight-node/64-NPU continuous Lance 6B training orchestration.
# Usage: bash launch_lance_mobile_o_small_20b_8node.sh preflight|prepare|launch|status
set -euo pipefail

MODE="${1:-preflight}"
HOSTS=(110.129.0.5 110.129.0.7 110.129.0.12 110.129.0.14 110.129.0.16 110.129.0.18 110.129.0.20 110.129.0.22)
CONTAINER="mindspeed-mm"
IMAGE="mindspeed-mm:latest"
IMAGE_TAR="/mnt/models/CODE/env/images/mindspeed-mm-latest.tar"
REPO="/mnt/models/CODE/MindSpeed-MM"
ENTRY="${REPO}/examples/lance/config/train_local/pretrain_lance_mobile_o_small_20b_8node.sh"
WATCHER="${REPO}/examples/lance/config/train_local/watch_lance_2b_eval.py"
OUTPUT="${LANCE_OUTPUT_DIR:-/mnt/models/outputs/lance-mobile-o-small-pt-6b-8node-planA}"
RUN_ID="${LANCE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=10)

preflight_host() {
  local host="$1"
  ssh "${SSH_OPTIONS[@]}" "${host}" bash -s -- "${IMAGE_TAR}" <<'REMOTE'
set -euo pipefail
image_tar="$1"
test -r "${image_tar}"
test -d /mnt/models/CODE/MindSpeed-MM
test -d /mnt/models/DATA_INIT/MULTI/T2I/Mobile-O-Pre-Train-packed-16k
busy="$(npu-smi info | grep -E '^\| [0-7][[:space:]]+0[[:space:]]+\|[[:space:]]+[0-9]+' || true)"
if [[ -n "${busy}" ]]; then
  echo "warning: sharing NPUs with existing processes:" >&2
  echo "${busy}" >&2
fi
echo "preflight ok"
REMOTE
}

container_matches_baseline() {
  local host="$1"
  ssh "${SSH_OPTIONS[@]}" "${host}" \
    "docker inspect '${CONTAINER}' --format '{{.Config.Image}} {{.HostConfig.ShmSize}} {{.HostConfig.IpcMode}} {{.HostConfig.NetworkMode}} {{.HostConfig.Privileged}}' 2>/dev/null" \
    | grep -qx 'mindspeed-mm:latest 53687091200 host host true'
}

prepare_host() {
  local host="$1"
  echo "[image sync] ${host}"
  ssh "${SSH_OPTIONS[@]}" "${host}" "docker load -i '${IMAGE_TAR}'"
  if container_matches_baseline "${host}"; then
    echo "[container reuse] ${host}: already matches .16 baseline"
    return
  fi
  echo "[container rebuild] ${host}"
  ssh "${SSH_OPTIONS[@]}" "${host}" bash -s -- "${CONTAINER}" "${IMAGE}" <<'REMOTE'
set -euo pipefail
container="$1"
image="$2"
if docker container inspect "${container}" >/dev/null 2>&1; then
  docker stop "${container}"
  docker rm "${container}"
fi
docker run -d --network=host --ipc=host --shm-size=50g \
  --name "${container}" --privileged=true \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 \
  --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /tmp:/tmp -v /mnt:/mnt -v /home:/home \
  -it "${image}" /bin/bash
REMOTE
  container_matches_baseline "${host}"
}

launch_rank() {
  local host="$1" rank="$2"
  local inner quoted_inner
  inner="cd '${REPO}' && if bash '${ENTRY}' > '${OUTPUT}/node${rank}.log' 2>&1; then touch '${OUTPUT}/run-${RUN_ID}-rank-${rank}.success'; else touch '${OUTPUT}/run-${RUN_ID}-rank-${rank}.failed' '${OUTPUT}/run-${RUN_ID}.failed'; exit 1; fi"
  printf -v quoted_inner '%q' "${inner}"
  ssh "${SSH_OPTIONS[@]}" "${host}" \
    "docker exec -d -e NODE_RANK=${rank} -e NNODES=8 -e NPROC_PER_NODE=8 -e MASTER_ADDR=110.129.0.5 -e MASTER_PORT=6002 -e RANK_TABLE_FILE=/mnt/models/rank_table_generated.json -e GLOO_SOCKET_IFNAME=enp189s0f0 -e HCCL_CONNECT_TIMEOUT=7200 -e HCCL_EXEC_TIMEOUT=0 -e LANCE_TRAIN_TOKENS=6000000000 -e LANCE_TRAIN_ITERS=6906 -e LANCE_OUTPUT_DIR='${OUTPUT}' -e LANCE_RUN_ID='${RUN_ID}' '${CONTAINER}' bash -lc ${quoted_inner}"
}

case "${MODE}" in
  preflight)
    for host in "${HOSTS[@]}"; do
      echo "[preflight] ${host}"
      preflight_host "${host}"
    done
    ;;
  prepare)
    for host in "${HOSTS[@]}"; do
      preflight_host "${host}"
    done
    for host in "${HOSTS[@]}"; do
      prepare_host "${host}"
    done
    ;;
  launch)
    mkdir -p "${OUTPUT}"
    printf '%s\n' "${RUN_ID}" > "${OUTPUT}/active_run_id.txt"
    # Independent evaluator runs on physical NPU 6 of the local .16 node.
    # Training remains authoritative; evaluator failures do not signal or stop it.
    watcher_inner="cd '${REPO}' && python '${WATCHER}' --checkpoint-root '${OUTPUT}' --packed-data '/mnt/models/DATA_INIT/MULTI/T2I/Mobile-O-Pre-Train-packed-16k' --target 6000000000 --device-id 6 --training-failed '${OUTPUT}/run-${RUN_ID}.failed' > '${OUTPUT}/eval_watcher.log' 2>&1"
    printf -v quoted_watcher_inner '%q' "${watcher_inner}"
    ssh "${SSH_OPTIONS[@]}" "${HOSTS[4]}" \
      "docker exec -d -e ASCEND_RT_VISIBLE_DEVICES=6 '${CONTAINER}' bash -lc ${quoted_watcher_inner}"
    for rank in "${!HOSTS[@]}"; do
      echo "[launch rank ${rank}] ${HOSTS[${rank}]}"
      launch_rank "${HOSTS[${rank}]}" "${rank}"
    done
    echo "continuous 64-card run launched: ${RUN_ID}; output=${OUTPUT}"
    ;;
  status)
    for rank in "${!HOSTS[@]}"; do
      echo "[rank ${rank}] ${HOSTS[${rank}]}"
      ssh "${SSH_OPTIONS[@]}" "${HOSTS[${rank}]}" \
        "docker exec '${CONTAINER}' bash -lc \"pgrep -af 'torchrun|trainer.py|watch_lance_2b_eval' || true\"; tail -3 '${OUTPUT}/node${rank}.log' 2>/dev/null || true"
    done
    ;;
  *)
    echo "usage: $0 preflight|prepare|launch|status" >&2
    exit 2
    ;;
esac
