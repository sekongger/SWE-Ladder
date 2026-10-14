#!/usr/bin/env bash
# Auto-retry launcher for the end-to-end OPSD run.
#
# Why: the colocated vLLM/HCCL/Ray machinery on Ascend intermittently fails during the
# startup of a step (Ray GCS init, vLLM first wake, or the actor->vLLM weight-sync
# HcclAllGather). These are transient: a clean retry usually gets past the hump, and once
# a run is past step 1 it tends to keep going. This wrapper cleans stale state and retries.
#
# Resume: after the first checkpoint (save_freq), set RESUME_DIR=<the run's default_local_dir>
# and RESUME_STEP=<N> to continue from global_step_N instead of restarting from base.
#
# Usage (CONFIG is a RENDERED config -- see scripts/tools/render_config.py):
#   CONFIG=/tmp/run.yaml bash DeepSWE_RL/rllm/examples/swe/run_e2e_with_retry.sh
#   CONFIG=/tmp/run.yaml MAX_RETRIES=10 bash .../run_e2e_with_retry.sh
#   CONFIG=/tmp/run.yaml RESUME_DIR=/abs/.../<run_dir> RESUME_STEP=10 bash .../run_e2e_with_retry.sh

set -u
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${WORKSPACE_ROOT}"

CONFIG="${CONFIG:-/tmp/run.yaml}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "[retry] ERROR: CONFIG not found: ${CONFIG}" >&2
  echo "[retry] render one first: scripts/tools/render_config.py <config> -o /tmp/run.yaml" >&2
  exit 1
fi
LAUNCHER="DeepSWE_RL/rllm/examples/swe/run_npu_train_from_yaml.sh"
MAX_RETRIES="${MAX_RETRIES:-8}"
RAY_BIN="${RAY_BIN:-$(command -v ray || true)}"
if [[ -z "${RAY_BIN}" ]]; then
  echo "[retry] ERROR: no ray on PATH; activate the training env or set RAY_BIN" >&2
  exit 1
fi

cleanup() {
  echo "[retry] cleaning stale Ray/vLLM state ..."
  "$RAY_BIN" stop --force >/dev/null 2>&1 || true
  # kill only LIVE (non-zombie) ray/vllm workers by PID; never pkill -f (would match self)
  for pid in $(ps -eo pid,stat,comm 2>/dev/null | awk '$2 !~ /Z/ && ($3 ~ /raylet|gcs_server|ray|vllm|VLLM/){print $1}'); do
    kill -9 "$pid" >/dev/null 2>&1 || true
  done
  rm -rf /tmp/ray/session_* 2>/dev/null || true
  sleep 5
}

# Optional resume: point the launcher at an existing checkpoint to continue progress.
export_resume() {
  if [[ -n "${RESUME_DIR:-}" && -n "${RESUME_STEP:-}" ]]; then
    export TRAINER_RESUME_MODE="resume_path"
    export TRAINER_RESUME_FROM_PATH="${RESUME_DIR}/global_step_${RESUME_STEP}"
    export TRAINER_STAGE_GLOBAL_STEP_OFFSET="0"
    echo "[retry] RESUME from ${TRAINER_RESUME_FROM_PATH}"
  fi
}

attempt=1
while (( attempt <= MAX_RETRIES )); do
  echo "=================================================================="
  echo "[retry] attempt ${attempt}/${MAX_RETRIES}  $(date '+%F %T')"
  echo "=================================================================="
  cleanup
  export_resume
  LOG="/tmp/e2e_retry_attempt${attempt}.log"
  bash "$LAUNCHER" "$CONFIG" 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  # Success signal: reached at least one finished training step.
  if grep -q "\[train\] finished global_step" "$LOG" 2>/dev/null; then
    echo "[retry] a training step completed (rc=$rc). Not auto-restarting; check the run."
    echo "[retry] if it died mid-run, resume with RESUME_DIR/RESUME_STEP from its last checkpoint."
    exit 0
  fi
  echo "[retry] attempt ${attempt} failed before completing a step (rc=$rc). Retrying after cleanup."
  attempt=$(( attempt + 1 ))
done
echo "[retry] exhausted ${MAX_RETRIES} attempts without completing a step. Environment may be unhealthy; check NPU/driver."
exit 1
