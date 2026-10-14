#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

CONFIG_PATH="${1:-${SCRIPT_DIR}/configs/smoke_stage_routing.yaml}"
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${WORKSPACE_ROOT}/${CONFIG_PATH}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: config yaml not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# The released configs keep machine-specific paths as {{PLACEHOLDER}} slots. Launching
# one unrendered would export the literal braces and fail much later, inside the trainer.
if grep -q '{{[A-Z0-9_]\+}}' "${CONFIG_PATH}"; then
  echo "ERROR: ${CONFIG_PATH} still contains unrendered placeholders:" >&2
  grep -o '{{[A-Z0-9_]\+}}' "${CONFIG_PATH}" | sort -u | sed 's/^/  /' >&2
  echo "Render it first: scripts/tools/render_config.py <config> --site <site.env> -o /tmp/run.yaml" >&2
  exit 1
fi

# Activate the training env if conda is installed. CONDA_SH/CONDA_ENV override the
# discovery; with no conda at all we assume the interpreter on PATH is already correct.
# runtime.conda_env has to be read with grep rather than the YAML parser below, because
# that parser needs the very interpreter this block is about to put on PATH.
if [[ -z "${CONDA_ENV:-}" ]]; then
  CONDA_ENV="$(sed -n "s/^[[:space:]]*conda_env:[[:space:]]*['\"]\?\([^'\"]*\)['\"]\?[[:space:]]*$/\1/p" "${CONFIG_PATH}" | head -n 1)"
fi
CONDA_ENV="${CONDA_ENV:-swe-ladder}"
if [[ -z "${CONDA_SH:-}" ]]; then
  for candidate in \
    "${CONDA_EXE:+$(dirname -- "$(dirname -- "${CONDA_EXE}")")/etc/profile.d/conda.sh}" \
    "${CONDA_PREFIX:-}/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
      CONDA_SH="${candidate}"
      break
    fi
  done
fi
if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# After `conda activate` this is the env's interpreter; outside conda, `python` may not
# exist at all, so fall back to python3.
YAML_PYTHON="${YAML_PYTHON:-$(command -v python || command -v python3 || true)}"
if [[ -z "${YAML_PYTHON}" ]]; then
  echo "ERROR: no python interpreter found to parse the config" >&2
  exit 1
fi

eval "$(
  "${YAML_PYTHON}" - "${CONFIG_PATH}" <<'PY'
import shlex
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
runtime = data.get("runtime") or {}
env = data.get("env") or {}

if not isinstance(runtime, dict):
    raise SystemExit("runtime must be a mapping")
if not isinstance(env, dict):
    raise SystemExit("env must be a mapping")

for key, value in env.items():
    if value is None:
        continue
    if not key.replace("_", "").isalnum() or key[0].isdigit():
        raise SystemExit(f"invalid env key: {key!r}")
    print(f"export {key}={shlex.quote(str(value))}")

for key in ("entrypoint", "log_dir", "log_prefix"):
    value = runtime.get(key)
    if value is not None:
        print(f"RUN_{key.upper()}={shlex.quote(str(value))}")

clean_ray = runtime.get("clean_ray_before_start", True)
print(f"RUN_CLEAN_RAY={shlex.quote('1' if clean_ray else '0')}")
PY
)"

ENTRYPOINT="${RUN_ENTRYPOINT:-DeepSWE_RL/rllm/examples/swe/train_entrypoint_npu.sh}"
LOG_DIR="${RUN_LOG_DIR:-output/swe_ladder_npu}"
LOG_PREFIX="${RUN_LOG_PREFIX:-swe_ladder_npu_train}"

if [[ "${ENTRYPOINT}" != /* ]]; then
  ENTRYPOINT="${WORKSPACE_ROOT}/${ENTRYPOINT}"
fi
if [[ "${LOG_DIR}" != /* ]]; then
  LOG_DIR="${WORKSPACE_ROOT}/${LOG_DIR}"
fi
if [[ ! -x "${ENTRYPOINT}" ]]; then
  echo "ERROR: entrypoint is not executable: ${ENTRYPOINT}" >&2
  exit 1
fi
SUPERVISOR="${SCRIPT_DIR}/process_tree_supervisor.py"
if [[ ! -f "${SUPERVISOR}" ]]; then
  echo "ERROR: process supervisor not found: ${SUPERVISOR}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${LOG_PREFIX}_$(date +%Y%m%d_%H%M%S).log"

echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "ENTRYPOINT: ${ENTRYPOINT}"
echo "LOG_PATH: ${LOG_PATH}"
echo "ASCEND_RT_VISIBLE_DEVICES: ${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
# Scaffold identity must be readable from the log, not inferred from the config filename.
# These two always travel together: the tool triple installed in the container and the
# prompt templates that name those tools. A missing OPD_PROMPT_TEMPLATE_DIR falls back to
# the built-in r2egym templates SILENTLY, so echo it explicitly.
echo "SWE_SCAFFOLD: ${SWE_SCAFFOLD:-<unset -> entrypoint default r2egym>}"
echo "OPD_PROMPT_TEMPLATE_DIR: ${OPD_PROMPT_TEMPLATE_DIR:-<unset -> built-in r2egym templates>}"
echo "MODEL_PATH: ${MODEL_PATH:-<unset>}"
echo "TRAIN_DATASET: ${TRAIN_DATASET:-<unset>}"
echo "VAL_DATASET: ${VAL_DATASET:-<unset>}"
echo "TRAIN_LOGGERS: ${TRAIN_LOGGERS:-<unset>}"
echo "WANDB_ENABLE: ${WANDB_ENABLE:-<unset>}"
echo "WANDB_MODE: ${WANDB_MODE:-<unset>}"
echo "OPSD_ENABLE: ${OPSD_ENABLE:-<unset>}"
echo "OPSD_TEACHER_SOURCE: ${OPSD_TEACHER_SOURCE:-<unset>}"
echo "OPSD_PRIVILEGED_STAGES: ${OPSD_PRIVILEGED_STAGES:-<all>}"
echo "OPSD_REF_LOAD_CHECKPOINT_FROM_RESUME: ${OPSD_REF_LOAD_CHECKPOINT_FROM_RESUME:-<unset>}"
echo "OPSD_MAX_TOKENS_PER_SAMPLE: ${OPSD_MAX_TOKENS_PER_SAMPLE:-<unset>}"
echo "USE_RLSD: ${USE_RLSD:-<unset>}"
echo "RLSD_USE_ROLLOUT_LOGP_AS_STUDENT_PRIOR: ${RLSD_USE_ROLLOUT_LOGP_AS_STUDENT_PRIOR:-<unset>}"
echo "DISABLE_EXECUTION_REWARD: ${DISABLE_EXECUTION_REWARD:-<unset>}"
echo "ISSUE_REPRODUCTION_EVAL_ENABLE: ${ISSUE_REPRODUCTION_EVAL_ENABLE:-<unset>}"
echo "ISSUE_REPRODUCTION_REWARD_ENABLE: ${ISSUE_REPRODUCTION_REWARD_ENABLE:-<unset>}"
echo "MAX_RESPONSE_LENGTH: ${MAX_RESPONSE_LENGTH:-<unset>}"
echo "AGENT_MAX_RESPONSE_TOKENS_FOR_TRAINING: ${AGENT_MAX_RESPONSE_TOKENS_FOR_TRAINING:-<unset>}"
echo "ROLLOUT_MAX_MODEL_LEN: ${ROLLOUT_MAX_MODEL_LEN:-<unset>}"
echo "ROLLOUT_MAX_NUM_BATCHED_TOKENS: ${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-<unset>}"
echo "ROLLOUT_MAX_NUM_SEQS: ${ROLLOUT_MAX_NUM_SEQS:-<unset>}"
echo "ROLLOUT_ACTIVE_TRAJECTORIES: ${ROLLOUT_ACTIVE_TRAJECTORIES:-<unset>}"
echo "ROLLOUT_PER_STEP_MAX_TOKENS: ${ROLLOUT_PER_STEP_MAX_TOKENS:-<unset>}"
echo "ROLLOUT_FIRST_STEP_MAX_TOKENS: ${ROLLOUT_FIRST_STEP_MAX_TOKENS:-<unset>}"
echo "ROLLOUT_INVALID_TOOL_CALL_MAX_TOKENS: ${ROLLOUT_INVALID_TOOL_CALL_MAX_TOKENS:-<unset>}"
echo "ROLLOUT_INVALID_TOOL_CALL_RETRY_LIMIT: ${ROLLOUT_INVALID_TOOL_CALL_RETRY_LIMIT:-<unset>}"
echo "ROLLOUT_STOP_AFTER_FUNCTION_CALL: ${ROLLOUT_STOP_AFTER_FUNCTION_CALL:-<unset>}"
echo "ROLLOUT_GPU_MEMORY_UTILIZATION: ${ROLLOUT_GPU_MEMORY_UTILIZATION:-<unset>}"
echo "ROLLOUT_ENFORCE_EAGER: ${ROLLOUT_ENFORCE_EAGER:-<unset>}"
echo "ROLLOUT_ENABLE_CHUNKED_PREFILL: ${ROLLOUT_ENABLE_CHUNKED_PREFILL:-<unset>}"
echo "ROLLOUT_ENABLE_PREFIX_CACHING: ${ROLLOUT_ENABLE_PREFIX_CACHING:-<unset>}"
echo "ROLLOUT_BLOCK_SIZE: ${ROLLOUT_BLOCK_SIZE:-<unset>}"
echo "ROLLOUT_ASCEND_SCHEDULER_ENABLE: ${ROLLOUT_ASCEND_SCHEDULER_ENABLE:-<unset>}"
echo "ROLLOUT_ASCEND_SCHEDULER_ENABLE_CHUNKED_PREFILL: ${ROLLOUT_ASCEND_SCHEDULER_ENABLE_CHUNKED_PREFILL:-<unset>}"
echo "ROLLOUT_ASCEND_SCHEDULER_MAX_LONG_PARTIAL_PREFILLS: ${ROLLOUT_ASCEND_SCHEDULER_MAX_LONG_PARTIAL_PREFILLS:-<unset>}"
echo "ROLLOUT_ASCEND_SCHEDULER_LONG_PREFILL_TOKEN_THRESHOLD: ${ROLLOUT_ASCEND_SCHEDULER_LONG_PREFILL_TOKEN_THRESHOLD:-<unset>}"
echo "ROLLOUT_CUDAGRAPH_CAPTURE_SIZES: ${ROLLOUT_CUDAGRAPH_CAPTURE_SIZES:-<unset>}"
echo "RLLM_FORCE_VLLM_WEIGHT_SYNC_SHM: ${RLLM_FORCE_VLLM_WEIGHT_SYNC_SHM:-<unset>}"
echo "RLLM_DISABLE_FSDP_SYNC_MODULE_STATES: ${RLLM_DISABLE_FSDP_SYNC_MODULE_STATES:-<unset>}"
echo "HCCL_OP_EXPANSION_MODE: ${HCCL_OP_EXPANSION_MODE:-<unset>}"
echo "TASK_QUEUE_ENABLE: ${TASK_QUEUE_ENABLE:-<unset>}"
echo "PYTORCH_NPU_ALLOC_CONF: ${PYTORCH_NPU_ALLOC_CONF:-<unset>}"
echo "CHECKPOINT_ENGINE_BUCKET_MB: ${CHECKPOINT_ENGINE_BUCKET_MB:-<unset>}"
echo "PIPELINE_TRAIN_AFTER_FULL_ROLLOUT: ${PIPELINE_TRAIN_AFTER_FULL_ROLLOUT:-<unset>}"
echo "PIPELINE_LENGTH_BUCKET_ROLLOUTS: ${PIPELINE_LENGTH_BUCKET_ROLLOUTS:-<unset>}"
echo "R2EGYM_TIMING: ${R2EGYM_TIMING:-<unset>}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1, skip training launch."
  exit 0
fi

SUPERVISOR_PID=""

pid_is_live_non_zombie() {
  local pid="${1:-}"
  [[ -z "${pid}" ]] && return 1
  ps -o stat= -p "${pid}" 2>/dev/null | awk 'NF && $1 !~ /^Z/ { found=1; exit } END { exit(found ? 0 : 1) }'
}

stop_supervisor() {
  local reason="${1:-EXIT}"
  local supervisor_grace="${RUN_SUPERVISOR_SHUTDOWN_GRACE_SECONDS:-60}"
  [[ -n "${SUPERVISOR_PID}" ]] || return 0

  echo "[cleanup] ${reason}: stopping process-tree supervisor pid=${SUPERVISOR_PID} signal=TERM"
  kill -TERM "${SUPERVISOR_PID}" >/dev/null 2>&1 || true

  local waited=0
  while (( waited < supervisor_grace )); do
    if ! pid_is_live_non_zombie "${SUPERVISOR_PID}"; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done

  if pid_is_live_non_zombie "${SUPERVISOR_PID}"; then
    echo "[cleanup] ${reason}: supervisor pid=${SUPERVISOR_PID} still alive after ${supervisor_grace}s; sending KILL"
    kill -KILL "${SUPERVISOR_PID}" >/dev/null 2>&1 || true
  fi
}

on_signal() {
  local sig="${1:-TERM}"
  trap - EXIT INT TERM
  stop_supervisor "${sig}"
  case "${sig}" in
    INT) exit 130 ;;
    TERM) exit 143 ;;
    *) exit 1 ;;
  esac
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ "${rc}" != "0" ]]; then
    stop_supervisor "EXIT(rc=${rc})"
  fi
  exit "${rc}"
}

if [[ "${RUN_CLEAN_RAY:-1}" == "1" ]]; then
  timeout "${RUN_RAY_STOP_TIMEOUT:-30}" ray stop --force >/dev/null 2>&1 || true
  # Ray's temp root is /tmp/ray-<user>; a stale session_latest symlink makes the next
  # start attach to a dead cluster.
  rm -rf "/tmp/ray-$(id -un)/session_latest" /tmp/ray/session_latest || true
fi

ln -sf "${LOG_PATH}" "${LOG_DIR}/latest.log"

trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap on_exit EXIT

set +e
python "${SUPERVISOR}" \
  --setsid-self \
  --grace "${RUN_TERMINATE_GRACE_SECONDS:-12}" \
  --kill-grace "${RUN_TERMINATE_KILL_GRACE_SECONDS:-2}" \
  -- bash "${ENTRYPOINT}" > >(tee -a "${LOG_PATH}") 2>&1 &
SUPERVISOR_PID=$!
wait "${SUPERVISOR_PID}"
run_rc=$?
SUPERVISOR_PID=""
set -e

echo "LOG_PATH=${LOG_PATH}"
echo "RUN_RC=${run_rc}"
exit "${run_rc}"
