#!/bin/bash
set -euo pipefail
if [[ "${VERBOSE_SHELL_TRACE:-0}" == "1" ]]; then
  set -x
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
# Default this training entry to 4 NPUs unless user overrides externally.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export ASCEND_TOOLKIT_VERSION_DIR="${ASCEND_TOOLKIT_VERSION_DIR:-8.3.RC1}"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/ascend_runtime.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/swe_dataset_local.sh"

normalize_npu_id_csv() {
  local raw="${1:-}"
  local -a raw_items=()
  local -a items=()
  local item
  IFS=',' read -r -a raw_items <<< "${raw}"
  for item in "${raw_items[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -z "${item}" ]] && continue
    items+=("${item}")
  done
  local IFS=,
  printf '%s' "${items[*]}"
}

sort_unique_npu_id_csv() {
  local raw="${1:-}"
  local -a raw_items=()
  local -a sorted_items=()
  local item
  IFS=',' read -r -a raw_items <<< "${raw}"
  mapfile -t sorted_items < <(
    for item in "${raw_items[@]}"; do
      item="${item//[[:space:]]/}"
      [[ -z "${item}" ]] && continue
      printf '%s\n' "${item}"
    done | sort -n | awk '!seen[$0]++'
  )
  local IFS=,
  printf '%s' "${sorted_items[*]}"
}

bool_is_true() {
  case "${1:-}" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
    *) return 1 ;;
  esac
}

bool_to_json() {
  if bool_is_true "${1:-}"; then
    printf 'true'
  else
    printf 'false'
  fi
}

dense_npu_id_csv_for_ray() {
  local fallback_raw="${1:-}"
  local max_id=-1
  local path item id
  for path in /dev/davinci[0-9]*; do
    [[ -e "${path}" ]] || continue
    id="${path##*/davinci}"
    [[ "${id}" =~ ^[0-9]+$ ]] || continue
    if (( id > max_id )); then
      max_id="${id}"
    fi
  done
  if (( max_id < 0 )); then
    local -a raw_items=()
    IFS=',' read -r -a raw_items <<< "${fallback_raw}"
    for item in "${raw_items[@]}"; do
      item="${item//[[:space:]]/}"
      [[ "${item}" =~ ^[0-9]+$ ]] || continue
      if (( item > max_id )); then
        max_id="${item}"
      fi
    done
  fi
  local -a ids=()
  if (( max_id >= 0 )); then
    for (( id = 0; id <= max_id; id++ )); do
      ids+=("${id}")
    done
  fi
  local IFS=,
  printf '%s' "${ids[*]}"
}

normalize_pipeline_npu_visibility() {
  if [[ -z "${ROLLOUT_NPU_IDS:-}" && -z "${TRAINING_NPU_IDS:-}" ]]; then
    return 0
  fi
  if [[ -z "${ROLLOUT_NPU_IDS:-}" || -z "${TRAINING_NPU_IDS:-}" ]]; then
    echo "ERROR: ROLLOUT_NPU_IDS and TRAINING_NPU_IDS must be set together." >&2
    exit 1
  fi

  ROLLOUT_NPU_IDS_EFFECTIVE="$(normalize_npu_id_csv "${ROLLOUT_NPU_IDS}")"
  TRAINING_NPU_IDS_EFFECTIVE="$(normalize_npu_id_csv "${TRAINING_NPU_IDS}")"
  ASCEND_RT_VISIBLE_DEVICES_EFFECTIVE="$(dense_npu_id_csv_for_ray "${ROLLOUT_NPU_IDS_EFFECTIVE},${TRAINING_NPU_IDS_EFFECTIVE}")"
  export ROLLOUT_NPU_IDS="${ROLLOUT_NPU_IDS_EFFECTIVE}"
  export TRAINING_NPU_IDS="${TRAINING_NPU_IDS_EFFECTIVE}"
  export RLLM_ROLLOUT_NPU_IDS="${ROLLOUT_NPU_IDS_EFFECTIVE}"
  export RLLM_TRAINING_NPU_IDS="${TRAINING_NPU_IDS_EFFECTIVE}"
  # Ray's NPU resource IDs are physical/dense. If ASCEND_RT_VISIBLE_DEVICES
  # skips an ID, Ray may index past the visible list before our worker code
  # runs. Keep role placement in RLLM_*_NPU_IDS, but expose a dense global
  # list for Ray's resource accounting.
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES_EFFECTIVE}"
  export NPU_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}"
}

if [[ "${TRAINER_LAYOUT_MODE:-}" == "colocated" ]]; then
  unset ROLLOUT_NPU_IDS TRAINING_NPU_IDS RLLM_ROLLOUT_NPU_IDS RLLM_TRAINING_NPU_IDS
  unset ROLLOUT_N_GPUS_PER_NODE TRAINING_N_GPUS_PER_NODE
fi
normalize_pipeline_npu_visibility
setup_ascend_runtime "${WORKSPACE_ROOT}"
check_npu_runtime
check_npu_headroom

if [[ "${NPU_ENV_CHECK_ONLY:-0}" == "1" ]]; then
  echo "NPU_ENV_CHECK_ONLY=1, skip training launch."
  exit 0
fi

TRAIN_CMD_PID=""

list_descendant_pids() {
  local root_pid="${1:-}"
  [[ -n "${root_pid}" ]] || return 0
  ps -eo pid=,ppid= | awk -v root="${root_pid}" '
    {
      pid=$1
      ppid=$2
      children[ppid] = children[ppid] " " pid
    }
    function walk(pid, child, n, arr, i) {
      n = split(children[pid], arr, " ")
      for (i = 1; i <= n; i++) {
        child = arr[i]
        if (child == "") {
          continue
        }
        print child
        walk(child)
      }
    }
    END {
      walk(root)
    }
  '
}

cleanup_orphan_ray_vllm() {
  local signal_name="${1:-TERM}"
  local pids
  pids="$(
    ps -u "${USER}" -o pid=,ppid=,cmd= | awk '
      $2 != 1 { next }
      /train_agent_ppo/ ||
      /ray::train_agent/ ||
      /vLLMHttpServer/ ||
      /EngineCore/ ||
      /Worker_TP/ ||
      /VLLM::EngineCore/ ||
      /VLLM::Worker_TP/ ||
      /VLLMWorker_TP/ { print $1 }
    '
  )"
  if [[ -n "${pids}" ]]; then
    echo "[cleanup] orphan Ray/vLLM processes detected (signal=${signal_name}): ${pids}"
    # shellcheck disable=SC2086
    kill "-${signal_name}" ${pids} >/dev/null 2>&1 || true
  fi
}

cleanup_training_processes() {
  [[ -n "${TRAIN_CMD_PID}" ]] || return 0

  local pids
  pids="$(list_descendant_pids "${TRAIN_CMD_PID}" | awk '{ rows[NR]=$0 } END { for (i=NR; i>=1; i--) print rows[i] }' || true)"
  if kill -0 "${TRAIN_CMD_PID}" >/dev/null 2>&1; then
    pids="${pids}"$'\n'"${TRAIN_CMD_PID}"
  fi
  pids="$(printf '%s\n' "${pids}" | awk 'NF && !seen[$1]++')"
  [[ -n "${pids}" ]] || return 0

  echo "[cleanup] stopping training process tree rooted at pid=${TRAIN_CMD_PID}"
  # shellcheck disable=SC2086
  kill -TERM ${pids} >/dev/null 2>&1 || true
  sleep "${TRAIN_CLEANUP_GRACE_SECONDS:-2}"

  local alive_pids=""
  local pid
  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    if kill -0 "${pid}" >/dev/null 2>&1; then
      alive_pids="${alive_pids} ${pid}"
    fi
  done <<< "${pids}"
  if [[ -n "${alive_pids}" ]]; then
    echo "[cleanup] force killing remaining training processes:${alive_pids}"
    # shellcheck disable=SC2086
    kill -KILL ${alive_pids} >/dev/null 2>&1 || true
  fi
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  cleanup_training_processes
  if [[ "${CLEANUP_ORPHAN_RAY_VLLM_ON_EXIT:-${CLEANUP_ORPHAN_VLLM_ON_EXIT:-1}}" == "1" ]]; then
    cleanup_orphan_ray_vllm TERM
    sleep 1
    cleanup_orphan_ray_vllm KILL
  fi
  exit "${rc}"
}

trap on_exit EXIT INT TERM

if [[ "${CLEANUP_ORPHAN_RAY_VLLM_BEFORE_START:-${CLEANUP_ORPHAN_VLLM_BEFORE_START:-0}}" == "1" ]]; then
  cleanup_orphan_ray_vllm TERM
  sleep 1
  cleanup_orphan_ray_vllm KILL
fi

PROJECT_NAME="${PROJECT_NAME:-swe_agent_hls_rl_4b_npu}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-swe-agent-rl_docker_all_4b_npu}"

# Fallback only: every released config sets MODEL_PATH explicitly. This exists so the
# entrypoint can be smoke-tested without one, and so a missing MODEL_PATH fails with a
# path in the log rather than an empty Hydra override.
default_model_path() {
  local -a candidates=(
    "${WORKSPACE_ROOT}/models/SWE-Master-4B-SFT"
    "${WORKSPACE_ROOT}/models/Qwen3-4B-Instruct-2507"
    "${WORKSPACE_ROOT}/SWE-Master-4B-SFT"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done
  printf '%s' "${candidates[0]}"
}

MODEL_PATH="${MODEL_PATH:-$(default_model_path)}"
MODEL_PATH="${MODEL_PATH%/}"
MODEL_SERVE_NAME="${MODEL_SERVE_NAME:-${MODEL_PATH##*/}}"
# Optional HF RoPE override used when a model's native context is too short for
# the requested rollout envelope.  The values are deliberately opt-in so older
# recipes keep their model config unchanged.
YARN_ENABLE="${YARN_ENABLE:-0}"
YARN_FACTOR="${YARN_FACTOR:-2.0}"
YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-32768}"
YARN_MAX_POSITION_EMBEDDINGS="${YARN_MAX_POSITION_EMBEDDINGS:-65536}"
SWE_SCAFFOLD="${SWE_SCAFFOLD:-r2egym}"
SWE_GENERIC_SANDBOX="${SWE_GENERIC_SANDBOX:-1}"
# Executable-repair regime: always load the execution-capable command set (python/pytest/pip
# allowed) so training matches the eval scaffold. 1 = force execution (recommended); 0 = defer
# to runtime.generic_sandbox. Exported so env-construction workers inherit it.
RLLM_R2EGYM_FORCE_EXECUTION_COMMANDS="${RLLM_R2EGYM_FORCE_EXECUTION_COMMANDS:-1}"
export RLLM_R2EGYM_FORCE_EXECUTION_COMMANDS
if [[ "${SWE_GENERIC_SANDBOX}" == "1" ]]; then
  RLLM_R2EGYM_SRC="${RLLM_R2EGYM_SRC:-${WORKSPACE_ROOT}/R2E-Gym-generic-sandbox/src}"
  R2EGYM_GENERIC_SANDBOX_IMAGE="${R2EGYM_GENERIC_SANDBOX_IMAGE:-skongger/r2egym-scaffold-base:latest}"
  FILTER_TO_LOCAL_DOCKER_IMAGES="${FILTER_TO_LOCAL_DOCKER_IMAGES:-0}"
  export RLLM_R2EGYM_SRC R2EGYM_GENERIC_SANDBOX_IMAGE
fi

TS="$(TZ="Asia/Shanghai" date +"%Y-%m-%d_%H-%M-%S")"
EXPERIMENT_NAME="${EXPERIMENT_NAME}_${TS}"

DEFAULT_LOCAL_DIR="${WORKSPACE_ROOT}/DeepSWE_RL/rllm/rllm/train_output/${PROJECT_NAME}/${EXPERIMENT_NAME}"
ENABLE_ROLLOUT_DATA_DUMP="${ENABLE_ROLLOUT_DATA_DUMP:-0}"
ROLLOUT_DATA_DIR=""
if [[ "${ENABLE_ROLLOUT_DATA_DUMP}" == "1" ]]; then
  ROLLOUT_DATA_DIR="${WORKSPACE_ROOT}/DeepSWE_RL/rllm/rllm/train_output_rollout/${PROJECT_NAME}/${EXPERIMENT_NAME}"
fi
if [[ "${SWE_GENERIC_SANDBOX}" == "1" ]]; then
  DEFAULT_TRAIN_DATASET="${WORKSPACE_ROOT}/data_examples/rl_data/generic_sandbox/swebench_verified_sympy_smoke/train_verl.parquet"
  DEFAULT_VAL_DATASET="${DEFAULT_TRAIN_DATASET}"
else
  DEFAULT_TRAIN_DATASET="${WORKSPACE_ROOT}/swe-gym-subset/train_verl.parquet"
  DEFAULT_VAL_DATASET="${WORKSPACE_ROOT}/swe-gym-subset/validation_verl.parquet"
fi
TRAIN_DATASET="${TRAIN_DATASET:-${DEFAULT_TRAIN_DATASET}}"
VAL_DATASET="${VAL_DATASET:-${DEFAULT_VAL_DATASET}}"

RL_ALGO="${RL_ALGO:-grpo}"
case "${RL_ALGO}" in
  grpo)
    ALGORITHM_ADV_ESTIMATOR="grpo"
    DEFAULT_CRITIC_ENABLE="False"
    DEFAULT_ACTOR_USE_KL_LOSS="False"
    DEFAULT_ACTOR_KL_LOSS_COEF="0.0"
    DEFAULT_ROLLOUT_N="4"
    ;;
  gae|ppo)
    ALGORITHM_ADV_ESTIMATOR="gae"
    DEFAULT_CRITIC_ENABLE="True"
    DEFAULT_ACTOR_USE_KL_LOSS="False"
    DEFAULT_ACTOR_KL_LOSS_COEF="0.0"
    DEFAULT_ROLLOUT_N="1"
    ;;
  *)
    echo "ERROR: unsupported RL_ALGO=${RL_ALGO}. Use one of: grpo, gae, ppo"
    exit 1
    ;;
esac

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
VISIBLE_NPUS_COUNT=8
if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _visible_npus <<< "${ASCEND_RT_VISIBLE_DEVICES}"
  VISIBLE_NPUS_COUNT="${#_visible_npus[@]}"
fi
if (( VISIBLE_NPUS_COUNT <= 0 )); then
  VISIBLE_NPUS_COUNT=1
fi
TRAIN_N_GPUS_PER_NODE="${TRAIN_N_GPUS_PER_NODE:-${VISIBLE_NPUS_COUNT}}"
TRAINER_LAYOUT_MODE="${TRAINER_LAYOUT_MODE:-}"
if [[ -n "${ROLLOUT_NPU_IDS:-}" || -n "${TRAINING_NPU_IDS:-}" ]]; then
  if [[ -z "${ROLLOUT_NPU_IDS:-}" || -z "${TRAINING_NPU_IDS:-}" ]]; then
    echo "ERROR: ROLLOUT_NPU_IDS and TRAINING_NPU_IDS must be set together." >&2
    exit 1
  fi
  normalize_pipeline_npu_visibility
  IFS=',' read -r -a _rollout_npus <<< "${ROLLOUT_NPU_IDS}"
  IFS=',' read -r -a _training_npus <<< "${TRAINING_NPU_IDS}"
  ROLLOUT_N_GPUS_PER_NODE="${ROLLOUT_N_GPUS_PER_NODE:-${#_rollout_npus[@]}}"
  TRAINING_N_GPUS_PER_NODE="${TRAINING_N_GPUS_PER_NODE:-${#_training_npus[@]}}"
  TRAIN_N_GPUS_PER_NODE="${TRAIN_N_GPUS_PER_NODE:-$(( ROLLOUT_N_GPUS_PER_NODE + TRAINING_N_GPUS_PER_NODE ))}"
  TRAINER_LAYOUT_MODE="${TRAINER_LAYOUT_MODE:-pipeline}"
fi
TRAINER_LAYOUT_MODE="${TRAINER_LAYOUT_MODE:-colocated}"
TRAINING_N_GPUS_PER_NODE="${TRAINING_N_GPUS_PER_NODE:-${TRAIN_N_GPUS_PER_NODE}}"
ROLLOUT_N_GPUS_PER_NODE="${ROLLOUT_N_GPUS_PER_NODE:-$(( TRAIN_N_GPUS_PER_NODE - TRAINING_N_GPUS_PER_NODE ))}"
if (( ROLLOUT_N_GPUS_PER_NODE <= 0 )); then
  ROLLOUT_N_GPUS_PER_NODE="${TRAIN_N_GPUS_PER_NODE}"
fi
case "${TRAINER_LAYOUT_MODE}" in
  colocated)
    HYBRID_ENGINE="${HYBRID_ENGINE:-True}"
    ROLLOUT_PARALLEL_N_GPUS_PER_NODE="${TRAIN_N_GPUS_PER_NODE}"
    TRAINING_PARALLEL_N_GPUS_PER_NODE="${TRAIN_N_GPUS_PER_NODE}"
    ;;
  pipeline)
    HYBRID_ENGINE="${HYBRID_ENGINE:-False}"
    ROLLOUT_PARALLEL_N_GPUS_PER_NODE="${ROLLOUT_N_GPUS_PER_NODE}"
    TRAINING_PARALLEL_N_GPUS_PER_NODE="${TRAINING_N_GPUS_PER_NODE}"
    ;;
  *)
    echo "ERROR: unsupported TRAINER_LAYOUT_MODE=${TRAINER_LAYOUT_MODE}. Use colocated or pipeline." >&2
    exit 1
    ;;
esac
MEMORY_PROFILE="${MEMORY_PROFILE:-high_context}"
case "${MEMORY_PROFILE}" in
  stable)
    PROFILE_MAX_PROMPT_LENGTH=4096
    PROFILE_MAX_RESPONSE_LENGTH=4096
    PROFILE_PPO_MAX_TOKEN_LEN_PER_GPU=12288
    PROFILE_LOGPROB_MAX_TOKEN_LEN_PER_GPU=12288
    PROFILE_ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192
    PROFILE_ROLLOUT_GPU_MEMORY_UTILIZATION=0.60
    PROFILE_ROLLOUT_TP_SIZE="${TRAIN_N_GPUS_PER_NODE}"
    PROFILE_LOGPROB_USE_DYNAMIC_BSZ=False
    PROFILE_ROLLOUT_ENFORCE_EAGER=True
    PROFILE_ACTOR_USE_DYNAMIC_BSZ=False
    PROFILE_ACTOR_ENTROPY_CHUNKING=True
    PROFILE_ACTOR_ENTROPY_CHECKPOINTING=False
    PROFILE_ACTOR_ENTROPY_CHUNK_SIZE=1024
    PROFILE_MODEL_ENABLE_ACTIVATION_OFFLOAD=False
    PROFILE_AGENT_MAX_STEPS=40
    PROFILE_MODEL_USE_REMOVE_PADDING=False
    PROFILE_ULYSSES_SEQUENCE_PARALLEL_SIZE=1
    ;;
  balanced)
    PROFILE_MAX_PROMPT_LENGTH=4096
    PROFILE_MAX_RESPONSE_LENGTH=6144
    PROFILE_PPO_MAX_TOKEN_LEN_PER_GPU=14336
    PROFILE_LOGPROB_MAX_TOKEN_LEN_PER_GPU=14336
    PROFILE_ROLLOUT_MAX_NUM_BATCHED_TOKENS=10240
    PROFILE_ROLLOUT_GPU_MEMORY_UTILIZATION=0.65
    PROFILE_ROLLOUT_TP_SIZE="${TRAIN_N_GPUS_PER_NODE}"
    PROFILE_LOGPROB_USE_DYNAMIC_BSZ=False
    PROFILE_ROLLOUT_ENFORCE_EAGER=True
    PROFILE_ACTOR_USE_DYNAMIC_BSZ=False
    PROFILE_ACTOR_ENTROPY_CHUNKING=True
    PROFILE_ACTOR_ENTROPY_CHECKPOINTING=False
    PROFILE_ACTOR_ENTROPY_CHUNK_SIZE=768
    PROFILE_MODEL_ENABLE_ACTIVATION_OFFLOAD=False
    PROFILE_AGENT_MAX_STEPS=40
    PROFILE_MODEL_USE_REMOVE_PADDING=False
    PROFILE_ULYSSES_SEQUENCE_PARALLEL_SIZE=1
    ;;
  high_context)
    PROFILE_MAX_PROMPT_LENGTH=8192
    # Keep prompt budget high, but lower response budget to avoid forcing
    # min(prompt+response)=16384 which auto-bumps token caps and triggers OOM in log_prob CE.
    PROFILE_MAX_RESPONSE_LENGTH=19200
    PROFILE_PPO_MAX_TOKEN_LEN_PER_GPU=12288
    PROFILE_LOGPROB_MAX_TOKEN_LEN_PER_GPU=12288
    PROFILE_ROLLOUT_MAX_NUM_BATCHED_TOKENS=12288
    PROFILE_ROLLOUT_GPU_MEMORY_UTILIZATION=0.40
    PROFILE_ROLLOUT_TP_SIZE="${TRAIN_N_GPUS_PER_NODE}"
    PROFILE_LOGPROB_USE_DYNAMIC_BSZ=True
    PROFILE_ROLLOUT_ENFORCE_EAGER=True
    PROFILE_ACTOR_USE_DYNAMIC_BSZ=True
    PROFILE_ACTOR_ENTROPY_CHUNKING=True
    PROFILE_ACTOR_ENTROPY_CHECKPOINTING=True
    PROFILE_ACTOR_ENTROPY_CHUNK_SIZE=128
    PROFILE_MODEL_ENABLE_ACTIVATION_OFFLOAD=True
    PROFILE_AGENT_MAX_STEPS=80
    PROFILE_MODEL_USE_REMOVE_PADDING=True
    PROFILE_ULYSSES_SEQUENCE_PARALLEL_SIZE=4
    ;;
  *)
    echo "ERROR: unsupported MEMORY_PROFILE=${MEMORY_PROFILE}. Use one of: stable, balanced, high_context"
    exit 1
    ;;
esac

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-${PROFILE_MAX_PROMPT_LENGTH}}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-${PROFILE_MAX_RESPONSE_LENGTH}}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU="${CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-${PROFILE_PPO_MAX_TOKEN_LEN_PER_GPU}}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-${PROFILE_ULYSSES_SEQUENCE_PARALLEL_SIZE}}"
if [[ "${TRAINER_LAYOUT_MODE}" == "pipeline" ]]; then
  PROFILE_ROLLOUT_TP_SIZE="${ROLLOUT_N_GPUS_PER_NODE}"
fi
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-${PROFILE_ROLLOUT_TP_SIZE}}"
MIN_ROLLOUT_MAX_MODEL_LEN=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-${MIN_ROLLOUT_MAX_MODEL_LEN}}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${PROFILE_ROLLOUT_MAX_NUM_BATCHED_TOKENS}}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-12}"
ROLLOUT_ACTIVE_TRAJECTORIES="${ROLLOUT_ACTIVE_TRAJECTORIES:-${ROLLOUT_MAX_NUM_SEQS}}"
ROLLOUT_CONTEXT_WINDOW_SAFETY_MARGIN="${ROLLOUT_CONTEXT_WINDOW_SAFETY_MARGIN:-2048}"
ROLLOUT_PER_STEP_MAX_TOKENS="${ROLLOUT_PER_STEP_MAX_TOKENS:-4096}"
ROLLOUT_FIRST_STEP_MAX_TOKENS="${ROLLOUT_FIRST_STEP_MAX_TOKENS:-}"
ROLLOUT_INVALID_TOOL_CALL_MAX_TOKENS="${ROLLOUT_INVALID_TOOL_CALL_MAX_TOKENS:-}"
ROLLOUT_INVALID_TOOL_CALL_RETRY_LIMIT="${ROLLOUT_INVALID_TOOL_CALL_RETRY_LIMIT:-}"
ROLLOUT_STOP_AFTER_FUNCTION_CALL="${ROLLOUT_STOP_AFTER_FUNCTION_CALL:-False}"
ROLLOUT_N="${ROLLOUT_N:-${DEFAULT_ROLLOUT_N}}"
# ---- ARPO (arXiv:2507.19849) entropy-driven adaptive branching rollout ----
# Read from os.environ by rllm/engine/agent_execution_engine.py (same pattern as
# OPSD_*); forwarded to Ray workers via the ARPO_ prefix in train_agent_ppo.py.
# ARPO_ENABLE=0 keeps every rollout path byte-identical to plain GRPO.
ARPO_ENABLE="${ARPO_ENABLE:-0}"
ARPO_GROUP_SIZE="${ARPO_GROUP_SIZE:-${ROLLOUT_N}}"
ARPO_INITIAL_ROLLOUTS="${ARPO_INITIAL_ROLLOUTS:-2}"
ARPO_BEAM_SIZE="${ARPO_BEAM_SIZE:-2}"
ARPO_BRANCH_PROBABILITY="${ARPO_BRANCH_PROBABILITY:-0.5}"
ARPO_ENTROPY_WEIGHT="${ARPO_ENTROPY_WEIGHT:-0.2}"
ARPO_TOPK_LOGPROBS="${ARPO_TOPK_LOGPROBS:-10}"
ARPO_FILL_MODE="${ARPO_FILL_MODE:-duplicate}"
ARPO_MAX_REPLAY_DIVERGENCE="${ARPO_MAX_REPLAY_DIVERGENCE:-0.2}"
ARPO_MIN_SEGMENT_TOKENS="${ARPO_MIN_SEGMENT_TOKENS:-8}"
ARPO_BRANCH_PARALLEL="${ARPO_BRANCH_PARALLEL:-16}"
export ARPO_ENABLE ARPO_GROUP_SIZE ARPO_INITIAL_ROLLOUTS ARPO_BEAM_SIZE \
  ARPO_BRANCH_PROBABILITY ARPO_ENTROPY_WEIGHT ARPO_TOPK_LOGPROBS ARPO_FILL_MODE \
  ARPO_MAX_REPLAY_DIVERGENCE ARPO_MIN_SEGMENT_TOKENS ARPO_BRANCH_PARALLEL
ROLLOUT_VAL_N="${ROLLOUT_VAL_N:-1}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.2}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-${PROFILE_ROLLOUT_GPU_MEMORY_UTILIZATION}}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-${PROFILE_ROLLOUT_ENFORCE_EAGER}}"
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-True}"
ROLLOUT_ENABLE_SLEEP_MODE="${ROLLOUT_ENABLE_SLEEP_MODE:-True}"
ROLLOUT_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}"
ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-True}"
ROLLOUT_BLOCK_SIZE="${ROLLOUT_BLOCK_SIZE:-128}"
ROLLOUT_ASCEND_SCHEDULER_ENABLE="${ROLLOUT_ASCEND_SCHEDULER_ENABLE:-True}"
ROLLOUT_ASCEND_SCHEDULER_ENABLE_CHUNKED_PREFILL="${ROLLOUT_ASCEND_SCHEDULER_ENABLE_CHUNKED_PREFILL:-${ROLLOUT_ENABLE_CHUNKED_PREFILL}}"
ROLLOUT_ASCEND_SCHEDULER_MAX_LONG_PARTIAL_PREFILLS="${ROLLOUT_ASCEND_SCHEDULER_MAX_LONG_PARTIAL_PREFILLS:-}"
ROLLOUT_ASCEND_SCHEDULER_LONG_PREFILL_TOKEN_THRESHOLD="${ROLLOUT_ASCEND_SCHEDULER_LONG_PREFILL_TOKEN_THRESHOLD:-}"
ROLLOUT_ASCEND_SCHEDULER_REFRESH="${ROLLOUT_ASCEND_SCHEDULER_REFRESH:-True}"
ROLLOUT_CUDAGRAPH_CAPTURE_SIZES="${ROLLOUT_CUDAGRAPH_CAPTURE_SIZES:-}"
ROLLOUT_HCCL_OP_EXPANSION_MODE="${ROLLOUT_HCCL_OP_EXPANSION_MODE:-${HCCL_OP_EXPANSION_MODE:-}}"
if [[ -n "${ROLLOUT_HCCL_OP_EXPANSION_MODE}" ]]; then
  export HCCL_OP_EXPANSION_MODE="${ROLLOUT_HCCL_OP_EXPANSION_MODE}"
fi
DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS:-${NCCL_TIMEOUT_SECONDS:-}}"
if [[ "${TRAINER_LAYOUT_MODE}" == "pipeline" ]]; then
  DEFAULT_CHECKPOINT_ENGINE_BUCKET_MB=896
  CHECKPOINT_ENGINE_BUCKET_MB="${CHECKPOINT_ENGINE_BUCKET_MB:-${DEFAULT_CHECKPOINT_ENGINE_BUCKET_MB}}"
  CHECKPOINT_ENGINE_WEIGHT_TRANSFER_DTYPE="${CHECKPOINT_ENGINE_WEIGHT_TRANSFER_DTYPE:-bf16}"
fi
PIPELINE_TRAIN_AFTER_FULL_ROLLOUT="${PIPELINE_TRAIN_AFTER_FULL_ROLLOUT:-0}"
PIPELINE_LENGTH_BUCKET_ROLLOUTS="${PIPELINE_LENGTH_BUCKET_ROLLOUTS:-1}"
ACTOR_USE_DYNAMIC_BSZ="${ACTOR_USE_DYNAMIC_BSZ:-${PROFILE_ACTOR_USE_DYNAMIC_BSZ}}"
ACTOR_ENTROPY_CHUNKING="${ACTOR_ENTROPY_CHUNKING:-${PROFILE_ACTOR_ENTROPY_CHUNKING}}"
ACTOR_ENTROPY_CHECKPOINTING="${ACTOR_ENTROPY_CHECKPOINTING:-${PROFILE_ACTOR_ENTROPY_CHECKPOINTING}}"
ACTOR_ENTROPY_CHUNK_SIZE="${ACTOR_ENTROPY_CHUNK_SIZE:-${PROFILE_ACTOR_ENTROPY_CHUNK_SIZE}}"
MODEL_ENABLE_ACTIVATION_OFFLOAD="${MODEL_ENABLE_ACTIVATION_OFFLOAD:-${PROFILE_MODEL_ENABLE_ACTIVATION_OFFLOAD}}"
MODEL_USE_REMOVE_PADDING="${MODEL_USE_REMOVE_PADDING:-${PROFILE_MODEL_USE_REMOVE_PADDING}}"
LOGPROB_USE_DYNAMIC_BSZ="${LOGPROB_USE_DYNAMIC_BSZ:-${ACTOR_USE_DYNAMIC_BSZ}}"
LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOGPROB_MAX_TOKEN_LEN_PER_GPU="${LOGPROB_MAX_TOKEN_LEN_PER_GPU:-${PROFILE_LOGPROB_MAX_TOKEN_LEN_PER_GPU}}"
CRITIC_ENABLE="${CRITIC_ENABLE:-${DEFAULT_CRITIC_ENABLE}}"
ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-${DEFAULT_ACTOR_USE_KL_LOSS}}"
ACTOR_KL_LOSS_COEF="${ACTOR_KL_LOSS_COEF:-${DEFAULT_ACTOR_KL_LOSS_COEF}}"
ACTOR_KL_LOSS_TYPE="${ACTOR_KL_LOSS_TYPE:-low_var_kl}"
ACTOR_CLIP_RATIO_LOW="${ACTOR_CLIP_RATIO_LOW:-0.2}"
ACTOR_CLIP_RATIO_HIGH="${ACTOR_CLIP_RATIO_HIGH:-0.28}"
# Entropy bonus (0.0 keeps legacy behavior). Set >0 to resist entropy/length collapse.
ACTOR_ENTROPY_COEFF="${ACTOR_ENTROPY_COEFF:-0.0}"
# Reward multiplier for soft-overlong trajectories (TRUNCATION/MAX_STEPS/TIMEOUT).
# 0.5 = legacy discount; 1.0 disables it so genuine long attempts are not punished.
SOFT_OVERLONG_REWARD_FACTOR="${SOFT_OVERLONG_REWARD_FACTOR:-0.5}"
# Penalty applied when the agent voluntarily finishes without solving and without any
# successful source edit (empty patch). 0.0 keeps legacy behavior; negative to enable.
GIVE_UP_EMPTY_PATCH_PENALTY="${GIVE_UP_EMPTY_PATCH_PENALTY:-0.0}"
PG_LOSS_AGG_MODE="${PG_LOSS_AGG_MODE:-seq-mean-token-sum}"
ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-False}"
ACTOR_OPTIMIZER_FOREACH="${ACTOR_OPTIMIZER_FOREACH:-auto}"
WANDB_ENABLE="${WANDB_ENABLE:-0}"
TRAIN_LOGGERS="${TRAIN_LOGGERS:-console,tensorboard}"
OPSD_ENABLE="${OPSD_ENABLE:-0}"
OPSD_LOSS_COEF="${OPSD_LOSS_COEF:-0.4}"
# SDAR baseline (arXiv 2605.15155): sigmoid-gated sampled-token distillation replaces
# the top-k reverse KL when OPSD_SDAR_GATE=1. Paper hyperparams: coef 0.01, beta 5.0.
OPSD_SDAR_GATE="${OPSD_SDAR_GATE:-0}"
SDAR_GATE_BETA="${SDAR_GATE_BETA:-5.0}"
OPSD_POLICY_LOSS_MODE="${OPSD_POLICY_LOSS_MODE:-sdpo}"
OPSD_TOPK_K="${OPSD_TOPK_K:-16}"
OPSD_MAX_TOKENS_PER_SAMPLE="${OPSD_MAX_TOKENS_PER_SAMPLE:-0}"
OPSD_MAX_RESPONSE_TOKENS_FOR_TRAINING="${OPSD_MAX_RESPONSE_TOKENS_FOR_TRAINING:-0}"
OPSD_ROUTE_SUCCESS_GRPO_FAILURE_OPSD="${OPSD_ROUTE_SUCCESS_GRPO_FAILURE_OPSD:-0}"
OPSD_TEACHER_SOURCE="${OPSD_TEACHER_SOURCE:-ref_policy}"
OPSD_REF_LOAD_CHECKPOINT_FROM_RESUME="${OPSD_REF_LOAD_CHECKPOINT_FROM_RESUME:-False}"
# Separate ref/teacher init path (Option B: teacher != student for ref_policy OPSD).
# The frozen ref worker (= the OPSD teacher) loads from this path while the actor warm-starts
# from MODEL_PATH. verl's fsdp ref worker reads actor_rollout_ref.ref.model.path (falls back to
# model.path), so a distinct value here makes teacher=SFT while student=base, and per-token
# stage routing (ref_policy path) keeps working. Defaults to MODEL_PATH => ref==actor (unchanged
# behavior for every existing config).
OPSD_REF_MODEL_PATH="${OPSD_REF_MODEL_PATH:-${MODEL_PATH}}"
OPSD_TEACHER_BASE_URL="${OPSD_TEACHER_BASE_URL:-${OPENAI_BASE_URL:-}}"
OPSD_TEACHER_MODEL="${OPSD_TEACHER_MODEL:-${MODEL_SERVE_NAME}}"
OPSD_TEACHER_TIMEOUT_SEC="${OPSD_TEACHER_TIMEOUT_SEC:-120}"
OPSD_FAIL_OPEN="${OPSD_FAIL_OPEN:-True}"
OPSD_NORM_TO_ONE_FOR_KL="${OPSD_NORM_TO_ONE_FOR_KL:-True}"
USE_RLSD="${USE_RLSD:-False}"
USE_OPSD_KL="${USE_OPSD_KL:-${OPSD_ENABLE}}"
RLSD_LAMBDA="${RLSD_LAMBDA:-1.0}"
RLSD_LAMBDA_INIT="${RLSD_LAMBDA_INIT:-${RLSD_LAMBDA}}"
RLSD_LAMBDA_FINAL="${RLSD_LAMBDA_FINAL:-${RLSD_LAMBDA_INIT}}"
RLSD_LAMBDA_DECAY_STEPS="${RLSD_LAMBDA_DECAY_STEPS:-0}"
RLSD_WEIGHT_CLIP_EPS="${RLSD_WEIGHT_CLIP_EPS:-0.2}"
RLSD_DELTA_CLIP="${RLSD_DELTA_CLIP:-5.0}"
RLSD_USE_OLD_STUDENT_LOGP="${RLSD_USE_OLD_STUDENT_LOGP:-True}"
RLSD_USE_ROLLOUT_LOGP_AS_STUDENT_PRIOR="${RLSD_USE_ROLLOUT_LOGP_AS_STUDENT_PRIOR:-False}"
RLSD_TEACHER_NO_GRAD="${RLSD_TEACHER_NO_GRAD:-True}"
RLSD_LOG_STATS="${RLSD_LOG_STATS:-True}"
RLSD_ACTOR_MAX_PACKED_CONTEXT_TOKENS="${RLSD_ACTOR_MAX_PACKED_CONTEXT_TOKENS:-0}"
TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-auto}"
TRAINER_RESUME_FROM_PATH="${TRAINER_RESUME_FROM_PATH:-null}"
TRAINER_RESUME_LOAD_DATALOADER="${TRAINER_RESUME_LOAD_DATALOADER:-True}"
TRAINER_STAGE_GLOBAL_STEP_OFFSET="${TRAINER_STAGE_GLOBAL_STEP_OFFSET:-0}"
EXPLORATION_TOOL_REWARD_COEF="${EXPLORATION_TOOL_REWARD_COEF:-0.0}"
EXPLORATION_TOOL_REWARD_MAX="${EXPLORATION_TOOL_REWARD_MAX:-0.0}"
EXPLORATION_TOOL_MIN_CALLS="${EXPLORATION_TOOL_MIN_CALLS:-0}"
EXPLORATION_TOOL_MISSING_PENALTY="${EXPLORATION_TOOL_MISSING_PENALTY:-0.0}"
EXPLORATION_NO_TOOL_CALL_PENALTY="${EXPLORATION_NO_TOOL_CALL_PENALTY:-0.0}"
EXPLORATION_REWARD_CLAMP_MIN="${EXPLORATION_REWARD_CLAMP_MIN:-null}"
EXPLORATION_REWARD_CLAMP_MAX="${EXPLORATION_REWARD_CLAMP_MAX:-null}"
CRITIC_PARAM_OFFLOAD="${CRITIC_PARAM_OFFLOAD:-True}"
CRITIC_OPTIMIZER_OFFLOAD="${CRITIC_OPTIMIZER_OFFLOAD:-True}"
CRITIC_ENABLE_ACTIVATION_OFFLOAD="${CRITIC_ENABLE_ACTIVATION_OFFLOAD:-False}"
AGENT_MAX_STEPS="${AGENT_MAX_STEPS:-${PROFILE_AGENT_MAX_STEPS}}"
AGENT_TRAJECTORY_TIMEOUT="${AGENT_TRAJECTORY_TIMEOUT:-3600}"
AGENT_MAX_RESPONSE_TOKENS_FOR_TRAINING="${AGENT_MAX_RESPONSE_TOKENS_FOR_TRAINING:-0}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-}"
PREPULL_DOCKER_IMAGES="${PREPULL_DOCKER_IMAGES:-0}"
FILTER_TO_LOCAL_DOCKER_IMAGES="${FILTER_TO_LOCAL_DOCKER_IMAGES:-1}"
ENABLE_VALIDATION="${ENABLE_VALIDATION:-0}"
if [[ "${ENABLE_VALIDATION}" != "1" ]]; then
  VAL_DATASET="${TRAIN_DATASET}"
fi
if [[ "${OPSD_ENABLE}" == "1" ]]; then
  if [[ "${ULYSSES_SEQUENCE_PARALLEL_SIZE}" != "1" && "${MODEL_USE_REMOVE_PADDING}" != "True" ]]; then
    echo "ERROR: OPSD_ENABLE=1 supports Ulysses SP only when MODEL_USE_REMOVE_PADDING=True." >&2
    exit 1
  fi
fi
if (( ULYSSES_SEQUENCE_PARALLEL_SIZE > TRAINING_PARALLEL_N_GPUS_PER_NODE )); then
  echo "Adjust ULYSSES_SEQUENCE_PARALLEL_SIZE from ${ULYSSES_SEQUENCE_PARALLEL_SIZE} to ${TRAINING_PARALLEL_N_GPUS_PER_NODE}." >&2
  ULYSSES_SEQUENCE_PARALLEL_SIZE="${TRAINING_PARALLEL_N_GPUS_PER_NODE}"
fi
if (( TRAINING_PARALLEL_N_GPUS_PER_NODE % ULYSSES_SEQUENCE_PARALLEL_SIZE != 0 )); then
  echo "ERROR: TRAINING_N_GPUS_PER_NODE=${TRAINING_PARALLEL_N_GPUS_PER_NODE} must be divisible by ULYSSES_SEQUENCE_PARALLEL_SIZE=${ULYSSES_SEQUENCE_PARALLEL_SIZE}." >&2
  exit 1
fi

if (( ROLLOUT_MAX_MODEL_LEN < MIN_ROLLOUT_MAX_MODEL_LEN )); then
  echo "Adjust ROLLOUT_MAX_MODEL_LEN from ${ROLLOUT_MAX_MODEL_LEN} to ${MIN_ROLLOUT_MAX_MODEL_LEN}." >&2
  ROLLOUT_MAX_MODEL_LEN="${MIN_ROLLOUT_MAX_MODEL_LEN}"
fi
if (( ROLLOUT_MAX_NUM_BATCHED_TOKENS < ROLLOUT_MAX_MODEL_LEN )) && ! bool_is_true "${ROLLOUT_ENABLE_CHUNKED_PREFILL}"; then
  echo "Adjust ROLLOUT_MAX_NUM_BATCHED_TOKENS from ${ROLLOUT_MAX_NUM_BATCHED_TOKENS} to ${ROLLOUT_MAX_MODEL_LEN}." >&2
  ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_MODEL_LEN}"
fi
if (( PPO_MAX_TOKEN_LEN_PER_GPU < MIN_ROLLOUT_MAX_MODEL_LEN )); then
  echo "Adjust PPO_MAX_TOKEN_LEN_PER_GPU from ${PPO_MAX_TOKEN_LEN_PER_GPU} to ${MIN_ROLLOUT_MAX_MODEL_LEN}." >&2
  PPO_MAX_TOKEN_LEN_PER_GPU="${MIN_ROLLOUT_MAX_MODEL_LEN}"
fi
if (( LOGPROB_MAX_TOKEN_LEN_PER_GPU < MIN_ROLLOUT_MAX_MODEL_LEN )); then
  echo "Adjust LOGPROB_MAX_TOKEN_LEN_PER_GPU from ${LOGPROB_MAX_TOKEN_LEN_PER_GPU} to ${MIN_ROLLOUT_MAX_MODEL_LEN}." >&2
  LOGPROB_MAX_TOKEN_LEN_PER_GPU="${MIN_ROLLOUT_MAX_MODEL_LEN}"
fi
if (( ROLLOUT_TP_SIZE > ROLLOUT_PARALLEL_N_GPUS_PER_NODE )); then
  echo "Adjust ROLLOUT_TP_SIZE from ${ROLLOUT_TP_SIZE} to ${ROLLOUT_PARALLEL_N_GPUS_PER_NODE}." >&2
  ROLLOUT_TP_SIZE="${ROLLOUT_PARALLEL_N_GPUS_PER_NODE}"
fi
if (( ROLLOUT_PARALLEL_N_GPUS_PER_NODE % ROLLOUT_TP_SIZE != 0 )); then
  original_rollout_tp_size="${ROLLOUT_TP_SIZE}"
  while (( ROLLOUT_TP_SIZE > 1 )); do
    ROLLOUT_TP_SIZE=$(( ROLLOUT_TP_SIZE - 1 ))
    if (( ROLLOUT_PARALLEL_N_GPUS_PER_NODE % ROLLOUT_TP_SIZE == 0 )); then
      break
    fi
  done
  echo "Adjust ROLLOUT_TP_SIZE from ${original_rollout_tp_size} to ${ROLLOUT_TP_SIZE} so rollout pool size % ROLLOUT_TP_SIZE == 0." >&2
fi

MODEL_ATTENTION_HEADS="$(python3 - "${MODEL_PATH}" <<'PY'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
config_path = model_path / "config.json"
if not config_path.exists():
    print("")
    raise SystemExit(0)

try:
    config = json.loads(config_path.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)

value = config.get("num_attention_heads")
print("" if value is None else value)
PY
)"

if [[ -n "${MODEL_ATTENTION_HEADS}" ]]; then
  if (( MODEL_ATTENTION_HEADS % ROLLOUT_TP_SIZE != 0 )); then
    original_rollout_tp_size="${ROLLOUT_TP_SIZE}"
    while (( ROLLOUT_TP_SIZE > 1 )); do
      ROLLOUT_TP_SIZE=$(( ROLLOUT_TP_SIZE - 1 ))
      if (( ROLLOUT_PARALLEL_N_GPUS_PER_NODE % ROLLOUT_TP_SIZE == 0 && MODEL_ATTENTION_HEADS % ROLLOUT_TP_SIZE == 0 )); then
        break
      fi
    done
    if (( MODEL_ATTENTION_HEADS % ROLLOUT_TP_SIZE != 0 )); then
      echo "ERROR: could not find a valid ROLLOUT_TP_SIZE for model attention heads=${MODEL_ATTENTION_HEADS} with rollout pool size=${ROLLOUT_PARALLEL_N_GPUS_PER_NODE}." >&2
      exit 1
    fi
    echo "Adjust ROLLOUT_TP_SIZE from ${original_rollout_tp_size} to ${ROLLOUT_TP_SIZE} so num_attention_heads=${MODEL_ATTENTION_HEADS} is divisible by tensor parallel size." >&2
  fi
fi


TRAIN_BATCH_SIZE="$(normalize_divisible_value TRAIN_BATCH_SIZE "${TRAIN_BATCH_SIZE}" "${TRAINING_PARALLEL_N_GPUS_PER_NODE}")"
VAL_BATCH_SIZE="$(normalize_divisible_value VAL_BATCH_SIZE "${VAL_BATCH_SIZE}" "${TRAINING_PARALLEL_N_GPUS_PER_NODE}")"
PPO_MINI_BATCH_SIZE="$(normalize_divisible_value PPO_MINI_BATCH_SIZE "${PPO_MINI_BATCH_SIZE}" "${TRAINING_PARALLEL_N_GPUS_PER_NODE}")"
PPO_MINI_BATCH_SIZE="$(
  normalize_actor_ppo_mini_batch_size \
    "${PPO_MINI_BATCH_SIZE}" \
    "${ROLLOUT_N}" \
    "${TRAINING_PARALLEL_N_GPUS_PER_NODE}" \
    "${ULYSSES_SEQUENCE_PARALLEL_SIZE}"
)"
TRAIN_BATCH_SIZE="$(normalize_train_batch_size_for_ppo "${TRAIN_BATCH_SIZE}" "${PPO_MINI_BATCH_SIZE}")"

if [[ "${WANDB_ENABLE}" == "1" ]]; then
  case ",${TRAIN_LOGGERS}," in
    *,wandb,*) ;;
    *) TRAIN_LOGGERS="${TRAIN_LOGGERS},wandb" ;;
  esac
fi

format_hydra_logger_list() {
  local raw="$1"
  if [[ "${raw}" == \[* ]]; then
    printf '%s' "${raw}"
    return 0
  fi

  local -a items=()
  local item
  IFS=',' read -r -a items <<< "${raw}"
  local output="["
  local first=1
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -z "${item}" ]] && continue
    if (( first == 0 )); then
      output+=","
    fi
    output+="'${item}'"
    first=0
  done
  output+="]"
  printf '%s' "${output}"
}

TRAIN_LOGGERS_HYDRA="$(format_hydra_logger_list "${TRAIN_LOGGERS}")"

RLLM_ZMQ_SOCKET_DIR="${RLLM_ZMQ_SOCKET_DIR:-/tmp/rllm-zmq-${USER:-user}-$$}"
export RLLM_ZMQ_SOCKET_DIR
mkdir -p "${DEFAULT_LOCAL_DIR}" "${RLLM_ZMQ_SOCKET_DIR}"
if [[ -n "${ROLLOUT_DATA_DIR}" ]]; then
  mkdir -p "${ROLLOUT_DATA_DIR}"
fi
find "${RLLM_ZMQ_SOCKET_DIR}" -maxdepth 1 -type s -name 'rl-colocate-zmq-*' -delete 2>/dev/null || true

echo "Project: ${PROJECT_NAME}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "TRAIN_LOGGERS: ${TRAIN_LOGGERS_HYDRA}"
echo "WANDB_ENABLE: ${WANDB_ENABLE}"
echo "WANDB_PROJECT: ${PROJECT_NAME}"
echo "WANDB_RUN_NAME: ${EXPERIMENT_NAME}"
echo "TRAIN_DATASET: ${TRAIN_DATASET}"
echo "VAL_DATASET: ${VAL_DATASET}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "MODEL_SERVE_NAME: ${MODEL_SERVE_NAME}"
echo "YARN_ENABLE: ${YARN_ENABLE}"
echo "YARN_FACTOR: ${YARN_FACTOR}"
echo "YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS: ${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}"
echo "YARN_MAX_POSITION_EMBEDDINGS: ${YARN_MAX_POSITION_EMBEDDINGS}"
echo "MODEL_ATTENTION_HEADS: ${MODEL_ATTENTION_HEADS:-unknown}"
echo "SWE_SCAFFOLD: ${SWE_SCAFFOLD}"
echo "RL_ALGO: ${RL_ALGO}"
echo "MEMORY_PROFILE: ${MEMORY_PROFILE}"
echo "ENABLE_VALIDATION: ${ENABLE_VALIDATION}"
echo "TRAINER_LAYOUT_MODE: ${TRAINER_LAYOUT_MODE}"
echo "HYBRID_ENGINE: ${HYBRID_ENGINE}"
echo "ASCEND_RT_VISIBLE_DEVICES: ${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
echo "ROLLOUT_NPU_IDS: ${ROLLOUT_NPU_IDS:-<unset>}"
echo "TRAINING_NPU_IDS: ${TRAINING_NPU_IDS:-<unset>}"
echo "RLLM_ZMQ_SOCKET_DIR: ${RLLM_ZMQ_SOCKET_DIR}"
echo "TRAIN_N_GPUS_PER_NODE: ${TRAIN_N_GPUS_PER_NODE}"
echo "TRAINING_N_GPUS_PER_NODE: ${TRAINING_PARALLEL_N_GPUS_PER_NODE}"
echo "ROLLOUT_N_GPUS_PER_NODE: ${ROLLOUT_PARALLEL_N_GPUS_PER_NODE}"
echo "TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE}"
echo "PPO_MINI_BATCH_SIZE: ${PPO_MINI_BATCH_SIZE}"
echo "ROLLOUT_TP_SIZE: ${ROLLOUT_TP_SIZE}"
echo "MAX_PROMPT_LENGTH: ${MAX_PROMPT_LENGTH}"
echo "MAX_RESPONSE_LENGTH: ${MAX_RESPONSE_LENGTH}"
echo "PPO_MAX_TOKEN_LEN_PER_GPU: ${PPO_MAX_TOKEN_LEN_PER_GPU}"
echo "LOGPROB_MAX_TOKEN_LEN_PER_GPU: ${LOGPROB_MAX_TOKEN_LEN_PER_GPU}"
echo "ROLLOUT_MAX_NUM_BATCHED_TOKENS: ${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
echo "ROLLOUT_MAX_NUM_SEQS: ${ROLLOUT_MAX_NUM_SEQS}"
echo "ROLLOUT_ACTIVE_TRAJECTORIES: ${ROLLOUT_ACTIVE_TRAJECTORIES}"
echo "ROLLOUT_CONTEXT_WINDOW_SAFETY_MARGIN: ${ROLLOUT_CONTEXT_WINDOW_SAFETY_MARGIN}"
echo "ROLLOUT_PER_STEP_MAX_TOKENS: ${ROLLOUT_PER_STEP_MAX_TOKENS}"
echo "ROLLOUT_FIRST_STEP_MAX_TOKENS: ${ROLLOUT_FIRST_STEP_MAX_TOKENS:-<unset>}"
echo "ROLLOUT_INVALID_TOOL_CALL_MAX_TOKENS: ${ROLLOUT_INVALID_TOOL_CALL_MAX_TOKENS:-<unset>}"
echo "ROLLOUT_INVALID_TOOL_CALL_RETRY_LIMIT: ${ROLLOUT_INVALID_TOOL_CALL_RETRY_LIMIT:-<unset>}"
echo "ROLLOUT_STOP_AFTER_FUNCTION_CALL: ${ROLLOUT_STOP_AFTER_FUNCTION_CALL}"
echo "ROLLOUT_GPU_MEMORY_UTILIZATION: ${ROLLOUT_GPU_MEMORY_UTILIZATION}"
echo "ROLLOUT_FREE_CACHE_ENGINE: ${ROLLOUT_FREE_CACHE_ENGINE}"
echo "ROLLOUT_ENABLE_SLEEP_MODE: ${ROLLOUT_ENABLE_SLEEP_MODE}"
echo "ROLLOUT_ENABLE_CHUNKED_PREFILL: ${ROLLOUT_ENABLE_CHUNKED_PREFILL}"
echo "ROLLOUT_ENABLE_PREFIX_CACHING: ${ROLLOUT_ENABLE_PREFIX_CACHING}"
echo "ROLLOUT_BLOCK_SIZE: ${ROLLOUT_BLOCK_SIZE}"
echo "HCCL_OP_EXPANSION_MODE: ${HCCL_OP_EXPANSION_MODE:-<unset>}"
echo "DISTRIBUTED_TIMEOUT_SECONDS: ${DISTRIBUTED_TIMEOUT_SECONDS:-<default>}"
echo "CHECKPOINT_ENGINE_BUCKET_MB: ${CHECKPOINT_ENGINE_BUCKET_MB:-<unused>}"
echo "CHECKPOINT_ENGINE_WEIGHT_TRANSFER_DTYPE: ${CHECKPOINT_ENGINE_WEIGHT_TRANSFER_DTYPE:-<unused>}"
echo "PIPELINE_TRAIN_AFTER_FULL_ROLLOUT: ${PIPELINE_TRAIN_AFTER_FULL_ROLLOUT}"
echo "PIPELINE_LENGTH_BUCKET_ROLLOUTS: ${PIPELINE_LENGTH_BUCKET_ROLLOUTS}"
echo "ROLLOUT_ENFORCE_EAGER: ${ROLLOUT_ENFORCE_EAGER}"
echo "ACTOR_USE_DYNAMIC_BSZ: ${ACTOR_USE_DYNAMIC_BSZ}"
echo "ACTOR_ENTROPY_CHUNKING: ${ACTOR_ENTROPY_CHUNKING}"
echo "ACTOR_ENTROPY_CHECKPOINTING: ${ACTOR_ENTROPY_CHECKPOINTING}"
echo "ACTOR_ENTROPY_CHUNK_SIZE: ${ACTOR_ENTROPY_CHUNK_SIZE}"
echo "OPSD_ENABLE: ${OPSD_ENABLE}"
echo "OPSD_POLICY_LOSS_MODE: ${OPSD_POLICY_LOSS_MODE}"
echo "OPSD_TEACHER_SOURCE: ${OPSD_TEACHER_SOURCE}"
echo "OPSD_STAGE_CLASSIFIER: ${OPSD_STAGE_CLASSIFIER:-<rule-based>}"
echo "OPSD_PRIVILEGED_STAGES: ${OPSD_PRIVILEGED_STAGES:-<all>}"
echo "OPSD_REF_LOAD_CHECKPOINT_FROM_RESUME: ${OPSD_REF_LOAD_CHECKPOINT_FROM_RESUME}"
echo "OPSD_REF_MODEL_PATH (ref/teacher init): ${OPSD_REF_MODEL_PATH}"
echo "OPSD_TEACHER_BASE_URL: ${OPSD_TEACHER_BASE_URL:-<unset>}"
echo "OPSD_TEACHER_MODEL: ${OPSD_TEACHER_MODEL}"
echo "OPSD_TOPK_K: ${OPSD_TOPK_K}"
echo "OPSD_MAX_TOKENS_PER_SAMPLE: ${OPSD_MAX_TOKENS_PER_SAMPLE}"
echo "OPSD_MAX_RESPONSE_TOKENS_FOR_TRAINING: ${OPSD_MAX_RESPONSE_TOKENS_FOR_TRAINING}"
echo "AGENT_MAX_RESPONSE_TOKENS_FOR_TRAINING: ${AGENT_MAX_RESPONSE_TOKENS_FOR_TRAINING}"
echo "AGENT_TRAJECTORY_TIMEOUT: ${AGENT_TRAJECTORY_TIMEOUT}"
echo "OPSD_ROUTE_SUCCESS_GRPO_FAILURE_OPSD: ${OPSD_ROUTE_SUCCESS_GRPO_FAILURE_OPSD}"
echo "USE_RLSD: ${USE_RLSD}"
echo "USE_OPSD_KL: ${USE_OPSD_KL}"
echo "RLSD_LAMBDA_INIT: ${RLSD_LAMBDA_INIT}"
echo "RLSD_LAMBDA_FINAL: ${RLSD_LAMBDA_FINAL}"
echo "RLSD_LAMBDA_DECAY_STEPS: ${RLSD_LAMBDA_DECAY_STEPS}"
echo "RLSD_USE_ROLLOUT_LOGP_AS_STUDENT_PRIOR: ${RLSD_USE_ROLLOUT_LOGP_AS_STUDENT_PRIOR}"
echo "RLSD_ACTOR_MAX_PACKED_CONTEXT_TOKENS: ${RLSD_ACTOR_MAX_PACKED_CONTEXT_TOKENS}"
echo "TRAINER_RESUME_MODE: ${TRAINER_RESUME_MODE}"
echo "TRAINER_RESUME_FROM_PATH: ${TRAINER_RESUME_FROM_PATH}"
echo "TRAINER_RESUME_LOAD_DATALOADER: ${TRAINER_RESUME_LOAD_DATALOADER}"
echo "TRAINER_STAGE_GLOBAL_STEP_OFFSET: ${TRAINER_STAGE_GLOBAL_STEP_OFFSET}"
echo "TRAINER_TOTAL_TRAINING_STEPS: ${TRAINER_TOTAL_TRAINING_STEPS:-<unset>}"
echo "EXPLORATION_TOOL_REWARD_COEF: ${EXPLORATION_TOOL_REWARD_COEF}"
echo "EXPLORATION_TOOL_REWARD_MAX: ${EXPLORATION_TOOL_REWARD_MAX}"
echo "EXPLORATION_TOOL_MIN_CALLS: ${EXPLORATION_TOOL_MIN_CALLS}"
echo "EXPLORATION_TOOL_MISSING_PENALTY: ${EXPLORATION_TOOL_MISSING_PENALTY}"
echo "MODEL_ENABLE_ACTIVATION_OFFLOAD: ${MODEL_ENABLE_ACTIVATION_OFFLOAD}"
echo "MODEL_USE_REMOVE_PADDING: ${MODEL_USE_REMOVE_PADDING}"
echo "ACTOR_OPTIMIZER_FOREACH: ${ACTOR_OPTIMIZER_FOREACH}"
echo "ROLLOUT_TEMPERATURE: ${ROLLOUT_TEMPERATURE}"
echo "ARPO_ENABLE: ${ARPO_ENABLE}"
echo "ARPO_GROUP_SIZE: ${ARPO_GROUP_SIZE}"
echo "ARPO_INITIAL_ROLLOUTS: ${ARPO_INITIAL_ROLLOUTS}"
echo "ARPO_BEAM_SIZE: ${ARPO_BEAM_SIZE}"
echo "ARPO_BRANCH_PROBABILITY: ${ARPO_BRANCH_PROBABILITY}"
echo "ARPO_ENTROPY_WEIGHT: ${ARPO_ENTROPY_WEIGHT}"
echo "ARPO_TOPK_LOGPROBS: ${ARPO_TOPK_LOGPROBS}"
echo "ARPO_FILL_MODE: ${ARPO_FILL_MODE}"
echo "ARPO_MAX_REPLAY_DIVERGENCE: ${ARPO_MAX_REPLAY_DIVERGENCE}"
echo "ARPO_MIN_SEGMENT_TOKENS: ${ARPO_MIN_SEGMENT_TOKENS}"
echo "ARPO_BRANCH_PARALLEL: ${ARPO_BRANCH_PARALLEL}"
echo "ROLLOUT_ASCEND_SCHEDULER_ENABLE: ${ROLLOUT_ASCEND_SCHEDULER_ENABLE}"
echo "ROLLOUT_ASCEND_SCHEDULER_ENABLE_CHUNKED_PREFILL: ${ROLLOUT_ASCEND_SCHEDULER_ENABLE_CHUNKED_PREFILL}"
echo "ROLLOUT_ASCEND_SCHEDULER_MAX_LONG_PARTIAL_PREFILLS: ${ROLLOUT_ASCEND_SCHEDULER_MAX_LONG_PARTIAL_PREFILLS:-<unset>}"
echo "ROLLOUT_ASCEND_SCHEDULER_LONG_PREFILL_TOKEN_THRESHOLD: ${ROLLOUT_ASCEND_SCHEDULER_LONG_PREFILL_TOKEN_THRESHOLD:-<unset>}"
echo "ROLLOUT_CUDAGRAPH_CAPTURE_SIZES: ${ROLLOUT_CUDAGRAPH_CAPTURE_SIZES:-<unset>}"
echo "RLLM_FORCE_VLLM_WEIGHT_SYNC_SHM: ${RLLM_FORCE_VLLM_WEIGHT_SYNC_SHM:-<unset>}"
echo "RLLM_DISABLE_FSDP_SYNC_MODULE_STATES: ${RLLM_DISABLE_FSDP_SYNC_MODULE_STATES:-<unset>}"

ensure_pyarrow_readable_parquet "${TRAIN_DATASET}"
if [[ "${VAL_DATASET}" != "${TRAIN_DATASET}" ]]; then
  ensure_pyarrow_readable_parquet "${VAL_DATASET}"
fi
summarize_dataset "${TRAIN_DATASET}"
if [[ "${PREPULL_DOCKER_IMAGES}" == "1" ]]; then
  prepull_dataset_images "${TRAIN_DATASET}"
fi
if [[ "${FILTER_TO_LOCAL_DOCKER_IMAGES}" == "1" && "${R2EGYM_DOCKER_MODE}" == "local" ]]; then
  FILTERED_DATASET_ROOT="${WORKSPACE_ROOT}/DeepSWE_RL/rllm/rllm/filtered_datasets/${PROJECT_NAME}/${EXPERIMENT_NAME}"
  prepare_local_image_filtered_datasets "${TRAIN_DATASET}" "${VAL_DATASET}" "${FILTERED_DATASET_ROOT}"
  ensure_pyarrow_readable_parquet "${TRAIN_DATASET}"
  if [[ "${VAL_DATASET}" != "${TRAIN_DATASET}" ]]; then
    ensure_pyarrow_readable_parquet "${VAL_DATASET}"
  fi
  summarize_dataset "${TRAIN_DATASET}"
fi

export RLLM_ENABLE_VALIDATION="${ENABLE_VALIDATION}"

cmd=(
  python3 -m rllm.trainer.verl.train_agent_ppo
  "algorithm.adv_estimator=${ALGORITHM_ADV_ESTIMATOR}"
  "data.train_files=${TRAIN_DATASET}"
  "data.val_files=${VAL_DATASET}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.val_batch_size=${VAL_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=True"
  "data.filter_overlong_prompts_workers=32"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  # Frozen ref/teacher loads from OPSD_REF_MODEL_PATH (defaults to MODEL_PATH). Set it to a
  # different checkpoint (e.g. the SFT model) to get teacher != student while keeping the
  # in-process ref_policy stage-routed OPSD.
  # NOTE: the base hydra config's `actor_rollout_ref.ref` struct has no `model` subkey, so this
  # MUST be a force-add (++), not a plain override, or hydra raises "Key 'model' is not in struct".
  # fsdp_workers reads self.config.ref.get("model", None) and uses .path when present.
  "++actor_rollout_ref.ref.model.path=${OPSD_REF_MODEL_PATH}"
  "critic.model.path=${MODEL_PATH}"
  "+critic.enable=${CRITIC_ENABLE}"
  "actor_rollout_ref.hybrid_engine=${HYBRID_ENGINE}"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.model.enable_activation_offload=${MODEL_ENABLE_ACTIVATION_OFFLOAD}"
  "actor_rollout_ref.model.use_remove_padding=${MODEL_USE_REMOVE_PADDING}"
  "actor_rollout_ref.actor.loss_agg_mode=${PG_LOSS_AGG_MODE}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.use_dynamic_bsz=${ACTOR_USE_DYNAMIC_BSZ}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}"
  "critic.ppo_micro_batch_size_per_gpu=${CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  "critic.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.actor.use_kl_loss=${ACTOR_USE_KL_LOSS}"
  "actor_rollout_ref.actor.clip_ratio_low=${ACTOR_CLIP_RATIO_LOW}"
  "actor_rollout_ref.actor.clip_ratio_high=${ACTOR_CLIP_RATIO_HIGH}"
  "actor_rollout_ref.actor.kl_loss_coef=${ACTOR_KL_LOSS_COEF}"
  "actor_rollout_ref.actor.kl_loss_type=${ACTOR_KL_LOSS_TYPE}"
  "actor_rollout_ref.actor.use_torch_compile=${ACTOR_USE_TORCH_COMPILE}"
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE}"
  "++actor_rollout_ref.actor.entropy_from_logits_with_chunking=${ACTOR_ENTROPY_CHUNKING}"
  "++actor_rollout_ref.actor.entropy_checkpointing=${ACTOR_ENTROPY_CHECKPOINTING}"
  "++actor_rollout_ref.actor.entropy_chunk_size=${ACTOR_ENTROPY_CHUNK_SIZE}"
  "++actor_rollout_ref.actor.opd_loss_enabled=${USE_OPSD_KL}"
  "++actor_rollout_ref.actor.opd_loss_coef=${OPSD_LOSS_COEF}"
  "++actor_rollout_ref.actor.opd_policy_loss_mode=${OPSD_POLICY_LOSS_MODE}"
  "++actor_rollout_ref.actor.opd_norm_to_one_for_kl=${OPSD_NORM_TO_ONE_FOR_KL}"
  "++actor_rollout_ref.actor.opd_topk_k=${OPSD_TOPK_K}"
  "++actor_rollout_ref.actor.opsd_sdar_gate=${OPSD_SDAR_GATE}"
  "++actor_rollout_ref.actor.sdar_gate_beta=${SDAR_GATE_BETA}"
  "++actor_rollout_ref.actor.rlsd_max_packed_context_tokens=${RLSD_ACTOR_MAX_PACKED_CONTEXT_TOKENS}"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.fsdp_config.param_offload=True"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}"
  "actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
  "++agent.engine_args.max_active_trajectories=${ROLLOUT_ACTIVE_TRAJECTORIES}"
  "++agent.engine_args.context_window_safety_margin=${ROLLOUT_CONTEXT_WINDOW_SAFETY_MARGIN}"
  "++agent.engine_args.per_step_max_tokens=${ROLLOUT_PER_STEP_MAX_TOKENS}"
  "++agent.engine_args.soft_overlong_reward_factor=${SOFT_OVERLONG_REWARD_FACTOR}"
  "++agent.engine_args.stop_after_function_call=${ROLLOUT_STOP_AFTER_FUNCTION_CALL}"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=async"
  "++actor_rollout_ref.rollout.prometheus.served_model_name=${MODEL_SERVE_NAME}"
  "actor_rollout_ref.rollout.load_format=auto"
  "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}"
  "actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE}"
  "++actor_rollout_ref.rollout.enable_sleep_mode=${ROLLOUT_ENABLE_SLEEP_MODE}"
  "++actor_rollout_ref.rollout.enable_chunked_prefill=${ROLLOUT_ENABLE_CHUNKED_PREFILL}"
  "++actor_rollout_ref.rollout.enable_prefix_caching=${ROLLOUT_ENABLE_PREFIX_CACHING}"
  "++actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=${ROLLOUT_BLOCK_SIZE}"
  "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.val_kwargs.n=${ROLLOUT_VAL_N}"
  "actor_rollout_ref.rollout.val_kwargs.temperature=0"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${LOGPROB_USE_DYNAMIC_BSZ}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOGPROB_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${LOGPROB_USE_DYNAMIC_BSZ}"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOGPROB_MAX_TOKEN_LEN_PER_GPU}"
  "++actor_rollout_ref.ref.entropy_from_logits_with_chunking=${ACTOR_ENTROPY_CHUNKING}"
  "++actor_rollout_ref.ref.entropy_checkpointing=${ACTOR_ENTROPY_CHECKPOINTING}"
  "++actor_rollout_ref.ref.entropy_chunk_size=${ACTOR_ENTROPY_CHUNK_SIZE}"
  "critic.model.fsdp_config.param_offload=${CRITIC_PARAM_OFFLOAD}"
  "critic.model.fsdp_config.optimizer_offload=${CRITIC_OPTIMIZER_OFFLOAD}"
  "critic.model.enable_activation_offload=${CRITIC_ENABLE_ACTIVATION_OFFLOAD}"
  "actor_rollout_ref.actor.entropy_coeff=${ACTOR_ENTROPY_COEFF}"
  "algorithm.kl_ctrl.kl_coef=0.0"
  "algorithm.mask_truncated_samples=False"
  "algorithm.clip_advantages=False"
  "++algorithm.use_rlsd=${USE_RLSD}"
  "++algorithm.use_opsd_kl=${USE_OPSD_KL}"
  "++algorithm.rlsd_lambda=${RLSD_LAMBDA}"
  "++algorithm.rlsd_lambda_init=${RLSD_LAMBDA_INIT}"
  "++algorithm.rlsd_lambda_final=${RLSD_LAMBDA_FINAL}"
  "++algorithm.rlsd_lambda_decay_steps=${RLSD_LAMBDA_DECAY_STEPS}"
  "++algorithm.rlsd_weight_clip_eps=${RLSD_WEIGHT_CLIP_EPS}"
  "++algorithm.rlsd_delta_clip=${RLSD_DELTA_CLIP}"
  "++algorithm.rlsd_use_old_student_logp=${RLSD_USE_OLD_STUDENT_LOGP}"
  "++algorithm.rlsd_use_rollout_logp_as_student_prior=${RLSD_USE_ROLLOUT_LOGP_AS_STUDENT_PRIOR}"
  "++algorithm.rlsd_teacher_no_grad=${RLSD_TEACHER_NO_GRAD}"
  "++algorithm.rlsd_log_stats=${RLSD_LOG_STATS}"
  "++opd.enabled=${OPSD_ENABLE}"
  "++opd.teacher_source=${OPSD_TEACHER_SOURCE}"
  "++opd.ref_load_checkpoint_from_resume=${OPSD_REF_LOAD_CHECKPOINT_FROM_RESUME}"
  "++opd.teacher_base_url=${OPSD_TEACHER_BASE_URL}"
  "++opd.teacher_model_name=${OPSD_TEACHER_MODEL}"
  "++opd.topk_k=${OPSD_TOPK_K}"
  "++opd.max_tokens_per_sample=${OPSD_MAX_TOKENS_PER_SAMPLE}"
  "++opd.max_response_tokens_for_training=${OPSD_MAX_RESPONSE_TOKENS_FOR_TRAINING}"
  "++opd.route_success_grpo_failure_opsd=${OPSD_ROUTE_SUCCESS_GRPO_FAILURE_OPSD}"
  "++opd.teacher_timeout_sec=${OPSD_TEACHER_TIMEOUT_SEC}"
  "++opd.fail_open=${OPSD_FAIL_OPEN}"
  "trainer.critic_warmup=0"
  "trainer.logger=${TRAIN_LOGGERS_HYDRA}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.default_local_dir=${DEFAULT_LOCAL_DIR}"
  "trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}"
  "trainer.val_before_train=False"
  "trainer.n_gpus_per_node=${TRAIN_N_GPUS_PER_NODE}"
  "++trainer.n_training_gpus_per_node=${TRAINING_PARALLEL_N_GPUS_PER_NODE}"
  "trainer.nnodes=1"
  "trainer.device=npu"
  "trainer.save_freq=10"
  "trainer.test_freq=-1"
  "trainer.default_hdfs_dir=null"
  "trainer.resume_mode=${TRAINER_RESUME_MODE}"
  "trainer.resume_from_path=${TRAINER_RESUME_FROM_PATH}"
  "++trainer.resume_load_dataloader=${TRAINER_RESUME_LOAD_DATALOADER}"
  "++trainer.stage_global_step_offset=${TRAINER_STAGE_GLOBAL_STEP_OFFSET}"
  "env.name=swe_arsenal"
  "+env.env_args.delete_image=False"
  "+env.env_args.scaffold=${SWE_SCAFFOLD}"
  "+env.env_args.backend=docker"
  "agent.name=sweagent"
  "agent.max_steps=${AGENT_MAX_STEPS}"
  "++agent.max_response_tokens_for_training=${AGENT_MAX_RESPONSE_TOKENS_FOR_TRAINING}"
  "agent.overlong_filter=True"
  "agent.trajectory_timeout=${AGENT_TRAJECTORY_TIMEOUT}"
  "agent.async_engine=True"
  "++agent.pipeline_train_after_full_rollout=${PIPELINE_TRAIN_AFTER_FULL_ROLLOUT}"
  "++agent.pipeline_length_bucket_rollouts=${PIPELINE_LENGTH_BUCKET_ROLLOUTS}"
  "++agent.exploration_tool_reward_coef=${EXPLORATION_TOOL_REWARD_COEF}"
  "++agent.exploration_tool_reward_max=${EXPLORATION_TOOL_REWARD_MAX}"
  "++agent.exploration_tool_min_calls=${EXPLORATION_TOOL_MIN_CALLS}"
  "++agent.exploration_tool_missing_penalty=${EXPLORATION_TOOL_MISSING_PENALTY}"
  "++agent.exploration_no_tool_call_penalty=${EXPLORATION_NO_TOOL_CALL_PENALTY}"
  "++agent.give_up_empty_patch_penalty=${GIVE_UP_EMPTY_PATCH_PENALTY}"
  "++agent.exploration_reward_clamp_min=${EXPLORATION_REWARD_CLAMP_MIN}"
  "++agent.exploration_reward_clamp_max=${EXPLORATION_REWARD_CLAMP_MAX}"
  "+agent.agent_args.scaffold=${SWE_SCAFFOLD}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
)

if [[ -n "${TRAINER_TOTAL_TRAINING_STEPS}" ]]; then
  cmd+=("trainer.total_training_steps=${TRAINER_TOTAL_TRAINING_STEPS}")
fi

if bool_is_true "${YARN_ENABLE}"; then
  yarn_rope_scaling_hydra="{rope_type:yarn,factor:${YARN_FACTOR},original_max_position_embeddings:${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}}"
  yarn_hf_overrides_hydra="{max_position_embeddings:${YARN_MAX_POSITION_EMBEDDINGS},rope_scaling:${yarn_rope_scaling_hydra}}"
  # The model override is consumed by FSDP actor/ref initialization.  The
  # explicit vLLM hf_overrides entry keeps the serving engine on the same HF
  # config; otherwise vLLM would reload the unscaled config.json from disk.
  cmd+=(
    "++actor_rollout_ref.model.override_config.max_position_embeddings=${YARN_MAX_POSITION_EMBEDDINGS}"
    "++actor_rollout_ref.model.override_config.rope_scaling=${yarn_rope_scaling_hydra}"
    "++actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides=${yarn_hf_overrides_hydra}"
  )
fi

if [[ -n "${ROLLOUT_FIRST_STEP_MAX_TOKENS}" ]]; then
  cmd+=("++agent.engine_args.first_step_max_tokens=${ROLLOUT_FIRST_STEP_MAX_TOKENS}")
fi
if [[ -n "${ROLLOUT_INVALID_TOOL_CALL_MAX_TOKENS}" ]]; then
  cmd+=("++agent.engine_args.invalid_tool_call_max_tokens=${ROLLOUT_INVALID_TOOL_CALL_MAX_TOKENS}")
fi
if [[ -n "${ROLLOUT_INVALID_TOOL_CALL_RETRY_LIMIT}" ]]; then
  cmd+=("++agent.engine_args.invalid_tool_call_retry_limit=${ROLLOUT_INVALID_TOOL_CALL_RETRY_LIMIT}")
fi

if [[ "${TRAINER_LAYOUT_MODE}" == "pipeline" ]]; then
  cmd+=(
    "++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${CHECKPOINT_ENGINE_BUCKET_MB}"
    "++actor_rollout_ref.rollout.checkpoint_engine.weight_transfer_dtype=${CHECKPOINT_ENGINE_WEIGHT_TRANSFER_DTYPE}"
  )
fi

if [[ -n "${ROLLOUT_CUDAGRAPH_CAPTURE_SIZES}" ]]; then
  cmd+=("+actor_rollout_ref.rollout.cudagraph_capture_sizes=${ROLLOUT_CUDAGRAPH_CAPTURE_SIZES}")
fi

if bool_is_true "${ROLLOUT_ASCEND_SCHEDULER_ENABLE}"; then
  if [[ -z "${ROLLOUT_VLLM_ADDITIONAL_CONFIG_JSON:-}" ]]; then
    ascend_scheduler_config_json="$(printf '{"enabled":true,"enable_chunked_prefill":%s' \
      "$(bool_to_json "${ROLLOUT_ASCEND_SCHEDULER_ENABLE_CHUNKED_PREFILL}")")"
    if [[ -n "${ROLLOUT_ASCEND_SCHEDULER_MAX_LONG_PARTIAL_PREFILLS}" ]]; then
      ascend_scheduler_config_json="${ascend_scheduler_config_json},\"max_long_partial_prefills\":${ROLLOUT_ASCEND_SCHEDULER_MAX_LONG_PARTIAL_PREFILLS}"
    fi
    if [[ -n "${ROLLOUT_ASCEND_SCHEDULER_LONG_PREFILL_TOKEN_THRESHOLD}" ]]; then
      ascend_scheduler_config_json="${ascend_scheduler_config_json},\"long_prefill_token_threshold\":${ROLLOUT_ASCEND_SCHEDULER_LONG_PREFILL_TOKEN_THRESHOLD}"
    fi
    ascend_scheduler_config_json="${ascend_scheduler_config_json}}"
    ROLLOUT_VLLM_ADDITIONAL_CONFIG_JSON="$(printf '{"ascend_scheduler_config":%s,"refresh":%s}' \
      "${ascend_scheduler_config_json}" \
      "$(bool_to_json "${ROLLOUT_ASCEND_SCHEDULER_REFRESH}")")"
  fi
  echo "ROLLOUT_VLLM_ADDITIONAL_CONFIG_JSON: ${ROLLOUT_VLLM_ADDITIONAL_CONFIG_JSON}"
  cmd+=("++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config='${ROLLOUT_VLLM_ADDITIONAL_CONFIG_JSON}'")
elif [[ -n "${ROLLOUT_VLLM_ADDITIONAL_CONFIG_JSON:-}" ]]; then
  echo "ROLLOUT_VLLM_ADDITIONAL_CONFIG_JSON: ${ROLLOUT_VLLM_ADDITIONAL_CONFIG_JSON}"
  cmd+=("++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config='${ROLLOUT_VLLM_ADDITIONAL_CONFIG_JSON}'")
fi

if [[ "${ACTOR_OPTIMIZER_FOREACH}" != "auto" ]]; then
  cmd+=("+actor_rollout_ref.actor.optim.override_optimizer_config={foreach:${ACTOR_OPTIMIZER_FOREACH}}")
fi
if [[ -n "${DISTRIBUTED_TIMEOUT_SECONDS}" ]]; then
  cmd+=("++actor_rollout_ref.nccl_timeout=${DISTRIBUTED_TIMEOUT_SECONDS}")
fi

"${cmd[@]}" &
TRAIN_CMD_PID=$!

set +e
wait "${TRAIN_CMD_PID}"
run_rc=$?
set -e

TRAIN_CMD_PID=""
exit "${run_rc}"
