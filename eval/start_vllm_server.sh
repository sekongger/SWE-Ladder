#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${REPO_ROOT}}"
# Ascend toolkit discovery helper. Ships in this repo; override if you keep it elsewhere.
ASCEND_RUNTIME_HELPER="${ASCEND_RUNTIME_HELPER:-${REPO_ROOT}/training/lib/ascend_runtime.sh}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEFAULT_BACKEND="${BACKEND:-gpu}"
DEFAULT_MODEL_PATH="${MODEL_PATH:-}"
DEFAULT_SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-swe-ladder-policy}"
DEFAULT_HOST="${HOST:-0.0.0.0}"
DEFAULT_PORT="${PORT:-8110}"
DEFAULT_TP="${TENSOR_PARALLEL_SIZE:-1}"
DEFAULT_MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
DEFAULT_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DEFAULT_LOG_DIR="${LOG_DIR:-${WORKSPACE_ROOT}/output/vllm_server}"
DEFAULT_WAIT_SECONDS="${WAIT_SECONDS:-600}"
DEFAULT_DTYPE="${DTYPE:-auto}"
DEFAULT_MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
DEFAULT_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
DEFAULT_MAX_LOGPROBS="${MAX_LOGPROBS:-}"
DEFAULT_ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
DEFAULT_LINES="${LINES:-200}"

usage() {
  cat <<'USAGE'
Usage:
  ./start_vllm_server.sh <start|stop|status|logs> [options]

Subcommands:
  start    Start a vLLM OpenAI-compatible server
  stop     Stop the server for the selected backend/port
  status   Show process and HTTP readiness state
  logs     Tail the server log

Options:
  --backend <ascend|gpu>         Backend to use (default: gpu)
  --model-path <path>            HuggingFace model directory (required)
  --served-model-name <name>     Exposed model name (default: swe-ladder-policy)
  --host <host>                  Listen host (default: 0.0.0.0)
  --port <port>                  Listen port (default: 8000)
  --tp <int>                     Tensor parallel size (default: 1)
  --max-model-len <int>          Max model length (default: 131072)
  --memory-utilization <float>   vLLM memory utilization (default: 0.8)
  --dtype <dtype>                vLLM dtype (default: auto)
  --enforce-eager                Disable ACL graph mode for better stability
  --max-num-seqs <int>           Limit concurrent sequences scheduled by vLLM
  --max-num-batched-tokens <int> Limit batched tokens scheduled by vLLM
  --max-logprobs <int>           Max logprobs/top-k values allowed by vLLM
  --no-enable-prefix-caching     Disable prefix caching for NPU stability tests
  --enable-prefix-caching        Explicitly enable prefix caching
  --devices <csv>                GPU/NPU device list
  --log-dir <path>               State and log directory (default: output/vllm_server)
  --wait-seconds <int>           Readiness timeout in seconds (default: 600)
  --lines <int>                  Lines for logs subcommand (default: 200)
  -h, --help                     Show this help

Examples:
  ./start_vllm_server.sh start --backend gpu --devices 0 --model-path /models/qwen3-4b
  ./start_vllm_server.sh start --backend ascend --devices 0 --port 8001 --model-path /models/qwen3-4b
  ./start_vllm_server.sh status --backend gpu --port 8000
  ./start_vllm_server.sh logs --backend ascend --port 8001
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

validate_backend() {
  case "${BACKEND}" in
    gpu|ascend) ;;
    *) die "unsupported backend: ${BACKEND}. Use ascend or gpu." ;;
  esac
}

ensure_model_exists() {
  [[ -n "${MODEL_PATH}" ]] || die "--model-path (or MODEL_PATH) is required"
  [[ -d "${MODEL_PATH}" || -f "${MODEL_PATH}" ]] || die "model path does not exist: ${MODEL_PATH}"
}

ensure_log_dir() {
  mkdir -p "${LOG_DIR}"
}

state_prefix() {
  printf '%s/vllm_%s_%s' "${LOG_DIR}" "${BACKEND}" "${PORT}"
}

log_file() {
  printf '%s.log' "$(state_prefix)"
}

pid_file() {
  printf '%s.pid' "$(state_prefix)"
}

meta_file() {
  printf '%s.meta' "$(state_prefix)"
}

models_url() {
  printf 'http://127.0.0.1:%s/v1/models' "${PORT}"
}

port_is_available() {
  "${PYTHON_BIN}" - "${HOST}" "${PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
if host in {"0.0.0.0", "::"}:
    host = ""
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

endpoint_ready() {
  curl -fsS "$(models_url)" >/dev/null 2>&1
}

running_pid() {
  local pid_path
  pid_path="$(pid_file)"
  [[ -f "${pid_path}" ]] || return 1
  local pid
  pid="$(cat "${pid_path}")"
  [[ -n "${pid}" ]] || return 1
  [[ -d "/proc/${pid}" ]] || return 1
  # kill -0 succeeds for zombies; they cannot serve requests or be restarted.
  [[ "$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null || true)" != "Z" ]] || return 1
  kill -0 "${pid}" >/dev/null 2>&1
}

write_metadata() {
  cat > "$(meta_file)" <<EOF_META
backend=${BACKEND}
model_path=${MODEL_PATH}
served_model_name=${SERVED_MODEL_NAME}
host=${HOST}
port=${PORT}
tensor_parallel_size=${TP}
max_model_len=${MAX_MODEL_LEN}
memory_utilization=${MEMORY_UTILIZATION}
dtype=${DTYPE}
enforce_eager=${ENFORCE_EAGER}
max_num_seqs=${MAX_NUM_SEQS}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
max_logprobs=${MAX_LOGPROBS}
prefix_caching=${PREFIX_CACHING_MODE}
pytorch_npu_alloc_conf=${PYTORCH_NPU_ALLOC_CONF:-}
hccl_op_expansion_mode=${HCCL_OP_EXPANSION_MODE:-}
devices=${DEVICES}
python=${PYTHON_BIN}
command=${CMD_STRING}
EOF_META
}

prepare_common_env() {
  require_command curl
  export PYTHONUNBUFFERED=1
  export NO_PROXY="127.0.0.1,localhost"
  export no_proxy="${NO_PROXY}"
}

prepare_backend_env() {
  prepare_common_env
  if [[ "${BACKEND}" == "gpu" ]]; then
    if [[ -n "${DEVICES}" ]]; then
      export CUDA_VISIBLE_DEVICES="${DEVICES}"
    fi
    return 0
  fi

  [[ -f "${ASCEND_RUNTIME_HELPER}" ]] || die "Ascend runtime helper not found: ${ASCEND_RUNTIME_HELPER}"
  export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:256}"
  export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
  # shellcheck source=/dev/null
  source "${ASCEND_RUNTIME_HELPER}"
  setup_ascend_runtime "${WORKSPACE_ROOT}" >/dev/null
  check_npu_runtime >/dev/null
  if [[ -n "${DEVICES}" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"
  fi
  export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend}"
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
}

start_server() {
  validate_backend
  ensure_model_exists
  ensure_log_dir

  if running_pid; then
    echo "Server already running: PID $(cat "$(pid_file)")"
    echo "Log: $(log_file)"
    return 0
  fi

  if endpoint_ready; then
    die "port ${PORT} is already serving a model"
  fi

  if ! port_is_available; then
    die "port ${PORT} is already in use"
  fi

  prepare_backend_env
  require_command setsid

  local -a cmd=(
    "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server
    --model "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --dtype "${DTYPE}"
    --tensor-parallel-size "${TP}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${MEMORY_UTILIZATION}"
  )
  if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    cmd+=(--enforce-eager)
  fi
  if [[ -n "${MAX_NUM_SEQS}" ]]; then
    cmd+=(--max-num-seqs "${MAX_NUM_SEQS}")
  fi
  if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
    cmd+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
  fi
  if [[ -n "${MAX_LOGPROBS}" ]]; then
    cmd+=(--max-logprobs "${MAX_LOGPROBS}")
  fi
  # Native function calling is opt-in; existing XML launchers retain their defaults.
  if [[ -n "${VLLM_TOOL_CALL_PARSER:-}" ]]; then
    cmd+=(--enable-auto-tool-choice --tool-call-parser "${VLLM_TOOL_CALL_PARSER}")
  fi
  case "${PREFIX_CACHING_MODE}" in
    enable)
      cmd+=(--enable-prefix-caching)
      ;;
    disable)
      cmd+=(--no-enable-prefix-caching)
      ;;
  esac

  printf -v CMD_STRING '%q ' "${cmd[@]}"
  write_metadata

  # setsid: API server + EngineCore share one process group so stop can
  # "kill -- -PGID" and release GPU (plain nohup left EngineCore orphans).
  setsid "${cmd[@]}" > "$(log_file)" 2>&1 &
  local server_pid=$!
  echo "${server_pid}" > "$(pid_file)"

  echo "Starting vLLM server"
  echo "Backend: ${BACKEND}"
  echo "Model: ${MODEL_PATH}"
  echo "Served model name: ${SERVED_MODEL_NAME}"
  echo "PID: ${server_pid}"
  echo "Log: $(log_file)"

  local ready=0
  for _ in $(seq 1 "${WAIT_SECONDS}"); do
    if endpoint_ready; then
      ready=1
      break
    fi
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "Server exited during startup. Tail of log:" >&2
      tail -n 200 "$(log_file)" >&2 || true
      rm -f "$(pid_file)"
      exit 1
    fi
    sleep 1
  done

  if [[ "${ready}" != "1" ]]; then
    echo "Server did not become ready within ${WAIT_SECONDS}s. Tail of log:" >&2
    tail -n 200 "$(log_file)" >&2 || true
    exit 1
  fi

  echo "Ready: $(models_url)"
}

stop_server() {
  ensure_log_dir
  local pid_path
  pid_path="$(pid_file)"
  if [[ ! -f "${pid_path}" ]]; then
    echo "No PID file found for ${BACKEND}:${PORT}"
    return 0
  fi

  local pid
  pid="$(cat "${pid_path}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    # Negative PID = whole process group (matches setsid start above).
    kill -- -"${pid}" >/dev/null 2>&1 || kill "${pid}" >/dev/null 2>&1 || true
    sleep 1
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -9 -- -"${pid}" >/dev/null 2>&1 || kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
    wait "${pid}" >/dev/null 2>&1 || true
    echo "Stopped process group ${pid}"
  else
    echo "PID file exists but process is not running: ${pid}"
  fi
  rm -f "${pid_path}"
}

status_server() {
  ensure_log_dir
  local pid_state="stopped"
  local http_state="not-ready"
  if running_pid; then
    pid_state="running (PID $(cat "$(pid_file)"))"
  fi
  if endpoint_ready; then
    http_state="ready"
  fi
  echo "Backend: ${BACKEND}"
  echo "Port: ${PORT}"
  echo "Process: ${pid_state}"
  echo "HTTP: ${http_state}"
  echo "Log: $(log_file)"
}

show_logs() {
  ensure_log_dir
  local current_log
  current_log="$(log_file)"
  [[ -f "${current_log}" ]] || die "log file does not exist: ${current_log}"
  tail -n "${LINES}" -f "${current_log}"
}

COMMAND="${1:-}"
if [[ -z "${COMMAND}" ]]; then
  usage
  exit 1
fi
shift || true

BACKEND="${DEFAULT_BACKEND}"
MODEL_PATH="${DEFAULT_MODEL_PATH}"
SERVED_MODEL_NAME="${DEFAULT_SERVED_MODEL_NAME}"
HOST="${DEFAULT_HOST}"
PORT="${DEFAULT_PORT}"
TP="${DEFAULT_TP}"
MAX_MODEL_LEN="${DEFAULT_MAX_MODEL_LEN}"
MEMORY_UTILIZATION="${DEFAULT_MEMORY_UTILIZATION}"
DTYPE="${DEFAULT_DTYPE}"
ENFORCE_EAGER="${DEFAULT_ENFORCE_EAGER}"
MAX_NUM_SEQS="${DEFAULT_MAX_NUM_SEQS}"
MAX_NUM_BATCHED_TOKENS="${DEFAULT_MAX_NUM_BATCHED_TOKENS}"
MAX_LOGPROBS="${DEFAULT_MAX_LOGPROBS}"
PREFIX_CACHING_MODE="default"
DEVICES="${DEVICES:-}"
LOG_DIR="${DEFAULT_LOG_DIR}"
WAIT_SECONDS="${DEFAULT_WAIT_SECONDS}"
LINES="${DEFAULT_LINES}"
CMD_STRING=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --served-model-name)
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --tp)
      TP="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --memory-utilization)
      MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --enforce-eager)
      ENFORCE_EAGER="1"
      shift
      ;;
    --max-num-seqs)
      MAX_NUM_SEQS="$2"
      shift 2
      ;;
    --max-num-batched-tokens)
      MAX_NUM_BATCHED_TOKENS="$2"
      shift 2
      ;;
    --max-logprobs)
      MAX_LOGPROBS="$2"
      shift 2
      ;;
    --no-enable-prefix-caching)
      PREFIX_CACHING_MODE="disable"
      shift
      ;;
    --enable-prefix-caching)
      PREFIX_CACHING_MODE="enable"
      shift
      ;;
    --devices)
      DEVICES="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --wait-seconds)
      WAIT_SECONDS="$2"
      shift 2
      ;;
    --lines)
      LINES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

case "${COMMAND}" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  status)
    status_server
    ;;
  logs)
    show_logs
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    die "unknown subcommand: ${COMMAND}"
    ;;
esac
