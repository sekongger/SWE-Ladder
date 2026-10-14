#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

DRY_RUN=0
MERGE_ONLY=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if [[ "${1:-}" == "--merge-only" ]]; then
  MERGE_ONLY=1
  shift
fi

CONFIG_PATH="${1:-${SCRIPT_DIR}/configs/vllm_ascend_opsd_location_gs110.yaml}"
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${WORKSPACE_ROOT}/${CONFIG_PATH}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: config yaml not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# Activate the serving env if conda is installed. CONDA_SH/CONDA_ENV override the
# discovery; with no conda at all we assume the interpreter on PATH is already correct.
CONDA_ENV="${CONDA_ENV:-swe-ladder}"
if [[ -z "${CONDA_SH:-}" ]]; then
  for candidate in \
    "${CONDA_EXE:+$(dirname -- "$(dirname -- "${CONDA_EXE}")")/etc/profile.d/conda.sh}" \
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

YAML_PYTHON="${YAML_PYTHON:-$(command -v python || command -v python3 || true)}"
if [[ -z "${YAML_PYTHON}" ]]; then
  echo "ERROR: no python interpreter found to parse the config" >&2
  exit 1
fi

eval "$(
  "${YAML_PYTHON}" - "${CONFIG_PATH}" "${WORKSPACE_ROOT}" <<'PY'
import shlex
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
workspace_root = sys.argv[2]
data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
runtime = data.get("runtime") or {}
env = data.get("env") or {}
server = data.get("server") or {}
checkpoint = data.get("checkpoint") or {}

for section_name, section in (
    ("runtime", runtime),
    ("env", env),
    ("server", server),
    ("checkpoint", checkpoint),
):
    if not isinstance(section, dict):
        raise SystemExit(f"{section_name} must be a mapping")

def expand(value):
    if value is None:
        return None
    return str(value).replace("${WORKSPACE_ROOT}", workspace_root)

for key, value in env.items():
    if value is None:
        continue
    if not key.replace("_", "").isalnum() or key[0].isdigit():
        raise SystemExit(f"invalid env key: {key!r}")
    print(f"export {key}={shlex.quote(expand(value))}")

for key, value in runtime.items():
    if value is not None:
        print(f"RUN_{key.upper()}={shlex.quote(expand(value))}")

for key, value in server.items():
    if value is not None:
        print(f"SERVER_{key.upper()}={shlex.quote(expand(value))}")

for key, value in checkpoint.items():
    if value is not None:
        print(f"CHECKPOINT_{key.upper()}={shlex.quote(expand(value))}")
PY
)"

LAUNCHER="${RUN_LAUNCHER:-start_vllm_server.sh}"
if [[ "${LAUNCHER}" != /* ]]; then
  LAUNCHER="${WORKSPACE_ROOT}/${LAUNCHER}"
fi
if [[ ! -x "${LAUNCHER}" ]]; then
  echo "ERROR: launcher is not executable: ${LAUNCHER}" >&2
  exit 1
fi

COMMAND="${RUN_COMMAND:-start}"
if [[ "${MERGE_ONLY}" == "1" ]]; then
  COMMAND="merge-only"
fi

MODEL_PATH="${SERVER_MODEL_PATH:-${CHECKPOINT_HF_OUTPUT_DIR:-}}"
SOURCE_ACTOR_DIR="${CHECKPOINT_SOURCE_ACTOR_DIR:-}"
MERGE_IF_MISSING="${CHECKPOINT_MERGE_IF_MISSING:-false}"

is_truthy() {
  case "${1:-}" in
    1|true|True|TRUE|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

has_hf_weights() {
  local model_dir="$1"
  compgen -G "${model_dir}/*.safetensors" >/dev/null || \
    compgen -G "${model_dir}/pytorch_model*.bin" >/dev/null
}

sync_hf_metadata_if_available() {
  if [[ -z "${MODEL_PATH}" || -z "${SOURCE_ACTOR_DIR}" ]]; then
    return 0
  fi
  local src="${SOURCE_ACTOR_DIR}/huggingface"
  if [[ ! -d "${src}" || ! -d "${MODEL_PATH}" ]]; then
    return 0
  fi

  echo "[vllm-yaml] syncing HF metadata from checkpoint: ${src}"
  local name
  for name in \
    added_tokens.json \
    chat_template.jinja \
    config.json \
    generation_config.json \
    merges.txt \
    special_tokens_map.json \
    tokenizer.json \
    tokenizer_config.json \
    vocab.json
  do
    if [[ -f "${src}/${name}" ]]; then
      cp -f "${src}/${name}" "${MODEL_PATH}/${name}"
    fi
  done
}

merge_checkpoint_if_needed() {
  if [[ -z "${MODEL_PATH}" || -z "${SOURCE_ACTOR_DIR}" ]]; then
    return 0
  fi
  if [[ -d "${MODEL_PATH}" ]] && has_hf_weights "${MODEL_PATH}"; then
    echo "[vllm-yaml] HF weights already present: ${MODEL_PATH}"
    sync_hf_metadata_if_available
    return 0
  fi
  if ! is_truthy "${MERGE_IF_MISSING}"; then
    return 0
  fi
  if [[ ! -d "${SOURCE_ACTOR_DIR}" ]]; then
    echo "ERROR: checkpoint source actor dir not found: ${SOURCE_ACTOR_DIR}" >&2
    exit 1
  fi

  echo "[vllm-yaml] merging FSDP checkpoint to HF model"
  echo "[vllm-yaml] source: ${SOURCE_ACTOR_DIR}"
  echo "[vllm-yaml] target: ${MODEL_PATH}"
  mkdir -p "${MODEL_PATH}"
  PYTHONNOUSERSITE=1 \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    DS_ACCELERATOR="${DS_ACCELERATOR:-cpu}" \
    PYTHONPATH="${WORKSPACE_ROOT}/DeepSWE_RL/rllm/verl:${PYTHONPATH:-}" \
    python "${WORKSPACE_ROOT}/DeepSWE_RL/rllm/verl/scripts/legacy_model_merger.py" merge \
      --backend fsdp \
      --local_dir "${SOURCE_ACTOR_DIR}" \
      --target_dir "${MODEL_PATH}"
  sync_hf_metadata_if_available
}

if [[ "${DRY_RUN}" != "1" ]]; then
  merge_checkpoint_if_needed
fi

if [[ "${COMMAND}" == "merge-only" ]]; then
  echo "[vllm-yaml] merge-only done."
  exit 0
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: server.model_path or checkpoint.hf_output_dir must be set." >&2
  exit 1
fi
if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: model path does not exist: ${MODEL_PATH}" >&2
    exit 1
  fi
  if ! has_hf_weights "${MODEL_PATH}"; then
    echo "ERROR: model path has no HF weight files: ${MODEL_PATH}" >&2
    echo "Run with checkpoint.merge_if_missing: true or run --merge-only first." >&2
    exit 1
  fi
fi

export PYTHON_BIN="${RUN_PYTHON_BIN:-${PYTHON_BIN:-python3}}"

cmd=(
  "${LAUNCHER}" "${COMMAND}"
  --backend "${SERVER_BACKEND:-ascend}"
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVER_SERVED_MODEL_NAME:-swe-opsd-location-gs110}"
  --host "${SERVER_HOST:-0.0.0.0}"
  --port "${SERVER_PORT:-8122}"
  --tp "${SERVER_TENSOR_PARALLEL_SIZE:-1}"
  --max-model-len "${SERVER_MAX_MODEL_LEN:-32768}"
  --memory-utilization "${SERVER_MEMORY_UTILIZATION:-0.80}"
  --dtype "${SERVER_DTYPE:-auto}"
  --log-dir "${RUN_LOG_DIR:-${WORKSPACE_ROOT}/output/vllm_server}"
  --wait-seconds "${RUN_WAIT_SECONDS:-600}"
)

if [[ -n "${SERVER_DEVICES:-}" ]]; then
  cmd+=(--devices "${SERVER_DEVICES}")
fi
if is_truthy "${SERVER_ENFORCE_EAGER:-false}"; then
  cmd+=(--enforce-eager)
fi
if [[ -n "${SERVER_MAX_NUM_SEQS:-}" ]]; then
  cmd+=(--max-num-seqs "${SERVER_MAX_NUM_SEQS}")
fi
if [[ -n "${SERVER_MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  cmd+=(--max-num-batched-tokens "${SERVER_MAX_NUM_BATCHED_TOKENS}")
fi
if [[ -n "${SERVER_MAX_LOGPROBS:-}" ]]; then
  cmd+=(--max-logprobs "${SERVER_MAX_LOGPROBS}")
fi
case "${SERVER_PREFIX_CACHING:-default}" in
  enable)
    cmd+=(--enable-prefix-caching)
    ;;
  disable)
    cmd+=(--no-enable-prefix-caching)
    ;;
  default|"")
    ;;
  *)
    echo "ERROR: server.prefix_caching must be one of: default, enable, disable" >&2
    exit 1
    ;;
esac

echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "LAUNCHER: ${LAUNCHER}"
echo "COMMAND: ${COMMAND}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "SERVED_MODEL_NAME: ${SERVER_SERVED_MODEL_NAME:-swe-opsd-location-gs110}"
echo "PORT: ${SERVER_PORT:-8122}"
echo "DEVICES: ${SERVER_DEVICES:-<unset>}"
echo "PYTHON_BIN: ${PYTHON_BIN}"
printf 'CMD:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1, skip vLLM launch."
  exit 0
fi

"${cmd[@]}"
