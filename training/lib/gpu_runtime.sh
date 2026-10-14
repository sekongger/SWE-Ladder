#!/bin/bash

safe_source_gpu() {
  local file="$1"
  if [[ -f "${file}" ]]; then
    set +u
    # shellcheck source=/dev/null
    source "${file}"
    set -u
  fi
}

prepend_colon_path_if_exists_gpu() {
  local var_name="$1"
  local candidate="$2"
  [[ -d "${candidate}" ]] || return 0
  local current="${!var_name:-}"
  if [[ -z "${current}" ]]; then
    export "${var_name}=${candidate}"
    return 0
  fi
  case ":${current}:" in
    *":${candidate}:"*) ;;
    *) export "${var_name}=${candidate}:${current}" ;;
  esac
}

setup_gpu_runtime() {
  local workspace_root="$1"
  local venv_path="${GPU_RL_VENV:-${workspace_root}/.venv-gpu-rl}"

  unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES ASCEND_HOME ASCEND_TOOLKIT_HOME ASCEND_TOOLKIT_VERSION_DIR || true
  unset VLLM_TARGET_DEVICE VLLM_PLUGINS VLLM_VERSION || true
  export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
  export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
  export RLLM_ALLOW_USER_SITE="${RLLM_ALLOW_USER_SITE:-0}"

  if [[ -f "${venv_path}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${venv_path}/bin/activate"
  else
    echo "[gpu-runtime] virtualenv not found: ${venv_path}" >&2
    echo "[gpu-runtime] run setup_rl_gpu_env.sh before launching training." >&2
  fi

  export PYTHONPATH="${workspace_root}/DeepSWE_RL/rllm:${workspace_root}/DeepSWE_RL/rllm/rllm:${PYTHONPATH:-}"
  export RLLM_R2EGYM_SRC="${RLLM_R2EGYM_SRC:-${workspace_root}/R2E-Gym-generic-sandbox/src}"
  export R2EGYM_GENERIC_SANDBOX_IMAGE="${R2EGYM_GENERIC_SANDBOX_IMAGE:-skongger/r2egym-scaffold-base:latest}"

  export R2EGYM_DOCKER_MODE="${R2EGYM_DOCKER_MODE:-local}"
  if [[ "${R2EGYM_DOCKER_MODE}" == "local" ]]; then
    export R2EGYM_LOCAL_DOCKER_HOST="${R2EGYM_LOCAL_DOCKER_HOST:-unix:///var/run/docker.sock}"
    unset R2EGYM_DOCKER_HOST R2EGYM_REMOTE_DOCKER_PORT DOCKER_HOST || true
  else
    export R2EGYM_DOCKER_HOST="${R2EGYM_DOCKER_HOST:-tcp://127.0.0.1:2375}"
    export R2EGYM_REMOTE_DOCKER_PORT="${R2EGYM_REMOTE_DOCKER_PORT:-2375}"
    export DOCKER_HOST="${DOCKER_HOST:-${R2EGYM_DOCKER_HOST}}"
  fi

  export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-0}"
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
  export VLLM_USE_V1="${VLLM_USE_V1:-1}"
  if [[ -z "${VLLM_ATTENTION_BACKEND:-}" ]]; then
    if command -v python3 >/dev/null 2>&1 && python3 - <<'PY' >/dev/null 2>&1
import flash_attn  # noqa: F401
PY
    then
      export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
    else
      export VLLM_ATTENTION_BACKEND="XFORMERS"
    fi
  else
    export VLLM_ATTENTION_BACKEND
  fi

  prepend_colon_path_if_exists_gpu LD_LIBRARY_PATH "/usr/local/cuda/lib64"
}

print_gpu_runtime_summary() {
  echo "Python: $(command -v python3 || true)"
  python3 -V || true
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "R2EGYM_DOCKER_MODE: ${R2EGYM_DOCKER_MODE:-<unset>}"
  echo "R2EGYM_DOCKER_HOST: ${R2EGYM_DOCKER_HOST:-${R2EGYM_LOCAL_DOCKER_HOST:-<unset>}}"
  echo "RLLM_R2EGYM_SRC: ${RLLM_R2EGYM_SRC:-<unset>}"
  echo "R2EGYM_GENERIC_SANDBOX_IMAGE: ${R2EGYM_GENERIC_SANDBOX_IMAGE:-<unset>}"
  echo "VLLM_ATTENTION_BACKEND: ${VLLM_ATTENTION_BACKEND:-<unset>}"
}
