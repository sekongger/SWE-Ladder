#!/bin/bash

safe_source() {
  local file="$1"
  if [[ -f "${file}" ]]; then
    set +u
    # shellcheck source=/dev/null
    source "${file}"
    set -u
  fi
}

safe_source_first_existing() {
  local file
  for file in "$@"; do
    if [[ -f "${file}" ]]; then
      safe_source "${file}"
      return 0
    fi
  done
  return 1
}

sanitize_ld_library_path() {
  local raw="${LD_LIBRARY_PATH:-}"
  local cleaned=""
  local p
  IFS=':' read -r -a _paths <<< "${raw}"
  for p in "${_paths[@]}"; do
    [[ -z "${p}" ]] && continue
    if [[ "${p}" == *"/devlib/linux/"* ]]; then
      continue
    fi
    if [[ -z "${cleaned}" ]]; then
      cleaned="${p}"
    else
      cleaned="${cleaned}:${p}"
    fi
  done
  export LD_LIBRARY_PATH="${cleaned}"
}

filter_colon_path_var() {
  local var_name="$1"
  shift
  local raw="${!var_name:-}"
  local cleaned=""
  local entry pattern skip
  IFS=':' read -r -a _entries <<< "${raw}"
  for entry in "${_entries[@]}"; do
    [[ -z "${entry}" ]] && continue
    skip=0
    for pattern in "$@"; do
      if [[ "${entry}" == *"${pattern}"* ]]; then
        skip=1
        break
      fi
    done
    [[ "${skip}" == "1" ]] && continue
    if [[ -z "${cleaned}" ]]; then
      cleaned="${entry}"
    else
      cleaned="${cleaned}:${entry}"
    fi
  done
  export "${var_name}=${cleaned}"
}

prepend_colon_path_if_exists() {
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

prepend_ld_library_path_if_exists() {
  local candidate="$1"
  [[ -d "${candidate}" ]] || return 0
  if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${candidate}"
  else
    case ":${LD_LIBRARY_PATH}:" in
      *":${candidate}:"*) ;;
      *) export LD_LIBRARY_PATH="${candidate}:${LD_LIBRARY_PATH}" ;;
    esac
  fi
}

ensure_hccl_runtime() {
  local ascend_home="$1"

  # Some recent system installs make /usr/local/Ascend/ascend-toolkit/latest point to CANN 8.5,
  # whose lib64 no longer contains libhccl.so. torch_npu in this environment still requires the
  # classic libhccl.so shipped with the 8.3.RC1 toolkit. Prefer the active runtime if present and
  # otherwise fall back to the known-compatible 8.3.RC1 paths.
  if ldconfig -p 2>/dev/null | grep -q 'libhccl\.so'; then
    return 0
  fi

  local -a candidates=(
    "${ascend_home}/ascend-toolkit/latest/lib64"
    "${ascend_home}/ascend-toolkit/latest/hccl/lib64"
    "${ascend_home}/cann-8.5.0/lib64"
    "${ascend_home}/cann-8.5.0/aarch64-linux/lib64"
    "${ascend_home}/ascend-toolkit/8.3.RC1/lib64"
    "${ascend_home}/ascend-toolkit/8.3.RC1/hccl/lib64"
    "${ascend_home}/ascend-toolkit/8.3.RC1/aarch64-linux/lib64"
  )

  local path
  for path in "${candidates[@]}"; do
    if [[ -f "${path}/libhccl.so" ]]; then
      prepend_ld_library_path_if_exists "${path}"
      break
    fi
  done
}

resolve_ascend_toolkit_root() {
  local ascend_home="$1"
  local selector="${2:-auto}"
  local -a candidates=()
  local candidate

  if [[ -z "${selector}" || "${selector}" == "auto" ]]; then
    candidates+=(
      "${ascend_home}/ascend-toolkit/latest"
      "${ascend_home}/ascend-toolkit/8.3.RC1"
      "${ascend_home}/cann-8.5.0"
    )
  elif [[ "${selector}" == /* ]]; then
    candidates+=("${selector}")
  else
    candidates+=(
      "${ascend_home}/ascend-toolkit/${selector}"
      "${ascend_home}/${selector}"
      "${ascend_home}/cann-${selector}"
    )
    if [[ "${selector}" == "latest" ]]; then
      candidates+=("${ascend_home}/ascend-toolkit/latest")
    fi
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}" ]]; then
      (cd "${candidate}" 2>/dev/null && pwd -P)
      return 0
    fi
  done

  return 1
}

resolve_ascend_devlib_path() {
  local ascend_home="$1"
  local toolkit_root="$2"
  local cann_version_dir="${3:-}"
  local arch
  arch="$(uname -m)"

  local -a candidates=()
  local candidate
  if [[ -n "${toolkit_root}" ]]; then
    candidates+=("${toolkit_root}/${arch}-linux/devlib/linux/${arch}")
  fi
  if [[ -n "${cann_version_dir}" ]]; then
    if [[ "${cann_version_dir}" == /* ]]; then
      candidates+=("${cann_version_dir}/${arch}-linux/devlib/linux/${arch}")
    else
      candidates+=(
        "${ascend_home}/ascend-toolkit/${cann_version_dir}/${arch}-linux/devlib/linux/${arch}"
        "${ascend_home}/cann-${cann_version_dir}/${arch}-linux/devlib/linux/${arch}"
      )
    fi
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done

  return 1
}

setup_ascend_toolkit_env_manual() {
  local ascend_home="$1"
  local toolkit_selector="$2"
  local toolkit_root=""

  if ! toolkit_root="$(resolve_ascend_toolkit_root "${ascend_home}" "${toolkit_selector}")"; then
    echo "ERROR: Ascend toolkit directory not found for selector: ${toolkit_selector}" >&2
    if [[ -d "${ascend_home}/ascend-toolkit" ]]; then
      echo "INFO: available under ${ascend_home}/ascend-toolkit:" >&2
      ls -1 "${ascend_home}/ascend-toolkit" >&2 || true
    fi
    if [[ -d "${ascend_home}" ]]; then
      echo "INFO: available under ${ascend_home}:" >&2
      ls -1 "${ascend_home}" >&2 || true
    fi
    exit 1
  fi

  # Remove conflicting toolkit/runtime entries before pinning the selected stack.
  filter_colon_path_var PATH \
    "/usr/local/Ascend/cann-" \
    "/usr/local/Ascend/ascend-toolkit/latest" \
    "/Ascend/ascend-toolkit/latest" \
    "/Ascend/cann-" \
    "/Ascend/ascend-toolkit/"
  filter_colon_path_var LD_LIBRARY_PATH \
    "/usr/local/Ascend/cann-" \
    "/usr/local/Ascend/ascend-toolkit/latest" \
    "/Ascend/ascend-toolkit/latest" \
    "/Ascend/cann-" \
    "/Ascend/ascend-toolkit/"
  filter_colon_path_var PYTHONPATH \
    "/usr/local/Ascend/cann-" \
    "/usr/local/Ascend/ascend-toolkit/latest" \
    "/Ascend/ascend-toolkit/latest" \
    "/Ascend/cann-" \
    "/Ascend/ascend-toolkit/"

  prepend_colon_path_if_exists PATH "${toolkit_root}/opp_kernel/bin"
  prepend_colon_path_if_exists PATH "${toolkit_root}/opp/bin"
  prepend_colon_path_if_exists PATH "${toolkit_root}/hccl/bin"
  prepend_colon_path_if_exists PATH "${toolkit_root}/toolkit/bin"
  prepend_colon_path_if_exists PATH "${toolkit_root}/compiler/ccec_compiler/bin"
  prepend_colon_path_if_exists PATH "${toolkit_root}/compiler/bin"
  prepend_colon_path_if_exists PATH "${toolkit_root}/runtime/bin"
  prepend_colon_path_if_exists PATH "${toolkit_root}/aarch64-linux/bin"

  prepend_colon_path_if_exists LD_LIBRARY_PATH "${toolkit_root}/atc/lib64"
  prepend_colon_path_if_exists LD_LIBRARY_PATH "${toolkit_root}/fwkacllib/lib64"
  prepend_colon_path_if_exists LD_LIBRARY_PATH "${toolkit_root}/aarch64-linux/lib64"
  prepend_colon_path_if_exists LD_LIBRARY_PATH "${toolkit_root}/opp/lib64"
  prepend_colon_path_if_exists LD_LIBRARY_PATH "${toolkit_root}/toolkit/lib64"
  prepend_colon_path_if_exists LD_LIBRARY_PATH "${toolkit_root}/hccl/lib64"
  prepend_colon_path_if_exists LD_LIBRARY_PATH "${toolkit_root}/compiler/lib64/plugin/nnengine"
  prepend_colon_path_if_exists LD_LIBRARY_PATH "${toolkit_root}/compiler/lib64/plugin/opskernel"
  prepend_colon_path_if_exists LD_LIBRARY_PATH "${toolkit_root}/compiler/lib64"
  prepend_colon_path_if_exists LD_LIBRARY_PATH "${toolkit_root}/runtime/lib64"

  prepend_colon_path_if_exists PYTHONPATH "${toolkit_root}/pyACL/python/site-packages/acl"
  prepend_colon_path_if_exists PYTHONPATH "${toolkit_root}/python/site-packages"
  prepend_colon_path_if_exists PYTHONPATH "${toolkit_root}/toolkit/python/site-packages"
  prepend_colon_path_if_exists PYTHONPATH "${toolkit_root}/hccl/python/site-packages"
  prepend_colon_path_if_exists PYTHONPATH "${toolkit_root}/compiler/python/site-packages"

  export ASCEND_OPP_PATH="${toolkit_root}/opp"
  export ASCEND_AICPU_PATH="${toolkit_root}"
  export ASCEND_HOME_PATH="${toolkit_root}"
  export ASCEND_TOOLKIT_HOME="${toolkit_root}"
  export ASCEND_TOOLKIT_ROOT="${toolkit_root}"
  export ASCEND_INSTALL_PATH="${ascend_home}/"
}

ensure_local_wheels_installed() {
  local workspace_root="$1"
  local -a wheel_specs=(
    "swebench-fork-swegym:swebench_fork_swegym-2.0.13-py3-none-any.whl"
    "swebench-fork-swerebench:swebench_fork_swerebench-4.0.3-py3-none-any.whl"
  )
  if [[ "${RLLM_INCLUDE_SWESMITH:-0}" == "1" ]]; then
    wheel_specs+=("swesmith:swesmith-0.0.7-py3-none-any.whl")
  fi
  local spec pkg wheel_path
  for spec in "${wheel_specs[@]}"; do
    pkg="${spec%%:*}"
    wheel_path="${workspace_root}/${spec#*:}"
    if ! python3 -m pip show "${pkg}" >/dev/null 2>&1; then
      if [[ ! -f "${wheel_path}" ]]; then
        echo "ERROR: local wheel not found: ${wheel_path}"
        exit 1
      fi
      python3 -m pip install --no-deps "${wheel_path}"
    fi
  done
}

ensure_runtime_deps_installed() {
  local -a required_pkgs=(
    "simple-parsing"
    "libtmux"
    "bashlex"
    "litellm"
    "seaborn"
    "gpustat"
  )
  local -a missing_pkgs=()
  local pkg
  for pkg in "${required_pkgs[@]}"; do
    if ! python3 -m pip show "${pkg}" >/dev/null 2>&1; then
      missing_pkgs+=("${pkg}")
    fi
  done
  if [[ "${#missing_pkgs[@]}" -gt 0 ]]; then
    if ! python3 -m pip install "${missing_pkgs[@]}"; then
      (
        unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
        python3 -m pip install "${missing_pkgs[@]}"
      ) || {
        echo "ERROR: failed to install required runtime packages: ${missing_pkgs[*]}"
        exit 1
      }
    fi
  fi
}

ensure_pkg_resources_available() {
  if python3 - <<'PY' >/dev/null 2>&1
import pkg_resources  # noqa: F401
PY
  then
    return 0
  fi

  echo "pkg_resources is missing; pin setuptools<81 for torch_npu/torchair compatibility."
  if ! python3 -m pip install "setuptools<81"; then
    (
      unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
      python3 -m pip install "setuptools<81"
    ) || {
      echo "ERROR: failed to install setuptools<81, pkg_resources remains unavailable."
      exit 1
    }
  fi
}

check_rl_python_stack() {
  python3 - <<'PY'
import importlib
import sys

def version_of(name):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return f"IMPORT_FAIL: {exc!r}"

transformers_version = version_of("transformers")
tokenizers_version = version_of("tokenizers")

print(f"transformers: {transformers_version}")
print(f"tokenizers: {tokenizers_version}")

try:
    import transformers
    has_auto_video = hasattr(transformers, "AutoVideoProcessor")
except Exception as exc:
    print(f"[runtime] WARNING: failed to inspect transformers: {exc!r}")
    has_auto_video = False

if not has_auto_video:
    print("[runtime] WARNING: transformers in this env has no AutoVideoProcessor.")
    print("[runtime] WARNING: This usually means the shared Ascend env was modified by SFT/MindFormers")
    print("[runtime] WARNING: and is older than the vLLM code expected by RL.")
    print("[runtime] WARNING: A compatibility patch is applied in third_party/vllm,")
    print("[runtime] WARNING: but RL and SFT should ideally use separate envs.")
PY
}

configure_docker_runtime() {
  local workspace_root="$1"
  export HYDRA_FULL_ERROR=1
  export PYTHONPATH="${workspace_root}/DeepSWE_RL/rllm:${workspace_root}/DeepSWE_RL/rllm/rllm:${PYTHONPATH:-}"
  # Default to the long-lived remote x86_64 Docker host because the current NPU node is aarch64
  # and cannot execute the SWE x86_64 images locally. Users can still override these defaults.
  export R2EGYM_DOCKER_MODE="${R2EGYM_DOCKER_MODE:-remote}"
  export R2EGYM_KEEP_CONTAINERS="${R2EGYM_KEEP_CONTAINERS:-0}"
  if [[ "${R2EGYM_DOCKER_MODE}" == "local" ]]; then
    export R2EGYM_LOCAL_DOCKER_HOST="${R2EGYM_LOCAL_DOCKER_HOST:-unix:///var/run/docker.sock}"
    unset R2EGYM_DOCKER_HOST R2EGYM_REMOTE_DOCKER_PORT || true
  else
    export R2EGYM_DOCKER_HOST="${R2EGYM_DOCKER_HOST:-tcp://10.162.232.247:2375}"
    export R2EGYM_DOCKER_FALLBACK_HOSTS="${R2EGYM_DOCKER_FALLBACK_HOSTS:-tcp://10.162.232.247:2375}"
    export R2EGYM_REMOTE_DOCKER_PORT="${R2EGYM_REMOTE_DOCKER_PORT:-2375}"
    # SWE-Master currently uses plain Docker TCP endpoint by default.
    export R2EGYM_DOCKER_TLS_VERIFY="${R2EGYM_DOCKER_TLS_VERIFY:-0}"
    if [[ "${R2EGYM_DOCKER_TLS_VERIFY}" == "0" ]]; then
      unset DOCKER_TLS_VERIFY DOCKER_CERT_PATH R2EGYM_DOCKER_CERT_PATH || true
    fi
    # Windows Docker API over TCP can have transient timeout spikes; keep retrying before failing a trajectory.
    export R2EGYM_DOCKER_CONNECT_RETRIES="${R2EGYM_DOCKER_CONNECT_RETRIES:-6}"
    export R2EGYM_DOCKER_CONNECT_RETRY_SLEEP_S="${R2EGYM_DOCKER_CONNECT_RETRY_SLEEP_S:-10}"
  fi
}

remote_docker_ping_once() {
  local docker_host="$1"
  python3 - "${docker_host}" <<'PY'
import os
import sys
from urllib import request, error

docker_host = sys.argv[1]
timeout_s = int(os.environ.get("R2EGYM_DOCKER_STARTUP_TIMEOUT_S", "15"))

try:
    if docker_host.startswith("tcp://"):
        ping_url = "http://" + docker_host[len("tcp://") :].rstrip("/") + "/_ping"
    elif docker_host.startswith("http://") or docker_host.startswith("https://"):
        ping_url = docker_host.rstrip("/") + "/_ping"
    else:
        raise RuntimeError(f"Unsupported remote docker host format: {docker_host}")
    opener = request.build_opener(request.ProxyHandler({}))
    with opener.open(ping_url, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="ignore").strip()
    if body.upper() != "OK":
        raise RuntimeError(f"Unexpected Docker ping response from {ping_url}: {body!r}")
except (error.URLError, TimeoutError, OSError, RuntimeError):
    raise SystemExit(2)
PY
}

select_reachable_remote_docker_host() {
  local primary="$1"
  local fallback_raw="${R2EGYM_DOCKER_FALLBACK_HOSTS:-}"
  local normalized="${fallback_raw//,/ }"
  local -a candidates=()
  local candidate existing seen

  for candidate in "${primary}" ${normalized}; do
    [[ -n "${candidate}" ]] || continue
    seen=0
    for existing in "${candidates[@]}"; do
      if [[ "${existing}" == "${candidate}" ]]; then
        seen=1
        break
      fi
    done
    [[ "${seen}" == "1" ]] && continue
    candidates+=("${candidate}")
  done

  for candidate in "${candidates[@]}"; do
    if remote_docker_ping_once "${candidate}" >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

docker_host_port() {
  local docker_host="$1"
  local endpoint="${docker_host#*://}"
  endpoint="${endpoint%%/*}"
  endpoint="${endpoint##*@}"
  if [[ "${endpoint}" == *:* ]]; then
    printf '%s\n' "${endpoint##*:}"
  fi
}

check_remote_docker_connectivity() {
  if [[ "${R2EGYM_DOCKER_MODE:-local}" != "remote" ]]; then
    return 0
  fi
  if [[ "${R2EGYM_CHECK_REMOTE_DOCKER_AT_START:-1}" != "1" ]]; then
    return 0
  fi

  local docker_host="${R2EGYM_DOCKER_HOST:-}"
  if [[ -z "${docker_host}" ]]; then
    echo "[runtime] ERROR: remote docker mode is enabled but R2EGYM_DOCKER_HOST is empty."
    exit 1
  fi

  local retries="${R2EGYM_DOCKER_STARTUP_RETRIES:-3}"
  local sleep_s="${R2EGYM_DOCKER_STARTUP_RETRY_SLEEP_S:-5}"
  local attempt selected selected_port
  if (( retries < 1 )); then
    retries=1
  fi

  echo "[runtime] checking remote docker connectivity: ${docker_host}"
  if [[ -n "${R2EGYM_DOCKER_FALLBACK_HOSTS:-}" ]]; then
    echo "[runtime] remote docker fallback hosts: ${R2EGYM_DOCKER_FALLBACK_HOSTS}"
  fi
  for (( attempt = 1; attempt <= retries; attempt++ )); do
    if selected="$(select_reachable_remote_docker_host "${docker_host}")"; then
      if [[ "${selected}" != "${docker_host}" ]]; then
        echo "[runtime] remote docker primary unavailable; using fallback: ${selected}"
      fi
      export R2EGYM_DOCKER_HOST="${selected}"
      selected_port="$(docker_host_port "${selected}")"
      if [[ -n "${selected_port}" ]]; then
        export R2EGYM_REMOTE_DOCKER_PORT="${selected_port}"
      fi
      echo "[runtime] remote docker ping ok on attempt ${attempt}/${retries}: ${R2EGYM_DOCKER_HOST}"
      return 0
    fi
    echo "[runtime] remote docker ping failed on attempt ${attempt}/${retries}: tried ${docker_host}${R2EGYM_DOCKER_FALLBACK_HOSTS:+ and ${R2EGYM_DOCKER_FALLBACK_HOSTS}}"
    if (( attempt < retries )); then
      sleep "${sleep_s}"
    fi
  done

  echo "[runtime] ERROR: remote docker is unreachable; aborting before training starts."
  exit 2
}

clear_proxy_env() {
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
}

setup_ascend_runtime() {
  local workspace_root="$1"

  # Default to the currently available 8 NPUs on this node. Users can still override this.
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
  export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
  export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-63000-63200}"
  unset CUDA_VISIBLE_DEVICES || true
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${LIBRARY_PATH:-}"

  export VLLM_VERSION="${VLLM_VERSION:-0.11.0}"
  export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend}"
  export VLLM_TARGET_DEVICE=npu
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
  export VLLM_USE_V1=1
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
  # Smaller split size reduces fragmentation spikes for long-sequence PPO/GRPO runs.
  export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:64}"
  export RLLM_DISABLE_RUNTIME_ENV_PIP="${RLLM_DISABLE_RUNTIME_ENV_PIP:-1}"
  export RLLM_INHERIT_PROXY_ENV="${RLLM_INHERIT_PROXY_ENV:-0}"
  if [[ "$(uname -m)" == "aarch64" && "${RLLM_DISABLE_RUNTIME_ENV_PIP}" == "0" && "${ALLOW_RUNTIME_ENV_PIP_ON_ARM:-0}" != "1" ]]; then
    echo "aarch64 detected: force RLLM_DISABLE_RUNTIME_ENV_PIP=1. Set ALLOW_RUNTIME_ENV_PIP_ON_ARM=1 to bypass."
    export RLLM_DISABLE_RUNTIME_ENV_PIP=1
  fi
  if [[ "${RLLM_INHERIT_PROXY_ENV}" != "1" ]]; then
    clear_proxy_env
  fi

  export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${USER}}"
  mkdir -p "${RAY_TMPDIR}"

  local ascend_home="${ASCEND_HOME:-/usr/local/Ascend}"
  local preferred_ascend_version="${ASCEND_TOOLKIT_VERSION_DIR:-8.3.RC1}"
  setup_ascend_toolkit_env_manual "${ascend_home}" "${preferred_ascend_version}"
  safe_source_first_existing \
    "${ASCEND_TOOLKIT_ROOT:-}/set_env.sh" \
    "${ASCEND_TOOLKIT_ROOT:-}/aarch64-linux/script/set_env.sh" \
    "${ASCEND_TOOLKIT_ROOT:-}/x86_64-linux/script/set_env.sh"
  if [[ -f "${ascend_home}/nnal/atb/set_env.sh" ]]; then
    safe_source "${ascend_home}/nnal/atb/set_env.sh"
  elif [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/Ascend/nnal/atb/set_env.sh" ]]; then
    safe_source "${CONDA_PREFIX}/Ascend/nnal/atb/set_env.sh"
  fi

  sanitize_ld_library_path
  ensure_hccl_runtime "${ascend_home}"
  if [[ "${ENABLE_ASCEND_DEVLIB:-0}" == "1" ]]; then
    local cann_version_dir="${CANN_VERSION_DIR:-}"
    local ascend_devlib=""
    if ascend_devlib="$(resolve_ascend_devlib_path "${ascend_home}" "${ASCEND_TOOLKIT_ROOT:-}" "${cann_version_dir}")"; then
      export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${ascend_devlib}"
    else
      echo "WARNING: ENABLE_ASCEND_DEVLIB=1 but no devlib path found." >&2
    fi
  fi

  ensure_local_wheels_installed "${workspace_root}"
  ensure_pkg_resources_available
  ensure_runtime_deps_installed
  check_rl_python_stack

  # Do not synthesize a fake W&B key here. If WANDB_API_KEY is unset,
  # wandb can use credentials from `wandb login`; exporting a placeholder
  # causes online runs to fail with 401.
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
  fi
  export WANDB_MODE="${WANDB_MODE:-online}"

  configure_docker_runtime "${workspace_root}"
  check_remote_docker_connectivity

  echo "RAY_TMPDIR: ${RAY_TMPDIR}"
  echo "ASCEND_RT_VISIBLE_DEVICES: ${ASCEND_RT_VISIBLE_DEVICES}"
  echo "PYTORCH_NPU_ALLOC_CONF: ${PYTORCH_NPU_ALLOC_CONF}"
  echo "R2EGYM_DOCKER_MODE: ${R2EGYM_DOCKER_MODE}"
  echo "R2EGYM_DOCKER_HOST: ${R2EGYM_DOCKER_HOST:-${R2EGYM_LOCAL_DOCKER_HOST:-<auto>}}"
  echo "R2EGYM_DOCKER_TLS_VERIFY: ${R2EGYM_DOCKER_TLS_VERIFY:-0}"
  echo "R2EGYM_DOCKER_CONNECT_RETRIES: ${R2EGYM_DOCKER_CONNECT_RETRIES:-1}"
  echo "R2EGYM_DOCKER_CONNECT_RETRY_SLEEP_S: ${R2EGYM_DOCKER_CONNECT_RETRY_SLEEP_S:-0}"
  echo "R2EGYM_KEEP_CONTAINERS: ${R2EGYM_KEEP_CONTAINERS}"
}

check_npu_runtime() {
  python3 - <<'PY'
import torch
import torch_npu
print('torch:', torch.__version__)
print('torch_npu import: OK')
print('npu available:', torch.npu.is_available())
PY
}

check_npu_headroom() {
  if [[ "${ALLOW_BUSY_NPUS:-0}" == "1" ]]; then
    echo "[runtime] skip NPU occupancy precheck because ALLOW_BUSY_NPUS=1"
    return 0
  fi

  python3 - <<'PY'
import os
import subprocess
import sys

visible = {
    int(x.strip())
    for x in (
        ",".join(
            value
            for value in (
                os.environ.get("RLLM_ROLLOUT_NPU_IDS", ""),
                os.environ.get("RLLM_TRAINING_NPU_IDS", ""),
            )
            if value
        )
        or os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    ).split(",")
    if x.strip()
}
threshold_mb = int(os.environ.get("NPU_BUSY_PROCESS_THRESHOLD_MB", "2048"))

try:
    text = subprocess.check_output(["npu-smi", "info"], text=True, stderr=subprocess.STDOUT)
except Exception as exc:
    print(f"[runtime] WARNING: failed to query npu-smi info: {exc}")
    raise SystemExit(0)

hits = []
for raw_line in text.splitlines():
    line = raw_line.strip()
    if not line.startswith("|"):
        continue
    parts = [part.strip() for part in line.split("|")[1:-1]]
    if len(parts) != 4:
        continue
    first = parts[0].split()
    if len(first) < 1 or not first[0].isdigit():
        continue
    if not parts[1].isdigit():
        continue
    mem_token = parts[3].split()
    if not mem_token or not mem_token[0].isdigit():
        continue

    npu_id = int(first[0])
    pid = int(parts[1])
    proc_name = parts[2]
    mem_mb = int(mem_token[0])

    if visible and npu_id not in visible:
        continue
    if mem_mb < threshold_mb:
        continue

    try:
        user = subprocess.check_output(
            ["ps", "-o", "user=", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or "unknown"
    except Exception:
        user = "unknown"

    hits.append((npu_id, pid, user, proc_name, mem_mb))

if not hits:
    print("[runtime] NPU occupancy precheck: OK")
    raise SystemExit(0)

print("[runtime] ERROR: visible NPUs are already occupied by large processes.")
print(f"[runtime] threshold_mb={threshold_mb}")
for npu_id, pid, user, proc_name, mem_mb in sorted(hits):
    print(
        f"[runtime] busy_npu={npu_id} pid={pid} user={user} "
        f"mem_mb={mem_mb} process={proc_name}"
    )

parents = {}
for _, pid, _, _, _ in hits:
    try:
        ppid = subprocess.check_output(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if ppid.isdigit():
            parents[int(ppid)] = None
    except Exception:
        pass

for ppid in sorted(parents):
    try:
        parent_cmd = subprocess.check_output(
            ["ps", "-fp", str(ppid)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        continue
    if parent_cmd:
        print(f"[runtime] parent_process:\n{parent_cmd}")

raise SystemExit(2)
PY
}
