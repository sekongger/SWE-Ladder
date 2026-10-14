#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

CONFIG_PATH="${1:-${SCRIPT_DIR}/configs/baseline_grpo_8b_cuda.yaml}"
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

find_yaml_python() {
  local candidate
  local -a candidates=()
  if [[ -n "${YAML_PYTHON:-}" ]]; then
    candidates+=("${YAML_PYTHON}")
  fi
  candidates+=(
    python3
    python
    "${HOME}/miniconda3/bin/python"
    "/root/miniconda3/bin/python"
    "/opt/conda/bin/python"
  )
  for candidate in "${candidates[@]}"; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      candidate="$(command -v "${candidate}")"
    elif [[ ! -x "${candidate}" ]]; then
      continue
    fi
    if "${candidate}" - <<'PY' >/dev/null 2>&1
import yaml
PY
    then
      printf '%s' "${candidate}"
      return 0
    fi
  done
  return 1
}

YAML_PARSER_CMD=()
if YAML_PYTHON_BIN="$(find_yaml_python)"; then
  YAML_PARSER_CMD=("${YAML_PYTHON_BIN}")
elif command -v uv >/dev/null 2>&1; then
  YAML_PARSER_CMD=(uv run --with PyYAML python)
else
  echo "ERROR: no Python with PyYAML found. Install pyyaml in base/miniconda or set YAML_PYTHON." >&2
  exit 1
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

eval "$(
  "${YAML_PARSER_CMD[@]}" - "${CONFIG_PATH}" "${WORKSPACE_ROOT}" <<'PY'
import re
import shlex
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
workspace_root = sys.argv[2]
data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
runtime = data.get("runtime") or {}
env = data.get("env") or {}

if not isinstance(runtime, dict):
    raise SystemExit("runtime must be a mapping")
if not isinstance(env, dict):
    raise SystemExit("env must be a mapping")

def expand_value(value):
    if isinstance(value, str):
        return value.replace("${WORKSPACE_ROOT}", workspace_root)
    return value

env_key_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
for key, value in env.items():
    if value is None:
        continue
    value = expand_value(value)
    if not env_key_re.match(str(key)):
        raise SystemExit(f"invalid env key: {key!r}")
    print(f"export {key}={shlex.quote(str(value))}")

for key in ("entrypoint", "log_dir", "log_prefix", "venv_path", "conda_sh", "conda_env"):
    value = runtime.get(key)
    if value is not None:
        value = expand_value(value)
        print(f"RUN_{key.upper()}={shlex.quote(str(value))}")

clean_ray = runtime.get("clean_ray_before_start", True)
print(f"RUN_CLEAN_RAY={shlex.quote('1' if clean_ray else '0')}")
PY
)"

# runtime.conda_sh / runtime.conda_env name the env explicitly; when the config only
# names the env (the usual case), discover the conda install the same way the NPU
# launcher does. With no conda at all, the interpreter on PATH is assumed correct.
if [[ -z "${RUN_CONDA_SH:-}" ]]; then
  for candidate in \
    "${CONDA_EXE:+$(dirname -- "$(dirname -- "${CONDA_EXE}")")/etc/profile.d/conda.sh}" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
      RUN_CONDA_SH="${candidate}"
      break
    fi
  done
fi
if [[ -n "${RUN_CONDA_SH:-}" && -f "${RUN_CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${RUN_CONDA_SH}"
  if [[ -n "${RUN_CONDA_ENV:-}" ]]; then
    conda activate "${RUN_CONDA_ENV}"
  fi
fi

ENTRYPOINT="${RUN_ENTRYPOINT:-DeepSWE_RL/rllm/examples/swe/train_entrypoint_gpu.sh}"
LOG_DIR="${RUN_LOG_DIR:-output/swe_ladder_cuda}"
LOG_PREFIX="${RUN_LOG_PREFIX:-swe_ladder_cuda_train}"
VENV_PATH="${RUN_VENV_PATH:-${GPU_RL_VENV:-${WORKSPACE_ROOT}/.venv-gpu-rl}}"

if [[ "${ENTRYPOINT}" != /* ]]; then
  ENTRYPOINT="${WORKSPACE_ROOT}/${ENTRYPOINT}"
fi
if [[ "${LOG_DIR}" != /* ]]; then
  LOG_DIR="${WORKSPACE_ROOT}/${LOG_DIR}"
fi
if [[ "${VENV_PATH}" != /* ]]; then
  VENV_PATH="${WORKSPACE_ROOT}/${VENV_PATH}"
fi
export GPU_RL_VENV="${VENV_PATH}"

if [[ ! -x "${ENTRYPOINT}" ]]; then
  echo "ERROR: entrypoint is not executable: ${ENTRYPOINT}" >&2
  exit 1
fi

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
else
  echo "WARNING: GPU venv not found: ${VENV_PATH}" >&2
  echo "WARNING: run setup_rl_gpu_env.sh before launching a real training job." >&2
fi

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${LOG_PREFIX}_$(date +%Y%m%d_%H%M%S).log"

echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "ENTRYPOINT: ${ENTRYPOINT}"
echo "LOG_PATH: ${LOG_PATH}"
echo "GPU_RL_VENV: ${GPU_RL_VENV}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "TRAIN_DATASET: ${TRAIN_DATASET:-<unset>}"
echo "VAL_DATASET: ${VAL_DATASET:-<unset>}"
echo "MODEL_PATH: ${MODEL_PATH:-<unset>}"
# Scaffold identity must be readable from the log, not inferred from the config filename.
# These two always travel together: the tool triple installed in the container and the
# prompt templates that name those tools.
echo "SWE_SCAFFOLD: ${SWE_SCAFFOLD:-<unset -> entrypoint default r2egym>}"
echo "OPD_PROMPT_TEMPLATE_DIR: ${OPD_PROMPT_TEMPLATE_DIR:-<unset -> built-in r2egym templates>}"
echo "TRAIN_LOGGERS: ${TRAIN_LOGGERS:-<unset>}"
echo "WANDB_ENABLE: ${WANDB_ENABLE:-<unset>}"
echo "WANDB_MODE: ${WANDB_MODE:-<unset>}"
echo "OPSD_ENABLE: ${OPSD_ENABLE:-<unset>}"
echo "OPSD_POLICY_LOSS_MODE: ${OPSD_POLICY_LOSS_MODE:-<unset>}"
echo "OPSD_TOPK_K: ${OPSD_TOPK_K:-<unset>}"
echo "R2EGYM_DOCKER_MODE: ${R2EGYM_DOCKER_MODE:-<unset>}"
echo "R2EGYM_DOCKER_HOST: ${R2EGYM_DOCKER_HOST:-<unset>}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1, skip training launch."
  exit 0
fi

if [[ "${RUN_CLEAN_RAY:-1}" == "1" ]]; then
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
  # Ray's temp root is /tmp/ray-<user>; a stale session_latest symlink makes the next
  # start attach to a dead cluster.
  rm -rf "/tmp/ray-$(id -un)/session_latest" /tmp/ray-root/session_latest /tmp/ray/session_latest || true
fi

set +e
bash "${ENTRYPOINT}" 2>&1 | tee "${LOG_PATH}"
run_rc=${PIPESTATUS[0]}
set -e

ln -sf "${LOG_PATH}" "${LOG_DIR}/latest.log"
echo "LOG_PATH=${LOG_PATH}"
echo "RUN_RC=${run_rc}"
exit "${run_rc}"
