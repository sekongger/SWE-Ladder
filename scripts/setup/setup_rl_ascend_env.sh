#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The workspace holding the upstream checkouts (DeepSWE_RL/rllm, R2E-Gym-*, wheels).
# Defaults to this repository's parent directory; override when it sits elsewhere.
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"

SOURCE_ENV="${SOURCE_ENV:-ascend}"
TARGET_ENV="${TARGET_ENV:-ascend-rl}"

# Conda install that owns SOURCE_ENV. A conda-initialized shell exports CONDA_EXE;
# set MINICONDA_HOME explicitly when running from a shell that has not been init'd.
if [[ -z "${MINICONDA_HOME:-}" && -n "${CONDA_EXE:-}" ]]; then
  MINICONDA_HOME="$(dirname -- "$(dirname -- "${CONDA_EXE}")")"
fi
if [[ -z "${MINICONDA_HOME:-}" ]]; then
  for candidate in "${HOME}/miniconda3" "${HOME}/anaconda3" /opt/conda; do
    if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
      MINICONDA_HOME="${candidate}"
      break
    fi
  done
fi
MINICONDA_HOME="${MINICONDA_HOME:-${HOME}/miniconda3}"
CONDA_SH="${CONDA_SH:-${MINICONDA_HOME}/etc/profile.d/conda.sh}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"
STRIP_SFT_PKGS="${STRIP_SFT_PKGS:-1}"
CONDA_CLONE_OFFLINE="${CONDA_CLONE_OFFLINE:-1}"
CONDA_CLONE_RETRY_ONLINE="${CONDA_CLONE_RETRY_ONLINE:-1}"
TARGET_PREFIX="${MINICONDA_HOME}/envs/${TARGET_ENV}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[rl-env] ERROR: conda init script not found: ${CONDA_SH}"
  exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_SH}"

if ! conda env list | awk '{print $1}' | grep -Fxq "${SOURCE_ENV}"; then
  echo "[rl-env] ERROR: source env does not exist: ${SOURCE_ENV}"
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "${TARGET_ENV}"; then
  if [[ "${FORCE_RECREATE}" == "1" ]]; then
    echo "[rl-env] removing existing env: ${TARGET_ENV}"
    conda remove -n "${TARGET_ENV}" --all -y
  else
    echo "[rl-env] target env already exists: ${TARGET_ENV}"
    echo "[rl-env] reuse it, or rerun with FORCE_RECREATE=1"
    exit 0
  fi
fi

if [[ -d "${TARGET_PREFIX}" ]] && [[ ! -f "${TARGET_PREFIX}/conda-meta/history" ]]; then
  echo "[rl-env] found incomplete env directory: ${TARGET_PREFIX}"
  echo "[rl-env] removing incomplete directory before clone"
  rm -rf "${TARGET_PREFIX}"
fi

# Avoid accidental concurrent clone jobs that leave partial envs.
stale_clone_pids="$(pgrep -f "conda create -n ${TARGET_ENV} --clone ${SOURCE_ENV}" || true)"
if [[ -n "${stale_clone_pids}" ]]; then
  echo "[rl-env] found stale clone process(es): ${stale_clone_pids}"
  echo "[rl-env] terminating stale clone process(es)"
  kill ${stale_clone_pids} || true
  sleep 2
fi

sanitize_proxy() {
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
}

echo "[rl-env] cloning ${SOURCE_ENV} -> ${TARGET_ENV}"
clone_rc=0
if [[ "${CONDA_CLONE_OFFLINE}" == "1" ]]; then
  echo "[rl-env] clone mode: offline"
  sanitize_proxy
  conda create -n "${TARGET_ENV}" --clone "${SOURCE_ENV}" --offline -y || clone_rc=$?
else
  echo "[rl-env] clone mode: online"
  conda create -n "${TARGET_ENV}" --clone "${SOURCE_ENV}" -y || clone_rc=$?
fi

if [[ "${clone_rc}" != "0" ]]; then
  echo "[rl-env] initial clone failed with code ${clone_rc}"
  if [[ "${CONDA_CLONE_RETRY_ONLINE}" == "1" ]]; then
    echo "[rl-env] retry clone in online mode"
    rm -rf "${TARGET_PREFIX}"
    conda create -n "${TARGET_ENV}" --clone "${SOURCE_ENV}" -y
  else
    exit "${clone_rc}"
  fi
fi

if [[ ! -f "${TARGET_PREFIX}/conda-meta/history" ]]; then
  echo "[rl-env] ERROR: clone appears incomplete (missing conda-meta/history)"
  echo "[rl-env] target path: ${TARGET_PREFIX}"
  exit 1
fi

conda activate "${TARGET_ENV}"

if [[ "${STRIP_SFT_PKGS}" == "1" ]]; then
  echo "[rl-env] uninstalling SFT-only packages from ${TARGET_ENV}"
  python3 -m pip uninstall -y mindformers openrlhf || true
fi

echo "[rl-env] reinstalling local editable packages"
python3 -m pip install -e "${WORKSPACE_ROOT}/DeepSWE_RL/rllm/verl"
python3 -m pip install -e "${WORKSPACE_ROOT}/DeepSWE_RL/rllm"

for wheel in \
  "${WORKSPACE_ROOT}/swebench_fork_swegym-2.0.13-py3-none-any.whl" \
  "${WORKSPACE_ROOT}/swebench_fork_swerebench-4.0.3-py3-none-any.whl" \
  "${WORKSPACE_ROOT}/swesmith-0.0.7-py3-none-any.whl"; do
  if [[ -f "${wheel}" ]]; then
    echo "[rl-env] installing local wheel: $(basename "${wheel}")"
    python3 -m pip install --no-deps "${wheel}"
  fi
done

echo "[rl-env] snapshot package versions"
mkdir -p "${WORKSPACE_ROOT}/DeepSWE_RL/rllm/examples/swe/env_snapshots"
conda list > "${WORKSPACE_ROOT}/DeepSWE_RL/rllm/examples/swe/env_snapshots/${TARGET_ENV}_conda_list.txt"

echo "[rl-env] done"
echo "[rl-env] activate via:"
echo "source ${CONDA_SH}"
echo "conda activate ${TARGET_ENV}"
