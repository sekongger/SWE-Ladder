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
TARGET_PREFIX="${TARGET_PREFIX:-${MINICONDA_HOME}/envs/${TARGET_ENV}}"
PACK_DIR="${PACK_DIR:-${WORKSPACE_ROOT}/.conda_packs}"
PACK_FILE="${PACK_FILE:-${PACK_DIR}/${SOURCE_ENV}_to_${TARGET_ENV}.tar.gz}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"
STRIP_SFT_PKGS="${STRIP_SFT_PKGS:-1}"
PACK_IGNORE_EDITABLE="${PACK_IGNORE_EDITABLE:-1}"
PACK_IGNORE_MISSING_FILES="${PACK_IGNORE_MISSING_FILES:-0}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[rl-pack] ERROR: conda init script not found: ${CONDA_SH}"
  exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_SH}"

if ! conda env list | awk '{print $1}' | grep -Fxq "${SOURCE_ENV}"; then
  echo "[rl-pack] ERROR: source env does not exist: ${SOURCE_ENV}"
  exit 1
fi

if [[ -d "${TARGET_PREFIX}" ]]; then
  if [[ "${FORCE_RECREATE}" == "1" ]]; then
    echo "[rl-pack] removing existing target prefix: ${TARGET_PREFIX}"
    rm -rf "${TARGET_PREFIX}"
  else
    if [[ -f "${TARGET_PREFIX}/conda-meta/history" ]]; then
      echo "[rl-pack] target env already exists and looks complete: ${TARGET_PREFIX}"
      echo "[rl-pack] rerun with FORCE_RECREATE=1 to rebuild"
      exit 0
    fi
    echo "[rl-pack] found incomplete target prefix, removing: ${TARGET_PREFIX}"
    rm -rf "${TARGET_PREFIX}"
  fi
fi

mkdir -p "${PACK_DIR}"

echo "[rl-pack] packing source env: ${SOURCE_ENV}"
conda activate "${SOURCE_ENV}"
PACK_EXTRA_ARGS=()
if [[ "${PACK_IGNORE_EDITABLE}" == "1" ]]; then
  PACK_EXTRA_ARGS+=(--ignore-editable-packages)
fi
if [[ "${PACK_IGNORE_MISSING_FILES}" == "1" ]]; then
  PACK_EXTRA_ARGS+=(--ignore-missing-files)
fi
if command -v conda-pack >/dev/null 2>&1; then
  conda-pack -n "${SOURCE_ENV}" -o "${PACK_FILE}" "${PACK_EXTRA_ARGS[@]}"
else
  python -m conda_pack.cli -n "${SOURCE_ENV}" -o "${PACK_FILE}" "${PACK_EXTRA_ARGS[@]}"
fi

echo "[rl-pack] unpacking to ${TARGET_PREFIX}"
mkdir -p "${TARGET_PREFIX}"
tar -xzf "${PACK_FILE}" -C "${TARGET_PREFIX}"

if [[ ! -x "${TARGET_PREFIX}/bin/conda-unpack" ]]; then
  echo "[rl-pack] ERROR: conda-unpack not found in target env"
  exit 1
fi

echo "[rl-pack] running conda-unpack"
"${TARGET_PREFIX}/bin/conda-unpack"

# shellcheck disable=SC1090
source "${TARGET_PREFIX}/bin/activate"

if [[ "${STRIP_SFT_PKGS}" == "1" ]]; then
  echo "[rl-pack] uninstalling SFT-only packages"
  python -m pip uninstall -y mindformers openrlhf || true
fi

echo "[rl-pack] reinstalling local editable packages"
python -m pip install -e "${WORKSPACE_ROOT}/DeepSWE_RL/rllm/verl"
python -m pip install -e "${WORKSPACE_ROOT}/DeepSWE_RL/rllm"

for wheel in \
  "${WORKSPACE_ROOT}/swebench_fork_swegym-2.0.13-py3-none-any.whl" \
  "${WORKSPACE_ROOT}/swebench_fork_swerebench-4.0.3-py3-none-any.whl" \
  "${WORKSPACE_ROOT}/swesmith-0.0.7-py3-none-any.whl"; do
  if [[ -f "${wheel}" ]]; then
    echo "[rl-pack] installing local wheel: $(basename "${wheel}")"
    python -m pip install --no-deps "${wheel}"
  fi
done

mkdir -p "${WORKSPACE_ROOT}/DeepSWE_RL/rllm/examples/swe/env_snapshots"
python -m pip freeze > "${WORKSPACE_ROOT}/DeepSWE_RL/rllm/examples/swe/env_snapshots/${TARGET_ENV}_pip_freeze.txt"
conda list > "${WORKSPACE_ROOT}/DeepSWE_RL/rllm/examples/swe/env_snapshots/${TARGET_ENV}_conda_list.txt"

echo "[rl-pack] done"
echo "[rl-pack] activate via:"
echo "source ${CONDA_SH}"
echo "conda activate ${TARGET_ENV}  # if not listed, use: source ${TARGET_PREFIX}/bin/activate"
