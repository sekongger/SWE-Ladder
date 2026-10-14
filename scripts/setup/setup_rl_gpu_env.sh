#!/bin/bash
set -euo pipefail
if [[ "${VERBOSE_SHELL_TRACE:-0}" == "1" ]]; then
  set -x
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The workspace holding the upstream checkouts (DeepSWE_RL/rllm, R2E-Gym-*, wheels).
# Defaults to this repository's parent directory; override when it sits elsewhere.
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
VENV_PATH="${GPU_RL_VENV:-${WORKSPACE_ROOT}/.venv-gpu-rl}"
CONDA_BIN="${CONDA_BIN:-}"

if [[ -z "${CONDA_BIN}" ]]; then
  for candidate in "${HOME}/miniconda3/bin/conda" "/root/miniconda3/bin/conda" "/opt/conda/bin/conda"; do
    if [[ -x "${candidate}" ]]; then
      CONDA_BIN="${candidate}"
      break
    fi
  done
fi

if [[ -n "${CONDA_BIN}" ]]; then
  export PATH="$(dirname "${CONDA_BIN}"):${PATH}"
fi

# Do not touch busy GPUs while preparing the Python environment.
export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
export UV_NO_CACHE="${UV_NO_CACHE:-1}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${WORKSPACE_ROOT}/.cache/conda-pkgs}"
mkdir -p "${CONDA_PKGS_DIRS}"

if ! command -v uv >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-}"
  if [[ -z "${BOOTSTRAP_PYTHON}" ]]; then
    for candidate in python3 python "${HOME}/miniconda3/bin/python" "/root/miniconda3/bin/python" "/opt/conda/bin/python"; do
      if command -v "${candidate}" >/dev/null 2>&1; then
        BOOTSTRAP_PYTHON="$(command -v "${candidate}")"
        break
      elif [[ -x "${candidate}" ]]; then
        BOOTSTRAP_PYTHON="${candidate}"
        break
      fi
    done
  fi
  if [[ -z "${BOOTSTRAP_PYTHON}" ]]; then
    echo "ERROR: uv is required and no bootstrap Python was found." >&2
    exit 1
  fi
  "${BOOTSTRAP_PYTHON}" -m pip install --upgrade uv
  BOOTSTRAP_BIN="$(dirname "${BOOTSTRAP_PYTHON}")"
  if [[ -x "${BOOTSTRAP_BIN}/uv" ]]; then
    export PATH="${BOOTSTRAP_BIN}:${PATH}"
  fi
  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv was installed via ${BOOTSTRAP_PYTHON}, but uv is still not on PATH." >&2
    exit 1
  fi
fi

if [[ -z "${GPU_RL_PYTHON:-}" && -n "${CONDA_BIN}" ]]; then
  PY310_ENV="${GPU_RL_PY310_ENV:-${WORKSPACE_ROOT}/.conda-gpu-py310}"
  if [[ ! -x "${PY310_ENV}/bin/python" ]]; then
    "${CONDA_BIN}" create -y -p "${PY310_ENV}" python=3.10 pip
  fi
  GPU_RL_PYTHON="${PY310_ENV}/bin/python"
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  uv venv --seed --python "${GPU_RL_PYTHON:-python3}" "${VENV_PATH}"
fi

# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"

if ! python -m pip --version >/dev/null 2>&1; then
  python -m ensurepip --upgrade || true
fi
if ! python -m pip --version >/dev/null 2>&1; then
  echo "[gpu-env] pip is missing in ${VENV_PATH}; recreating venv with uv --seed." >&2
  deactivate 2>/dev/null || true
  rm -rf "${VENV_PATH}"
  uv venv --seed --python "${GPU_RL_PYTHON:-python3}" "${VENV_PATH}"
  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel

# Install CUDA stack explicitly outside the project pyproject to avoid pulling
# NPU/Ascend-only packages. Override TORCH_INDEX_URL if the cluster uses a mirror.
#
# H200 (Hopper, sm_90) notes: torch 2.6.0 + cu124 and vllm 0.8.5.post1 both support
# Hopper out of the box -- no arch-specific rebuild is needed (H200 is the same sm_90 as
# H100, just more HBM). If the H200 host ships a newer CUDA driver, cu124 wheels still run
# (driver is backward-compatible with the 12.4 runtime). Only override TORCH_VERSION/
# TORCH_INDEX_URL if the cluster mandates a specific CUDA minor.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"
python -m pip install --index-url "${TORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}"

# Keep the broad verl requirement set, but install CUDA/flash-attn explicitly
# below so a failed source build does not poison the environment setup.
python -m pip install -r "${WORKSPACE_ROOT}/DeepSWE_RL/rllm/verl/requirements.txt"

python -m pip install -e "${WORKSPACE_ROOT}/DeepSWE_RL/rllm/verl" --no-deps

# rllm's pyproject includes broad research/demo dependencies such as sglang,
# browsergym and loose torch/vllm ranges. Installing it with dependencies can
# replace the pinned CUDA torch stack above, so keep the source package editable
# and install only the SWE RL runtime dependencies we actually need here.
python -m pip install -e "${WORKSPACE_ROOT}/DeepSWE_RL/rllm" --no-deps
python -m pip install -e "${WORKSPACE_ROOT}/R2E-Gym-generic-sandbox" --no-deps
python -m pip install -e "${WORKSPACE_ROOT}/R2E-Gym-eval" --no-deps

python -m pip install \
  anthropic==0.43.0 \
  astor \
  bashlex \
  beautifulsoup4 \
  chardet \
  diff-parser \
  docker \
  fire \
  ghapi \
  gitpython \
  gym==0.26.2 \
  hydra-core \
  jsonlines \
  kubernetes \
  libcst \
  'litellm>=1.58.2' \
  matplotlib \
  numpy \
  omegaconf \
  openai \
  pandas \
  polars \
  pyarrow \
  pytest \
  python-dotenv \
  requests \
  simple-parsing \
  swebench==3.0.2 \
  tenacity \
  tree-sitter \
  tree-sitter-c-sharp \
  tree-sitter-cpp \
  tree-sitter-go \
  tree-sitter-java \
  tree-sitter-javascript \
  tree-sitter-php \
  tree-sitter-ruby \
  tree-sitter-rust \
  tree_sitter_c \
  unidiff \
  uvloop \
  python-multipart

VLLM_VERSION="${VLLM_VERSION:-0.8.5.post1}"
python -m pip install "vllm==${VLLM_VERSION}" --no-deps || {
  echo "[gpu-env] WARNING: vllm==${VLLM_VERSION} could not be installed." >&2
  echo "[gpu-env] Install a cluster-compatible vllm wheel before launching GPU rollout." >&2
}
python -m pip install --no-deps \
  apache-tvm-ffi \
  airportsdata \
  blake3 \
  cachetools \
  compressed-tensors==0.9.3 \
  deprecated \
  depyf==0.18.0 \
  diskcache \
  einops \
  gguf \
  importlib-metadata==8.0.0 \
  interegular \
  lark==1.2.2 \
  llguidance==0.7.30 \
  lm-format-enforcer==0.10.12 \
  llvmlite==0.44.0 \
  mistral-common \
  msgspec \
  nest-asyncio \
  ninja \
  numba==0.61.2 \
  opentelemetry-api==1.26.0 \
  opentelemetry-exporter-otlp==1.26.0 \
  opentelemetry-exporter-otlp-proto-grpc==1.26.0 \
  opentelemetry-exporter-otlp-proto-http==1.26.0 \
  opentelemetry-sdk==1.26.0 \
  opentelemetry-semantic-conventions==0.47b0 \
  opentelemetry-semantic-conventions-ai==0.4.1 \
  opencv-python-headless==4.11.0.86 \
  outlines==0.1.11 \
  outlines-core==0.1.26 \
  partial-json-parser \
  protobuf==4.25.9 \
  prometheus-fastapi-instrumentator \
  py-cpuinfo \
  pycountry \
  pydantic-extra-types \
  python-json-logger \
  pyzmq \
  scipy \
  sentencepiece \
  starlette==0.50.0 \
  tiktoken \
  watchfiles \
  xformers==0.0.29.post2 \
  xgrammar==0.1.18

# vLLM 0.8.5 is built against the transformers 4.x tokenizer API. Newer
# transformers 5.x removes properties such as all_special_tokens_extended,
# which makes vLLM fail during tokenizer initialization.
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.52.4}"
TOKENIZERS_VERSION="${TOKENIZERS_VERSION:-0.21.1}"
python -m pip install --upgrade \
  "transformers[hf_xet]==${TRANSFORMERS_VERSION}" \
  "tokenizers==${TOKENIZERS_VERSION}"

# verl's broad requirements may install the latest prometheus exporter, which
# expects a newer OpenTelemetry stack than vllm==0.8.5.post1. The vLLM HTTP
# server path uses prometheus-fastapi-instrumentator instead.
python -m pip uninstall -y opentelemetry-exporter-prometheus || true

FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-${WORKSPACE_ROOT}/flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
if [[ -f "${FLASH_ATTN_WHEEL}" ]]; then
  python -m pip install --no-deps "${FLASH_ATTN_WHEEL}"
else
  echo "[gpu-env] WARNING: flash-attn wheel not found: ${FLASH_ATTN_WHEEL}" >&2
  echo "[gpu-env] Set FLASH_ATTN_WHEEL to a matching prebuilt wheel, or install flash-attn manually." >&2
fi

python -m pip install --no-deps \
  "${WORKSPACE_ROOT}/swesmith-0.0.7-py3-none-any.whl" \
  "${WORKSPACE_ROOT}/swebench_fork_swerebench-4.0.3-py3-none-any.whl" \
  "${WORKSPACE_ROOT}/swebench_fork_swegym-2.0.13-py3-none-any.whl" || true

python - <<'PY'
import importlib.util
import sys

mods = ["torch", "transformers", "ray", "verl", "rllm", "r2egym"]
for mod in mods:
    print(f"{mod}: {'ok' if importlib.util.find_spec(mod) else 'missing'}")
print(f"python: {sys.executable}")
PY

echo "[gpu-env] Environment prepared at ${VENV_PATH}"
echo "[gpu-env] CUDA_VISIBLE_DEVICES was empty during setup; no GPU validation was run."
