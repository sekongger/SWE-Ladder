#!/bin/bash
set -euo pipefail
if [[ "${VERBOSE_SHELL_TRACE:-0}" == "1" ]]; then set -x; fi

# ==========================================================================
# Blackwell (RTX 6000D / sm_120) RL environment setup.
#
# WHY THIS EXISTS (vs setup_rl_gpu_env.sh):
#   setup_rl_gpu_env.sh targets H200 (Hopper/sm_90) with torch 2.6.0+cu124 +
#   vllm 0.8.5.post1 -- that stack has NO sm_120 kernels and dies on Blackwell
#   with "no kernel image is available for execution on the device".
#   Blackwell (compute cap 12.0) needs CUDA 12.8 kernels: torch >= 2.7 + cu128,
#   and a vllm new enough to ship sm_120 kernels.
#
# VERSION CHOICE (validated against this repo, 2026-08):
#   * vllm 0.11.2  -- the vendored verl (0.8.0.dev) has explicit version gates
#     up to 0.13.0 (vllm_async_server.py: >0.11.0, >=0.11.1 assert, ==0.12.0,
#     >=0.13.0). 0.11.2 lands on verl's MAINLINE path (satisfies >0.11.0 and the
#     >=0.11.1 assert; avoids the newer ==0.12.0 / >=0.13.0 branches). rllm has
#     NO direct vllm import and no version gate (only vllm>=0.8.3): it talks to
#     vllm through the OpenAI HTTP API + verl's checkpoint_manager, both version
#     tolerant. So the whole vllm coupling lives in verl, which supports 0.11.2.
#   * torch 2.9.0 + cu128 -- vllm 0.11.2 pins torch==2.9.0; the cu128 build has
#     sm_120 kernels. (vllm 0.11.2 requires_dist: torch==2.9.0, torchvision
#     0.24.0, torchaudio 2.9.0, xformers 0.0.33.post1, flashinfer-python 0.5.2.)
#
# NO-NVCC CONSTRAINT:
#   The box has the NVIDIA driver but NO CUDA toolkit / nvcc. So EVERYTHING must
#   come from prebuilt wheels -- no source builds (flash-attn source), no JIT.
#   * vllm's prebuilt cu128 wheel already contains sm_120 attention kernels ->
#     run rollout with VLLM_ATTENTION_BACKEND=FLASH_ATTN (vllm's bundled
#     flash-attn) to AVOID flashinfer's runtime JIT (which needs nvcc).
#   * Training-side flash-attn (transformers attn_implementation=flash_attention_2)
#     is BEST-EFFORT here: a prebuilt sm_120 wheel if one is provided via
#     FLASH_ATTN_WHEEL, else the entrypoint auto-falls back to sdpa (works,
#     a bit slower). Do NOT trigger a flash-attn source build (no nvcc).
#
# CONFLICT NOTE:
#   verl/requirements.txt pins numpy<2.0.0 and an unpinned transformers, which
#   conflict with the torch 2.9 / vllm 0.11.2 world (numpy 2.x, vllm-pinned
#   transformers). So we install vllm FIRST (it establishes torch/numpy2/
#   transformers/ray), then install verl's OTHER deps explicitly WITHOUT the
#   numpy/transformers pins. Editable installs are --no-deps.
# ==========================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The workspace holding the upstream checkouts (DeepSWE_RL/rllm, R2E-Gym-*, wheels).
# Defaults to this repository's parent directory; override when it sits elsewhere.
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
VENV_PATH="${GPU_RL_VENV:-${WORKSPACE_ROOT}/.venv-gpu-rl}"

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WORKSPACE_ROOT%/SWE-Master}/pip-cache}"
# Do not touch busy GPUs while preparing the environment.
export CUDA_VISIBLE_DEVICES=""

# --- venv (py3.10+; py3.12 from the box's miniconda is fine) ---------------
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  BOOTSTRAP_PY="${BOOTSTRAP_PYTHON:-}"
  if [[ -z "${BOOTSTRAP_PY}" ]]; then
    for c in /root/miniconda3/bin/python3.12 python3.12 python3 /opt/conda/bin/python; do
      command -v "$c" >/dev/null 2>&1 && { BOOTSTRAP_PY="$(command -v "$c")"; break; }
      [[ -x "$c" ]] && { BOOTSTRAP_PY="$c"; break; }
    done
  fi
  [[ -z "${BOOTSTRAP_PY}" ]] && { echo "ERROR: no bootstrap python found" >&2; exit 1; }
  echo "[bw-env] creating venv at ${VENV_PATH} from ${BOOTSTRAP_PY}"
  "${BOOTSTRAP_PY}" -m venv "${VENV_PATH}"
fi
# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# --- torch 2.9.0 + cu128 (sm_120 kernels) ---------------------------------
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
python -m pip install --index-url "${TORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION:-2.9.0}" \
  "torchvision==${TORCHVISION_VERSION:-0.24.0}" \
  "torchaudio==${TORCHAUDIO_VERSION:-2.9.0}"

# --- vllm 0.11.2 (pulls xformers/flashinfer/transformers/ray; torch already
#     satisfied by the cu128 build above) --------------------------------
VLLM_VERSION="${VLLM_VERSION:-0.11.2}"
# Constrain torch to the installed cu128 build so vllm's deps never swap it out.
TORCH_LOCK="$(python -c 'import torch; print("torch=="+torch.__version__)')"
python -m pip install "vllm==${VLLM_VERSION}" --constraint <(echo "${TORCH_LOCK}")

# --- editable source packages (no deps) ------------------------------------
python -m pip install -e "${WORKSPACE_ROOT}/DeepSWE_RL/rllm/verl" --no-deps
python -m pip install -e "${WORKSPACE_ROOT}/DeepSWE_RL/rllm" --no-deps
python -m pip install -e "${WORKSPACE_ROOT}/R2E-Gym-generic-sandbox" --no-deps
python -m pip install -e "${WORKSPACE_ROOT}/R2E-Gym-eval" --no-deps

# --- verl deps MINUS numpy<2 / transformers (vllm owns those) --------------
python -m pip install --constraint <(echo "${TORCH_LOCK}") \
  accelerate codetiming datasets dill hydra-core liger-kernel pandas peft \
  pybind11 pylatexenc "pyarrow>=19.0.0" "ray[default]" \
  "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata wandb "packaging>=20.0" \
  uvicorn fastapi latex2sympy2_extended math_verify tensorboard

# --- r2egym / SWE agent runtime deps ---------------------------------------
python -m pip install --constraint <(echo "${TORCH_LOCK}") \
  anthropic==0.43.0 astor bashlex beautifulsoup4 chardet diff-parser docker fire \
  ghapi gitpython gym==0.26.2 jsonlines kubernetes libcst 'litellm>=1.58.2' \
  matplotlib omegaconf openai polars pytest python-dotenv requests simple-parsing \
  swebench==3.0.2 tenacity tree-sitter tree-sitter-c-sharp tree-sitter-cpp \
  tree-sitter-go tree-sitter-java tree-sitter-javascript tree-sitter-php \
  tree-sitter-ruby tree-sitter-rust tree_sitter_c unidiff uvloop python-multipart

# --- SWE bench forks (no deps) ---------------------------------------------
python -m pip install --no-deps \
  "${WORKSPACE_ROOT}/swesmith-0.0.7-py3-none-any.whl" \
  "${WORKSPACE_ROOT}/swebench_fork_swerebench-4.0.3-py3-none-any.whl" \
  "${WORKSPACE_ROOT}/swebench_fork_swegym-2.0.13-py3-none-any.whl" || true

# --- training flash-attn: prebuilt sm_120 wheel ONLY (no source build) -----
# Provide a torch2.9/cu128/sm_120 wheel via FLASH_ATTN_WHEEL to enable
# attn_implementation=flash_attention_2. Otherwise the entrypoint uses sdpa.
if [[ -n "${FLASH_ATTN_WHEEL:-}" && -f "${FLASH_ATTN_WHEEL}" ]]; then
  python -m pip install --no-deps "${FLASH_ATTN_WHEEL}" || \
    echo "[bw-env] WARN: flash-attn wheel install failed; will fall back to sdpa." >&2
else
  echo "[bw-env] NOTE: no FLASH_ATTN_WHEEL set; training will use sdpa (no source build on a no-nvcc box)."
fi

# --- self-check ------------------------------------------------------------
python - <<'PY'
import importlib.util as u, sys
for m in ["torch","transformers","ray","vllm","verl","rllm","r2egym","xformers"]:
    print(f"{m}: {'ok' if u.find_spec(m) else 'MISSING'}")
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("python", sys.executable)
PY
echo "[bw-env] done. Runtime tips: export VLLM_ATTENTION_BACKEND=FLASH_ATTN (avoid flashinfer JIT / no nvcc)."
