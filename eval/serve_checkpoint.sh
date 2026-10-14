#!/usr/bin/env bash
set -euo pipefail

# Merge an RL checkpoint (sharded FSDP actor -> HuggingFace) and serve it with vLLM.
#
# This single script replaces the per-checkpoint launchers we used during the
# experiments: everything that differed between them is an environment variable.
#
#   ACTOR_DIR        sharded actor dir, e.g. <train_output>/.../global_step_40/actor
#   HF_MODEL_DIR     where the merged HF weights are written (reused if already there)
#   SERVED_MODEL_NAME name the OpenAI-compatible endpoint exposes (must match the
#                    `--llm_name hosted_vllm/<name>` the eval runner asks for)
#   BACKEND          ascend | gpu
#   RLLM_ROOT        checkout that provides verl/scripts/legacy_model_merger.py
#                    (only needed for `merge`; serving a plain HF dir does not)
#
# Usage: ./serve_checkpoint.sh <merge|start|stop|status|logs>
#
# Example (Ascend, 8 cards, TP=8):
#   ACTOR_DIR=/ckpt/run/global_step_40/actor \
#   HF_MODEL_DIR=/models/run_gs40 \
#   SERVED_MODEL_NAME=swe-ladder-loc-gs40 \
#   BACKEND=ascend TP=8 PORT=8127 \
#   RLLM_ROOT=/opt/SWE-Master/DeepSWE_RL/rllm \
#   ./serve_checkpoint.sh start

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKEND="${BACKEND:-ascend}"
ACTOR_DIR="${ACTOR_DIR:-}"
HF_MODEL_DIR="${HF_MODEL_DIR:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-swe-ladder-policy}"
RLLM_ROOT="${RLLM_ROOT:-}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8127}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
TP="${TP:-8}"
# 131072 = the upstream SWE-Master context budget; shrink it if the card cannot hold the KV cache.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MEMORY_UTILIZATION="${MEMORY_UTILIZATION:-0.50}"
DTYPE="${DTYPE:-auto}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-20}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_LOGPROBS="${MAX_LOGPROBS:-64}"
# Prefix caching OFF matches the protocol all reported numbers were measured under.
PREFIX_CACHING="${PREFIX_CACHING:-off}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/output/vllm_server}"
WAIT_SECONDS="${WAIT_SECONDS:-900}"

SERVER_SCRIPT="${SERVER_SCRIPT:-${SCRIPT_DIR}/start_vllm_server.sh}"
ASCEND_RUNTIME_HELPER="${ASCEND_RUNTIME_HELPER:-${REPO_ROOT}/training/lib/ascend_runtime.sh}"
ASCEND_TOOLKIT_VERSION_DIR="${ASCEND_TOOLKIT_VERSION_DIR:-8.3.RC1}"

LOGP="[serve-checkpoint:${SERVED_MODEL_NAME}]"

die() { echo "ERROR: $*" >&2; exit 1; }

has_hf_weights() {
  local dir="$1"
  compgen -G "${dir}/*.safetensors" >/dev/null || compgen -G "${dir}/pytorch_model*.bin" >/dev/null
}

prepare_ascend_env() {
  [[ -f "${ASCEND_RUNTIME_HELPER}" ]] || die "Ascend runtime helper not found: ${ASCEND_RUNTIME_HELPER}"
  # torch_npu autoload + the toolkit's own set_env.sh fight over LD_LIBRARY_PATH; let the
  # helper assemble it instead (see training/lib/ascend_runtime.sh).
  unset TORCH_DEVICE_BACKEND_AUTOLOAD
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${DEVICES}}"
  export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
  export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:256}"
  # shellcheck source=/dev/null
  source "${ASCEND_RUNTIME_HELPER}"
  local ascend_home="${ASCEND_HOME:-/usr/local/Ascend}"
  setup_ascend_toolkit_env_manual "${ascend_home}" "${ASCEND_TOOLKIT_VERSION_DIR}"
  safe_source_first_existing \
    "${ASCEND_TOOLKIT_ROOT:-}/set_env.sh" \
    "${ASCEND_TOOLKIT_ROOT:-}/aarch64-linux/script/set_env.sh" \
    "${ASCEND_TOOLKIT_ROOT:-}/x86_64-linux/script/set_env.sh" >/dev/null || true
  sanitize_ld_library_path
  ensure_hccl_runtime "${ascend_home}"
}

# The sharded actor dir keeps tokenizer/config under huggingface/; the merger only
# writes weights, so copy the metadata across or vLLM will not find a tokenizer.
sync_hf_metadata() {
  local src="${ACTOR_DIR}/huggingface"
  [[ -d "${src}" && -d "${HF_MODEL_DIR}" ]] || return 0
  echo "${LOGP} syncing tokenizer/config from ${src}"
  local name
  for name in added_tokens.json chat_template.jinja config.json generation_config.json \
              merges.txt special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json; do
    [[ -f "${src}/${name}" ]] && cp -f "${src}/${name}" "${HF_MODEL_DIR}/${name}"
  done
  return 0
}

merge_checkpoint() {
  [[ -n "${HF_MODEL_DIR}" ]] || die "HF_MODEL_DIR is required"
  if [[ -d "${HF_MODEL_DIR}" ]] && has_hf_weights "${HF_MODEL_DIR}"; then
    echo "${LOGP} HF weights already present: ${HF_MODEL_DIR}"
    sync_hf_metadata
    return 0
  fi
  [[ -n "${ACTOR_DIR}" ]] || die "ACTOR_DIR is required to merge (or point HF_MODEL_DIR at merged weights)"
  [[ -d "${ACTOR_DIR}" ]] || die "ACTOR_DIR not found: ${ACTOR_DIR}"
  [[ -n "${RLLM_ROOT}" ]] || die "RLLM_ROOT is required to merge (provides verl/scripts/legacy_model_merger.py)"
  local merger="${RLLM_ROOT}/verl/scripts/legacy_model_merger.py"
  [[ -f "${merger}" ]] || die "model merger not found: ${merger}"

  mkdir -p "${HF_MODEL_DIR}"
  echo "${LOGP} merging FSDP shards -> HF"
  echo "${LOGP}   source: ${ACTOR_DIR}"
  echo "${LOGP}   target: ${HF_MODEL_DIR}"
  [[ "${BACKEND}" == "ascend" ]] && prepare_ascend_env
  PYTHONNOUSERSITE=1 PYTHONPATH="${RLLM_ROOT}/verl:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${merger}" merge \
      --backend fsdp \
      --local_dir "${ACTOR_DIR}" \
      --target_dir "${HF_MODEL_DIR}"
  sync_hf_metadata
}

vllm_cmd() {
  local subcommand="$1"; shift || true
  [[ -x "${SERVER_SCRIPT}" ]] || die "server script not executable: ${SERVER_SCRIPT}"
  local -a cmd=(
    "${SERVER_SCRIPT}" "${subcommand}"
    --backend "${BACKEND}"
    --model-path "${HF_MODEL_DIR}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --devices "${DEVICES}"
    --tp "${TP}"
    --max-model-len "${MAX_MODEL_LEN}"
    --memory-utilization "${MEMORY_UTILIZATION}"
    --dtype "${DTYPE}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --max-logprobs "${MAX_LOGPROBS}"
    --log-dir "${LOG_DIR}"
    --wait-seconds "${WAIT_SECONDS}"
  )
  case "${PREFIX_CACHING}" in
    off|0|false) cmd+=(--no-enable-prefix-caching) ;;
    on|1|true)   cmd+=(--enable-prefix-caching) ;;
  esac
  [[ "${ENFORCE_EAGER}" == "1" ]] && cmd+=(--enforce-eager)

  # start_vllm_server.sh resolves `python -m vllm...` from PATH, so make sure the
  # interpreter we were handed wins over whatever the login shell had.
  local python_dir
  python_dir="$(cd -- "$(dirname -- "$(command -v "${PYTHON_BIN}")")" && pwd)"
  PATH="${python_dir}:${PATH}" \
    ASCEND_RUNTIME_HELPER="${ASCEND_RUNTIME_HELPER}" \
    R2EGYM_CHECK_REMOTE_DOCKER_AT_START=0 \
    PYTHON_BIN="${PYTHON_BIN}" \
    "${cmd[@]}" "$@"
}

case "${1:-}" in
  merge)             merge_checkpoint ;;
  start)             merge_checkpoint; vllm_cmd start ;;
  stop|status|logs)  vllm_cmd "${1}" ;;
  -h|--help|help|"")
    sed -n '3,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
  *) die "unknown command: ${1}" ;;
esac
