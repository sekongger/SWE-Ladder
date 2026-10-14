#!/usr/bin/env bash
# Serve the OPSD external teacher behind an OpenAI-compatible /completions endpoint.
#
# Pairs with configs/opsd_grpo_e2e_external_teacher_local8npu.yaml (OPSD_TEACHER_SOURCE=external).
# The training run scores each student response with ONE /completions call per sample:
#   prompt = prompt_ids + response_ids (raw token ids), echo=True, logprobs=K, max_tokens=1.
# So this server MUST:
#   * accept token-id prompts + echo=True + logprobs (stock vLLM OpenAI server does),
#   * use a SAME-FAMILY (identical) tokenizer as the Qwen3-4B student, because the client sends
#     raw token ids -- a different tokenizer would score the wrong tokens. Use a Qwen3-family
#     teacher (e.g. Qwen3-8B/-14B/-32B or another Qwen3 checkpoint).
#
# Run this on a box WITH SPARE GPUs/NPUs -- NOT the local 8-NPU box while the actor run uses it.
#
# Usage:
#   TEACHER_MODEL_PATH=/path/to/Qwen3-8B \
#   TEACHER_SERVED_NAME=opsd-teacher \
#   TEACHER_PORT=8000 TEACHER_TP=2 \
#   bash DeepSWE_RL/rllm/examples/swe/serve_opsd_teacher_vllm.sh
#
# Then on the training box:
#   export OPSD_TEACHER_BASE_URL=http://<this-host>:${TEACHER_PORT}/v1
#   export OPSD_TEACHER_MODEL=${TEACHER_SERVED_NAME}
set -euo pipefail

TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:?set TEACHER_MODEL_PATH to the teacher checkpoint dir}"
TEACHER_SERVED_NAME="${TEACHER_SERVED_NAME:-opsd-teacher}"
TEACHER_HOST="${TEACHER_HOST:-0.0.0.0}"
TEACHER_PORT="${TEACHER_PORT:-8000}"
TEACHER_TP="${TEACHER_TP:-2}"
TEACHER_GPU_UTIL="${TEACHER_GPU_UTIL:-0.90}"
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-65536}"
TEACHER_DTYPE="${TEACHER_DTYPE:-auto}"
# top-K logprobs the client requests per position. vLLM caps returned logprobs at
# --max-logprobs; keep it >= OPSD_TOPK_K (default 16) with headroom.
TEACHER_MAX_LOGPROBS="${TEACHER_MAX_LOGPROBS:-32}"

echo "[serve-teacher] model=${TEACHER_MODEL_PATH}"
echo "[serve-teacher] served_name=${TEACHER_SERVED_NAME} host=${TEACHER_HOST} port=${TEACHER_PORT} tp=${TEACHER_TP}"
echo "[serve-teacher] max_model_len=${TEACHER_MAX_MODEL_LEN} max_logprobs=${TEACHER_MAX_LOGPROBS}"
echo "[serve-teacher] REMINDER: teacher tokenizer MUST match the student (Qwen3 family)."

# --enable-prefix-caching helps a lot here: successive samples in a batch share the same long
# task prompt prefix, so the teacher forward reuses cached KV across echo calls.
exec vllm serve "${TEACHER_MODEL_PATH}" \
  --served-model-name "${TEACHER_SERVED_NAME}" \
  --host "${TEACHER_HOST}" \
  --port "${TEACHER_PORT}" \
  --tensor-parallel-size "${TEACHER_TP}" \
  --gpu-memory-utilization "${TEACHER_GPU_UTIL}" \
  --max-model-len "${TEACHER_MAX_MODEL_LEN}" \
  --max-logprobs "${TEACHER_MAX_LOGPROBS}" \
  --dtype "${TEACHER_DTYPE}" \
  --enable-prefix-caching
