#!/usr/bin/env bash
set -euo pipefail

# Roll out a served policy over SWE-bench Verified through an R2E-Gym scaffold.
#
# One script for all three scaffolds we report. The scaffold is the single most
# consequential knob: it decides which tools exist inside the container, so numbers
# from different scaffolds are NOT comparable. See eval/README.md.
#
#   SCAFFOLD=openhands  str_replace_editor + execute_bash + submit   (upstream recipe)
#   SCAFFOLD=r2egym     file_editor + search + execute_bash + finish (R2E-Gym minimal ReAct)
#   SCAFFOLD=sweagent   str_replace_editor + execute_bash + submit   (SWE-agent prompt set)
#
# Required:
#   R2EGYM_ROOT         R2E-Gym checkout (provides src/r2egym/agenthub/run/edit.py)
#   SERVED_MODEL_NAME   must match what the vLLM server exposes
# Common:
#   VLLM_BASE_URL       default http://127.0.0.1:8127
#   DOCKER_BASE_URL     default http://127.0.0.1:23750 (local dockerd: http://127.0.0.1:2375)
#   PARQUET_DATASET     SWE-bench Verified parquet (converted to JSON once, cached)
#   EXP_NAME / TRAJ_DIR where trajectories land (<TRAJ_DIR>/<EXP_NAME>.jsonl)
#   K / START_IDX / MAX_WORKERS / TEMPERATURE / MAX_STEPS / MAX_TOKENS
#
# Rerunning with USE_EXISTING=True reuses rows already in the jsonl and only executes
# the missing instances, which is what makes the recovery loop in
# run_eval_until_complete.sh cheap.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
R2EGYM_ROOT="${R2EGYM_ROOT:-}"
SCAFFOLD="${SCAFFOLD:-openhands}"
USED_YAML="${USED_YAML:-}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-swe-ladder-policy}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8127}"
DOCKER_BASE_URL="${DOCKER_BASE_URL:-http://127.0.0.1:23750}"
DOCKER_TUNNEL_HELPER="${DOCKER_TUNNEL_HELPER:-${SCRIPT_DIR}/ensure_docker_tunnel.sh}"

PARQUET_DATASET="${PARQUET_DATASET:-}"
DATASET_JSON="${DATASET_JSON:-${REPO_ROOT}/output/swebench_verified_dataset.json}"
SPLIT="${SPLIT:-test}"

EXP_NAME="${EXP_NAME:-swebench_verified_${SERVED_MODEL_NAME}_${SCAFFOLD}}"
TRAJ_DIR="${TRAJ_DIR:-${REPO_ROOT}/output/${EXP_NAME}}"

MAX_WORKERS="${MAX_WORKERS:-16}"
START_IDX="${START_IDX:-0}"
K="${K:-500}"
MAX_STEPS="${MAX_STEPS:-150}"
MAX_STEPS_ABSOLUTE="${MAX_STEPS_ABSOLUTE:-150}"
MAX_TOKENS="${MAX_TOKENS:-131072}"
TEMPERATURE="${TEMPERATURE:-0.7}"
USE_FN_CALLING="${USE_FN_CALLING:-False}"
USE_EXISTING="${USE_EXISTING:-True}"
SKIP_EXISTING="${SKIP_EXISTING:-False}"
PREPULL_IMAGES="${PREPULL_IMAGES:-False}"

CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-5}"
CURL_MAX_TIME="${CURL_MAX_TIME:-10}"

EVAL_PID=""
EVAL_KILL_TARGET=""
LOGP="[swebench-eval:${SCAFFOLD}:${SERVED_MODEL_NAME}]"

die() { echo "ERROR: $*" >&2; exit 1; }

# Per-scaffold default prompt/tool config. stepout16384 caps a single assistant turn at
# 16384 tokens; the unsuffixed openhands yaml leaves the per-step cap at its 2048 default,
# which truncates long edits (see docs/EVALUATION.md).
default_used_yaml() {
  case "${SCAFFOLD}" in
    openhands) echo "./src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling_stepout16384.yaml" ;;
    r2egym)    echo "./src/r2egym/agenthub/config/r2egym/edit_non_fn_calling_stepout16384.yaml" ;;
    sweagent)  echo "./src/r2egym/agenthub/config/sweagent/edit_non_fn_calling.yaml" ;;
    *)         die "unknown SCAFFOLD: ${SCAFFOLD} (expected openhands|r2egym|sweagent)" ;;
  esac
}

write_summary() {
  "${PYTHON_BIN}" - "${TRAJ_DIR}/${EXP_NAME}.jsonl" "${TRAJ_DIR}/${EXP_NAME}.summary.json" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path

jsonl, summary = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
if jsonl.exists():
    with jsonl.open() as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
rewards = [float(r.get("reward") or 0.0) for r in rows]
steps = [len(r.get("trajectory_steps") or []) for r in rows]
success = sum(1 for x in rewards if x == 1.0)
# Rows whose patch starts with "Error:" are infrastructure failures (container/socket),
# not model failures; they are reported separately so solve rates stay honest.
infra_errors = sum(1 for r in rows if str(r.get("output_patch") or "").startswith("Error:"))
data = {
    "jsonl": str(jsonl),
    "total_trajectories": len(rows),
    "unique_instances": len({r.get("instance_id") or (r.get("ds") or {}).get("instance_id") for r in rows}),
    "num_success": success,
    "solve_rate": success / len(rows) if rows else 0.0,
    "infra_error_rows": infra_errors,
    "nonempty_patch": sum(1 for r in rows if str(r.get("output_patch") or "").strip()),
    "exit_reasons": dict(Counter(str(r.get("exit_reason")) for r in rows)),
    "avg_steps": sum(steps) / len(steps) if steps else 0.0,
}
summary.parent.mkdir(parents=True, exist_ok=True)
summary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
print(f"summary: {summary}")
print(f"rows={data['total_trajectories']} unique={data['unique_instances']} "
      f"success={success} solve={data['solve_rate'] * 100:.1f}% infra_errors={infra_errors}")
PY
}

cleanup_eval() {
  if [[ -n "${EVAL_KILL_TARGET}" ]]; then
    echo "${LOGP} stopping eval process group: ${EVAL_KILL_TARGET}" >&2
    kill -TERM -- "${EVAL_KILL_TARGET}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "${EVAL_KILL_TARGET}" 2>/dev/null || true
  fi
  write_summary || true
  exit 130
}

docker_host_from_base_url() {
  "${PYTHON_BIN}" - "${DOCKER_BASE_URL}" <<'PY'
import sys
url = sys.argv[1].strip().rstrip("/")
for scheme in ("http://", "https://"):
    if url.startswith(scheme):
        print("tcp://" + url[len(scheme):])
        break
else:
    print(url)
PY
}

maybe_start_docker_tunnel() {
  # Only meaningful when the dockerd is remote and reached through the bundled
  # SSH-tunnel helper; a local dockerd needs nothing.
  [[ "${DOCKER_TUNNEL:-0}" == "1" ]] || return 0
  [[ -x "${DOCKER_TUNNEL_HELPER}" ]] || die "docker tunnel helper not executable: ${DOCKER_TUNNEL_HELPER}"
  "${DOCKER_TUNNEL_HELPER}" start
}

ensure_dataset_json() {
  [[ -n "${PARQUET_DATASET}" ]] || die "PARQUET_DATASET (SWE-bench Verified parquet) is required"
  [[ -f "${PARQUET_DATASET}" ]] || die "dataset parquet not found: ${PARQUET_DATASET}"
  if [[ -f "${DATASET_JSON}" && ! "${PARQUET_DATASET}" -nt "${DATASET_JSON}" ]]; then
    return 0
  fi
  echo "${LOGP} converting parquet -> json: ${DATASET_JSON}"
  "${PYTHON_BIN}" - "${PARQUET_DATASET}" "${DATASET_JSON}" <<'PY'
import json, math, sys
from pathlib import Path
import pandas as pd

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
df = pd.read_parquet(src)


def norm(v):
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return [norm(x) for x in v.tolist()]
        if isinstance(v, np.generic):
            return v.item()
    except Exception:
        pass
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {str(k): norm(x) for k, x in v.items()}
    return v


records = [{str(k): norm(v) for k, v in row.items()} for row in df.to_dict(orient="records")]
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(records)} records to {dst}")
PY
}

[[ -n "${R2EGYM_ROOT}" ]] || die "R2EGYM_ROOT is required (R2E-Gym checkout)"
[[ -d "${R2EGYM_ROOT}/src/r2egym" ]] || die "not an R2E-Gym checkout: ${R2EGYM_ROOT}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "PYTHON_BIN not found: ${PYTHON_BIN}"
[[ -n "${USED_YAML}" ]] || USED_YAML="$(default_used_yaml)"

mkdir -p "${TRAJ_DIR}"
VLLM_BASE_URL="${VLLM_BASE_URL%/}"
DOCKER_BASE_URL="${DOCKER_BASE_URL%/}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_API_BASE="${VLLM_BASE_URL}/v1"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
# torch_npu autoload inside the eval workers is pure overhead and can abort on a busy card.
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
export R2EGYM_DOCKER_HOST="$(docker_host_from_base_url)"
export R2EGYM_DOCKER_TLS_VERIFY="${R2EGYM_DOCKER_TLS_VERIFY:-}"
# A stale corporate proxy silently swallows localhost traffic to vLLM and dockerd.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

maybe_start_docker_tunnel
ensure_dataset_json

if command -v curl >/dev/null 2>&1; then
  models_json="$(curl --connect-timeout "${CURL_CONNECT_TIMEOUT}" --max-time "${CURL_MAX_TIME}" \
    -fsS "${VLLM_BASE_URL}/v1/models" 2>/dev/null || true)"
  [[ -n "${models_json}" ]] || die "vLLM endpoint not reachable: ${VLLM_BASE_URL}/v1/models (start eval/serve_checkpoint.sh)"
  [[ "${models_json}" == *"${SERVED_MODEL_NAME}"* ]] || \
    die "vLLM is up but does not serve '${SERVED_MODEL_NAME}'. Models: ${models_json}"
  curl --connect-timeout "${CURL_CONNECT_TIMEOUT}" --max-time "${CURL_MAX_TIME}" \
    -fsS "${DOCKER_BASE_URL}/version" >/dev/null || die "Docker endpoint not reachable: ${DOCKER_BASE_URL}/version"
fi

PYTHONPATH="${R2EGYM_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<'PY'
import importlib.util
missing = [m for m in ("litellm", "r2egym", "docker", "pandas") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("missing eval dependencies: " + ", ".join(missing))
print("dependency check passed")
PY

cd "${R2EGYM_ROOT}"
export PYTHONPATH="${R2EGYM_ROOT}/src:${PYTHONPATH:-}"

CMD=(
  "${PYTHON_BIN}" "src/r2egym/agenthub/run/edit.py" runagent_multiple
  --traj_dir "${TRAJ_DIR}"
  --max_workers "${MAX_WORKERS}"
  --start_idx "${START_IDX}"
  --k "${K}"
  --dataset "${DATASET_JSON}"
  --split "${SPLIT}"
  --llm_name "hosted_vllm/${SERVED_MODEL_NAME}"
  --use_fn_calling "${USE_FN_CALLING}"
  --exp_name "${EXP_NAME}"
  --temperature "${TEMPERATURE}"
  --max_steps "${MAX_STEPS}"
  --max_steps_absolute "${MAX_STEPS_ABSOLUTE}"
  --max_tokens "${MAX_TOKENS}"
  --backend docker
  --prepull_images "${PREPULL_IMAGES}"
  --scaffold "${SCAFFOLD}"
  --ip "127.0.0.1"
  --used_yaml "${USED_YAML}"
  --use_existing "${USE_EXISTING}"
  --skip_existing "${SKIP_EXISTING}"
)

echo "${LOGP} R2E-Gym root : ${R2EGYM_ROOT}"
echo "${LOGP} scaffold yaml: ${USED_YAML}"
echo "${LOGP} vLLM         : ${OPENAI_API_BASE}"
echo "${LOGP} docker host  : ${R2EGYM_DOCKER_HOST}"
echo "${LOGP} trajectories : ${TRAJ_DIR}/${EXP_NAME}.jsonl"
echo "${LOGP} protocol     : temp=${TEMPERATURE} steps=${MAX_STEPS}/${MAX_STEPS_ABSOLUTE} ctx=${MAX_TOKENS} k=${K} workers=${MAX_WORKERS}"

# setsid so Ctrl-C / SIGTERM can take down the whole worker pool, not just the parent.
if command -v setsid >/dev/null 2>&1; then
  setsid "${CMD[@]}" &
  EVAL_PID="$!"; EVAL_KILL_TARGET="-${EVAL_PID}"
else
  "${CMD[@]}" &
  EVAL_PID="$!"; EVAL_KILL_TARGET="${EVAL_PID}"
fi
trap cleanup_eval INT TERM TSTP
set +e
wait "${EVAL_PID}"
RC="$?"
set -e
trap - INT TERM TSTP
EVAL_PID=""; EVAL_KILL_TARGET=""
write_summary || true
exit "${RC}"
