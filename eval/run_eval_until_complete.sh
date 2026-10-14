#!/usr/bin/env bash
# Drive run_swebench_verified_eval.sh in rounds until the run has TARGET clean rows.
#
# Why this exists: on a shared dockerd a fraction of instances die on container or socket
# errors rather than on the model. Those rows land in the jsonl with output_patch starting
# with "Error:". Counting them as failures understates the policy; leaving them in place
# blocks the rerun because `use_existing` would treat them as done. So each round:
#
#   1. drop "Error:" rows from the jsonl (the only destructive step, and it only removes
#      infrastructure failures -- never a real model attempt),
#   2. stop when clean rows >= TARGET,
#   3. otherwise rerun the eval, which `use_existing` narrows to the missing instances,
#   4. abort the round early if many new rows arrive but almost none are clean -- that is
#      the signature of a host-side container killer, and hammering it just burns rows.
#
# Usage:
#   SERVED_MODEL_NAME=... R2EGYM_ROOT=... PARQUET_DATASET=... \
#   EXP_NAME=... TRAJ_DIR=... TARGET=500 ./run_eval_until_complete.sh
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/run_swebench_verified_eval.sh}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-swe-ladder-policy}"
SCAFFOLD="${SCAFFOLD:-openhands}"
EXP_NAME="${EXP_NAME:-swebench_verified_${SERVED_MODEL_NAME}_${SCAFFOLD}}"
TRAJ_DIR="${TRAJ_DIR:-${REPO_ROOT}/output/${EXP_NAME}}"
TARGET="${TARGET:-500}"
ROUNDS="${ROUNDS:-10}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8127}"
DOCKER_BASE_URL="${DOCKER_BASE_URL:-http://127.0.0.1:23750}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/output/eval_logs}"
LOG="${LOG:-${LOG_DIR}/${EXP_NAME}.log}"
# How long to wait for a server that vanished mid-run, and how long to back off after an abort.
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-1800}"
BACKOFF_SECONDS="${BACKOFF_SECONDS:-1800}"
POLL_SECONDS="${POLL_SECONDS:-600}"

export SERVED_MODEL_NAME SCAFFOLD EXP_NAME TRAJ_DIR VLLM_BASE_URL DOCKER_BASE_URL
export USE_EXISTING="${USE_EXISTING:-True}"
export SKIP_EXISTING="${SKIP_EXISTING:-False}"

JSONL="${TRAJ_DIR}/${EXP_NAME}.jsonl"
mkdir -p "${LOG_DIR}" "${TRAJ_DIR}"

log() { echo "[${EXP_NAME} $(date '+%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }

serve_ok() {
  curl -s -m 8 --noproxy '*' "${VLLM_BASE_URL%/}/v1/models" 2>/dev/null | grep -q "${SERVED_MODEL_NAME}"
}
docker_ok() {
  curl -s -m 6 --noproxy '*' "${DOCKER_BASE_URL%/}/version" >/dev/null 2>&1
}
row_count() { [[ -f "$1" ]] && wc -l < "$1" || echo 0; }

# Rewrite the jsonl without infrastructure-error rows; print the surviving count.
filter_clean() {
  [[ -f "${JSONL}" ]] || { echo 0; return 0; }
  "${PYTHON_BIN}" - "${JSONL}" <<'PY'
import json, sys

path = sys.argv[1]
rows = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
clean = [r for r in rows if not str(r.get("output_patch") or "").startswith("Error:")]
with open(path, "w") as f:
    for r in clean:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(clean))
PY
}

new_clean_since() {
  "${PYTHON_BIN}" - "${JSONL}" "$1" <<'PY'
import json, sys

path, base = sys.argv[1], int(sys.argv[2])
total = errors = 0
try:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                if str(json.loads(line).get("output_patch") or "").startswith("Error:"):
                    errors += 1
            except Exception:
                pass
except FileNotFoundError:
    pass
print(total - errors - base)
PY
}

log "==== start: scaffold=${SCAFFOLD} target=${TARGET} rounds=${ROUNDS} server=${VLLM_BASE_URL} ===="

waited=0
while ! serve_ok; do
  [[ "${waited}" == 0 ]] && log "waiting for vLLM to serve ${SERVED_MODEL_NAME} ..."
  sleep 10
  waited=$((waited + 10))
  if (( waited >= SERVER_WAIT_SECONDS )); then
    log "FATAL vLLM did not serve ${SERVED_MODEL_NAME} within ${SERVER_WAIT_SECONDS}s"
    exit 1
  fi
done
log "server OK"

stall=0
prev=-1
for r in $(seq 1 "${ROUNDS}"); do
  if ! docker_ok; then
    log "docker endpoint down; waiting 5min"
    sleep 300
    continue
  fi
  if ! serve_ok; then
    log "server lost mid-run; waiting (this driver never relaunches it, to avoid a double-launch race)"
    waited=0
    while ! serve_ok && (( waited < SERVER_WAIT_SECONDS )); do sleep 10; waited=$((waited + 10)); done
    serve_ok || { log "FATAL server gone for ${SERVER_WAIT_SECONDS}s"; exit 1; }
  fi

  clean="$(filter_clean)"
  log "round ${r} start: clean=${clean:-0}/${TARGET}"
  if (( ${clean:-0} >= TARGET )); then
    log "TARGET REACHED"
    break
  fi

  if [[ "${clean:-0}" == "${prev}" ]]; then
    stall=$((stall + 1))
    if (( stall >= 2 )); then
      log "no progress for 2 rounds; backing off ${BACKOFF_SECONDS}s"
      sleep "${BACKOFF_SECONDS}"
      stall=0
    fi
  else
    stall=0
  fi
  prev="${clean:-0}"

  base="$(row_count "${JSONL}")"
  setsid bash "${RUNNER}" > "${LOG_DIR}/${EXP_NAME}.round${r}.log" 2>&1 &
  epid=$!
  aborted=0
  while kill -0 "${epid}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
    n="$(row_count "${JSONL}")"
    log "round ${r}: +$((n - base)) rows (total ${n}/${TARGET})"
    if (( n - base >= 20 )); then
      newclean="$(new_clean_since "${base}" 2>/dev/null || echo 99)"
      if (( ${newclean:-99} < 3 )); then
        log "round ${r} ABORT: +$((n - base)) rows but only ${newclean} clean -- host is killing containers"
        kill -TERM -- "-${epid}" 2>/dev/null
        sleep 10
        kill -KILL -- "-${epid}" 2>/dev/null
        aborted=1
        break
      fi
    fi
  done
  wait "${epid}" 2>/dev/null
  if (( aborted != 0 )); then
    log "backing off ${BACKOFF_SECONDS}s after abort"
    sleep "${BACKOFF_SECONDS}"
  fi
done

final="$(filter_clean)"
"${PYTHON_BIN}" - "${JSONL}" <<'PY' | tee -a "${LOG}"
import json, sys
from collections import Counter

rows = []
try:
    with open(sys.argv[1]) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
except FileNotFoundError:
    pass
ok = sum(1 for r in rows if float(r.get("reward") or 0) == 1.0)
steps = [len(r.get("trajectory_steps") or []) for r in rows]
print(f"FINAL clean_rows={len(rows)} success={ok} "
      f"solve={ok / max(len(rows), 1) * 100:.1f}% "
      f"avg_steps={sum(steps) / max(len(steps), 1):.1f} "
      f"exits={dict(Counter(r.get('exit_reason') for r in rows))}")
PY
log "ALL_DONE clean=${final}"
