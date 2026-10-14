#!/usr/bin/env bash
set -euo pipefail

# Keep a local TCP port forwarded to a remote dockerd, with a watchdog that reconnects.
#
# You only need this when the Docker daemon that runs SWE-bench containers lives on
# another machine. With a local daemon, point DOCKER_BASE_URL at it directly
# (e.g. http://127.0.0.1:2375) and skip this script.
#
# The watchdog is not optional in practice: a plain `ssh -N -L` tunnel dies on network
# flaps and does not reconnect, and a dead tunnel does not look like a dead tunnel --
# every container start fails as NO_CONTAINER, so evaluation silently scores 0 and
# training silently degenerates to near-empty rollouts. Training runs against a remote
# daemon want this running too (`start`, then set R2EGYM_DOCKER_HOST to CHECK_URL's host).
#
# Required when starting:
#   REMOTE_HOST   ssh host running dockerd
#   SSH_KEY       private key for REMOTE_HOST (never commit one)
# Optional:
#   REMOTE_USER (root) SSH_PORT (22)
#   REMOTE_DOCKER_SOCKET  forward to this unix socket instead of a TCP port, for a
#                         daemon that does not listen on TCP (e.g. /var/run/docker.sock)
#   REMOTE_DOCKER_HOST (127.0.0.1) REMOTE_DOCKER_PORT (2375)
#   LOCAL_HOST (127.0.0.1) LOCAL_PORT (23750)
#
# Usage: ./ensure_docker_tunnel.sh {start|status|stop|watch}

LOCAL_HOST="${LOCAL_HOST:-127.0.0.1}"
LOCAL_PORT="${LOCAL_PORT:-23750}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:-}"
SSH_PORT="${SSH_PORT:-22}"
REMOTE_DOCKER_SOCKET="${REMOTE_DOCKER_SOCKET:-}"
REMOTE_DOCKER_HOST="${REMOTE_DOCKER_HOST:-127.0.0.1}"
REMOTE_DOCKER_PORT="${REMOTE_DOCKER_PORT:-2375}"
SSH_KEY="${SSH_KEY:-}"
CHECK_INTERVAL="${CHECK_INTERVAL:-2}"
RESTART_DELAY="${RESTART_DELAY:-2}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-10}"
STATE_DIR="${STATE_DIR:-/tmp/swe_ladder_docker_tunnel_${LOCAL_PORT}}"

CHECK_URL="http://${LOCAL_HOST}:${LOCAL_PORT}/version"
LOG_FILE="${LOG_FILE:-${STATE_DIR}/watchdog.log}"
PID_FILE="${PID_FILE:-${STATE_DIR}/watchdog.pid}"
LOCK_FILE="${LOCK_FILE:-${STATE_DIR}/watchdog.lock}"

# ssh -L accepts a remote unix socket path in place of host:port (OpenSSH >= 6.7).
if [[ -n "${REMOTE_DOCKER_SOCKET}" ]]; then
  FORWARD_SPEC="${LOCAL_HOST}:${LOCAL_PORT}:${REMOTE_DOCKER_SOCKET}"
  REMOTE_DESC="${REMOTE_HOST}:${REMOTE_DOCKER_SOCKET}"
else
  FORWARD_SPEC="${LOCAL_HOST}:${LOCAL_PORT}:${REMOTE_DOCKER_HOST}:${REMOTE_DOCKER_PORT}"
  REMOTE_DESC="${REMOTE_HOST}:${REMOTE_DOCKER_PORT}"
fi

mkdir -p "${STATE_DIR}"

endpoint_ok() {
  curl --connect-timeout 2 --max-time 5 --noproxy '*' -fsS "${CHECK_URL}" >/dev/null 2>&1
}

watchdog_alive() {
  [[ -s "${PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${PID_FILE}")"
  [[ -d "/proc/${pid}" ]] || return 1
  # kill -0 also succeeds for zombies; reject them so a stale pid never masks a dead watchdog.
  [[ "$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null || true)" != "Z" ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

start_watchdog() {
  if endpoint_ok; then
    echo "[docker-tunnel] endpoint already reachable: ${CHECK_URL}"
    return 0
  fi
  if watchdog_alive; then
    echo "[docker-tunnel] watchdog already running: pid=$(cat "${PID_FILE}")"
    return 0
  fi
  if [[ -z "${REMOTE_HOST}" ]]; then
    echo "ERROR: REMOTE_HOST is required to open a tunnel (or set DOCKER_BASE_URL to a local daemon)" >&2
    return 1
  fi
  if [[ -z "${SSH_KEY}" || ! -r "${SSH_KEY}" ]]; then
    echo "ERROR: SSH_KEY is unset or unreadable: ${SSH_KEY:-<unset>}" >&2
    return 1
  fi
  rm -f "${PID_FILE}"
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid "$0" watch >>"${LOG_FILE}" 2>&1 &
  else
    nohup "$0" watch >>"${LOG_FILE}" 2>&1 &
  fi
  local i
  for ((i = 1; i <= 20; i++)); do
    if watchdog_alive; then
      echo "[docker-tunnel] watchdog started: pid=$(cat "${PID_FILE}") log=${LOG_FILE}"
      return 0
    fi
    if endpoint_ok; then
      echo "[docker-tunnel] endpoint ready: ${CHECK_URL}"
      return 0
    fi
    sleep 0.1
  done
  echo "ERROR: failed to start docker tunnel watchdog; see ${LOG_FILE}" >&2
  return 1
}

wait_until_ready() {
  local attempts="${1:-15}"
  local i
  for ((i = 1; i <= attempts; i++)); do
    if endpoint_ok; then
      echo "[docker-tunnel] endpoint ready: ${CHECK_URL}"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: Docker tunnel endpoint is not reachable: ${CHECK_URL}" >&2
  echo "Last watchdog log:" >&2
  tail -n 40 "${LOG_FILE}" >&2 2>/dev/null || true
  return 1
}

watch() {
  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    echo "[$(date -Is)] another docker tunnel watchdog is already running"
    exit 0
  fi
  echo "$$" >"${PID_FILE}"
  trap 'rm -f "${PID_FILE}"; exit 0' INT TERM

  while true; do
    if endpoint_ok; then
      sleep "${CHECK_INTERVAL}"
      continue
    fi
    echo "[$(date -Is)] starting ssh tunnel ${LOCAL_HOST}:${LOCAL_PORT} -> ${REMOTE_DESC}"
    ssh -N \
      -i "${SSH_KEY}" \
      -p "${SSH_PORT}" \
      -o BatchMode=yes \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=2 \
      -o TCPKeepAlive=yes \
      -o ConnectTimeout="${CONNECT_TIMEOUT}" \
      -o StrictHostKeyChecking=accept-new \
      -L "${FORWARD_SPEC}" \
      "${REMOTE_USER}@${REMOTE_HOST}" || true
    echo "[$(date -Is)] ssh tunnel exited; retrying in ${RESTART_DELAY}s"
    sleep "${RESTART_DELAY}"
  done
}

case "${1:-start}" in
  start)
    start_watchdog
    wait_until_ready 15
    ;;
  status)
    if watchdog_alive; then
      echo "[docker-tunnel] watchdog running: pid=$(cat "${PID_FILE}")"
    else
      echo "[docker-tunnel] watchdog not running"
    fi
    if endpoint_ok; then
      echo "[docker-tunnel] endpoint reachable: ${CHECK_URL}"
    else
      echo "[docker-tunnel] endpoint unreachable: ${CHECK_URL}"
      exit 1
    fi
    ;;
  stop)
    if watchdog_alive; then
      pid="$(cat "${PID_FILE}")"
      kill "${pid}" 2>/dev/null || true
      for _ in {1..20}; do
        [[ -d "/proc/${pid}" ]] || break
        sleep 0.1
      done
      rm -f "${PID_FILE}"
      echo "[docker-tunnel] watchdog stopped"
    fi
    ;;
  watch)
    watch
    ;;
  *)
    echo "Usage: $0 {start|status|stop|watch}" >&2
    exit 2
    ;;
esac
