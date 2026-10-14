#!/usr/bin/env python3
"""Run a command as a reaper and stop its whole process group on signals."""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import signal
import subprocess
import sys
import time


PR_SET_CHILD_SUBREAPER = 36


def set_child_subreaper() -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        rc = libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    except Exception as exc:  # pragma: no cover - platform guard
        print(f"[supervisor] warning: failed to enable child subreaper: {exc}", file=sys.stderr)
        return
    if rc != 0:
        err = ctypes.get_errno()
        print(f"[supervisor] warning: prctl(PR_SET_CHILD_SUBREAPER) failed: errno={err}", file=sys.stderr)


def reap_children() -> None:
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        except OSError as exc:
            if exc.errno == errno.ECHILD:
                return
            raise
        if pid == 0:
            return


def signal_pgid(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return


def stop_process_group(pgid: int, first_signal: signal.Signals, grace: float, kill_grace: float) -> None:
    signal_pgid(pgid, first_signal)
    deadline = time.monotonic() + max(grace, 0.0)
    while time.monotonic() < deadline:
        reap_children()
        time.sleep(0.2)

    print(f"[supervisor] sending SIGTERM to process group pgid={pgid}", flush=True)
    signal_pgid(pgid, signal.SIGTERM)
    deadline = time.monotonic() + max(kill_grace, 0.0)
    while time.monotonic() < deadline:
        reap_children()
        time.sleep(0.2)

    print(f"[supervisor] sending SIGKILL to process group pgid={pgid}", flush=True)
    signal_pgid(pgid, signal.SIGKILL)
    deadline = time.monotonic() + max(kill_grace, 0.0)
    while time.monotonic() < deadline:
        reap_children()
        time.sleep(0.2)
    reap_children()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setsid-self", action="store_true")
    parser.add_argument("--grace", type=float, default=12.0)
    parser.add_argument("--kill-grace", type=float, default=2.0)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("missing command")

    if args.setsid_self:
        try:
            os.setsid()
        except OSError as exc:
            print(f"[supervisor] warning: os.setsid() failed: {exc}", file=sys.stderr, flush=True)

    set_child_subreaper()
    child = subprocess.Popen(cmd, start_new_session=True)
    child_pgid = os.getpgid(child.pid)
    stopping_signal: signal.Signals | None = None

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stopping_signal
        sig = signal.Signals(signum)
        if stopping_signal is None:
            stopping_signal = sig
            print(
                f"[supervisor] received {sig.name}; stopping process group pgid={child_pgid} root_pid={child.pid}",
                flush=True,
            )
            signal_pgid(child_pgid, sig)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    while True:
        reap_children()
        rc = child.poll()
        if rc is not None:
            break
        if stopping_signal is not None:
            stop_process_group(child_pgid, stopping_signal, args.grace, args.kill_grace)
            return 128 + stopping_signal.value
        time.sleep(0.2)

    # If the root process exited by itself, terminate any same-PGID stragglers.
    signal_pgid(child_pgid, signal.SIGTERM)
    time.sleep(max(args.kill_grace, 0.0))
    reap_children()
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
