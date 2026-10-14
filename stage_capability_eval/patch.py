#!/usr/bin/env python3
"""PATCH_GIVEN_LOC evaluation primitives.

The public task package is deliberately made from the public task manifest and
the public LOC oracle only.  Gold source patches and verifier test names are
inputs to :func:`evaluate_patch`, kept in the evaluator's private boundary.
Execution is injected through a tiny runner interface so this module has no
Docker or third-party dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .package_common import public_task_row
except ImportError:  # pragma: no cover - exercised by direct CLI invocation
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from stage_capability_eval.package_common import public_task_row


SCHEMA_VERSION = "stage_capability_patch_v1"
STAGE = "PATCH_GIVEN_LOC"
PRIVATE_FIELDS = {"patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "run_tests", "parsed_commit"}


@dataclass(frozen=True)
class PatchRunnerResult:
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    infra_error: bool = False
    duration_seconds: float | None = None


def _sha256(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _jsonl(source: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        return [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [dict(row) for row in source]


def _public_oracle_loc(label: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the declared location hint, excluding gold patch hashes."""
    files = []
    for item in label.get("files", []) or []:
        if not isinstance(item, Mapping) or not item.get("path"):
            continue
        files.append({
            "path": item["path"],
            "functions": list(item.get("functions", []) or []),
            "old_context": list(item.get("old_context", []) or []),
            "mapping_status": item.get("mapping_status"),
        })
    return {
        "schema_version": label.get("schema_version", "stage_capability_loc_v1"),
        "instance_id": label.get("instance_id"),
        "mapping_status": label.get("mapping_status"),
        "files": files,
    }


def build_public_package(
    manifest: str | Path | Iterable[Mapping[str, Any]],
    oracle_loc: str | Path | Iterable[Mapping[str, Any]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic public PATCH_GIVEN_LOC inputs.

    This function never copies fields containing gold patch or verifier data,
    even when a source manifest accidentally contains them.
    """
    rows = _jsonl(manifest)
    labels = {str(row.get("instance_id")): dict(row) for row in _jsonl(oracle_loc)}
    if limit is not None:
        rows = rows[:limit]
    package: list[dict[str, Any]] = []
    for row in rows:
        iid = str(row.get("instance_id", ""))
        if not iid or iid not in labels:
            continue
        public = public_task_row(row)
        public["oracle_loc"] = _public_oracle_loc(labels[iid])
        public["stage"] = STAGE
        public["schema_version"] = SCHEMA_VERSION
        package.append(public)
    package.sort(key=lambda row: str(row["instance_id"]))
    return package


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    target.write_text(payload, encoding="utf-8")
    return _sha256(payload)


def _value(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _tests(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _coerce_result(value: Any) -> PatchRunnerResult:
    if isinstance(value, PatchRunnerResult):
        return value
    if isinstance(value, Mapping):
        return PatchRunnerResult(
            returncode=value.get("returncode"), stdout=str(value.get("stdout", "") or ""),
            stderr=str(value.get("stderr", "") or ""), timed_out=bool(value.get("timed_out", False)),
            infra_error=bool(value.get("infra_error", False)), duration_seconds=value.get("duration_seconds"),
        )
    return PatchRunnerResult(
        returncode=getattr(value, "returncode", None), stdout=str(getattr(value, "stdout", "") or ""),
        stderr=str(getattr(value, "stderr", "") or ""), timed_out=bool(getattr(value, "timed_out", False)),
        infra_error=bool(getattr(value, "infra_error", False)), duration_seconds=getattr(value, "duration_seconds", None),
    )


class SubprocessPatchRunner:
    """Small local runner; callers can replace it with a sandbox adapter."""

    def apply_patch(self, workspace: str | Path, patch: str, timeout_seconds: float) -> PatchRunnerResult:
        target = Path(workspace)
        started = time.monotonic()
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".patch", dir=target, delete=False) as handle:
                handle.write(patch)
                patch_file = Path(handle.name)
            try:
                proc = subprocess.run(["git", "apply", str(patch_file)], cwd=target, capture_output=True, text=True, timeout=timeout_seconds, check=False)
            finally:
                patch_file.unlink(missing_ok=True)
        except subprocess.TimeoutExpired as exc:
            return PatchRunnerResult(None, str(exc.stdout or ""), str(exc.stderr or ""), timed_out=True, duration_seconds=time.monotonic() - started)
        except (OSError, ValueError) as exc:
            return PatchRunnerResult(None, stderr=str(exc), infra_error=True, duration_seconds=time.monotonic() - started)
        return PatchRunnerResult(proc.returncode, proc.stdout, proc.stderr, duration_seconds=time.monotonic() - started)

    def run_test(self, workspace: str | Path, test_name: str, timeout_seconds: float) -> PatchRunnerResult:
        command = shlex.split(test_name)
        if not command:
            return PatchRunnerResult(None, stderr="empty test command")
        started = time.monotonic()
        try:
            proc = subprocess.run(command, cwd=workspace, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired as exc:
            return PatchRunnerResult(None, str(exc.stdout or ""), str(exc.stderr or ""), timed_out=True, duration_seconds=time.monotonic() - started)
        except (OSError, ValueError) as exc:
            return PatchRunnerResult(None, stderr=str(exc), infra_error=True, duration_seconds=time.monotonic() - started)
        return PatchRunnerResult(proc.returncode, proc.stdout, proc.stderr, duration_seconds=time.monotonic() - started)


def _run_method(runner: Any, method: str, workspace: str | Path, item: Any, timeout: float) -> PatchRunnerResult:
    fn = getattr(runner, method, None)
    if fn is None:
        fn = getattr(runner, {"apply_patch": "apply", "run_test": "run"}[method], None)
    if fn is None and callable(runner):
        try:
            return _coerce_result(runner(operation=method, workspace=workspace, value=item, timeout_seconds=timeout))
        except TypeError as first:
            try:
                return _coerce_result(runner(workspace, item, timeout))
            except TypeError:
                raise first
    if not callable(fn):
        raise TypeError(f"runner must provide {method}(workspace, value, timeout_seconds)")
    try:
        return _coerce_result(fn(workspace, item, timeout_seconds=timeout))
    except TypeError as first:
        try:
            return _coerce_result(fn(workspace, item, timeout))
        except TypeError:
            raise first


def _result_record(name: str, result: PatchRunnerResult) -> dict[str, Any]:
    return {
        "name": name,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "infra_error": result.infra_error,
        "stdout_sha256": _sha256(result.stdout),
        "stderr_sha256": _sha256(result.stderr),
        "duration_seconds": result.duration_seconds,
    }


def evaluate_patch(
    task: Mapping[str, Any],
    candidate_patch: str,
    verifier: Mapping[str, Any],
    *,
    workspace: str | Path,
    runner: Any | None = None,
    timeout_seconds: float = 300.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply one candidate and execute private F2P/P2P tests.

    The workspace is assumed to be an isolated buggy checkout supplied by the
    caller. This function does not create commits, inspect gold source, or
    assume Docker.
    """
    iid = str(task.get("instance_id", ""))
    base = {"schema_version": SCHEMA_VERSION, "stage": STAGE, "instance_id": iid,
            "candidate_patch_sha256": _sha256(candidate_patch), "patch_applied": False,
            "f2p": {"total": 0, "passed": 0, "failed": 0, "results": []},
            "p2p": {"total": 0, "passed": 0, "regressions": 0, "results": []}}
    f2p = _tests(_value(verifier, "FAIL_TO_PASS", "fail_to_pass", "f2p"))
    p2p = _tests(_value(verifier, "PASS_TO_PASS", "pass_to_pass", "p2p"))
    base["f2p"]["total"], base["p2p"]["total"] = len(f2p), len(p2p)
    if dry_run:
        return {**base, "status": "not_applicable", "failure_category": "dry_run"}
    if not isinstance(candidate_patch, str) or "diff --git " not in candidate_patch:
        return {**base, "status": "invalid", "failure_category": "patch_apply"}
    if not Path(workspace).is_dir() or not f2p and not p2p:
        return {**base, "status": "invalid", "failure_category": "missing_verifier" if Path(workspace).is_dir() else "workspace"}
    if runner is None:
        runner = SubprocessPatchRunner()
    try:
        applied = _run_method(runner, "apply_patch", workspace, candidate_patch, timeout_seconds)
    except Exception as exc:
        return {**base, "status": "infra_error", "failure_category": "runner", "detail": str(exc)}
    base["apply"] = _result_record("<patch>", applied)
    if applied.timed_out:
        return {**base, "status": "timeout", "failure_category": "timeout"}
    if applied.infra_error:
        return {**base, "status": "infra_error", "failure_category": "infrastructure"}
    if applied.returncode is None:
        return {**base, "status": "invalid", "failure_category": "invalid_runner_result"}
    if applied.returncode != 0:
        return {**base, "status": "invalid", "failure_category": "patch_apply"}
    base["patch_applied"] = True
    saw_timeout = False
    saw_infra = False
    saw_invalid = False
    saw_f2p_failure = False
    saw_p2p_regression = False
    for name in f2p:
        try:
            result = _run_method(runner, "run_test", workspace, name, timeout_seconds)
        except Exception as exc:
            return {**base, "status": "infra_error", "failure_category": "runner", "detail": str(exc)}
        base["f2p"]["results"].append(_result_record(name, result))
        if result.timed_out:
            saw_timeout = True
            continue
        if result.infra_error:
            saw_infra = True
            continue
        if result.returncode is None:
            saw_invalid = True
            continue
        if result.returncode == 0:
            base["f2p"]["passed"] += 1
        else:
            base["f2p"]["failed"] += 1
            saw_f2p_failure = True
    for name in p2p:
        try:
            result = _run_method(runner, "run_test", workspace, name, timeout_seconds)
        except Exception as exc:
            return {**base, "status": "infra_error", "failure_category": "runner", "detail": str(exc)}
        base["p2p"]["results"].append(_result_record(name, result))
        if result.timed_out:
            saw_timeout = True
            continue
        if result.infra_error:
            saw_infra = True
            continue
        if result.returncode is None:
            saw_invalid = True
            continue
        if result.returncode == 0:
            base["p2p"]["passed"] += 1
        else:
            base["p2p"]["regressions"] += 1
            saw_p2p_regression = True
    if saw_timeout:
        return {**base, "status": "timeout", "failure_category": "timeout"}
    if saw_infra:
        return {**base, "status": "infra_error", "failure_category": "infrastructure"}
    if saw_invalid:
        return {**base, "status": "invalid", "failure_category": "invalid_runner_result"}
    if saw_f2p_failure:
        return {**base, "status": "failed", "failure_category": "f2p_failure"}
    if saw_p2p_regression:
        return {**base, "status": "failed", "failure_category": "p2p_regression"}
    return {**base, "status": "conditional_resolved", "failure_category": None}


evaluate_task = evaluate_patch
build_patch_given_loc_package = build_public_package


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="stage_capability_eval/manifests/all_swebench_verified.jsonl")
    parser.add_argument("--oracle-loc", default="stage_capability_eval/oracle_loc/oracle_loc.jsonl")
    parser.add_argument("--out", default="stage_capability_eval/manifests/patch_given_loc.jsonl")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    rows = build_public_package(args.manifest, args.oracle_loc, limit=args.limit)
    digest = write_jsonl(args.out, rows)
    print(json.dumps({"stage": STAGE, "rows": len(rows), "sha256": digest, "dry_run": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
