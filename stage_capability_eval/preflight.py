#!/usr/bin/env python3
"""Old/fixed reference preflight with resumable, normalized status records.

The module deliberately keeps execution behind an injected runner.  This makes
the protocol testable without a Docker daemon and allows the R2E-Gym runner to
be wired in later without changing status semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


VALID_STATUSES = {
    "completed",
    "passed",
    "failed",
    "skipped",
    "timeout",
    "infra_error",
    "invalid",
    "not_applicable",
}


@dataclass(frozen=True)
class CommandResult:
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    infra_error: bool = False
    duration_seconds: float | None = None


@dataclass
class PreflightRecord:
    instance_id: str
    phase: str
    status: str
    failure_category: str | None = None
    returncode: int | None = None
    stdout_sha256: str | None = None
    stderr_sha256: str | None = None
    duration_seconds: float | None = None
    command: list[str] | None = None
    detail: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"unknown preflight status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def normalize_command_result(
    *,
    returncode: int | None,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
    infra_error: bool = False,
    duration_seconds: float | None = None,
) -> PreflightRecord:
    """Map process outcomes to stable evaluator statuses.

    A zero exit code is a successful command even with empty output; output
    content is retained only through hashes in the record to avoid leaking
    reference test details into the public manifest.
    """
    if timed_out:
        status, category = "timeout", "timeout"
    elif infra_error:
        status, category = "infra_error", "infrastructure"
    elif returncode is None:
        status, category = "invalid", "invalid_process_result"
    elif returncode == 0:
        status, category = "passed", None
    else:
        status, category = "failed", "test_failure"
    return PreflightRecord(
        instance_id="<command>",
        phase="command",
        status=status,
        failure_category=category,
        returncode=returncode,
        stdout_sha256=_sha256(stdout),
        stderr_sha256=_sha256(stderr),
        duration_seconds=duration_seconds,
    )


def parse_test_names(value: Any) -> list[str]:
    """Normalize SWE-bench F2P/P2P fields without accepting arbitrary objects."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [line.strip() for line in value.splitlines() if line.strip()]
    return parse_test_names(parsed)


def run_command(command: Sequence[str], *, timeout_seconds: float = 300.0, cwd: str | Path | None = None) -> CommandResult:
    """Run one command and classify missing executables as infrastructure errors."""
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            returncode=None,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            timed_out=True,
            duration_seconds=time.monotonic() - started,
        )
    except (OSError, ValueError) as exc:
        return CommandResult(
            returncode=None,
            stderr=str(exc),
            infra_error=True,
            duration_seconds=time.monotonic() - started,
        )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        duration_seconds=time.monotonic() - started,
    )


def apply_gold_patch(workspace: str | Path, patch: str) -> PreflightRecord:
    """Validate and apply a gold source patch in a fixed workspace.

    The check is performed before the mutating apply step.  A malformed or
    unappliable patch is an evaluator-invalid result, never a model failure.
    """
    target = Path(workspace)
    if not target.is_dir() or "diff --git " not in str(patch or ""):
        return PreflightRecord(
            instance_id="<patch>",
            phase="fixed_patch",
            status="invalid",
            failure_category="patch_apply",
            detail="workspace or unified diff is invalid",
        )
    patch_file: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".patch", dir=target, delete=False) as handle:
            handle.write(str(patch))
            patch_file = Path(handle.name)
        check = run_command(["git", "apply", "--check", str(patch_file)], cwd=target)
        if check.returncode != 0 or check.infra_error or check.timed_out:
            return PreflightRecord(
                instance_id="<patch>",
                phase="fixed_patch",
                status="invalid" if not check.infra_error else "infra_error",
                failure_category="patch_apply" if not check.infra_error else "infrastructure",
                returncode=check.returncode,
                stdout_sha256=_sha256(check.stdout),
                stderr_sha256=_sha256(check.stderr),
                duration_seconds=check.duration_seconds,
                detail="git apply --check failed",
            )
        applied = run_command(["git", "apply", str(patch_file)], cwd=target)
        if applied.returncode != 0 or applied.infra_error or applied.timed_out:
            return PreflightRecord(
                instance_id="<patch>",
                phase="fixed_patch",
                status="invalid" if not applied.infra_error else "infra_error",
                failure_category="patch_apply" if not applied.infra_error else "infrastructure",
                returncode=applied.returncode,
                stdout_sha256=_sha256(applied.stdout),
                stderr_sha256=_sha256(applied.stderr),
                duration_seconds=applied.duration_seconds,
                detail="git apply failed after successful check",
            )
        return PreflightRecord(
            instance_id="<patch>",
            phase="fixed_patch",
            status="passed",
            returncode=0,
            duration_seconds=(check.duration_seconds or 0.0) + (applied.duration_seconds or 0.0),
        )
    finally:
        if patch_file is not None:
            try:
                patch_file.unlink()
            except OSError:
                pass


def _record_for_task(
    row: Mapping[str, Any],
    *,
    phase: str,
    command: Sequence[str] | None,
    result: CommandResult | None,
    detail: str | None = None,
) -> PreflightRecord:
    if result is None:
        record = PreflightRecord(
            instance_id=str(row["instance_id"]),
            phase=phase,
            status="not_applicable",
            failure_category="dry_run",
            detail=detail or "execution not requested",
        )
    else:
        normalized = normalize_command_result(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
            infra_error=result.infra_error,
            duration_seconds=result.duration_seconds,
        )
        normalized.instance_id = str(row["instance_id"])
        normalized.phase = phase
        normalized.command = list(command) if command is not None else None
        normalized.detail = detail
        record = normalized
    record.metadata.update(
        {
            "repo": row.get("repo"),
            "base_commit": row.get("base_commit"),
            "environment_setup_commit": row.get("environment_setup_commit"),
            "docker_image": row.get("docker_image"),
            "f2p_count": len(parse_test_names(row.get("FAIL_TO_PASS"))),
            "p2p_count": len(parse_test_names(row.get("PASS_TO_PASS"))),
        }
    )
    return record


def collect_reference_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    runner: Callable[[Sequence[str]], CommandResult] | None = None,
    command_builder: Callable[[Mapping[str, Any], str], Sequence[str]] | None = None,
    limit: int | None = None,
    existing: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> list[PreflightRecord]:
    """Collect buggy/fixed phase records, resuming records already present.

    ``command_builder`` is intentionally injected: the default dry-run emits
    records but never assumes a Docker command or mutates a repository.
    """
    records: list[PreflightRecord] = []
    existing = existing or {}
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        iid = str(row["instance_id"])
        for phase in ("buggy", "fixed"):
            previous = existing.get((iid, phase))
            if previous is not None:
                records.append(PreflightRecord(**{k: previous[k] for k in PreflightRecord.__dataclass_fields__ if k in previous}))
                continue
            command = list(command_builder(row, phase)) if command_builder is not None else None
            result = runner(command) if runner is not None and command is not None else None
            records.append(_record_for_task(row, phase=phase, command=command, result=result))
    return records


def write_records(path: str | Path, records: Iterable[PreflightRecord]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) for record in records]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _load_manifest(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="stage_capability_eval/evaluator_private/source_records.jsonl")
    parser.add_argument("--out", default="stage_capability_eval/manifests/preflight_status.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true", help="write explicit not_applicable records without execution")
    args = parser.parse_args(argv)
    rows = _load_manifest(args.manifest)
    records = collect_reference_records(rows, limit=args.limit)
    digest = write_records(args.out, records)
    print(json.dumps({"records": len(records), "sha256": digest, "dry_run": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
