"""Public REPRO_GIVEN_LOC packaging and double-version test evaluation.

The evaluator treats a model artifact as a frozen value and scores the test
outcomes collected by the runner. A runner's process exit code or aggregate
``pass``/``fail`` status is never enough to establish a reproduction.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

try:
    from .package_common import public_task_row
except ImportError:  # pragma: no cover - exercised by direct CLI invocation
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from stage_capability_eval.package_common import public_task_row

PUBLIC_DROP_FIELDS = {
    "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_TEST", "PASS_TO_PASS", "parsed_commit",
    "run_tests", "source_records", "gold_patch", "fixed_repository",
}
RUN_STATUSES = {"completed", "timeout", "infra_error", "model_invalid"}
TEST_OUTCOMES = {"passed", "failed", "skipped", "error"}
_LEGACY_STATUSES = {"pass", "fail", "skipped", "timeout", "infra_error", "invalid", "model_invalid"}


def _rows(source: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        return [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [dict(row) for row in source]


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _public_oracle_loc(label: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the declared location assistance, never a gold patch hash."""
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


def build_public_repro_tasks(
    manifest: str | Path | Iterable[Mapping[str, Any]],
    oracle_loc: str | Path | Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join public manifest rows with oracle locations in stable ID order."""
    labels = {str(row.get("instance_id")): row for row in _rows(oracle_loc)}
    result: list[dict[str, Any]] = []
    for row in sorted(_rows(manifest), key=lambda item: str(item.get("instance_id", ""))):
        instance_id = str(row.get("instance_id", ""))
        label = labels.get(instance_id)
        if not instance_id or label is None:
            continue
        public = public_task_row(row)
        public["oracle_loc"] = _public_oracle_loc(label)
        public["task_type"] = "REPRO_GIVEN_LOC"
        public["schema_version"] = "stage_capability_repro_v1"
        result.append(public)
    return result


def build_private_metadata(source: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract hidden evaluator references without making them public inputs."""
    result = []
    for row in sorted(_rows(source), key=lambda item: str(item.get("instance_id", ""))):
        item = {"instance_id": row.get("instance_id"),
                "FAIL_TO_PASS": list(row.get("FAIL_TO_PASS") or []),
                "PASS_TO_PASS": list(row.get("PASS_TO_PASS") or []),
                "test_patch": row.get("test_patch", "")}
        item["test_patch_sha256"] = hashlib.sha256(str(item["test_patch"]).encode()).hexdigest()
        result.append(item)
    return result


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> str:
    payload = "".join(_canonical(row) + "\n" for row in rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def _legacy_test_ids(artifact: Any) -> list[str]:
    if not isinstance(artifact, Mapping) or "files" in artifact or "command" in artifact:
        return []
    tests = artifact.get("tests")
    if not isinstance(tests, (list, tuple)):
        return []
    return [item for item in tests if isinstance(item, str) and item.strip()]


def validate_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy a frozen generated artifact.

    The canonical artifact has ``files`` (a path-to-text mapping) and a
    non-empty ``command`` list. A small legacy ``tests`` form is accepted for
    old callers; it is converted into a deterministic synthetic command.
    """
    if not isinstance(artifact, Mapping):
        raise ValueError("artifact must be a mapping")
    legacy_ids = _legacy_test_ids(artifact)
    if legacy_ids and "files" not in artifact and "command" not in artifact:
        return {"files": {}, "command": ["legacy-tests", *legacy_ids]}
    files = artifact.get("files")
    command = artifact.get("command")
    if not isinstance(files, Mapping):
        raise ValueError("artifact.files must be a mapping")
    if not isinstance(command, (list, tuple)) or not command:
        raise ValueError("artifact.command must be a non-empty list")
    normalized_files: dict[str, str] = {}
    for raw_path, content in files.items():
        if not isinstance(raw_path, str) or not raw_path.strip() or "\x00" in raw_path:
            raise ValueError("artifact file paths must be non-empty strings")
        path_for_parts = raw_path.replace("\\", "/")
        posix_path = PurePosixPath(path_for_parts)
        if posix_path.is_absolute() or any(part in {"", ".", ".."} for part in posix_path.parts):
            raise ValueError(f"invalid artifact path: {raw_path!r}")
        if "\\" in raw_path or str(posix_path) != raw_path:
            raise ValueError(f"artifact path is not canonical: {raw_path!r}")
        if not isinstance(content, str):
            raise ValueError(f"artifact content for {raw_path!r} must be text")
        normalized_files[raw_path] = content
    normalized_command: list[str] = []
    for token in command:
        if not isinstance(token, str) or not token.strip() or "\x00" in token:
            raise ValueError("artifact command entries must be non-empty strings")
        normalized_command.append(token)
    result: dict[str, Any] = {"files": dict(sorted(normalized_files.items())), "command": normalized_command}
    if "source_provenance" in artifact:
        result["source_provenance"] = copy.deepcopy(artifact["source_provenance"])
    return result


def artifact_sha256(artifact: Mapping[str, Any]) -> str:
    """Hash canonical file contents and runner command, independent of key order."""
    normalized = validate_artifact(artifact)
    payload = {"files": normalized["files"], "command": normalized["command"]}
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


# Descriptive aliases used by integrations.
freeze_artifact = validate_artifact
hash_artifact = artifact_sha256


def _runner_result(value: Any) -> str:
    """Compatibility helper returning a normalized top-level status."""
    if isinstance(value, str):
        status = value.lower()
    elif isinstance(value, Mapping):
        status = str(value.get("status", "model_invalid")).lower()
    else:
        status = "model_invalid"
    if status in RUN_STATUSES:
        return status
    if status in {"pass", "fail"}:
        return "completed"
    return "model_invalid"


def _has_collection_error(result: Mapping[str, Any]) -> bool:
    for key in ("collection_error", "import_error", "collection_errors", "import_errors"):
        if result.get(key):
            return True
    return False


def _copy_raw(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return repr(value)


class ReproEvaluator:
    """Evaluate a frozen artifact against buggy and fixed versions."""

    def __init__(self, runner: Callable[[Mapping[str, Any], str, Any], Any], *, protocol_id: str = "stage_capability_v1"):
        self.runner = runner
        self.protocol_id = protocol_id

    def _invoke(
        self,
        task: Mapping[str, Any],
        version: str,
        artifact: dict[str, Any],
        expected_hash: str,
        legacy_ids: list[str],
    ) -> dict[str, Any]:
        # Each phase receives an independent copy. The pre/post hash catches a
        # runner that mutates its copy while keeping the evaluator's value safe.
        phase_artifact = copy.deepcopy(artifact)
        try:
            raw = self.runner(copy.deepcopy(dict(task)), version, phase_artifact)
        except TimeoutError as exc:
            return {"status": "timeout", "raw": {"status": "timeout", "error_type": type(exc).__name__}, "exception": True}
        except Exception as exc:  # runner infrastructure, never a model score
            return {"status": "infra_error", "raw": {"status": "infra_error", "error_type": type(exc).__name__, "error": str(exc)}, "exception": True}
        try:
            changed = artifact_sha256(phase_artifact) != expected_hash
        except Exception:
            changed = True
        if changed:
            return {"status": "infra_error", "failure_category": "artifact_tampered", "raw": _copy_raw(raw), "artifact_tampered": True}
        if isinstance(raw, str):
            raw = {"status": raw}
        if not isinstance(raw, Mapping):
            return {"status": "infra_error", "failure_category": "invalid_runner_result", "raw": _copy_raw(raw)}
        raw_copy = _copy_raw(raw)
        status = str(raw.get("status", "")).lower()
        # Legacy callers returned pass/fail without per-test records. Keep
        # that API only for the legacy artifact shape; new artifacts cannot be
        # scored from an aggregate status.
        if legacy_ids and status in {"pass", "fail"} and "tests" not in raw:
            value = "passed" if status == "pass" else "failed"
            raw = dict(raw)
            raw["status"] = "completed"
            raw["tests"] = {test_id: value for test_id in legacy_ids}
            status = "completed"
        if legacy_ids and status in _LEGACY_STATUSES and "artifact_sha256" not in raw:
            raw = dict(raw)
            raw["artifact_sha256"] = expected_hash
        if str(raw.get("artifact_sha256", "")) != expected_hash:
            return {"status": "infra_error", "failure_category": "artifact_hash_mismatch", "raw": raw_copy}
        if status not in RUN_STATUSES:
            return {"status": "infra_error", "failure_category": "invalid_runner_result", "raw": raw_copy}
        if status != "completed":
            return {"status": status, "raw": raw_copy, "tests": _copy_raw(raw.get("tests")) if "tests" in raw else None, "result": dict(raw)}
        tests = raw.get("tests")
        if not isinstance(tests, Mapping) or not tests:
            return {"status": "infra_error", "failure_category": "invalid_runner_result", "raw": raw_copy, "result": dict(raw)}
        normalized_tests: dict[str, str] = {}
        for test_id, outcome in tests.items():
            if not isinstance(test_id, str) or not test_id.strip() or outcome not in TEST_OUTCOMES:
                return {"status": "infra_error", "failure_category": "invalid_runner_result", "raw": raw_copy, "result": dict(raw)}
            normalized_tests[test_id] = outcome
        return {"status": "completed", "raw": raw_copy, "tests": normalized_tests, "result": dict(raw)}

    def _score_completed(self, buggy: dict[str, Any], fixed: dict[str, Any]) -> tuple[bool, str | None]:
        if _has_collection_error(buggy.get("result", {})) or _has_collection_error(fixed.get("result", {})):
            return False, "test_error"
        buggy_tests = buggy.get("tests") or {}
        fixed_tests = fixed.get("tests") or {}
        if set(buggy_tests) != set(fixed_tests):
            return False, "test_id_mismatch"
        if not buggy_tests:
            return False, "no_tests_collected"
        if any(value == "error" for value in (*buggy_tests.values(), *fixed_tests.values())):
            return False, "test_error"
        if all(value == "skipped" for value in buggy_tests.values()) and all(value == "skipped" for value in fixed_tests.values()):
            return False, "all_tests_skipped"
        if not all(value == "passed" for value in fixed_tests.values()):
            return False, "fixed_not_all_passed"
        if not any(value == "failed" for value in buggy_tests.values()):
            return False, "no_buggy_failure"
        return True, None

    def evaluate(
        self,
        tasks: Iterable[Mapping[str, Any]],
        artifacts: Mapping[str, Any],
        *,
        dry_run: bool = False,
        model_id: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        ordered = sorted(tasks, key=lambda item: str(item.get("instance_id", "")))
        records: list[dict[str, Any]] = []
        for task in ordered:
            instance_id = str(task.get("instance_id", ""))
            supplied = artifacts.get(instance_id) if isinstance(artifacts, Mapping) else None
            base: dict[str, Any] = {
                "schema_version": "stage_capability_repro_v1", "protocol_id": self.protocol_id,
                "instance_id": instance_id, "model_id": model_id, "seed": seed,
                "valid_reproduction": False, "artifact_sha256": None,
            }
            try:
                artifact = validate_artifact(supplied)
                expected_hash = artifact_sha256(artifact)
                legacy_ids = _legacy_test_ids(supplied)
                base["artifact_sha256"] = expected_hash
                if artifact.get("source_provenance") is not None:
                    base["source_provenance"] = _copy_raw(artifact["source_provenance"])
            except (TypeError, ValueError) as exc:
                base.update(status="model_invalid", failure_category="invalid_artifact", error=str(exc))
                records.append(base)
                continue
            if dry_run:
                base.update(status="dry_run", failure_category="dry_run")
                records.append(base)
                continue
            buggy = self._invoke(task, "buggy", artifact, expected_hash, legacy_ids)
            fixed = self._invoke(task, "fixed", artifact, expected_hash, legacy_ids)
            base["buggy_outcome"] = _copy_raw(buggy.get("raw"))
            base["fixed_outcome"] = _copy_raw(fixed.get("raw"))
            if "tests" in buggy:
                base["buggy_tests"] = _copy_raw(buggy.get("tests"))
            if "tests" in fixed:
                base["fixed_tests"] = _copy_raw(fixed.get("tests"))
            if buggy["status"] == "infra_error" or fixed["status"] == "infra_error":
                category = buggy.get("failure_category") or fixed.get("failure_category") or "runner_infrastructure"
                base.update(status="infra_error", failure_category=category, buggy_status=buggy["status"], fixed_status=fixed["status"])
            elif buggy["status"] == "timeout" or fixed["status"] == "timeout":
                base.update(status="timeout", failure_category="test_timeout", buggy_status=buggy["status"], fixed_status=fixed["status"])
            elif buggy["status"] == "model_invalid" or fixed["status"] == "model_invalid":
                base.update(status="model_invalid", failure_category="runner_model_invalid", buggy_status=buggy["status"], fixed_status=fixed["status"])
            else:
                valid, category = self._score_completed(buggy, fixed)
                base.update(status="completed", failure_category=category, valid_reproduction=valid,
                            buggy_status="completed", fixed_status="completed")
            records.append(base)
        planned = len(records)
        valid_count = sum(bool(record["valid_reproduction"]) for record in records)
        dry_count = sum(record["status"] == "dry_run" for record in records)
        infra_count = sum(record["status"] == "infra_error" for record in records)
        timeout_count = sum(record["status"] == "timeout" for record in records)
        pending_count = infra_count + timeout_count + dry_count
        evaluated_count = planned - pending_count
        metric = None if planned == 0 or pending_count else valid_count / planned
        aggregate = {
            "task_count": planned, "planned_count": planned, "evaluated_count": evaluated_count,
            "completed_count": sum(record["status"] == "completed" for record in records),
            "valid_count": None if dry_count else valid_count,
            "failure_count": None if dry_count else planned - valid_count,
            "model_invalid_count": sum(record["status"] == "model_invalid" for record in records),
            "infra_error_count": infra_count, "timeout_count": timeout_count,
            "pending_count": pending_count, "valid_reproduction_rate": metric,
            "headline_valid_reproduction_rate": metric, "denominator": planned,
        }
        return {"schema_version": "stage_capability_repro_v1", "protocol_id": self.protocol_id,
                "records": records, "aggregate": aggregate}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or smoke-test REPRO_GIVEN_LOC inputs")
    parser.add_argument("--manifest", default="stage_capability_eval/manifests/all_swebench_verified.jsonl")
    parser.add_argument("--oracle", default="stage_capability_eval/oracle_loc/oracle_loc.jsonl")
    parser.add_argument("--out", default="stage_capability_eval/manifests/repro_given_loc.jsonl")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    tasks = build_public_repro_tasks(args.manifest, args.oracle)
    if args.sample_size:
        tasks = tasks[:args.sample_size]
    digest = write_jsonl(args.out, tasks)
    print(json.dumps({"task_count": len(tasks), "sha256": digest, "path": args.out,
                      "status": "dry_run" if args.dry_run else "packaged"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
