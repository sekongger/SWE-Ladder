#!/usr/bin/env python3
"""Derive auditable location labels from SWE-bench unified diffs.

The module deliberately uses only the Python standard library.  A checkout at
``--repo-root/<repo>`` is preferred; the frozen manifest's ``parsed_commit``
old-file snapshots are a useful offline fallback.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "stage_capability_loc_v1"
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def parse_unified_diff(patch: str) -> list[dict[str, Any]]:
    """Parse file paths and old/new changed line numbers from a unified diff."""
    files: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    hunk: dict[str, Any] | None = None
    old_line = new_line = 0
    for line in str(patch or "").splitlines():
        if line.startswith("diff --git "):
            current = None
            hunk = None
            continue
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            raw = line[4:].strip().split("\t", 1)[0]
            path = raw[2:] if raw.startswith(("a/", "b/")) else raw
            if path == "/dev/null":
                path = ""
            current = {"path": path, "hunks": []}
            files.append(current)
            hunk = None
            continue
        match = _HUNK.match(line)
        if match and current is not None:
            old_line = int(match.group(1)); new_line = int(match.group(3))
            hunk = {
                "old_start": old_line, "old_count": int(match.group(2) or 1),
                "new_start": new_line, "new_count": int(match.group(4) or 1),
                "old_changed_lines": [], "new_changed_lines": [],
            }
            current["hunks"].append(hunk)
            continue
        if hunk is None or current is None or line.startswith("\\"):
            continue
        if line.startswith("-") and not line.startswith("---"):
            hunk["old_changed_lines"].append(old_line); old_line += 1
        elif line.startswith("+") and not line.startswith("+++"):
            hunk["new_changed_lines"].append(new_line); new_line += 1
        elif line.startswith(" "):
            old_line += 1; new_line += 1
    return files


def _snapshot_map(row: dict[str, Any]) -> dict[str, str]:
    raw = row.get("parsed_commit")
    if isinstance(raw, str):
        try: raw = json.loads(raw)
        except json.JSONDecodeError: raw = None
    result: dict[str, str] = {}
    for item in (raw or {}).get("file_diffs", []) if isinstance(raw, dict) else []:
        if not isinstance(item, dict): continue
        old = item.get("minus_file")
        path = item.get("old_path") or item.get("path") or item.get("filename")
        if not path and isinstance(old, dict): path = old.get("path")
        if not path:
            # The upstream parsed representation often stores paths only in
            # the patch; matching by content is intentionally not attempted.
            continue
        content = item.get("old_file_content")
        if isinstance(content, str):
            normalized_path = str(path)
            if normalized_path.startswith("a/"):
                normalized_path = normalized_path[2:]
            result[normalized_path] = content
    # Most records omit a path in file_diffs. Match in patch order as a safe
    # fallback, provided the cardinalities agree.
    if not result:
        items = (raw or {}).get("file_diffs", []) if isinstance(raw, dict) else []
        contents = [x.get("old_file_content") for x in items if isinstance(x, dict) and isinstance(x.get("old_file_content"), str)]
        paths = [x["path"] for x in parse_unified_diff(row.get("patch", ""))]
        if len(contents) == len(paths): result.update(zip(paths, contents))
    return result


def _entities(source: str) -> list[dict[str, Any]]:
    try: tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError): return []
    out: list[dict[str, Any]] = []
    stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            name = ".".join(stack + [node.name]); out.append({"qualified_name": name, "kind": "class", "start_line": node.lineno, "end_line": getattr(node, "end_lineno", node.lineno)})
            stack.append(node.name); self.generic_visit(node); stack.pop()
        def _function(self, node: ast.AST) -> None:
            name = ".".join(stack + [getattr(node, "name")]); out.append({"qualified_name": name, "kind": "function", "start_line": node.lineno, "end_line": getattr(node, "end_lineno", node.lineno)})
            stack.append(getattr(node, "name")); self.generic_visit(node); stack.pop()
        visit_FunctionDef = _function
        visit_AsyncFunctionDef = _function
    Visitor().visit(tree)
    return out


def _checkout_source(repo_root: str | Path | None, repo: str, path: str) -> str | None:
    if not repo_root: return None
    root = Path(repo_root) / repo
    candidate = root / path
    try:
        return candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError): return None


def label_row(row: dict[str, Any], repo_root: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    files = parse_unified_diff(row.get("patch", ""))
    snapshots = _snapshot_map(row)
    private_files: list[dict[str, Any]] = []; oracle_files: list[dict[str, Any]] = []; issues: list[dict[str, Any]] = []
    for file in files:
        path = file["path"]
        if not path or path.lower().startswith(("test", "tests/")) or "/test" in path.lower() or path.endswith("_test.py"):
            issues.append({"instance_id": row.get("instance_id"), "path": path, "status": "test_only"}); continue
        source = _checkout_source(repo_root, row.get("repo", ""), path)
        source_status = "checkout"
        if source is None: source = snapshots.get(path); source_status = "snapshot" if source is not None else "unavailable"
        entities = _entities(source) if source is not None and path.endswith(".py") else []
        labeled_hunks = []; all_names: set[str] = set(); any_hit = False; all_hit = True
        for h in file["hunks"]:
            lines = set(h["old_changed_lines"])
            if not lines and h["new_changed_lines"]:
                status = "new_function" if source_status != "unavailable" and any(e["start_line"] <= h["new_changed_lines"][0] <= e["end_line"] for e in entities) else "no_hit"
                hits = [e for e in entities if e["start_line"] <= h["new_changed_lines"][0] <= e["end_line"]] if status == "new_function" else []
            else:
                hits = [e for e in entities if any(e["start_line"] <= line <= e["end_line"] for line in lines)]
                status = "full" if hits and all(any(e["start_line"] <= line <= e["end_line"] for e in hits) for line in lines) else ("partial" if hits else "no_hit")
            if source_status == "unavailable": status = "unavailable"
            any_hit |= bool(hits); all_hit &= status in ("full", "new_function")
            all_names.update(e["qualified_name"] for e in hits)
            labeled_hunks.append({"old_start": h["old_start"], "old_count": h["old_count"], "new_start": h["new_start"], "new_count": h["new_count"], "old_changed_lines": h["old_changed_lines"], "mapping_status": status, "entities": hits})
        file_status = "unavailable" if source_status == "unavailable" else ("full" if all_hit and labeled_hunks else ("partial" if any_hit else "no_hit"))
        private_files.append({"path": path, "source_status": source_status, "mapping_status": file_status, "hunks": labeled_hunks})
        oracle_files.append({"path": path, "mapping_status": file_status, "functions": sorted(all_names), "old_context": [{"start": h["old_start"], "count": h["old_count"]} for h in file["hunks"]]})
        if file_status not in ("full",): issues.append({"instance_id": row.get("instance_id"), "path": path, "status": file_status})
    aggregate = "unavailable" if any(f["mapping_status"] == "unavailable" for f in private_files) else ("full" if private_files and all(f["mapping_status"] == "full" for f in private_files) else ("partial" if any(f["mapping_status"] in ("full", "partial", "new_function") for f in private_files) else "no_hit"))
    base = {"schema_version": SCHEMA_VERSION, "instance_id": row.get("instance_id"), "repo": row.get("repo"), "base_commit": row.get("base_commit"), "patch_sha256": _sha(str(row.get("patch", ""))), "mapping_status": aggregate}
    private = {**base, "files": private_files}; oracle = {**base, "files": oracle_files}
    return private, oracle, issues


def _update_task_sets(root: Path, labels: Iterable[dict[str, Any]]) -> None:
    """Publish the function-level denominator only after labels are available."""
    path = root / "manifests/task_sets.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    function_ids = sorted(
        str(row["instance_id"])
        for row in labels
        if row.get("mapping_status") in {"full", "partial"}
        and any(h.get("entities") for f in row.get("files", []) for h in f.get("hunks", []))
    )
    sets = data.setdefault("sets", {})
    sets["loc_function_set"] = function_ids
    common = set(sets.get("loc_file_set", [])) & set(sets.get("repro_set", [])) & set(sets.get("patch_cond_set", [])) & set(function_ids)
    sets["common_analysis_set"] = sorted(common)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    public_manifest = root / "manifests/all_swebench_verified.jsonl"
    if public_manifest.exists():
        rows = [json.loads(line) for line in public_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        function_set = set(function_ids)
        for row in rows:
            capabilities = row.setdefault("task_capabilities", {})
            capabilities["loc_function"] = row.get("instance_id") in function_set
        payload = "".join(canonical_json(row).decode("utf-8") for row in rows)
        public_manifest.write_text(payload, encoding="utf-8")
        data["manifest_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def collect(source: str | Path = "stage_capability_eval/evaluator_private/source_records.jsonl", out: str | Path = "stage_capability_eval", repo_root: str | Path | None = None) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    private: list[dict[str, Any]] = []; oracle: list[dict[str, Any]] = []; issues: list[dict[str, Any]] = []
    for row in rows:
        p, o, i = label_row(row, repo_root); private.append(p); oracle.append(o); issues.extend(i)
    root = Path(out); (root / "evaluator_private").mkdir(parents=True, exist_ok=True); (root / "oracle_loc").mkdir(parents=True, exist_ok=True); (root / "manifests").mkdir(parents=True, exist_ok=True)
    for path, records in ((root / "evaluator_private/loc_labels.jsonl", private), (root / "oracle_loc/oracle_loc.jsonl", oracle), (root / "manifests/loc_mapping_issues.jsonl", issues)):
        path.write_bytes(b"".join(canonical_json(x) for x in records))
    _update_task_sets(root, private)
    return {"rows": len(rows), "issues": len(issues), "private": str(root / "evaluator_private/loc_labels.jsonl"), "oracle": str(root / "oracle_loc/oracle_loc.jsonl")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--source", default="stage_capability_eval/evaluator_private/source_records.jsonl"); ap.add_argument("--out", default="stage_capability_eval"); ap.add_argument("--repo-root", default=None)
    args = ap.parse_args(argv)
    print(json.dumps(collect(args.source, args.out, args.repo_root), indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
