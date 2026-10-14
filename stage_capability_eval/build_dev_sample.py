#!/usr/bin/env python3
"""Build a deterministic, public development sample for evaluator smoke tests."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PRIVATE_FIELDS = {"patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "parsed_commit", "run_tests"}


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _bucket(row: Mapping[str, Any], labels: Mapping[str, Mapping[str, Any]]) -> tuple[str, int, str]:
    iid = str(row.get("instance_id", ""))
    status = str(labels.get(iid, {}).get("mapping_status", "unknown"))
    shape = len(row.get("source_patch_files", []))
    return str(row.get("repo", "")), min(shape, 3), status


def build_sample(rows: Iterable[Mapping[str, Any]], labels: Mapping[str, Mapping[str, Any]], *, size: int = 25) -> list[dict[str, Any]]:
    if size <= 0:
        raise ValueError("sample size must be positive")
    buckets: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_bucket(row, labels)].append(row)
    for values in buckets.values():
        values.sort(key=lambda item: str(item.get("instance_id", "")))
    ordered: list[Mapping[str, Any]] = []
    keys = sorted(buckets)
    while len(ordered) < size and keys:
        next_keys: list[tuple[str, int, str]] = []
        for key in keys:
            values = buckets[key]
            if values:
                ordered.append(values.pop(0))
                if len(ordered) >= size:
                    break
            if values:
                next_keys.append(key)
        keys = next_keys
    result: list[dict[str, Any]] = []
    for row in ordered:
        public = {key: value for key, value in dict(row).items() if key not in PRIVATE_FIELDS}
        public["sample_bucket"] = list(_bucket(row, labels))
        result.append(public)
    return result


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    target.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def build_report(sample: list[Mapping[str, Any]], *, manifest_path: str, label_path: str, sample_sha256: str) -> str:
    repos = sorted({str(row.get("repo", "")) for row in sample})
    statuses = defaultdict(int)
    for row in sample:
        statuses[str(row.get("sample_bucket", ["", "", "unknown"])[2])] += 1
    status_lines = "\n".join(f"- `{key}`: {value}" for key, value in sorted(statuses.items())) or "- none"
    return f"""# Stage capability development sample

The sample is a public-input smoke set. It contains no gold source patch,
test patch, test names, or parsed source snapshot.

- Sample rows: {len(sample)}
- Repositories: {', '.join(repos) or 'none'}
- Sample SHA-256: `{sample_sha256}`
- Public manifest: `{manifest_path}`
- Private LOC labels: `{label_path}`

Mapping status distribution:

{status_lines}

Collection status: **development only**. Full-scale model collection is not
approved until real buggy/fixed preflight and both evaluator entry modes pass.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="stage_capability_eval/manifests/all_swebench_verified.jsonl")
    parser.add_argument("--labels", default="stage_capability_eval/evaluator_private/loc_labels.jsonl")
    parser.add_argument("--out", default="stage_capability_eval")
    parser.add_argument("--size", type=int, default=25)
    args = parser.parse_args(argv)
    rows = _load_jsonl(args.manifest)
    label_rows = _load_jsonl(args.labels)
    labels = {str(row["instance_id"]): row for row in label_rows}
    sample = build_sample(rows, labels, size=args.size)
    root = Path(args.out)
    digest = write_jsonl(root / "manifests/dev_sample.jsonl", sample)
    report = build_report(sample, manifest_path=args.manifest, label_path=args.labels, sample_sha256=digest)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports/dev_sample.md").write_text(
        report + "\n\nRun `python -m unittest discover -s stage_capability_eval/tests -p 'test*.py'` before collection.\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(sample), "sha256": digest, "path": str(root / "manifests/dev_sample.jsonl")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
