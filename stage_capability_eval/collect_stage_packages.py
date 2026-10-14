#!/usr/bin/env python3
"""Materialize public REPRO/PATCH task packages and private verifier metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

try:  # support both ``python -m`` and the README's direct script command
    from .patch import build_public_package
    from .repro import build_private_metadata, build_public_repro_tasks, write_jsonl
except ImportError:  # pragma: no cover - exercised by direct CLI invocation
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from stage_capability_eval.patch import build_public_package
    from stage_capability_eval.repro import build_private_metadata, build_public_repro_tasks, write_jsonl


def _rows(source: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        return [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [dict(row) for row in source]


def _patch_metadata(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in sorted(rows, key=lambda item: str(item.get("instance_id", ""))):
        patch = str(row.get("patch", ""))
        result.append({
            "schema_version": "stage_capability_patch_private_v1",
            "instance_id": row.get("instance_id"),
            "base_commit": row.get("base_commit"),
            "docker_image": row.get("docker_image"),
            "FAIL_TO_PASS": list(row.get("FAIL_TO_PASS") or []),
            "PASS_TO_PASS": list(row.get("PASS_TO_PASS") or []),
            "gold_patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        })
    return result


def collect_packages(
    manifest: str | Path,
    oracle_loc: str | Path,
    source_records: str | Path,
    out: str | Path = "stage_capability_eval",
) -> dict[str, Any]:
    public_manifest = _rows(manifest)
    oracle = _rows(oracle_loc)
    private_rows = _rows(source_records)
    root = Path(out)
    public_repro = build_public_repro_tasks(public_manifest, oracle)
    public_patch = build_public_package(public_manifest, oracle)
    private_repro = build_private_metadata(private_rows)
    private_patch = _patch_metadata(private_rows)
    outputs = {
        root / "manifests/repro_given_loc.jsonl": public_repro,
        root / "manifests/patch_given_loc.jsonl": public_patch,
        root / "evaluator_private/repro_metadata.jsonl": private_repro,
        root / "evaluator_private/patch_verifier_metadata.jsonl": private_patch,
    }
    hashes = {}
    for path, rows in outputs.items():
        hashes[str(path)] = write_jsonl(path, rows)
    root.joinpath("manifests/stage_package_hashes.json").parent.mkdir(parents=True, exist_ok=True)
    root.joinpath("manifests/stage_package_hashes.json").write_text(
        json.dumps({"schema_version": "stage_capability_packages_v1", "hashes": hashes}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"repro_count": len(public_repro), "patch_count": len(public_patch), "private_repro_count": len(private_repro), "private_patch_count": len(private_patch), "hashes": hashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="stage_capability_eval/manifests/all_swebench_verified.jsonl")
    parser.add_argument("--oracle-loc", default="stage_capability_eval/oracle_loc/oracle_loc.jsonl")
    parser.add_argument("--source-records", default="stage_capability_eval/evaluator_private/source_records.jsonl")
    parser.add_argument("--out", default="stage_capability_eval")
    args = parser.parse_args(argv)
    print(json.dumps(collect_packages(args.manifest, args.oracle_loc, args.source_records, args.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
