#!/usr/bin/env python3
"""Keep only the instances that exist in all three stages, and write a manifest.

`build_staged_verl_dataset.py` converts each stage independently, so the three stages
can end up with slightly different instance sets (a row is dropped when its stage's
privileged evidence cannot be derived — no gold hunks, no FAIL_TO_PASS, and so on).
The ladder needs the *same* instances on every rung: otherwise LOC, REPRO and PATCH
are trained and compared on different task populations, and a leg-to-leg difference
could be the population rather than the stage.

This takes a family root produced by the builder and writes the per-split
intersection, plus `manifest.json` recording what was dropped and what remains — the
numbers quoted in `data/README.md` come from that file.

  python3 scripts/data/intersect_stages.py \\
      --input-root  "$DATA_ROOT/my_family" \\
      --output-root "$DATA_ROOT/my_family_task_intersection"

Row order is preserved, so the selection is deterministic; the manifest records a
sha256 per output file (the bytes also depend on the pyarrow version that wrote them).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

STAGES = ("causal_localization", "localized_issue_reproduction", "patch_generation")


def _extra_info(value) -> dict:
    return json.loads(value) if isinstance(value, str) else (value or {})


def _instance_ids(table: pa.Table) -> list[str | None]:
    return [_extra_info(v).get("instance_id") for v in table.column("extra_info").to_pylist()]


def _summarize(table: pa.Table, path: Path) -> dict:
    infos = [_extra_info(v) for v in table.column("extra_info").to_pylist()]
    ids = [i.get("instance_id") for i in infos]
    return {
        "rows": table.num_rows,
        "unique_instance_ids": len(set(ids)),
        "duplicates": len(ids) - len(set(ids)),
        "source_dataset": dict(Counter(i.get("source_dataset") for i in infos)),
        "stage_field": dict(Counter(i.get("stage") for i in infos)),
        "missing_docker_image": sum(1 for i in infos if not i.get("docker_image")),
        "missing_FAIL_TO_PASS": sum(1 for i in infos if not i.get("FAIL_TO_PASS")),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def intersect_split(input_root: Path, output_root: Path, split: str, stages: tuple[str, ...]) -> dict | None:
    """Write the intersected `<stage>/<split>_verl.parquet` files; return the manifest entry."""
    tables: dict[str, pa.Table] = {}
    for stage in stages:
        path = input_root / stage / f"{split}_verl.parquet"
        if not path.exists():
            print(f"skip {split}: {path} does not exist")
            return None
        tables[stage] = pq.read_table(path)

    ids = {stage: _instance_ids(table) for stage, table in tables.items()}
    keep: set[str] = set.intersection(*({i for i in v if i} for v in ids.values()))

    entry: dict = {
        "input_rows": {s: t.num_rows for s, t in tables.items()},
        "input_unique_instance_ids": {s: len({i for i in v if i}) for s, v in ids.items()},
        "intersection_size": len(keep),
        "dropped": {s: len({i for i in v if i}) - len(keep) for s, v in ids.items()},
        "outputs": {},
    }

    for stage, table in tables.items():
        filtered = table.filter(pa.array([i in keep for i in ids[stage]]))
        out_path = output_root / stage / f"{split}_verl.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(filtered, out_path)
        summary = _summarize(filtered, out_path)
        entry["outputs"][stage] = {"path": str(out_path), "summary": summary}
        print(
            f"[{split}/{stage}] {table.num_rows} -> {filtered.num_rows} rows  "
            f"by source {summary['source_dataset']}"
        )
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", required=True, help="family root holding <stage>/<split>_verl.parquet")
    ap.add_argument("--output-root", required=True, help="where the intersected family is written")
    ap.add_argument("--splits", default="train,valid")
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())

    manifest = {
        "source_root": str(input_root),
        "output_root": str(output_root),
        "selection": "per-split intersection of instance_id across all three stages",
        "stages": list(stages),
        "splits": {},
    }
    for split in (s.strip() for s in args.splits.split(",") if s.strip()):
        entry = intersect_split(input_root, output_root, split, stages)
        if entry is not None:
            manifest["splits"][split] = entry

    if not manifest["splits"]:
        print("nothing written: no split had all stages present")
        return 1
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {output_root} + manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
