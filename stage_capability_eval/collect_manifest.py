#!/usr/bin/env python3
"""Freeze the SWE-bench Verified task manifest used by stage-capability eval."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any, Iterable

REQUIRED = (
    "instance_id", "repo", "base_commit", "environment_setup_commit", "docker_image",
    "problem_statement", "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "run_tests",
)
DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
PROTOCOL_VERSION = "stage_capability_v1"

def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("manifest source must be a JSON list")
    return rows

def _parse_list(value: Any) -> list[Any]:
    if value is None: return []
    if isinstance(value, list): return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError: return []
    return []

def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    missing = [k for k in REQUIRED if k not in row]
    if missing: raise ValueError(f"{row.get('instance_id', '<unknown>')}: missing {missing}")
    out = dict(row)
    out["docker_image"] = str(out["docker_image"]).strip()
    out["repo"] = str(out["repo"]).strip()
    out["FAIL_TO_PASS"] = _parse_list(out["FAIL_TO_PASS"])
    out["PASS_TO_PASS"] = _parse_list(out["PASS_TO_PASS"])
    out["source_manifest"] = "princeton-nlp/SWE-bench_Verified"
    out["dataset_revision"] = DATASET_REVISION
    return out

def patch_source_files(patch: str) -> list[str]:
    files=[]
    for line in str(patch or "").splitlines():
        if line.startswith("+++ b/"):
            files.append(line[6:].strip())
    return list(dict.fromkeys(files))

def is_test_file(path: str) -> bool:
    p=path.lower()
    return ("/test" in p or p.startswith("test") or p.endswith("_test.py") or p.startswith("tests/"))

def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()

def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    data=b"".join(canonical_json(r) for r in rows)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


PUBLIC_EXCLUDED_FIELDS = {
    "patch",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "parsed_commit",
    "run_tests",
}


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return only model-visible task inputs and provenance metadata."""
    return {key: value for key, value in row.items() if key not in PUBLIC_EXCLUDED_FIELDS}

def build(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]], list[dict[str, Any]]]:
    seen=set(); normalized=[]; exclusions=[]
    for idx,row in enumerate(rows):
        iid=row.get("instance_id") if isinstance(row,dict) else None
        if not isinstance(row,dict): exclusions.append({"row":idx,"status":"invalid_row"}); continue
        if not iid or iid in seen:
            exclusions.append({"row":idx,"instance_id":iid,"status":"duplicate_or_missing_id"}); continue
        try: r=normalize_row(row)
        except ValueError as e: exclusions.append({"row":idx,"instance_id":iid,"status":"schema_error","reason":str(e)}); continue
        seen.add(iid); files=patch_source_files(r["patch"])
        non_test=[f for f in files if not is_test_file(f)]
        r["source_patch_files"]=files; r["non_test_source_files"]=non_test
        r["task_capabilities"]={"loc_file":bool(non_test),"loc_function":False,"repro":bool(r["FAIL_TO_PASS"] and r["test_patch"]),"patch_cond":bool(non_test and r["patch"])}
        normalized.append(r)
    ids=[r["instance_id"] for r in normalized]
    sets={"end_to_end_set":ids,
          "loc_file_set":[r["instance_id"] for r in normalized if r["task_capabilities"]["loc_file"]],
          "loc_function_set":[],
          "repro_set":[r["instance_id"] for r in normalized if r["task_capabilities"]["repro"]],
          "coverage_valid_set":[],
          "patch_cond_set":[r["instance_id"] for r in normalized if r["task_capabilities"]["patch_cond"]],
          "common_analysis_set":[]}
    return normalized,sets,exclusions

def main(argv=None)->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--source",default="output/rucaibox_original_scaffold_swebench_verified_dataset.json"); ap.add_argument("--out",default="stage_capability_eval"); args=ap.parse_args()
    out=Path(args.out); (out/"manifests").mkdir(parents=True,exist_ok=True); (out/"protocol").mkdir(parents=True,exist_ok=True)
    rows,sets,exclusions=build(load_rows(args.source))
    # Gold patches, test patches, test names, and source snapshots stay in the
    # evaluator-private area.  The public manifest contains the issue and task
    # identity needed to construct a buggy-repository task package.
    private_dir = out / "evaluator_private"
    private_dir.mkdir(parents=True, exist_ok=True)
    private_digest = write_jsonl(private_dir / "source_records.jsonl", rows)
    public_rows = [public_row(row) for row in rows]
    digest = write_jsonl(out/"manifests/all_swebench_verified.jsonl", public_rows)
    (out/"manifests/task_sets.json").write_text(json.dumps({"protocol_id":PROTOCOL_VERSION,"dataset_revision":DATASET_REVISION,"manifest_sha256":digest,"sets":sets},indent=2)+"\n")
    (out/"manifests/exclusions.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in exclusions))
    print(json.dumps({"rows":len(rows),"manifest_sha256":digest,"private_source_sha256":private_digest,"sets":{k:len(v) for k,v in sets.items()},"exclusions":len(exclusions)},indent=2))
    return 0
if __name__=="__main__": raise SystemExit(main())
