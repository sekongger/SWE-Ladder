"""Experiment 4: solved-set overlap and difficulty stratification.

  - solved instance sets per system (best ckpt each): overlap matrix, unique wins
  - solve rate stratified by gold-patch size (files changed / lines changed)
Zero GPU. Prints tables; saves analysis/solved_overlap.json
"""
import json, glob, re
from collections import defaultdict
import os
from pathlib import Path

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree
# that holds output/ and the model weights; it defaults to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[1])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
DEC = json.JSONDecoder()
def iter_records(path):
    for line in open(path):
        line = line.strip(); idx = 0
        while idx < len(line):
            try: obj, end = DEC.raw_decode(line, idx)
            except json.JSONDecodeError: break
            yield obj; idx = end
            while idx < len(line) and line[idx] in " \t": idx += 1

SYSTEMS = {
  "base":         "output/swebench_verified_qwen3_4b_instruct_2507_base_original_scaffold_eval",
  "grpo_gs30":    "output/swebench_verified_pure_grpo_baseline_end_to_end_qwen3_4b_colocated_8npu_2x4tp_20260717_gs30_original_scaffold_eval",
  "noroute_gs50": "output/swebench_verified_opsd_grpo_e2e_no_routing_wholeanswer_patch_resume_gs10_qwen3_4b_colocated_8npu_2x4tp_20260724_gs50_original_scaffold_eval",
  "routing_gs90": "output/swebench_verified_opsd_grpo_end_to_end_stage_routing_qwen3_4b_colocated_8npu_2x4tp_epoch2_from_gs70_20260714_gs90_original_scaffold_eval",
}

DIFF_HDR = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)

solved = {}
gold_size = {}   # iid -> (n_files, n_lines)
for name, d in SYSTEMS.items():
    f = glob.glob(f"{WORKSPACE}/{d}/*.jsonl")[0]
    s = set()
    for r in iter_records(f):
        ds = r.get("ds"); ds = json.loads(ds) if isinstance(ds, str) else ds
        iid = ds["instance_id"]
        if r.get("reward"): s.add(iid)
        if iid not in gold_size:
            p = ds.get("patch") or ""
            nf = len(DIFF_HDR.findall(p))
            nl = sum(1 for l in p.splitlines()
                     if (l.startswith("+") or l.startswith("-"))
                     and not l.startswith("+++") and not l.startswith("---"))
            gold_size[iid] = (nf, nl)
    solved[name] = s
    print(f"{name:13s} solved={len(s)}")

names = list(SYSTEMS)
print("\n=== overlap matrix (row∩col) ===")
print("             " + "".join(f"{n:>14s}" for n in names))
for a in names:
    print(f"{a:13s}" + "".join(f"{len(solved[a] & solved[b]):>14d}" for b in names))

routing = solved["routing_gs90"]
others = solved["base"] | solved["grpo_gs30"] | solved["noroute_gs50"]
print(f"\nrouting-only wins (vs union of all others): {len(routing - others)}")
print(f"lost by routing but solved by some other:    {len(others - routing)}")
print(f"routing vs noroute: +{len(routing - solved['noroute_gs50'])} / -{len(solved['noroute_gs50'] - routing)}")
print(f"routing vs base:    +{len(routing - solved['base'])} / -{len(solved['base'] - routing)}")

# difficulty stratification by gold-patch lines
bins = [(0, 10, "≤10 lines"), (11, 40, "11-40"), (41, 10**9, ">40")]
print("\n=== solve rate by gold-patch size (lines changed) ===")
all_iids = list(gold_size)
hdr = "bin          n   " + "".join(f"{n:>14s}" for n in names)
print(hdr)
strat = {}
for lo, hi, lbl in bins:
    ids = [i for i in all_iids if lo <= gold_size[i][1] <= hi]
    row = {n: len(solved[n] & set(ids)) / len(ids) for n in names}
    strat[lbl] = dict(n=len(ids), **row)
    print(f"{lbl:11s} {len(ids):4d}  " + "".join(f"{row[n]*100:13.1f}%" for n in names))

# by files changed
print("\n=== solve rate by gold files changed ===")
for k, lbl in [(1, "1 file"), (2, "2 files"), (3, "3+ files")]:
    ids = [i for i in all_iids if (gold_size[i][0] >= 3 if k == 3 else gold_size[i][0] == k)]
    if not ids: continue
    row = {n: len(solved[n] & set(ids)) / len(ids) for n in names}
    print(f"{lbl:11s} {len(ids):4d}  " + "".join(f"{row[n]*100:13.1f}%" for n in names))

json.dump({"solved": {k: sorted(v) for k, v in solved.items()},
           "gold_size": gold_size, "strat": strat},
          open(f"{ROOT}/analysis/solved_overlap.json", "w"))
print(f"\nwrote {ROOT}/analysis/solved_overlap.json")
