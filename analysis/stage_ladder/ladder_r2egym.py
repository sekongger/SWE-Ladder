"""Experiment 1: stage-capability ladder from existing SWE-bench Verified eval trajectories.

Per-trajectory behavioral funnel (heuristics documented inline):
  localize   : any gold changed (non-test) file is touched by a file_editor/search
               action (viewed) OR appears in the final patch (edited).
  loc_recall : |gold ∩ edited| / |gold| at file level (final output_patch).
  reproduce  : the agent created a new script AND executed it at least once whose
               output shows the failure (Traceback/AssertionError/FAILED/Error...).
  patch      : final output_patch is non-empty.
  verify     : after the LAST source edit, some created script was re-run with no
               failure signal in its output (agent confirmed its own fix).
  solve      : reward == 1 (official pass@1).
"""
import json, re, os, glob, sys
from collections import defaultdict
from pathlib import Path

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree
# that holds output/ and the model weights; it defaults to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
SUMMARY = f"{ROOT}/analysis/stage_ladder/ladder_r2egym_summary.json"
PER_TRAJ = f"{ROOT}/analysis/stage_ladder/ladder_r2egym_per_traj.json"

DEC = json.JSONDecoder()

def iter_records(path):
    for line in open(path):
        line = line.strip(); idx = 0
        while idx < len(line):
            try: obj, end = DEC.raw_decode(line, idx)
            except json.JSONDecodeError: break
            yield obj; idx = end
            while idx < len(line) and line[idx] in " \t": idx += 1

DIFF_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
FAIL_RE = re.compile(r"Traceback \(most recent call last\)|AssertionError|FAILED|"
                     r"Error:|error:|Exception|assert(ion)? failed|exit code [1-9]", re.I)
PARAM_PATH_RE = re.compile(r"<parameter=path>\s*([^<\s]+)\s*</parameter>")
PARAM_CMD_RE  = re.compile(r"<parameter=command>\s*(\w+)", )
FUNC_RE       = re.compile(r"<function=(\w+)>")

def norm(p):
    p = p.strip()
    if p.startswith("/testbed/"): p = p[len("/testbed/"):]
    return p.lstrip("./")

def is_test_path(p):
    p = p.lower()
    return p.startswith("test") or "/test" in p or p.endswith("_test.py")

def diff_files(patch_text):
    if not patch_text: return set()
    return {norm(m.group(2)) for m in DIFF_RE.finditer(patch_text)}

def analyze(rec):
    ds = rec.get("ds") or {}
    if isinstance(ds, str):
        try: ds = json.loads(ds)
        except Exception: ds = {}
    gold = {f for f in diff_files(ds.get("patch", "")) if not is_test_path(f)}
    edited = diff_files(rec.get("output_patch", ""))
    viewed, created = set(), set()
    # (created_script_ran, saw_failure) events in step order
    exec_events = []           # (step_idx, ran_created, failed)
    last_source_edit_idx = -1
    for i, s in enumerate(rec.get("trajectory_steps") or []):
        act = s.get("action") or ""
        obs = s.get("observation") or ""
        m = FUNC_RE.search(act)
        fn = m.group(1) if m else ""
        pm = PARAM_PATH_RE.search(act)
        path = norm(pm.group(1)) if pm else None
        cm = PARAM_CMD_RE.search(act)
        sub = cm.group(1) if cm else ""
        if fn in ("file_editor", "search", "grep") and path:
            viewed.add(path)
        if (fn == "create") or (fn == "file_editor" and sub == "create"):
            if path: created.add(path)
        if (fn == "str_replace") or (fn == "file_editor" and sub in ("str_replace", "insert")):
            if path and path not in created and not is_test_path(path):
                last_source_edit_idx = i
        if fn == "execute_bash":
            ran_created = any(os.path.basename(c) in act or c in act for c in created)
            exec_events.append((i, ran_created, bool(FAIL_RE.search(obs))))
    reproduce = any(rc and fl for _, rc, fl in exec_events)
    verify = any(idx > last_source_edit_idx and rc and not fl
                 for idx, rc, fl in exec_events) if last_source_edit_idx >= 0 else False
    localize_any = bool(gold & (viewed | edited))
    loc_recall = (len(gold & edited) / len(gold)) if gold else None
    return dict(
        instance_id=ds.get("instance_id"),
        localize=localize_any,
        loc_recall=loc_recall,
        reproduce=reproduce,
        patch_nonempty=bool((rec.get("output_patch") or "").strip()),
        verify=verify,
        solve=bool(rec.get("reward")),
        n_steps=len(rec.get("trajectory_steps") or []),
    )

SYSTEMS = {
  "base":          "output/swebench_verified_qwen3_4b_instruct_2507_base_original_scaffold_eval",
  "grpo_gs30":     "output/swebench_verified_pure_grpo_baseline_end_to_end_qwen3_4b_colocated_8npu_2x4tp_20260717_gs30_original_scaffold_eval",
  "noroute_gs30":  "output/swebench_verified_opsd_grpo_e2e_no_routing_wholeanswer_patch_resume_gs10_qwen3_4b_colocated_8npu_2x4tp_20260724_gs30_original_scaffold_eval",
  "noroute_gs50":  "output/swebench_verified_opsd_grpo_e2e_no_routing_wholeanswer_patch_resume_gs10_qwen3_4b_colocated_8npu_2x4tp_20260724_gs50_original_scaffold_eval",
  "noroute_gs70":  "output/swebench_verified_opsd_grpo_e2e_no_routing_wholeanswer_patch_resume_gs10_qwen3_4b_colocated_8npu_2x4tp_20260724_gs70_original_scaffold_eval",
  "routing_gs30":  "output/swebench_verified_opsd_grpo_end_to_end_stage_routing_qwen3_4b_colocated_8npu_2x4tp_20260711_gs30_original_scaffold_eval",
  "routing_gs50":  "output/swebench_verified_opsd_grpo_end_to_end_stage_routing_qwen3_4b_colocated_8npu_2x4tp_20260711_gs50_original_scaffold_eval",
  "routing_gs70":  "output/swebench_verified_opsd_grpo_end_to_end_stage_routing_qwen3_4b_colocated_8npu_2x4tp_20260711_gs70_original_scaffold_eval",
  "routing_gs90":  "output/swebench_verified_opsd_grpo_end_to_end_stage_routing_qwen3_4b_colocated_8npu_2x4tp_epoch2_from_gs70_20260714_gs90_original_scaffold_eval",
}

out = {}
per_traj = {}
for name, d in SYSTEMS.items():
    fs = glob.glob(f"{WORKSPACE}/{d}/*.jsonl")
    if not fs:
        print(f"!! {name}: no jsonl", file=sys.stderr); continue
    rows = [analyze(r) for r in iter_records(fs[0])]
    per_traj[name] = rows
    n = len(rows)
    recs = [r["loc_recall"] for r in rows if r["loc_recall"] is not None]
    out[name] = dict(
        n=n,
        localize=sum(r["localize"] for r in rows) / n,
        loc_recall=sum(recs) / len(recs),
        reproduce=sum(r["reproduce"] for r in rows) / n,
        patch_nonempty=sum(r["patch_nonempty"] for r in rows) / n,
        verify=sum(r["verify"] for r in rows) / n,
        solve=sum(r["solve"] for r in rows) / n,
        steps_mean=sum(r["n_steps"] for r in rows) / n,
    )
    m = out[name]
    print(f"{name:14s} n={n:4d}  localize={m['localize']:.3f}  loc_recall={m['loc_recall']:.3f}  "
          f"reproduce={m['reproduce']:.3f}  patch={m['patch_nonempty']:.3f}  "
          f"verify={m['verify']:.3f}  solve={m['solve']:.3f}")

if not out:
    print(f"no eval trajectories found under {WORKSPACE}/output - nothing written, the "
          f"shipped ladder_r2egym_summary.json is left alone. Point WORKSPACE_ROOT at the "
          f"tree holding output/.", file=sys.stderr)
    raise SystemExit(1)

json.dump(out, open(SUMMARY, "w"), indent=1)
json.dump({k: v for k, v in per_traj.items()}, open(PER_TRAJ, "w"))
print(f"\nwrote {SUMMARY}\n      {PER_TRAJ}")
