"""Experiment 3: behavioral-structure analysis of eval trajectories.

Per system, from the SWE-bench Verified eval jsonl:
  - stage traversal (training stage classifier, backtrack=3): reached repro/patch,
    full loc->repro->patch traversal, reproduce-before-patch ordering,
    per-stage step shares
  - repeated-edit rate: fraction of edit actions that are exact repeats of an
    earlier edit action in the same trajectory; same restricted to FAILED edits
  - action-type mix (search / view / edit / bash / finish per step)
  - step-count distribution (saved for CDF)
Zero GPU. Outputs analysis/behavior_structure.json
"""
import json, os, re, sys, glob, importlib.util
from collections import Counter
from pathlib import Path

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree
# that holds output/ and the model weights; it defaults to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[1])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
spec = importlib.util.spec_from_file_location(
    "stage_classifier", f"{ROOT}/swe_ladder/stage_classifier.py")
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)

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

FAIL_OBS = re.compile(r"ERROR|No replacement was performed|did not appear verbatim|"
                      r"Cannot find|not found in", re.I)

def action_kind(fn, params):
    if fn in ("file_editor", "str_replace_editor"):
        cmd = (params.get("command") or "").strip().lower()
        return "edit" if cmd in ("str_replace", "insert", "create") else "view"
    if fn in ("search", "grep"): return "search"
    if fn == "execute_bash": return "bash"
    if fn == "finish": return "finish"
    return "other"

def analyze_system(path):
    out = dict(n=0, reached_repro=0, reached_patch=0, full_traversal=0,
               repro_before_patch=0, step_counts=[], solved_step_counts=[],
               repeat_edit_rates=[], repeat_failed_edit_rates=[],
               stage_share=Counter(), action_mix=Counter())
    for rec in iter_records(path):
        steps = rec.get("trajectory_steps") or []
        if not steps: continue
        out["n"] += 1
        pairs = [(f"{s.get('thought') or ''}\n\n{s.get('action') or ''}",
                  str(s.get("observation") or "")) for s in steps]
        labels = sc.classify_trajectory(pairs, backtrack_threshold=3)
        stages_seen = set(labels)
        reached_r = sc.REPRODUCTION in stages_seen
        reached_p = sc.PATCH in stages_seen
        out["reached_repro"] += reached_r
        out["reached_patch"] += reached_p
        out["full_traversal"] += (reached_r and reached_p)
        if reached_r and reached_p:
            fr = labels.index(sc.REPRODUCTION); fp = labels.index(sc.PATCH)
            out["repro_before_patch"] += (fr < fp)
        for lab in labels: out["stage_share"][lab] += 1
        out["step_counts"].append(len(steps))
        if rec.get("reward"): out["solved_step_counts"].append(len(steps))
        # edits & repeats
        edits, failed_edits = [], []
        for s in steps:
            act = s.get("action") or ""
            fn, params = sc.parse_action(act)
            kind = action_kind(fn or "", params)
            out["action_mix"][kind] += 1
            if kind == "edit":
                edits.append(act)
                if FAIL_OBS.search(str(s.get("observation") or "")[:400]):
                    failed_edits.append(act)
        if len(edits) >= 2:
            seen, rep = set(), 0
            for a in edits:
                if a in seen: rep += 1
                seen.add(a)
            out["repeat_edit_rates"].append(rep / len(edits))
        if len(failed_edits) >= 2:
            seen, rep = set(), 0
            for a in failed_edits:
                if a in seen: rep += 1
                seen.add(a)
            out["repeat_failed_edit_rates"].append(rep / len(failed_edits))
    return out

def main():
    res = {}
    for name, d in SYSTEMS.items():
        fs = glob.glob(f"{WORKSPACE}/{d}/*.jsonl")
        if not fs: print(f"!! missing {name}"); continue
        o = analyze_system(fs[0])
        n = o["n"]
        tot_steps = sum(o["stage_share"].values())
        mean = lambda xs: sum(xs) / len(xs) if xs else None
        res[name] = dict(
            n=n,
            reached_repro=o["reached_repro"] / n,
            reached_patch=o["reached_patch"] / n,
            full_traversal=o["full_traversal"] / n,
            repro_before_patch=(o["repro_before_patch"] / max(o["full_traversal"], 1)),
            stage_share={k: v / tot_steps for k, v in o["stage_share"].items()},
            action_mix={k: v / tot_steps for k, v in o["action_mix"].items()},
            repeat_edit_rate=mean(o["repeat_edit_rates"]),
            repeat_failed_edit_rate=mean(o["repeat_failed_edit_rates"]),
            steps_mean=mean(o["step_counts"]),
            step_counts=o["step_counts"],
            solved_steps_mean=mean(o["solved_step_counts"]),
        )
        r = res[name]
        print(f"{name:13s} n={n:3d} full_trav={r['full_traversal']:.2f} "
              f"repro_first={r['repro_before_patch']:.2f} "
              f"rep_edit={r['repeat_edit_rate']:.3f} rep_failed={r['repeat_failed_edit_rate']:.3f} "
              f"share(loc/rep/patch)="
              f"{r['stage_share'].get('causal_localization',0):.2f}/"
              f"{r['stage_share'].get('localized_issue_reproduction',0):.2f}/"
              f"{r['stage_share'].get('patch_generation',0):.2f}")
    json.dump(res, open(f"{ROOT}/analysis/behavior_structure.json", "w"))
    print(f"wrote {ROOT}/analysis/behavior_structure.json")

if __name__ == "__main__":
    main()
