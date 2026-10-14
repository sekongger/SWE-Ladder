"""Gradient-interference experiment, step 1: data preparation.

For each selected base-model trajectory on SWE-bench Verified:
  - per-step stage labels via the TRAINING stage classifier (backtrack=3, same as runs)
  - rebuilt conversation exactly as the eval agent constructed it
    (assistant = thought + "\n\n" + action; user obs = obs + "\n" + stepcount message)
  - tokenization with the base model chat template, per-message token spans,
    per-token stage ids and assistant-token loss mask
  - teacher contexts: end-to-end student user prompt + the 3-line privileged block
    (exact training construction), for stages loc/repro/patch, plus SHUFFLED-privilege
    controls (evidence taken from a different instance)

Outputs one .json + .npz per trajectory under analysis/grad_interference/data/.
Run with:  PYTHONNOUSERSITE=1 <ascend-rl-pack python> prep_data.py
"""
import json, os, re, sys, importlib.util
import numpy as np

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree
# that holds output/ and the model weights; it defaults to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
EVAL_JSONL = f"{WORKSPACE}/output/swebench_verified_qwen3_4b_instruct_2507_base_original_scaffold_eval/swebench_verified_qwen3_4b_instruct_2507_base_original_scaffold_eval.jsonl"
TOKENIZER_DIR = f"{WORKSPACE}/hf_models/Qwen/Qwen3-4B-Instruct-2507"
TPL_DIR = f"{ROOT}/swe_ladder/prompt_templates"
OUT_DIR = f"{ROOT}/analysis/grad_interference/data"

N_TRAJ = 48
MIN_STAGE_ASSISTANT_TOKENS = 128
MAX_TOTAL_TOKENS = 40000

# ---- import the training stage classifier directly by path (zero-dep file) ----
spec = importlib.util.spec_from_file_location(
    "stage_classifier", f"{ROOT}/swe_ladder/stage_classifier.py")
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)

STAGES = [sc.LOCALIZATION, sc.REPRODUCTION, sc.PATCH]
HEADINGS = {
    sc.LOCALIZATION: "Code context hints",
    sc.REPRODUCTION: "Reproduction starting points",
    sc.PATCH: "Verified repository evidence",
}

DEC = json.JSONDecoder()
def iter_records(path):
    for line in open(path):
        line = line.strip(); idx = 0
        while idx < len(line):
            try: obj, end = DEC.raw_decode(line, idx)
            except json.JSONDecodeError: break
            yield obj; idx = end
            while idx < len(line) and line[idx] in " \t": idx += 1

# ---- gold-patch parsing -> teacher_pi (formats copied from the training parquet) ----
DIFF_HDR = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", re.M)

def is_test_path(p):
    p = p.lower()
    return "/tests/" in p or "/test/" in p or os.path.basename(p).startswith("test_") \
        or p.endswith("_test.py") or os.path.basename(p) == "conftest.py"

def split_file_diffs(patch):
    """yield (path, file_diff_text) per file in a unified diff."""
    parts = re.split(r"(?m)^(?=diff --git )", patch)
    for part in parts:
        m = DIFF_HDR.match(part)
        if m: yield m.group(2), part

def parse_hunks(file_diff):
    """yield (old_start, old_count, removed_lines['<lineno>: <content>'])."""
    lines = file_diff.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", lines[i])
        if not m: i += 1; continue
        old_start = int(m.group(1)); old_count = int(m.group(2) or "1")
        removed, ln = [], old_start
        i += 1
        while i < len(lines) and not lines[i].startswith("@@") and not lines[i].startswith("diff --git"):
            l = lines[i]
            if l.startswith("-") and not l.startswith("---"):
                removed.append(f"{ln}: {l[1:]}"); ln += 1
            elif l.startswith("+") and not l.startswith("+++"):
                pass
            elif l.startswith("\\"):
                pass
            else:
                ln += 1
            i += 1
        yield old_start, old_count, removed

def build_teacher_pi(ds):
    gold_patch = ds.get("patch") or ""
    files, hunk_ranges, buggy = [], [], []
    for path, fdiff in split_file_diffs(gold_patch):
        if is_test_path(path): continue
        files.append(path)
        hunks = []
        for hi, (os_, oc, removed) in enumerate(parse_hunks(fdiff)):
            hunk_ranges.append({"file": path, "hunk_index": hi, "old_start": os_,
                                "old_count": oc, "old_end": os_ + oc - 1})
            if removed:
                hunks.append({"hunk_index": hi, "old_start": os_, "old_count": oc,
                              "old_end": os_ + oc - 1, "removed_lines": removed})
        if hunks:
            buggy.append({"file": path, "hunks": hunks})
    try:
        f2p = json.loads(ds.get("FAIL_TO_PASS") or "[]")
    except Exception:
        f2p = []
    return {
        "gold_changed_non_test_files": files,
        "gold_hunk_ranges": hunk_ranges,
        "old_buggy_lines": buggy,
        "test_patch": ds.get("test_patch") or "",
        "fail_to_pass_test_names": f2p,
        "golden_patch": gold_patch,
    }

def evidence_for(stage, tp):
    """Verbatim re-implementation of _verified_repository_evidence (staged_dataset.py:338)."""
    if stage == sc.LOCALIZATION:
        return {
            "candidate_non_test_source_files_to_inspect": tp.get("gold_changed_non_test_files", []),
            "candidate_hunk_ranges_to_inspect": tp.get("gold_hunk_ranges", []),
            "candidate_modified_source_segments_to_inspect": tp.get("old_buggy_lines", []),
        }
    if stage == sc.REPRODUCTION:
        ev = {}
        diff = tp.get("fail_to_pass_tests", "") or tp.get("test_patch", "")
        if diff: ev["candidate_reproduction_test_changes_to_consider"] = diff
        if tp.get("fail_to_pass_test_ids"): ev["candidate_failing_test_ids_to_consider"] = tp["fail_to_pass_test_ids"]
        if tp.get("fail_to_pass_test_names"): ev["candidate_failing_test_names_to_consider"] = tp["fail_to_pass_test_names"]
        if not ev: ev["candidate_reproduction_test_changes_to_consider"] = ""
        return ev
    if stage == sc.PATCH:
        return {"verified_repair_diff": tp.get("golden_patch", "")}
    raise ValueError(stage)

def jblock(v): return json.dumps(v, ensure_ascii=False, indent=2)

# ---- prompts ----
STUDENT_SYSTEM = open(f"{TPL_DIR}/end_to_end_student_system.md").read()
STUDENT_USER_TPL = open(f"{TPL_DIR}/end_to_end_student_user.md").read()

def student_user(issue): return STUDENT_USER_TPL.replace("{issue_statement}", issue)

def teacher_user(issue, stage, tp):
    return (student_user(issue)
            + f"\n\n## {HEADINGS[stage]}\n{jblock(evidence_for(stage, tp))}")

def stepcount_msg(step_count, max_steps):
    rem = max_steps - step_count
    if rem > 0:
        return f"This is step {step_count} of a maximum of {max_steps}. Steps Remaining: {rem}."
    return "You have reached the maximum number of steps. Please submit your answer NOW."

def build_messages(rec, user1):
    msgs = [{"role": "system", "content": STUDENT_SYSTEM},
            {"role": "user", "content": user1}]
    max_steps = int(rec.get("max_steps") or 88)
    for s in rec["trajectory_steps"]:
        thought = s.get("thought") or ""
        action = s.get("action") or ""
        msgs.append({"role": "assistant", "content": f"{thought}\n\n{action}"})
        obs = str(s.get("observation") or "")
        msgs.append({"role": "user",
                     "content": f"{obs}\n{stepcount_msg(int(s.get('step_count') or 1), max_steps)}"})
    return msgs

# ---- tokenization with per-message spans ----
from transformers import AutoTokenizer
from pathlib import Path
TOK = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)

def tokenize_with_spans(msgs):
    """Return (ids, spans) where spans[i] = (start, end) token span of message i."""
    ids, spans = [], []
    prev = 0
    for i in range(1, len(msgs) + 1):
        cur = TOK.apply_chat_template(msgs[:i], tokenize=True, add_generation_prompt=False)
        # prefix stability check
        if cur[:prev] != ids[:prev]:
            raise RuntimeError(f"chat template not prefix-stable at message {i}")
        spans.append((prev, len(cur)))
        ids = cur
        prev = len(cur)
    return ids, spans

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # pass 1: light scan for candidates
    cands = []
    for ri, rec in enumerate(iter_records(EVAL_JSONL)):
        steps = rec.get("trajectory_steps") or []
        if len(steps) < 6: continue
        pairs = [(f"{s.get('thought') or ''}\n\n{s.get('action') or ''}",
                  str(s.get("observation") or "")) for s in steps]
        labels = sc.classify_trajectory(pairs, backtrack_threshold=3)
        # quick per-stage assistant token estimate via completion token usage
        tok_per_stage = {st: 0 for st in STAGES}
        for lab, s in zip(labels, steps):
            tok_per_stage[lab] += int(s.get("token_usage_completion") or 0)
        total_est = sum(int(s.get("token_usage_prompt") or 0) == 0 for s in steps)  # unused
        approx_total = int(steps[-1].get("token_usage_prompt") or 0) + \
                       int(steps[-1].get("token_usage_completion") or 0)
        if min(tok_per_stage.values()) < MIN_STAGE_ASSISTANT_TOKENS: continue
        if approx_total > MAX_TOTAL_TOKENS: continue
        cands.append((ri, bool(rec.get("reward")), tok_per_stage, rec))
    solved = [c for c in cands if c[1]]
    unsolved = [c for c in cands if not c[1]]
    picked = (solved + unsolved)[:N_TRAJ]
    print(f"candidates={len(cands)} (solved={len(solved)})  picked={len(picked)}")

    # shuffle-privilege pairing: rotate by 1
    tps = []
    for _, _, _, rec in picked:
        ds = rec["ds"]
        if isinstance(ds, str): ds = json.loads(ds)
        tps.append(build_teacher_pi(ds))

    manifest = []
    for k, (ri, is_solved, tok_ps, rec) in enumerate(picked):
        ds = rec["ds"]
        if isinstance(ds, str): ds = json.loads(ds)
        iid = ds["instance_id"]
        issue = ds.get("problem_statement") or rec.get("problem_statement") or ""
        tp, tp_sh = tps[k], tps[(k + 1) % len(picked)]

        steps = rec["trajectory_steps"]
        pairs = [(f"{s.get('thought') or ''}\n\n{s.get('action') or ''}",
                  str(s.get("observation") or "")) for s in steps]
        labels = sc.classify_trajectory(pairs, backtrack_threshold=3)

        msgs = build_messages(rec, student_user(issue))
        try:
            ids, spans = tokenize_with_spans(msgs)
        except RuntimeError as e:
            print(f"skip {iid}: {e}"); continue
        if len(ids) > MAX_TOTAL_TOKENS + 8000:
            print(f"skip {iid}: {len(ids)} tokens"); continue

        stage_ids = np.full(len(ids), -1, dtype=np.int8)
        assist_mask = np.zeros(len(ids), dtype=np.int8)
        # message 0=system, 1=user; then (assistant, user)*
        for si, lab in enumerate(labels):
            a_span = spans[2 + 2 * si]
            o_span = spans[3 + 2 * si]
            sid = sc.STAGE_ORDER[lab]
            stage_ids[a_span[0]:a_span[1]] = sid
            stage_ids[o_span[0]:o_span[1]] = sid
            assist_mask[a_span[0]:a_span[1]] = 1

        # teacher context prefixes: system+user1 token span from variant conversations
        variants = {}
        resp_start = spans[1][1]  # end of user1 in student conversation
        suffix = ids[resp_start:]
        ok = True
        for tag, (stage, tpx) in {
            "loc":      (sc.LOCALIZATION, tp),
            "repro":    (sc.REPRODUCTION, tp),
            "patch":    (sc.PATCH, tp),           # == global whole-answer privilege
            "loc_sh":   (sc.LOCALIZATION, tp_sh),
            "repro_sh": (sc.REPRODUCTION, tp_sh),
            "patch_sh": (sc.PATCH, tp_sh),
        }.items():
            vmsgs = build_messages(rec, teacher_user(issue, stage, tpx))
            vids, vspans = tokenize_with_spans(vmsgs)
            vstart = vspans[1][1]
            if vids[vstart:] != suffix:
                print(f"skip {iid}: teacher/student suffix mismatch ({tag})"); ok = False; break
            variants[tag] = vids[:vstart]
        if not ok: continue

        np.savez_compressed(
            f"{OUT_DIR}/{iid}.npz",
            ids=np.asarray(ids, dtype=np.int32),
            stage_ids=stage_ids, assist_mask=assist_mask,
            resp_start=np.int32(resp_start),
            **{f"prefix_{t}": np.asarray(v, dtype=np.int32) for t, v in variants.items()},
        )
        stage_tok = {st: int(((stage_ids == sc.STAGE_ORDER[st]) & (assist_mask == 1)).sum())
                     for st in STAGES}
        manifest.append(dict(instance_id=iid, solved=is_solved, n_tokens=len(ids),
                             resp_start=resp_start, stage_assist_tokens=stage_tok))
        print(f"[{k+1}/{len(picked)}] {iid} n={len(ids)} solved={is_solved} stage_tok={stage_tok}")

    json.dump(manifest, open(f"{OUT_DIR}/manifest.json", "w"), indent=1)
    print(f"\nwrote {len(manifest)} trajectories -> {OUT_DIR}")

if __name__ == "__main__":
    main()
