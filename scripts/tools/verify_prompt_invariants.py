#!/usr/bin/env python3
"""Verify the three prompt invariants of the staged curriculum, byte for byte.

I1  train student prompt == the prompt the eval harness dispatches.
I2  the student prompt is identical across all three curriculum stages.
I3  the teacher prompt == the student prompt plus exactly one privileged block,
    byte-identical everywhere else; a stage without privilege gets the student
    prompt verbatim.

Why a separate script when ``gen_openhands_prompt_templates.py`` already asserts I1:
that assertion covers the *files on disk*. This one goes through the real renderer
(``swe_ladder/runtime_prompt.py``, installed as ``rllm.opd.runtime_prompt``) with real
dataset rows, so it also catches a renderer that wraps, prefixes, or stage-conditions
the template after the fact.

Usage
-----
    python3 scripts/tools/verify_prompt_invariants.py \
        --parquet "$DATA_ROOT/<family>/patch_generation/train_verl.parquet" \
        [--rllm "$WS/DeepSWE_RL/rllm"] [--r2egym "$WS/R2E-Gym"] \
        [--template-dir DIR] [--rows N] [--traj EVAL_JSONL] [--train-traj DUMP_JSONL]

This imports the training stack (``rllm.opd``, ``rllm.agents.swe_agent``), so it has to
run after ``integration/install.sh`` and inside the RL environment -- but it needs no
accelerator. ``--traj`` cross-checks I1 against the ``agent_args.system_prompt`` actually
recorded in an eval trajectory instead of trusting the yaml alone; ``--train-traj``
cross-checks it against a training-time prompt dump. Exit code is 1 on any violation, so
this can gate a launch.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_DIR = REPO_ROOT / "swe_ladder" / "prompt_templates_openhands"
# Same eval prompt the generator reads, and for the same reason: parity is only
# meaningful against the file the evaluation harness itself loads.
EVAL_YAML_REL = Path("src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling.yaml")
RUNTIME_PROMPT_REL = Path("rllm/opd/runtime_prompt.py")
STAGES = ("causal_localization", "localized_issue_reproduction", "patch_generation")
PRIV = "causal_localization"  # the stage treated as privileged by the I3-train check
WORKING_DIR = "/testbed"


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def jl(v):
    return json.loads(v) if isinstance(v, str) else v


def head_diff(a: str, b: str, n: int = 24, fromfile: str = "A", tofile: str = "B") -> str:
    d = list(
        difflib.unified_diff(
            a.splitlines(), b.splitlines(), fromfile=fromfile, tofile=tofile, lineterm=""
        )
    )
    return "\n".join(d[:n]) + (f"\n  ... ({len(d)} diff lines total)" if len(d) > n else "")


def _resolve(explicit: str | None, env_names: tuple[str, ...], subdirs: tuple[str, ...], marker: Path) -> Path | None:
    """First checkout that holds ``marker``, from an explicit path then the environment."""
    if explicit:
        return Path(explicit)
    for name in env_names:
        value = os.environ.get(name)
        if not value:
            continue
        root = Path(value)
        for sub in ("",) + subdirs:
            candidate = root / sub if sub else root
            if (candidate / marker).exists():
                return candidate
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True, help="a staged verl parquet, e.g. <family>/patch_generation/train_verl.parquet")
    ap.add_argument("--rllm", default=None, help="rllm checkout with swe_ladder installed (or set RLLM_ROOT / WORKSPACE_ROOT)")
    ap.add_argument("--r2egym", default=None, help="R2E-Gym checkout holding the eval prompt yaml (or set R2EGYM_ROOT / WORKSPACE_ROOT)")
    ap.add_argument("--template-dir", default=str(DEFAULT_TEMPLATE_DIR))
    ap.add_argument("--rows", type=int, default=400)
    ap.add_argument("--traj", default="", help="eval jsonl to cross-check I1 against")
    ap.add_argument(
        "--train-traj",
        default="",
        help="training prompt dump chat_completions/{step}.jsonl - the ONLY proof that the "
        "prompt the policy actually read equals the eval prompt",
    )
    args = ap.parse_args()

    rllm_root = _resolve(args.rllm, ("RLLM_ROOT", "WORKSPACE_ROOT"), ("DeepSWE_RL/rllm", "rllm"), RUNTIME_PROMPT_REL)
    if rllm_root is None or not (rllm_root / RUNTIME_PROMPT_REL).exists():
        print(
            "ERROR: cannot find an rllm checkout with swe_ladder installed "
            f"(looking for {RUNTIME_PROMPT_REL}).\n"
            "       pass --rllm <checkout>, or set RLLM_ROOT / WORKSPACE_ROOT, and run "
            "integration/install.sh first.",
            file=sys.stderr,
        )
        return 2
    r2egym_root = _resolve(args.r2egym, ("R2EGYM_ROOT", "WORKSPACE_ROOT"), ("R2E-Gym",), EVAL_YAML_REL)
    eval_yaml = (r2egym_root / EVAL_YAML_REL) if r2egym_root else None
    if eval_yaml is None or not eval_yaml.exists():
        print(
            f"ERROR: eval prompt yaml not found ({EVAL_YAML_REL}).\n"
            "       pass --r2egym <checkout> or set R2EGYM_ROOT / WORKSPACE_ROOT.",
            file=sys.stderr,
        )
        return 2

    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["OPD_END_TO_END_PROMPT"] = "1"
    os.environ["OPD_RENDER_PROMPT_AT_RUNTIME"] = "1"
    os.environ["OPD_PROMPT_TEMPLATE_DIR"] = args.template_dir
    sys.path.insert(0, str(rllm_root))

    import pyarrow.parquet as pq
    import yaml

    from rllm.opd.runtime_prompt import (
        render_end_to_end_teacher_messages,
        render_runtime_staged_prompt_entry,
        render_student_messages_at_runtime,
    )

    tpl_dir = Path(args.template_dir)
    recs = []
    for row in pq.read_table(args.parquet).to_pylist()[: args.rows]:
        ei = jl(row["extra_info"])
        recs.append(
            {
                "task_id": ei.get("task_id") or ei.get("instance_id"),
                "stage": ei["stage"],
                "student_input": jl(ei["student_input"]),
                "teacher_pi": jl(ei["teacher_pi"]),
            }
        )
    print(f"rllm:          {rllm_root}")
    print(f"eval yaml:     {eval_yaml}")
    print(f"template_dir:  {tpl_dir}")
    print(f"rows: {len(recs)}  stage mix: { {s: sum(1 for r in recs if r['stage'] == s) for s in STAGES} }")
    failures: list[str] = []

    # ---------------- I1 ----------------
    print("\n" + "=" * 74 + "\nI1  train student prompt == the prompt eval dispatches (byte for byte)\n" + "=" * 74)
    eval_cfg = yaml.safe_load(eval_yaml.read_text(encoding="utf-8"))
    eval_system = eval_cfg["system_prompt"]
    train_system = render_student_messages_at_runtime(recs[0])[0]["content"]
    print(f"  eval yaml system    : {len(eval_system):6d} chars  sha={h(eval_system)}")
    print(f"  train-rendered system: {len(train_system):6d} chars  sha={h(train_system)}")
    if eval_system != train_system:
        failures.append("I1 system")
        print("  FAIL: differs\n" + head_diff(eval_system, train_system, 40, "EVAL", "TRAIN"))
    else:
        print("  ok: system is byte-identical")

    # user turn: compare through the renderer, not just the file on disk. The eval side
    # formats instance_prompt with problem_statement + working_dir (agent.py:471), so the
    # rendered student user must equal that template with the issue text spliced in.
    tpl_user = (tpl_dir / "end_to_end_student_user.md").read_text(encoding="utf-8")
    pre, _, post = tpl_user.partition("{issue_statement}")
    eval_user = eval_cfg["instance_prompt"].replace("{working_dir}", WORKING_DIR)
    e_pre, _, e_post = eval_user.partition("{problem_statement}")
    if (pre, post) != (e_pre, e_post):
        failures.append("I1 user template")
        print("  FAIL: user template != eval instance_prompt\n" + head_diff(eval_user, tpl_user, 40, "EVAL", "TRAIN"))
    else:
        print(f"  ok: user template is byte-identical ({len(tpl_user)} B; only the issue slot is renamed)")
    bad_user = 0
    for r in recs:
        u = render_student_messages_at_runtime(r)[1]["content"]
        if not (u.startswith(pre) and u.endswith(post)):
            bad_user += 1
            if bad_user == 1:
                print(f"  FAIL: the renderer added text outside the template, task={r['task_id']}")
                print(f"     head repr: {u[:120]!r}\n     tail repr: {u[-120:]!r}")
    if bad_user:
        failures.append("I1 user render")
    else:
        print(f"  ok: all {len(recs)} rows render as 'template + issue body', nothing added")

    if args.traj:
        seen = {json.loads(ln)["agent_args"]["system_prompt"] for ln in open(args.traj) if ln.strip()}
        print(f"  cross-check vs eval trajectory {Path(args.traj).name}: distinct system prompts = {len(seen)} (want 1)")
        if len(seen) != 1 or next(iter(seen)) != train_system:
            failures.append("I1 traj")
            print("  FAIL: the system prompt eval actually dispatched != the training render")
        else:
            print("  ok: the bytes eval dispatched == the bytes training renders")

    # -------- I1-live: the AGENT HANDOFF, not just the renderer --------
    # The renderer producing the right bytes is necessary but NOT sufficient. The env puts
    # them in entry["problem_statement"] as "[SYSTEM]\n..\n\n[USER]\n.." and the agent only
    # adopts them if swe_agent.split_staged_opd_observation() accepts that text; otherwise it
    # silently falls back to the OPENHANDS_SYSTEM_PROMPT/SWE_SYSTEM_PROMPT code constant and
    # nests the rendered prompt inside the generic template's <issue_description>. That was
    # the behaviour of every OPD_END_TO_END_PROMPT=1 run until the splitter was fixed: it
    # required a "## Stage" heading, which end-to-end templates do not have.
    print("\n  --- I1-live: env -> agent handoff (rendering right is not enough; the agent must adopt it) ---")
    from rllm.agents.swe_agent import merge_staged_system_prompt, split_staged_opd_observation

    entry = {
        "stage": recs[0]["stage"],
        "student_input": recs[0]["student_input"],
        "teacher_pi": recs[0]["teacher_pi"],
        "task_id": recs[0]["task_id"],
    }
    rendered_entry = render_runtime_staged_prompt_entry(dict(entry))
    framed = rendered_entry.get("problem_statement") or ""
    if not framed.startswith("[SYSTEM]"):
        failures.append("I1-live framing")
        print(f"  FAIL: render_runtime_staged_prompt_entry emitted no [SYSTEM]/[USER] framing: {framed[:80]!r}")
    split = split_staged_opd_observation(framed)
    if split is None:
        failures.append("I1-live split")
        print("  FAIL: split_staged_opd_observation rejected the render -> the agent falls back to the")
        print("     code constant and nests the whole rendered prompt inside <issue_description>.")
    else:
        staged_system, staged_user = split
        merged = merge_staged_system_prompt(staged_system, "BASE_CONSTANT_SHOULD_NOT_WIN")
        want_user = render_student_messages_at_runtime(recs[0])[1]["content"]
        ok_sys = merged == eval_system
        ok_user = staged_user == want_user.strip()
        print(f"  after handoff, system: {len(merged):6d} chars sha={h(merged)}  == eval? {ok_sys}")
        print(f"  after handoff, user  : {len(staged_user):6d} chars sha={h(staged_user)}  == rendered student user? {ok_user}")
        if not ok_sys:
            failures.append("I1-live system")
            print(head_diff(eval_system, merged, 30, "EVAL", "AGENT"))
        if not ok_user:
            failures.append("I1-live user")
            print(head_diff(want_user.strip(), staged_user, 30, "RENDER", "AGENT"))
        if ok_sys and ok_user:
            print("  ok: the agent adopted the render; system and user are byte-identical to eval")

    # -------- I1-train / I3-train: read a real training dump (the only end-to-end proof) --------
    # NOTE on the nesting predicate: `<uploaded_files>` is NOT evidence of the wrapper bug -
    # the eval instance_prompt itself opens with "<uploaded_files>\n/testbed\n</uploaded_files>".
    # The marker of the bug is the `[SYSTEM]` framing surviving into the user turn (plus a
    # 19k-char generic repair template around it). Judge by exact match instead: the dumped
    # (system, user) pair must equal some record's rendered (student system, student user).
    if args.train_traj:
        print(f"\n  --- I1-train / I3-train: real training dump {Path(args.train_traj).name} ---")
        # index every record's rendered student pair so each dumped trajectory can be matched
        all_rows = pq.read_table(args.parquet).to_pylist()
        pair_index = {}
        for row in all_rows:
            ei = jl(row["extra_info"])
            rec = {
                "task_id": ei.get("task_id") or ei.get("instance_id"),
                "stage": ei["stage"],
                "student_input": jl(ei["student_input"]),
                "teacher_pi": jl(ei["teacher_pi"]),
            }
            m = render_student_messages_at_runtime(rec)
            pair_index[(m[0]["content"], m[1]["content"])] = rec
        trows = [json.loads(l) for l in open(args.train_traj) if l.strip()]
        bad_sys = nested = tailed = unmatched = 0
        sys_shas = set()
        i3_bad = 0
        i3_checked = 0
        TAIL = re.compile(r"This is step \d+ of a maximum of \d+\. Steps Remaining: \d+\.\s*$")
        for r in trows:
            msgs = r.get("chat_completion") or []
            s = next((m["content"] for m in msgs if m.get("role") == "system"), None)
            u = next((m["content"] for m in msgs if m.get("role") == "user"), None)
            if s is None or u is None:
                continue
            sys_shas.add(h(s))
            if s != eval_system:
                bad_sys += 1
            if "[SYSTEM]" in u:
                nested += 1
            if TAIL.search(u):
                tailed += 1
            rec = pair_index.get((s, u))
            if rec is None:
                unmatched += 1
                continue
            # I3-train: with this record, the teacher the trainer feeds must be exactly this
            # student prompt plus one privileged block. Set the privilege gate explicitly
            # rather than reading the ambient env: unset means "ALL stages privileged"
            # (runtime_prompt._opsd_privileged_stages), which silently inverts the predicate.
            # Pinning one privileged stage tests both semantics in the same pass.
            os.environ["OPSD_PRIVILEGED_STAGES"] = PRIV
            i3_checked += 1
            for st in STAGES:
                te = render_end_to_end_teacher_messages(rec, st)
                if te[0]["content"] != s:
                    i3_bad += 1
                    break
                tu = te[1]["content"]
                if not tu.startswith(u):
                    i3_bad += 1
                    break
                heads = [l for l in tu[len(u) :].splitlines() if l.startswith("## ")]
                if st == PRIV:
                    if len(heads) != 1:
                        i3_bad += 1
                        break
                elif tu != u:
                    i3_bad += 1
                    break
        n = len(trows)
        print(f"  {n} trajectories; distinct system shas {sorted(sys_shas)} (want only {h(eval_system)})")
        print(f"  system != eval: {bad_sys}   user still carrying [SYSTEM] framing: {nested}   first user with a step-count tail: {tailed}")
        print(f"  (system,user) matching some record's render exactly: {n - unmatched}/{n}  unmatched {unmatched}")
        print(f"  I3-train: rebuilt the teacher prompt for {i3_checked} trajectories x 3 stages, violations {i3_bad}")
        if bad_sys or nested or tailed or unmatched or i3_bad or sys_shas != {h(eval_system)}:
            failures.append("I1-train/I3-train")
            print("  FAIL: the policy did not read the eval prompt, or the teacher is not student + one privileged block")
        else:
            print("  ok: the system/user the policy read is byte-identical to eval and to the render,")
            print("      with no wrapper and no tail; teacher == student + exactly one privileged block")
            print("      (non-privileged stages byte-identical to the student)")

    # ---------------- I2 ----------------
    print("\n" + "=" * 74 + "\nI2  is the student prompt byte-identical across stages?\n" + "=" * 74)
    viol2 = 0
    for r in recs:
        variants = {}
        for s in STAGES:
            rr = dict(r)
            rr["stage"] = s
            m = render_student_messages_at_runtime(rr)
            variants[s] = (m[0]["content"], m[1]["content"])
        if len(set(variants.values())) != 1:
            viol2 += 1
            if viol2 == 1:
                base = variants[STAGES[0]]
                for s in STAGES[1:]:
                    if variants[s] != base:
                        for i, (a, b) in enumerate(zip(base, variants[s])):
                            if a != b:
                                print(f"  first difference, task={r['task_id']} {STAGES[0]} vs {s}, message {i}:")
                                print(head_diff(a, b, 20, STAGES[0], s))
    print(f"  {len(recs)} rows x 3 stages: violations {viol2}  " + ("ok: all byte-identical" if viol2 == 0 else "FAIL"))
    if viol2:
        failures.append("I2")
    n_sys = len({h(render_student_messages_at_runtime(r)[0]["content"]) for r in recs})
    print(f"  distinct student system hashes = {n_sys} (want 1: independent of both task and stage)")
    if n_sys != 1:
        failures.append("I2 system hash")

    # ---------------- I3 ----------------
    print("\n" + "=" * 74 + "\nI3  teacher == student + exactly one privileged block?\n" + "=" * 74)
    os.environ["OPSD_PRIVILEGED_STAGES"] = ",".join(STAGES)  # all three privileged: strictest
    viol3 = 0
    blocks: dict[str, set[str]] = {}
    for r in recs:
        st = render_student_messages_at_runtime(r)
        for s in STAGES:
            te = render_end_to_end_teacher_messages(r, s)
            if te[0]["content"] != st[0]["content"]:
                viol3 += 1
                if viol3 == 1:
                    print(f"  FAIL: system differs, task={r['task_id']} stage={s}")
                continue
            su, tu = st[1]["content"], te[1]["content"]
            if not tu.startswith(su):
                viol3 += 1
                if viol3 == 1:
                    print(f"  FAIL: teacher user is not a prefix extension of the student user, task={r['task_id']} stage={s}")
                    print(head_diff(su, tu, 24, "STUDENT", "TEACHER"))
                continue
            extra = tu[len(su) :]
            titles = [l for l in extra.splitlines() if l.startswith("## ")]
            if len(titles) != 1:
                viol3 += 1
                if viol3 == 1:
                    print(f"  FAIL: appended block is not exactly one ({len(titles)} '## ' headings), task={r['task_id']} stage={s}")
                continue
            blocks.setdefault(s, set()).add(titles[0])
    print(
        f"  {len(recs)} rows x 3 stages: violations {viol3}  "
        + (
            "ok: system identical, user is a strict prefix extension, exactly one block appended"
            if viol3 == 0
            else "FAIL"
        )
    )
    if viol3:
        failures.append("I3")
    for s in STAGES:
        print(f"    {s:34s} appended block heading: {sorted(blocks.get(s, []))}")

    # A stage without privilege must be byte-identical to the student: that is what
    # "the ladder revokes privilege" means operationally.
    os.environ["OPSD_PRIVILEGED_STAGES"] = "causal_localization"
    r = recs[0]
    st = render_student_messages_at_runtime(r)
    for s in STAGES:
        te = render_end_to_end_teacher_messages(r, s)
        same = (te[0]["content"], te[1]["content"]) == (st[0]["content"], st[1]["content"])
        want = s != "causal_localization"
        ok = "ok  " if same == want else "FAIL"
        print(
            f"  {ok} with OPSD_PRIVILEGED_STAGES=causal_localization, teacher/{s:34s} == student ? {same}"
            f"  (want {want})"
        )
        if same != want:
            failures.append(f"I3 privilege gate {s}")

    print("\n" + "=" * 74)
    if failures:
        print(f"FAIL: {', '.join(failures)}")
        return 1
    print("PASS: invariants I1 / I2 / I3 all hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
