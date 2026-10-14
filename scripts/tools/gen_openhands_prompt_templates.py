#!/usr/bin/env python3
"""Generate the OpenHands-scaffold runtime prompt templates from two sources of truth.

Why this exists
---------------
``swe_ladder/prompt_templates/`` advertises the **r2egym** tool triple
(``file_editor`` / ``search`` / ``finish``). ``rllm/agents/swe_agent.py::merge_staged_system_prompt``
returns a staged template verbatim whenever it contains the string
"We have access to the following functions", so those templates - not
``OPENHANDS_SYSTEM_PROMPT`` - are what the policy actually reads. Flipping
``SWE_SCAFFOLD=openhands`` without new templates would advertise three functions that
the container does not have (it only gets ``str_replace_editor`` / ``execute_bash`` /
``submit``), which is exactly the silent-no-op failure described in ``docs/EVALUATION.md``.

Sources of truth
----------------
1. ``<R2E-Gym>/src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling.yaml``
   - the *eval* config selected by ``--scaffold openhands``. Using its text verbatim is
   what buys train/eval prompt parity.
2. ``swe_ladder/prompt_templates/*.md`` - the staged (LOC -> REPRO -> PATCH) scaffolding:
   stage preambles, required artifacts, teacher evidence blocks, placeholders.

Every r2egym-specific tool reference is rewritten through an explicit, asserted
replacement table: if upstream text drifts, this script fails loudly instead of
emitting a half-converted prompt. The generated tree is then checked against a
forbidden-token list so no r2egym tool name can survive.

Usage
-----
    python3 scripts/tools/gen_openhands_prompt_templates.py --r2egym "$WS/R2E-Gym" [--check]

``--check`` regenerates in memory and diffs against the files on disk (exit 1 on drift),
which is how to tell whether the shipped templates still match upstream's eval prompt.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "swe_ladder" / "prompt_templates"
OUT_DIR = REPO_ROOT / "swe_ladder" / "prompt_templates_openhands"
# Read from the R2E-Gym checkout the evaluation harness actually runs, never from a
# copy: a copy is how train/eval prompt parity rots without anyone noticing.
EVAL_YAML_REL = Path("src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling.yaml")

TOOL_MARKER = "We have access to the following functions:"
WORKING_DIR = "/testbed"

# Deliberately *not* embellished: the generated prompt is the stock eval prompt.
#
# This generator used to append two blocks: seven RL response-discipline bullets
# inside <IMPORTANT>, and an <ENVIRONMENT_CONVENTIONS> block. Both are gone, because
# they broke invariant I1 (train student prompt == eval prompt, byte for byte) by 23
# lines, and I1 is what makes a trained checkpoint's score comparable to the
# published baseline it is trained from. Do not re-add them here.
#
# Two claims that used to justify the conventions block were checked and are false:
#   - "the rule router needs /testbed/reproduce_issue.py named in the prompt" - no:
#     stage_classifier.py keys REPRODUCTION on a (repro|reproduce) basename regex
#     *plus* v2 rules (any test-file edit, any pytest run), so it fires on whatever
#     the policy names its script.
#   - "the policy must be told submit takes no parameters" - the stock eval prompt
#     already carries the full submit signature in its function list.
# The cost, accepted knowingly: a base student loses the format guardrails, so early
# malformed-call rate may rise. Watch mal% in the rollout logs; do not patch the
# prompt.

# r2egym -> openhands rewrites, applied per file. Every entry is asserted to match at
# least once, so upstream drift breaks generation instead of leaking r2egym text.
FINALIZE_LOC_SYSTEM = (
    "- To finalize, put the localization artifact in your final message text and then call the "
    "submit function, which takes no parameters: <function=submit>\n</function>. Never pass a "
    "parameter to submit."
)
INSPECT_TOOLS_PHRASE = "`grep -rn` through execute_bash or str_replace_editor view"

LOC_SYSTEM_COMMON: list[tuple[str, str]] = [
    (
        "- The first assistant turn MUST be a non-finish repository inspection call",
        "- The first assistant turn MUST be a non-submit repository inspection call",
    ),
    (
        "- A finish call is invalid until you have inspected repository output and cited evidence from that output.",
        "- A submit call is invalid until you have inspected repository output and cited evidence from that output.",
    ),
]

REMAPS: dict[str, list[tuple[str, str]]] = {
    # Only the student system template carries the finalize bullet; the teacher one stops
    # after the workflow constraints (7218 vs 7440 bytes).
    "causal_localization_student_system.md": LOC_SYSTEM_COMMON
    + [
        (
            "- To finalize, call the finish function with command=submit and put the localization "
            "artifact in the result parameter. Never use <function=submit>; submit is only the "
            "command parameter value for the finish function.",
            FINALIZE_LOC_SYSTEM,
        ),
    ],
    "causal_localization_teacher_system.md": LOC_SYSTEM_COMMON,
    "causal_localization_student_user.md": [
        ("Before calling finish, your final artifact", "Before calling submit, your final artifact"),
        (
            "Your first action must be repository inspection, not finish. Before finalizing, first "
            "make at least one successful non-finish repository inspection call such as search, "
            "file_editor view, or execute_bash. To finalize, call the finish function with "
            "command=submit and put the artifact in result: "
            "<function=finish><parameter=command>submit</parameter><parameter=result>...</parameter></function>. "
            "Never use <function=submit>.",
            "Your first action must be repository inspection, not submit. Before finalizing, first "
            f"make at least one successful non-submit repository inspection call such as {INSPECT_TOOLS_PHRASE}. "
            "To finalize, write the artifact in your final message text and then call "
            "<function=submit>\n</function>; the submit function takes no parameters.",
        ),
    ],
    "causal_localization_teacher_user.md": [
        ("Before calling finish, your final artifact", "Before calling submit, your final artifact"),
        (
            "Your first action must be repository inspection, not finish. Before submitting, first "
            "make at least one successful non-finish repository inspection call such as search, "
            "file_editor view, or execute_bash. Submit only with "
            "<function=finish><parameter=command>submit</parameter><parameter=result>...</parameter></function>.",
            "Your first action must be repository inspection, not submit. Before submitting, first "
            f"make at least one successful non-submit repository inspection call such as {INSPECT_TOOLS_PHRASE}. "
            "Write the artifact in your final message text and then call <function=submit>\n</function>; "
            "the submit function takes no parameters.",
        ),
    ],
}

# Tokens that must not survive anywhere in the generated tree. "search" as an English
# word is fine; only tool-call shapes are forbidden.
FORBIDDEN = (
    "file_editor",
    "<function=finish",
    "<function=search",
    "FUNCTION #3: search",
    "FUNCTION #4: finish",
    "parameter=cmd>",
    "the finish function",
    "`finish` tool",
    "non-finish",
)

STAGES = ("causal_localization", "localized_issue_reproduction", "patch_generation")
ROLES = ("student_system", "student_user", "teacher_system", "teacher_user")

TEACHER_EVIDENCE_SUFFIX = "\n\n## {teacher_evidence_heading}\n{teacher_evidence_json}\n"


def _apply(text: str, name: str, rules: list[tuple[str, str]]) -> str:
    for old, new in rules:
        if old not in text:
            raise SystemExit(
                f"[gen_openhands_prompt_templates] {name}: expected source text not found, "
                f"upstream template changed. Missing:\n---\n{old}\n---"
            )
        text = text.replace(old, new)
    return text


def build_openhands_system(eval_system: str) -> str:
    """The eval-yaml system prompt, verbatim. Invariant I1: train student == eval, byte for byte.

    Nothing is added and nothing is substituted: ``agent.py`` never formats
    ``system_prompt``, so the bytes in the yaml are the bytes the eval policy receives
    (verified: yaml 15513 chars == the ``agent_args.system_prompt`` recorded in a live
    eval trajectory). The two checks below are drift guards only.
    """
    if TOOL_MARKER not in eval_system:
        raise SystemExit("eval yaml system_prompt lost the tool marker")
    close = "</IMPORTANT>"
    if eval_system.count(close) != 1:
        raise SystemExit(f"expected exactly one {close} in the eval system prompt")
    for token in ("{working_dir}", "[working_dir]", "{problem_statement}"):
        if token in eval_system:
            raise SystemExit(
                f"eval system_prompt now contains {token!r}; agent.py does not format "
                "system_prompt, so substituting it here would break train/eval byte parity"
            )
    return eval_system


def build_openhands_user(eval_instance: str) -> str:
    """Eval-yaml instance prompt with our runtime placeholder name.

    Only two transforms, both parity-preserving: ``{working_dir}`` -> ``/testbed`` is
    exactly what ``agent.py:471`` passes at eval time, and ``{problem_statement}`` ->
    ``{issue_statement}`` is a rename of the same slot into our renderer's key. No
    trailing newline is added - the eval template ends without one.
    """
    text = eval_instance
    for token in ("{working_dir}", "[working_dir]"):
        text = text.replace(token, WORKING_DIR)
    replaced = False
    for token in ("{problem_statement}", "[problem_statement]"):
        if token in text:
            text = text.replace(token, "{issue_statement}")
            replaced = True
    if not replaced:
        raise SystemExit("eval yaml instance_prompt has no problem_statement placeholder")
    if "{issue_statement}" not in text:
        raise SystemExit("issue_statement placeholder missing after substitution")
    return text


def tool_block(openhands_system: str) -> str:
    """Everything from the function list onward (spec + format + IMPORTANT)."""
    return openhands_system[openhands_system.index(TOOL_MARKER) :]


def convert_stage_system(src_text: str, name: str, oh_tool_block: str) -> str:
    """Keep the stage preamble, swap the r2egym tool spec for the OpenHands one."""
    if TOOL_MARKER not in src_text:
        return ""  # caller handles templates with no tool spec
    preamble = src_text[: src_text.index(TOOL_MARKER)]
    preamble = _apply(preamble, name, REMAPS.get(name, []))
    return preamble.rstrip("\n") + "\n\n" + oh_tool_block.rstrip("\n") + "\n"


def check_placeholders(name: str, text: str) -> None:
    """A lone brace would raise inside str.format_map at render time."""
    stripped = re.sub(r"\{[a-z_]+\}", "", text)
    if "{" in stripped or "}" in stripped:
        raise SystemExit(f"{name}: stray single brace would break format_map rendering")


def check_forbidden(name: str, text: str) -> None:
    for token in FORBIDDEN:
        if token in text:
            line = next(
                (i for i, l in enumerate(text.splitlines(), 1) if token in l), -1
            )
            raise SystemExit(f"{name}:{line}: forbidden r2egym token {token!r} survived conversion")


def generate(eval_yaml: Path) -> dict[str, str]:
    eval_cfg = yaml.safe_load(eval_yaml.read_text(encoding="utf-8"))
    oh_system = build_openhands_system(eval_cfg["system_prompt"])
    oh_user = build_openhands_user(eval_cfg["instance_prompt"])
    oh_tools = tool_block(oh_system)

    out: dict[str, str] = {}

    # --- end_to_end: what OPD_END_TO_END_PROMPT=1 renders for student and, per stage,
    # for the OPSD teacher (render_end_to_end_teacher_messages). Student and teacher
    # differ ONLY by the privileged evidence block, mirroring the r2egym pair.
    out["end_to_end_student_system.md"] = oh_system
    out["end_to_end_teacher_system.md"] = oh_system
    student_user = oh_user
    out["end_to_end_student_user.md"] = student_user
    out["end_to_end_teacher_user.md"] = student_user.rstrip("\n") + TEACHER_EVIDENCE_SUFFIX

    # --- per-stage templates. render_teacher_messages_at_runtime uses the ROW stage,
    # and a missing file silently falls back to code-built r2egym prompts, so mirror
    # all of them.
    for stage in STAGES:
        for role in ROLES:
            name = f"{stage}_{role}.md"
            src = (SRC_DIR / name).read_text(encoding="utf-8")
            if role.endswith("_system"):
                if TOOL_MARKER in src:
                    out[name] = convert_stage_system(src, name, oh_tools)
                else:
                    # r2egym defers to {default_system_prompt} here, which is built from
                    # r2egym code paths; substitute the OpenHands system prompt so no
                    # r2egym tool spec can leak into this stage.
                    out[name] = oh_system
            else:
                out[name] = _apply(src, name, REMAPS.get(name, []))

    for name, text in out.items():
        check_placeholders(name, text)
        check_forbidden(name, text)
    check_eval_parity(out, eval_cfg)
    return out


def check_eval_parity(out: dict[str, str], eval_cfg: dict) -> None:
    """Invariant I1, enforced at generation time rather than documented.

    The student prompt the policy trains on must be the exact bytes the eval harness
    dispatches: system verbatim from the yaml, user identical once ``{working_dir}`` is
    resolved the way ``agent.py:471`` resolves it and the issue slot is renamed. Without
    this assertion, any future "just one more bullet" edit silently makes this run's
    score incomparable to the paper's.
    """
    eval_system = eval_cfg["system_prompt"]
    if out["end_to_end_student_system.md"] != eval_system:
        raise SystemExit("I1 violated: end_to_end_student_system.md != eval yaml system_prompt")
    eval_user = eval_cfg["instance_prompt"].replace("{working_dir}", WORKING_DIR)
    student_user = out["end_to_end_student_user.md"].replace("{issue_statement}", "{problem_statement}")
    if student_user != eval_user:
        raise SystemExit("I1 violated: end_to_end_student_user.md != eval yaml instance_prompt")


README = """# OpenHands-scaffold runtime prompt templates (generated)

Point a run at this directory with `OPD_PROMPT_TEMPLATE_DIR` (it is forwarded into Ray
workers automatically because of the `OPD_` prefix) together with `SWE_SCAFFOLD=openhands`.

**Do not hand-edit.** Regenerate with

    python3 scripts/tools/gen_openhands_prompt_templates.py --r2egym "$WS/R2E-Gym"

and verify with `--check`. Sources of truth:

* `<R2E-Gym>/src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling.yaml` - the
  prompt the eval side uses for `--scaffold openhands`; copied verbatim for train/eval parity.
* `swe_ladder/prompt_templates/` - the staged LOC -> REPRO -> PATCH scaffolding
  (stage preambles, required artifacts, teacher evidence blocks, placeholders).

The three prompt invariants this tree must satisfy (verify with
`scripts/tools/verify_prompt_invariants.py`):

* **I1 train == eval, byte for byte.** `end_to_end_student_system.md` is the eval yaml's
  `system_prompt` verbatim (`agent.py` never formats it), and `end_to_end_student_user.md`
  is its `instance_prompt` with the only two parity-preserving transforms below.
  `generate()` asserts this (`check_eval_parity`), so the build fails rather than drifts.
* **I2 stage-invariant student.** With `OPD_END_TO_END_PROMPT=1` the student always reads
  `end_to_end_student_*`; the stage changes the teacher's evidence and the token routing,
  never the student text.
* **I3 teacher == student + one privileged block.** `end_to_end_teacher_system.md` is
  byte-equal to the student system, and the teacher user is the student user plus exactly
  `\\n## {teacher_evidence_heading}\\n{teacher_evidence_json}`.

Transforms against the stock eval prompt - only these two, both parity-preserving:

1. `{working_dir}` -> `/testbed`, which is exactly what `agent.py:471` passes at eval time.
2. `{problem_statement}` -> `{issue_statement}`, a rename of the same slot into the key our
   runtime context uses.

Nothing is added. An earlier version of this generator appended seven RL
response-discipline bullets and an `<ENVIRONMENT_CONVENTIONS>` block (23 lines); both were
removed because they broke I1, and the two justifications for the conventions
block turned out to be false (the rule router keys REPRODUCTION on a `(repro|reproduce)`
basename regex *plus* test-edit / pytest v2 rules, so it does not need the prompt to name
`/testbed/reproduce_issue.py`; and the stock prompt already documents `submit`). The
accepted cost is that a base student has no format guardrails - watch mal% in the rollout
logs instead of patching the prompt.

Per-stage templates keep their r2egym stage preamble and required-artifact text; only the
tool spec after "We have access to the following functions:" is swapped, plus the
finish->submit wording. `localized_issue_reproduction_*_system.md` was `{default_system_prompt}`
in the r2egym set (which resolves through r2egym code paths) and is replaced by the OpenHands
system prompt so no r2egym tool spec can leak.
"""


def default_r2egym_root() -> Path:
    """The R2E-Gym checkout, from --r2egym, R2EGYM_ROOT or WORKSPACE_ROOT."""
    for value in (os.environ.get("R2EGYM_ROOT"), os.environ.get("WORKSPACE_ROOT")):
        if not value:
            continue
        root = Path(value)
        for candidate in (root, root / "R2E-Gym"):
            if (candidate / EVAL_YAML_REL).exists():
                return candidate
    return Path("R2E-Gym")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="diff against disk instead of writing")
    ap.add_argument("--r2egym", default=None, help="R2E-Gym checkout holding the eval prompt yaml")
    args = ap.parse_args()

    r2egym = Path(args.r2egym) if args.r2egym else default_r2egym_root()
    eval_yaml = r2egym / EVAL_YAML_REL
    if not eval_yaml.exists():
        print(f"ERROR: eval prompt yaml not found: {eval_yaml}\n"
              f"       pass --r2egym <checkout> or set R2EGYM_ROOT / WORKSPACE_ROOT", file=sys.stderr)
        return 2

    files = generate(eval_yaml)
    files["README.md"] = README

    if args.check:
        drift = False
        for name, text in sorted(files.items()):
            path = OUT_DIR / name
            on_disk = path.read_text(encoding="utf-8") if path.exists() else ""
            if on_disk != text:
                drift = True
                print(f"DRIFT {name}")
                sys.stdout.writelines(
                    difflib.unified_diff(
                        on_disk.splitlines(True), text.splitlines(True), f"disk/{name}", f"gen/{name}"
                    )
                )
        print("check: DRIFT" if drift else f"check: OK ({len(files)} files)")
        return 1 if drift else 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in sorted(files.items()):
        (OUT_DIR / name).write_text(text, encoding="utf-8")
        print(f"wrote {name:52s} {len(text):6d} chars")
    print(f"\n{len(files)} files -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
