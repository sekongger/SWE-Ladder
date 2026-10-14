# OpenHands-scaffold runtime prompt templates (generated)

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
  `\n## {teacher_evidence_heading}\n{teacher_evidence_json}`.

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
