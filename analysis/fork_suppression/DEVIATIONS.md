# Deviations: paper and design doc vs. the dataset and implementation we actually ran

> Frozen before any metric was computed, and folded into `REPORT.md` §10. Every
> deviation here is recorded as-is; no rule was changed to fit a prediction.

## D1. The SR teacher has no privileged evidence at LOC/REPRO (this reshapes what M4 measures)

**Fact (checked field by field, 2026-08-12).** In the staged dataset this analysis
ran on (`staged_opd_joint_swerebench500_r2e200_swegym100_swesmith500_task_intersection`),
all **1149** rows across four sources (swe_smith 500 / swe_rebench 431 / r2e_gym 155 /
swe_gym 63) carry an `extra_info.raw_record.teacher_pi` with **exactly one key,
`golden_patch`**.

`swe_ladder/staged_dataset.py::_verified_repository_evidence` **reads pre-stored
fields** (`gold_changed_non_test_files`, `gold_hunk_ranges`, `old_buggy_lines`,
`fail_to_pass_test_code`, `test_patch`, ...); it does **not** re-derive them from
`golden_patch` at runtime. Those fields are all absent in this dataset, so:

| SR stage teacher | `_verified_repository_evidence` dict | rendered text contains the gold patch |
|---|---|---|
| `causal_localization` (LOC) | `candidate_non_test_source_files_to_inspect` / `..._hunk_ranges` / `..._modified_source_segments` all empty | no |
| `localized_issue_reproduction` (REPRO) | `candidate_reproduction_test_changes_to_consider` empty | no |
| `patch_generation` (PATCH) | `verified_repair_diff` = the full gold patch (1026 chars in the sampled row) | yes |

**Restated semantics:**
- **SR (stage-routed)** is in fact "LOC/REPRO tokens see only the issue plus the
  stage instruction, with no gold-derived privilege; only PATCH tokens see the gold patch."
- **WA (force-PATCH, the clean definition)** is "**every** token sees the gold patch."

**Consequences for the analysis (no code changed, only the reading):**
1. **P3 is still an exact self-check.** At PATCH, WA and SR render the gold patch
   through the same function with the same arguments, so the two contexts are
   byte-identical and `NEP(WA) - NEP(SR)` must be exactly 0. P3 runs first and the
   pipeline stops if it fails (§9).
2. **M4's LOC/REPRO contrast is "full privilege vs. no privilege"**, not the
   "whole-answer privilege vs. stage privilege" the design doc assumed. That is the
   **strongest** available test of the fork-suppression hypothesis — if privilege
   suppresses exploration it has to show up here — but the semantics have changed, so
   conclusions must be worded as "full-answer privilege relative to no privilege".
3. **LOC is approximately REPRO by construction.** The two renderings differ by only
   ~82 characters (stage title, `required_artifact`, `localized_buggy_files` text) and
   both evidence blocks are empty. So M4's effective contrast is **binary — {LOC, REPRO}
   (pre-PATCH, unprivileged) vs. PATCH (privileged)** — not a three-level gradient. All
   three stages are still reported, with this noted on the figures and tables.

## D2. WA uses the force-PATCH clean definition

See `freeze_rules.py`: WA is `render_end_to_end_teacher_messages(rec, "patch_generation")`
applied to every token. Against a production WA run (`render_teacher_messages_at_runtime`,
per-stage templates) the **evidence is the same** (gold patch only) but the **outer
template differs** (a ~200-314 line diff). If M3 is ever run against a real
whole-answer checkpoint, that difference is the robustness caveat.

## D3. The primary fork criterion is student full-vocabulary entropy; the paper's criterion is reported alongside

The paper (arXiv:2607.05184 §4.1) defines fork = **teacher top-1 prob < 0.25** and
lock = **> 0.65**. Our primary criterion is the **student's full-vocabulary entropy
H_t** (top-p% percentile for M2/M4, an absolute threshold for M3). The student-entropy
criterion is the NEP judge; the paper's teacher-top-1 criterion is **reported alongside**
it in M1/M3 as a cross-check and is never used as the judge.

## D4. Instruction-tuned models have a flat fork rate (caveat)

The student is Qwen3-4B-Instruct-2507, an instruction-tuned model, and the paper
observes that fork rates for such models are naturally flat (~0.06). M1/M3 must note
that a uniformly low fork rate with little spread between conditions is a property of
the instruction-tuned model, and does not weaken the M2/M4 conclusions, which rest on
student-entropy percentiles.

**Confirmed empirically (2026-08-12, step-30 clean):** under the paper's criterion the
fork rate is WA=0.002 / SR=0.0005 with a lock rate of 0.93 — the base teacher is
extremely confident, so the paper's teacher-top-1 criterion carries almost no signal
here. That further settles the choice of student entropy as the primary criterion.

## D5. NEP changed from explore-vs-commit to an entropy contrast (variant B)

**Fact.** The frozen definition `NEP = mean(U | fork & EXPLORE) - mean(U | fork & COMMIT)`
is **structurally uncomputable** on a fork set defined by high entropy percentiles.
Measured (step 30, K=32): EXPLORE H_mean=0.231, COMMIT H_mean=0.0021 — COMMIT positions
are nearly deterministic and therefore almost never land in the top-p% student-entropy
fork set (zero COMMIT positions at the LOC stage), so the frozen NEP's COMMIT cell is
always empty and the metric returns NaN.

**The authorized replacement (variant B)** keeps NEP's semantics — is exploration
suppressed more strongly than commitment? — and redefines both cells by student entropy:
- top cell = **top-p% student entropy** within the pool/stage (the high-uncertainty,
  exploratory end);
- base cell = **bottom-50% student entropy** within the pool/stage (the low-uncertainty,
  committed end);
- `NEP_B = U(top) - U(base)`, and `delta = NEP_B(WA) - NEP_B(SR)`.

**Fidelity:**
1. The frozen explore-vs-commit NEP point estimates are **still reported alongside** in
   `results.json` (`M2_frozen_explore_vs_commit` / `M4_frozen_explore_vs_commit`, mostly
   NaN), so the deviation is fully auditable.
2. Variant B **does not break P3**: at PATCH, WA is identical to SR, so delta == 0 holds
   exactly (measured 0.0 on both the 94-instance / 8275-PATCH-position sample and the
   16-instance / 1370-position sample).
3. The placebo is correspondingly moved inside the low-entropy region (25th-50th
   percentile vs. below the 25th), which is still a negative control with no real forks;
   measured delta is approximately 0 (1e-8 to 1e-10 on the 94-instance sample).

**94-instance re-check (2026-08-12, steps 28-33 merged, fixed gs30 student):** for
variant B's `delta = NEP_B(WA) - NEP_B(SR)`, the 95% by-instance bootstrap CI is
**entirely above 0** in the M2 global pool (p5/10/20) and in all six M4 LOC/REPRO cells
(3 percentiles x 2 stages); the K=16 slice also clears 0 everywhere. On the 16-instance
sample (step-30, purely on-policy) the sign is the same but the CIs straddle 0. Going
from 16 to 94 instances left the sign unchanged and tightened the CIs past 0, which is
the signature of "underpowered, then powered", not of a criterion retuned to fit — the
percentiles and metrics were frozen in `freeze_rules.py` before any result was seen. The
94-instance sample is a fixed-instrument robustness extension; policy drift across ±3
steps is small and is recorded as a declared minor deviation (see `REPORT.md` §12,
fidelity statement).
