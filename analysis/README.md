# Analysis: what each experiment claims, and what it does not

Every script here is offline. Nothing in this directory updates weights; the
training code lives in `../swe_ladder` and `../training`. The experiments exist
to answer one question — *does routing privileged evidence by stage do anything
real, or is it an expensive no-op?* — from several independent directions, so
that a single flattering measurement cannot carry the claim.

Results that are small enough to audit are committed next to the script that
produced them (`*.json`, `*.png`, `*.pdf`). Everything large is regenerable and
is **not** shipped; each section says what you have to re-create and from what.

## Path contract

The scripts share four environment variables. All default to something sane, so
in a plain checkout most of them can be left alone.

| Variable | Default | What it points at |
|---|---|---|
| `SWE_LADDER_ROOT` | the repo root, inferred from `__file__` | this repository: `analysis/`, `swe_ladder/` |
| `WORKSPACE_ROOT` | `$SWE_LADDER_ROOT` | the tree holding `output/` (eval results) and `hf_models/` (weights) |
| `TRAIN_OUTPUT_ROOT` | `$WORKSPACE_ROOT/DeepSWE_RL/rllm/rllm/train_output` | per-run training telemetry (`metrics/train_curve.jsonl`, `chat_completions/`) |
| `RLLM_ROOT` | unset | an rllm checkout, for the few scripts that need the *engine's own* tokenization primitives rather than a re-implementation |

`SWE_LADDER_ROOT` is also how the analysis scripts import the training modules:
they put the repo root on `sys.path` and import `swe_ladder.stage_classifier`
directly, so the classifier under test is the released one, byte for byte, not a
copy. The modules the offline tools need — `stage_classifier`,
`stage_classifier_v1_legacy`, `semantic_stage_api`, `staged_eval`,
`process_graph_evidence` and the rest of the non-loss half — import on a plain
CPU Python with no training stack. The five that do not (`topk_loss`,
`teacher_topk_client`, `rlsd` need `torch`; `staged_dataset` and
`runtime_prompt` need `rllm`) are trainer-side code and no offline tool touches
them.

## 1. Stage-capability ladder — `stage_ladder/`

**Claim.** Stage-routed privilege improves the *middle* of the funnel
(localization recall, reproduction, verification), not just the end-to-end
solve rate, and the gains are monotone along the LOC → REPRO → PATCH schedule.

**How.** `ladder_r2egym.py` turns finished SWE-bench Verified eval trajectories
into a five-rung behavioral funnel — localize / loc_recall / reproduce /
patch_nonempty / verify — plus solve. Heuristics are documented inline at the
top of the file. `ladder_openhands.py` is the same funnel with bash-side script
creation detection (heredoc / redirect / touch / tee), needed because openhands
trajectories often author repro scripts through `execute_bash` instead of the
editor; r2egym-family numbers are unchanged by it, and `ladder_r2egym.py`
remains canonical there.

**Shipped.** `ladder_r2egym_summary.json` (9 systems × 500 instances),
`stage_ladder_12sys_den500_v2.json`, `stage_ladder_6more_den500.json`,
`onpolicy_kl_curves.json`, and the two figures. `plot_ladder.py` reads the
committed summary, so it runs in a plain checkout; `ladder_r2egym.py` rebuilds
that summary from eval trajectories under `$WORKSPACE_ROOT/output` and refuses to
write if it finds none. Headline rows, all n≈500:

| system | loc_recall | reproduce | verify | solve |
|---|---|---|---|---|
| base | 0.336 | 0.594 | 0.226 | 0.096 |
| no-router (whole-answer) gs70 | 0.380 | 0.694 | 0.162 | 0.116 |
| routed gs70 | 0.440 | 0.587 | 0.256 | 0.141 |
| routed gs90 | 0.467 | 0.660 | 0.268 | 0.156 |

**Caveats.** The funnel rungs are heuristics over trajectory text, not unit-test
verdicts — only `solve` is test-verified. `verify` in particular rewards
re-running a repro script after editing, which a model can do without
understanding. Panel A of `plot_ladder.py` is the only *matched* comparison
(everyone at gs30); Panel B is best-available checkpoint per method and is
therefore favourable to whichever run was trained longest.

**Also here.** `inspect_stage_router.py` (three views over a run's own rollout
dumps: step trace, agreement with independent ground truth, trajectory-level
failure modes) and `validate_stage_router.py` (silver-label check: each
single-stage run should stay inside its own rung). Both read
`$TRAIN_OUTPUT_ROOT/<run>/chat_completions/`, which is not shipped. The
gold-label router numbers are in the top-level README.

## 2. Causal intervention probe — `probe/`

**Claim.** The privileges are not interchangeable: each one helps *in its own
phase and nowhere else*. This is the strongest evidence here, because the
candidate actions are **executed in real containers** rather than scored.

**How.** `step_probe.py` samples "the model is retrieving" moments from real base
trajectories, restores the environment by replaying the recorded prefix in a
fresh container of that instance's image, then under four conditions (no
privilege / +LOC / +REPRO / +PATCH, identical history otherwise) samples N
candidate next actions, executes the read-only ones, and scores gold-file
contact. `repro_probe.py` does the same at reproduction moments, but scores
whether a *created and executed* script actually exhibits the bug, then deletes
the script to restore state. `aggregate.py` does the paired tests.

**Shipped.** `results/` (93 moments × 2 steps), `repro_results/` (46 moments),
`hit_rates.json`, `repro_hit_rates.json`, and the two figures.

| condition | gold-file contact @ LOC moments (93) | failing-repro production @ REPRO moments (46) |
|---|---|---|
| no privilege | 0.759 | 0.540 |
| + LOC privilege | **0.840** | 0.511 |
| + REPRO privilege | 0.790 | **0.656** |
| + whole-answer (PATCH) privilege | 0.781 | 0.565 |

Paired Wilcoxon, one-sided: LOC vs none `p=0.029`, LOC vs whole-answer
`p=0.035`; REPRO vs none `p=0.004`, REPRO vs whole-answer `p=0.014`. The
matched-stage privilege gives the largest gain in each column, while the
off-diagonal conditions sit much closer to the no-privilege baseline; LOC
privilege at a reproduction moment is slightly *below* it (0.511 vs. 0.540).
The cross-rung conditions do not reproduce the matched-stage gain.

Reproduce with `python analysis/probe/aggregate.py` (needs `scipy` for the
p-values; without it the means still print). Regenerating `results/` needs
Docker, the SWE-bench Verified images, and a served base model.

**Caveats.** 93 and 46 moments, 8 and 6 samples per moment — the effect sizes
are a few points with overlapping standard errors, and the p-values are from a
paired test over moments, not over instances. Moments come from *base*
trajectories, so this measures what privilege does to the base policy's next
action, not what RL eventually learns. Gold-file contact is a proxy for
localization, not localization.

## 3. Gradient geometry and knowledge transfer — `grad_interference/`

**Claim.** Whole-answer privilege and stage privilege produce *measurably
different* gradients on the same tokens, and after 70 RL steps the routed
student remains anchored to its stage teachers while the unrouted one drifts
away from all of them.

**How.** A four-stage offline pipeline over 48 base trajectories:
`prep_data.py` (stage labels + exact re-tokenization + teacher contexts) →
`score_teacher.py` / `score_teacher_torch.py` (per-position teacher top-k;
the torch edition exists because vllm-ascend's `prompt_logprobs` crashes under
chunked prefill) → `compute_grads.py` (8 backward passes per trajectory, then
pairwise cosines) and `compute_transfer.py` (truncated reverse KL from each
model to each privileged teacher) → `plot_interference.py` / `plot_transfer.py`.
`routing_validation.py` is the classifier-free audit of the router itself:
privilege gain by *observable action type*, token-level best-teacher confusion
with a label-shuffle null, and a step-offset sweep.

**Shipped.** `grads/` (48 instances), `transfer/{base,routing_gs70,noroute_gs70}.json`,
`routing_validation.json`, `human_label_sample.tsv`, `routing_steps.jsonl`, and
four figures. Stage-matched reverse KL (student prompt, no privilege, lower =
closer to that teacher):

| model | LOC tokens → LOC teacher | REPRO → REPRO | PATCH → PATCH |
|---|---|---|---|
| base | 0.136 | 0.042 | 0.029 |
| routed gs70 | 0.163 | 0.055 | 0.040 |
| no-router (whole-answer) gs70 | 0.276 | 0.138 | 0.156 |

**Caveats.** Read that table as a *cross-model* comparison within a column. The
off-diagonal cells (in `transfer/*.json`) show that for LOC and REPRO tokens the
stage-matched teacher is not the nearest teacher even for the base model, so
"anchoring" means "stayed as close as the base was", not "closest to its own
teacher". `routing_validation.json` v2 is the honest negative in this group: the
token-level best-teacher label agrees with the rule label on 18.9% of
informative tokens against a 29.4% shuffle null — i.e. per-token privilege
preference does *not* recover the rule labels. The claim the routing rests on is
the behavioral one (groups 1, 2, 4), not a per-token one. All of
`teacher/`, `data/`, `actual_lp/`, `logs/`, `vllm_logs/` are regenerable and not
shipped; they are hundreds of GB of NPZ.

## 4. Fork suppression — `fork_suppression/`

**Claim (replication).** Under whole-answer privilege, the distillation term
suppresses the student more at *exploratory* positions than at committed ones;
stage routing weakens that suppression. This replicates arXiv:2607.05184 on our
setup.

This group has its own writeups, which are the authoritative version:

- **`REPORT.md`** — §0–§12: metric definitions, M1–M4, placebo, every CI.
- **`DEVIATIONS.md`** — where this reproduction departs from the paper and why,
  frozen before any metric was computed.
- **`freeze_rules.py`** — the thresholds and tool→class mappings, frozen
  2026-08-12 with an explicit no-edit-after-seeing-results declaration. Only the
  parameters already declared sweepable there may be swept.

Headline: in the global pool the variant-B NEP delta is `+0.0302`
with a by-instance bootstrap CI of `[+0.0144, +0.0473]` at p5, and all six
LOC/REPRO cells clear zero. The **P3 self-check** runs first and gates
everything: at the PATCH rung the stage-routed context *is* the whole-answer
context byte for byte, so the delta must be exactly 0 — measured 0.0 on both
samples. The placebo (inside the low-entropy region) lands at 1e-8 to 1e-10.

**Shipped.** `results.json`, `results_gs2833.json`, `results_gs30_clean.json`,
`results_gs30_partial.json`, `m3.json`, and three figure sets.

**Caveats, in the authors' own words** — see `DEVIATIONS.md` D1–D5. The two that
change what the numbers mean: (D1) in the dataset this ran on, the stage teacher
has *no* gold-derived evidence at LOC/REPRO, so the contrast is "full-answer
privilege vs. no privilege", and LOC ≈ REPRO by construction (~82 characters
apart); (D5) the frozen explore-vs-commit NEP is structurally uncomputable on an
entropy-percentile fork set and was replaced by the entropy-contrast variant B —
the original point estimates are still reported alongside in `results.json`.
The 94-instance sample merges steps 28–33, a declared minor deviation from pure
on-policy.

**Reproduce.** `bash run_all.sh gate` (N=20, P3 only — stop if it fails), then
`bash run_all.sh full 100`. Required: `RUN_DIR`, `STAGED_PARQUET`, `BASE_MODEL`;
see the script header. Only the `forwards` stage needs an accelerator.
`out/*.pkl` and `merged_ckpts/` are regenerable and not shipped.

## 5. Behavior and overlap — `behavior_structure.py`, `solved_overlap.py`

**Claim.** The improvement is a change in *how* trajectories are structured, not
a uniformly better model: routed runs traverse all three rungs more often,
reproduce before patching, and repeat failed edits less.

**How.** Both read eval jsonls and need no accelerator. `behavior_structure.py`
reports stage traversal, repeated-edit and repeated-*failed*-edit rates, action
mix, and the step-count distribution per system. `solved_overlap.py` computes
the per-system solved-set overlap matrix, unique wins, and solve rate stratified
by gold-patch size.

**Shipped.** `behavior_structure.json`, `solved_overlap.json` (both include the
raw per-instance lists, so the tables can be recomputed without re-running evals).

**Caveats.** `repro_before_patch` is an ordering statistic over classifier
labels, so it inherits the classifier's errors. The difficulty stratification by
gold-patch size is a crude proxy for difficulty, and the per-stratum cells are
small.

## 6. Figures and training-curve comparisons

- `figures/plot_dynamics.py` — four-panel optimization stability: entropy,
  gradient norm, response length, training solve rate, ours vs. pure GRPO. Reads
  `$TRAIN_OUTPUT_ROOT/<run>/metrics/train_curve.jsonl`; also prints
  caption-ready endpoint stats.
- `figures/plot_eval_curve.py` — held-out pass@1 vs. training step. Reads each
  eval's `*.summary.json` from `$WORKSPACE_ROOT/output` live, so refreshed evals
  change the figure. The gs60 point is flagged in the figure itself as
  eval-infrastructure dropout (only 291/482 non-empty patches), not a policy
  result.
- `reward_compare/plot_reward_compare.py` — per-step training reward,
  from-base vs. from-SFT. Honest-by-construction: the from-SFT run's Docker
  tunnel died at gs29, so reward is exactly 0.0 from gs29 on; the script plots
  that run **only up to gs29** and says why in its docstring.
- `stage_ladder/plot_onpolicy_kl.py` — each run's on-policy reverse KL to *its
  own* teacher, straight from training telemetry. No off-policy replay and no
  cross-objective comparison, which is what makes it quotable.
- `plot_case_study.py` — per-step teacher-divergence curves over one real solved
  trajectory with rule-stage bands; `fig_case_study_django__django-7530.png` is
  the committed instance.

**Caveat shared by this group.** These read training telemetry from
`$TRAIN_OUTPUT_ROOT`, which is not shipped (it is tens of GB of rollout dumps
per run). Without it the plotters will exit on a missing path; the numbers they
produce are quoted in the top-level README and in `REPORT.md`.

## Cross-cutting caveats worth stating once

- **Scaffold non-comparability.** Numbers produced under the r2egym scaffold and
  under the openhands scaffold are *not* comparable, and neither is comparable to
  published numbers from a third harness. Every table above is single-scaffold.
  See `docs/EVALUATION.md`.
- **Single seed.** Each training run is one seed. Differences of a point or two
  in solve rate are not resolvable by these experiments.
- **The router is a rule.** Unless a script says otherwise, stage labels come
  from `StageStateMachine(backtrack_threshold=3)` — the same rule the training
  runs used. Its measured accuracy, and the accuracy of the semantic and learned
  alternatives, are in the top-level README.
