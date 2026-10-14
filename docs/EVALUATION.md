# Evaluation

A solve rate is not a property of a model. It is a property of a (model, scaffold,
step budget, context budget, temperature, denominator) tuple, and on agentic SWE
benchmarks the non-model half of that tuple moves the number by more than most
training interventions do. This document fixes the protocols we used, so that the
numbers in this repository mean something, and says which comparisons across them
are legitimate.

Three protocols appear here. They answer different questions and **their numbers do
not live on a common scale**:

| protocol | question | harness | denominator |
|---|---|---|---|
| end-to-end | did the model fix the bug | `eval/` | 500 SWE-bench Verified |
| per-stage capability | which rung did it fall off | `stage_capability_eval/` | 500, scored per rung |
| training valid split | is the run still healthy | the trainer itself | 32 held-out train instances |

The valid split is a **monitor, not a benchmark**. It is 32 instances drawn from the
same four sources as training (`data/README.md`), it moves several points on noise,
and no reported number comes from it. If you see a validation curve in a training
log, that is what it is.

## The end-to-end protocol

Everything reported end-to-end in this repository was produced at:

```
t07_ctx131072_steps150_stepout16384
```

which expands to, read off the evaluation rows themselves (`agent_args` / `env_args`):

| knob | value | why it is load-bearing |
|---|---|---|
| scaffold | `openhands` | decides which tools exist; see below |
| temperature | 0.7 | single sample, pass@1 |
| `max_steps` / `max_steps_absolute` | 150 / 150 | upstream's budget |
| `max_token_limit` | 131072 | cumulative context ceiling for the episode |
| per-assistant-turn cap | 16384 | a truncated edit is not a policy failure |
| `max_exec_time` | 90 s | per container command |
| `max_total_time` | 50000 s | per trajectory |
| instances | 500 | all of SWE-bench Verified, no sub-sampling |

Two of these are worth their own paragraph, because we got both wrong first and the
wrong versions produce numbers that look plausible.

**The per-turn cap (16384, not 2048).** The default per-assistant-turn cap in the
R2E-Gym runner is 2048 tokens. A `str_replace` whose arguments exceed that is cut off
mid-call; the edit then fails to apply, and the instance scores zero for a reason that
has nothing to do with the policy — the longer and more careful the edit, the more
likely it is truncated. This penalises exactly the behaviour training is trying to
produce. `eval/run_swebench_verified_eval.sh` therefore defaults `USED_YAML` to the
`*_stepout16384` variant wherever one exists. Override it only deliberately.

**The context ceiling (131072, not 61440).** We once ran with a self-imposed 61440
cumulative cap, which killed 52% of trajectories at the wall. Raising it to the
upstream default of 131072 took `token_limit` exits to zero. It did **not** rescue the
solve rate, which is the useful part of that result: the extra context was consumed by
the same failure mode, not converted into solved instances. Do not treat any number
produced under the 61440 cap as comparable to the ones here; none of ours are.

These two interact, and the interaction is the main caveat on our gross numbers.
Raising the per-turn cap to 16384 moves the context wall from "input > 129,024" to
"input > 114,688" (131072 − 16384), because the runner must reserve the full turn
budget. So we trade truncated edits for a context wall that bites ~19% of SFT rows.
That trade is right — a truncated edit is a silent corruption, a context wall is a
visible exit — but it costs gross solve rate, and it is why ours sits below the number
the upstream SWE-Master release reports for the same checkpoint (27.6% SFT / 33.4% RL).
Our net-of-context-wall figure, 25.4%, is within about a point of theirs. Neither of
those is a controlled comparison; they are two different harness instances.

## End-to-end anchors (4B, openhands scaffold)

Both arms are the same protocol byte for byte, the same 500 instances, the same server
configuration, 500/500 clean rows, zero infrastructure rows, all instance ids unique.
"SFT" is the released SWE-Master-4B-SFT; "RL" is its RL continuation.

| | SFT-4B | RL-4B |
|---|---|---|
| solve (gross) | **22.2%** (111/500) | **35.2%** (176/500) |
| solve (net of context wall) | **25.4%** (103/405) | **38.2%** (171/448) |
| finished on its own (`exit=agent`) | 30.2% | 60.2% |
| hit the 150-step wall | 50.8% | 29.4% |
| hit the context wall (`llm_query_error`) | 19.0% | 10.4% |
| mean / median steps | 118.1 / 150 | 109.4 / 111 |
| empty final patch | 32 | 3 |

Paired over the 500 common instances: both solve 93, RL only 83, SFT only 18, neither
306 → **Δ = +13.0 pp, exact McNemar p = 3.7e-11**.

The mechanism is worth more than the delta. Conditional solve rate *given the agent
terminated on its own* is essentially identical on the two arms — 53.6% (81/151) for
SFT, 53.2% (160/301) for RL. What RL changed is how often a trajectory reaches that
state at all: it moved mass out of the two low-yield buckets (step wall solves 8.7%,
context wall 8.4%) into the agent-done bucket, doubling the done rate and taking the
median trajectory from "ran out of steps" to "finished in 111". On this checkpoint
pair, the RL gain is almost entirely a *termination* gain, not a reasoning gain.

Gross and net are both given on purpose. Net is the fairer read of the policy, gross
is the number the benchmark actually produces; quoting only one of them is how
protocol differences get laundered into model differences.

## The scaffold is a measured effect, not a detail

`SWE_SCAFFOLD` decides which tools exist inside the container:

| scaffold | editor | finish | origin |
|---|---|---|---|
| `openhands` | `str_replace_editor` | `submit` | the upstream SWE-Master recipe |
| `r2egym` | `file_editor` + `search` | `finish` | R2E-Gym's minimal ReAct |
| `sweagent` | `str_replace_editor` | `submit` | SWE-agent's prompt set |

A model trained against one emits tool calls the others do not implement. The harness
does not report that as an error: for non-`execute_bash` tools, `observation.py`'s
`__str__` drops the exit code and stderr, so a call that died with exit 127 comes back
as an observation with no content, and the model retries the same edit until the step
budget is gone.

We measured it. One model (SWE-Master-4B-SFT, trained with `str_replace_editor`), one
server, one batch of 100 instances, identical temperature, step budget, context
ceiling, per-turn cap and worker count; the only difference is `--scaffold` and the
tool yaml it selects:

| | `r2egym` | `openhands` |
|---|---|---|
| solve (paired, n=100) | **10.0%** (10/100) | **26.0%** (26/100) |
| finished on its own | 5% | — |
| hit the step wall | 85% | — |
| mean steps | 142.2 | — |
| non-empty final patch | 69/100 | 93.6% (500-row arm) |

Paired McNemar on those 100: both 6, openhands only 20, r2egym only 4, neither 70 →
**exact p = 0.0015**. A **16-point** swing from a flag that most result tables do not
print.

The failure is fully legible in the trajectories once you know what to count:

* 3,646 bare `<function=str_replace>` calls (the openhands tool's signature under the
  r2egym prompt), of which **3,646 — 100.0% — return a zero-content observation**.
* Those are **25.6%** of all 14,224 tool steps in the arm (3,647 zero-content).
* The model called the scaffold's real editor, `file_editor`, 2,000 times — fewer than
  half as often as the tool that does not exist.

One predicate detail, because it cost us time: `not observation.strip()` finds **zero**
of these. A no-op observation is not empty, it is a 93-byte string —
`Execution output of [str_replace]:\n\nThis is step 4 of a maximum of 150. Steps
Remaining: 146.` — the header and the step footer with nothing between them. Strip both
before testing for emptiness.

So: **evaluate a checkpoint under the scaffold it was trained with, and print the
scaffold next to every number.** The r2egym-scaffold funnel in `analysis/README.md`
(base 0.096 → routed gs90 0.156) and the openhands anchors above are both internally
consistent and are *not* two points on one axis — different scaffold, and also a
different starting policy (that funnel's "base" is stock Qwen3-4B-Instruct-2507, not
the SWE-Master SFT checkpoint).

The same reasoning rules out comparing either to a published number from a third
harness. Our own `--scaffold leaf` is plumbed but has produced no reported number; see
`integration/README.md`.

## Per-stage capability

End-to-end resolve rate tells you the model failed, not which rung it fell off. The
`stage_capability_eval/` package scores the three rungs separately over the same 500
instances, handing each later rung the *oracle* answer to the earlier one so that an
upstream failure cannot be mistaken for a downstream one:

| rung | given | must produce | scored by |
|---|---|---|---|
| `LOC` | issue + buggy repo | files and functions to change | private LOC labels |
| `REPRO_GIVEN_LOC` | + oracle localization | a test that reproduces the issue | fails on buggy, passes on fixed |
| `PATCH_GIVEN_LOC` | + oracle localization | a source patch | F2P passes, P2P does not regress |

Two protocol properties make the scores trustworthy and both are enforced in code:

* **Reference preflight.** Before any model is scored, each task's environment must be
  shown to discriminate: the reference tests must fail on the buggy image and pass on
  the fixed one. A task whose reference run is broken cannot score a model. On the
  audited 25-task dev sample, 20 references are valid and 5 pending (three Django tasks
  error in the buggy image, two hit the 5-minute timeout).
* **Infrastructure is not failure.** `protocol/protocol.yaml` defines
  `status_values: [completed, model_invalid, budget_exhausted, not_applicable,
  infra_error, timeout]`, and pending or infra rows leave the denominator rather than
  counting as a model miss. This is the same rule the end-to-end loop applies by
  dropping `"Error:"` rows.

`manifests/stage_package_hashes.json` pins the SHA-256 of all four stage packages, so a
rebuilt package can be proven identical to the one a score was produced against.

**State of this protocol in the release.** The harness, manifests, oracle localization
and scorers all ship and run; the private split (gold patches, test patches, F2P/P2P
names, reference records) is generated locally and is deliberately absent. We have
collected a full base-model arm (503 task packages × 3 rungs, under the r2egym
scaffold) and the dev-sample reference audit above. **No per-stage score is reported in
this repository** — aggregation across the collected arms was not finished, and we are
not going to publish an aggregate we did not compute. The behavioral funnel in
`analysis/stage_ladder/` is the per-rung evidence that *is* reported, and it is a set of
heuristics over trajectory text, not test verdicts: only its `solve` column is
test-verified.

## Reading rules

These are the rules we applied to our own numbers; the scripts enforce the mechanical
ones.

1. **500/500 or say otherwise.** A partial sweep is not a benchmark score. Report the
   denominator next to the rate; `run_eval_until_complete.sh` exists to reach a clean
   500 rather than to let a run stop early.
2. **Infrastructure rows leave the denominator; model failures never do.** Rows whose
   `output_patch` begins `"Error:"` are container and socket failures. They are dropped
   and re-run — the one destructive operation in `eval/` — and are never scored as
   misses. A real attempt that failed stays.
3. **Compare paired, on the common instance set.** Both arms run the same 500
   instances, so McNemar's exact test on the discordant pairs is the right test and a
   difference of marginal rates is not. We report the full 2×2 so a reader can run
   their own test.
4. **One seed per configuration.** Every training run here is a single seed. A one- or
   two-point difference in solve rate is not resolvable by these experiments, and we do
   not claim any.
5. **Carry the protocol with the number.** Scaffold, per-turn cap, context ceiling and
   step budget, every time. Most of the surprises in this project were a protocol
   difference wearing a model difference's clothes — a 16-point scaffold swing, a 52%
   truncation rate, a 19% context wall — and all three were invisible in the logs until
   someone counted.
6. **Published numbers are reference rows, not baselines.** A number from another
   harness instance belongs in a separate, labelled row. It is not a control, and a gap
   to it is not a reproduction failure until the protocols have been matched.

## Where the artifacts live

Evaluation output is not shipped (`output/` is git-ignored; it is tens of GB). The runs
behind this document, by directory name, for anyone matching them against logs:

| what | directory under `output/` |
|---|---|
| RL-4B anchor | `swebench_verified_swe_master_4b_rl_openhands_t07_ctx131072_steps150_stepout16384_eval/` |
| SFT-4B anchor (arm O) | `swebench_verified_swe_master_4b_sft_repro_armO_openhands_t07_..._eval/` |
| scaffold control (arm R) | `swebench_verified_swe_master_4b_sft_repro_armR_r2egym_t07_..._eval/` |
| per-stage base arm | `stage_capability_eval/base_vs_gs70_20261008/base/` |

Each eval directory holds the per-instance `*.jsonl` and a `*.summary.json` written by
`run_swebench_verified_eval.sh`. Older summaries predate the `unique_instances` and
`infra_error_rows` keys the script writes today.
