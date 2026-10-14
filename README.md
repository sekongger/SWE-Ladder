# SWE-Ladder

**Staged privileged distillation for software-engineering agents.** One RL run is
split into three legs — *localize → reproduce → patch* — and in each leg only the
steps belonging to that rung are distilled from a teacher that can see gold-derived
evidence for *that* rung. The student never sees the evidence: it trains on its own
on-policy trajectories, with GRPO as the policy-gradient term and a reverse-KL to the
stage-matched teacher added on top.

The thing being tested is that **privilege is not interchangeable across rungs**. A
teacher that knows the answer helps a localization step and a reproduction step in
different ways, and handing a model whole-answer privilege at every token is both
weaker and more suppressive than routing each step to the privilege it can actually
use. The evidence for and against that claim is in [`analysis/`](analysis/README.md),
with the caveats attached to each number.

This repository is an **overlay**, not a fork: our code ships as files, the upstream
files we had to change ship as git patches, and one script puts them together. See
[`integration/README.md`](integration/README.md).

```bash
WS=~/swe-ladder-workspace

# 1. upstream checkouts (rllm + R2E-Gym from SWE-Master; verl nested inside rllm)
git clone https://github.com/RUCAIBox/SWE-Master "$WS"
git -C "$WS" checkout fb864a7d397b2c287ba6e38152da2228efc98e03
git clone https://github.com/verl-project/verl "$WS/DeepSWE_RL/rllm/verl"
git -C "$WS/DeepSWE_RL/rllm/verl" checkout d4399a673ae60748271af009d38e9a3dca86086b

# 2. graft this repository onto it
bash integration/install.sh --workspace "$WS" --check   # dry run first
bash integration/install.sh --workspace "$WS"
python3 integration/check_install.py --workspace "$WS"

# 3. point the configs at your machine, then smoke-test before anything long
cp training/configs/site.env.example training/configs/site.env && $EDITOR training/configs/site.env
scripts/tools/render_config.py training/configs/smoke_stage_routing.yaml -o /tmp/run.yaml
bash "$WS/DeepSWE_RL/rllm/examples/swe/run_npu_train_from_yaml.sh" /tmp/run.yaml
```

The smoke config is `n=2 × bs=2` and reaches the first `update_actor` in minutes,
which is where environment problems surface. Run it on every new machine.

## Map

| directory | what is in it |
|---|---|
| [`swe_ladder/`](swe_ladder/) | the method: stage router, privileged-context builders, top-k distillation loss, prompt templates. Installs as `rllm.opd`, so `swe_ladder.X` here is `rllm.opd.X` once grafted. |
| [`training/`](training/configs/README.md) | one YAML per run (24 of them), the launchers, and the entrypoints. A config is the *whole* run definition. |
| [`eval/`](eval/README.md) | end-to-end SWE-bench Verified: serve a checkpoint, roll out, recover from infrastructure failures until 500 clean rows exist. |
| [`stage_capability_eval/`](stage_capability_eval/README.md) | per-rung scoring with oracle localization handed to the later rungs, so a drop between rungs is attributable. |
| [`analysis/`](analysis/README.md) | six experiment groups, each with its claim, its shipped data, and its caveats. |
| [`scripts/`](scripts/data/README.md) | dataset construction, config rendering, prompt-template generation, environment setup. |
| [`data/`](data/README.md) | dataset recipe, staged-record samples, and the 119-step router gold set. No datasets or weights. |
| [`integration/`](integration/README.md) | the three upstream patches, the installer, and the install checker. |
| [`docs/`](docs/) | [TRAINING.md](docs/TRAINING.md) — how a run is launched and which knobs are load-bearing. [EVALUATION.md](docs/EVALUATION.md) — the protocols, the anchors, and the rules for reading a number. |

## The method

Three legs over one dataset, each granting privilege to one rung and resuming the
previous leg's checkpoint:

| leg | `OPSD_PRIVILEGED_STAGES` | steps | checkpoint |
|---|---|---|---|
| 1 | `causal_localization` | 30 | `global_step_30` |
| 2 | `localized_issue_reproduction` | 20 | `global_step_50` |
| 3 | `patch_generation` | 20 | `global_step_70` |

Per step: the router assigns the step a rung; if that rung is privileged this leg,
the teacher re-scores the step's tokens while conditioned on the privileged context
for that rung (localization evidence, reproduction evidence, or the gold patch), and
the student is pulled toward the teacher's top-k distribution on those tokens. Steps
on other rungs get the policy gradient alone. The teacher is the same frozen model as
the reference policy, so the only thing it has that the student doesn't is the
context.

The 70-step total is why `baseline_grpo_4b_openhands.yaml` runs 70 steps: same data,
same scaffold, same gradient budget, differing only in where privilege is applied.
`docs/TRAINING.md` covers the length envelope, the colocated-memory arithmetic, and
the device-specific knobs; `training/configs/README.md` is the catalogue of runs
including every ablation and the released ↔ internal name mapping.

## The stage router

Everything depends on the router, so it is measured rather than assumed. The gold set
is ten complete trajectories, every step adjudicated by hand as LOCATE / REPRODUCE /
PATCH / VERIFY — 119 steps, shipped at `data/stage_router_gold/gold_119.jsonl`. One
command, no accelerator, no network, importing the released classifier itself:

```bash
python3 analysis/stage_ladder/validate_router_gold.py            # v2, the rule set training used
python3 analysis/stage_ladder/validate_router_gold.py --version v1 --demo
```

| router | gold agreement | reproducible offline |
|---|---|---|
| rule **v2** (used by every training run here) | **98/119 = 82.4%** | yes |
| rule v1 (`stage_classifier_v1_legacy.py`) | **113/119 = 95.0%** | yes |
| semantic (hosted LLM, `semantic_stage_api.py`) | 112/119 = 94.1% | no — needs the gateway |

v2 scores *worse* than the rule set it replaced on this gold set, and we are keeping
the number visible rather than quietly reporting v1's. The confusion matrix says why
it is nonetheless the one the runs used: all 21 of its errors are in one direction —
gold-PATCH steps it still calls LOC (16) or REPRO (5) — and none the other way.

```
confusion (rows = gold, cols = rule router v2)
           LOC  REPRO  PATCH
    LOC     38      0      0
  REPRO      0      8      0
  PATCH     16      5     52
```

v2 is late to declare PATCH, never early. On the much larger silver check — every step
of the single-stage runs, where the run's own task stage is the label, ~10.4k steps —
the ordering reverses (v2 ≈ 92% against v1 ≈ 90%), because that check rewards staying
inside a rung rather than placing the boundary. Those silver numbers are **not**
reproducible from this repository: they are computed from `$TRAIN_OUTPUT_ROOT` rollout
dumps, which are tens of GB per run and not shipped (`validate_stage_router.py` is).

Two alternatives ship as ablations: `OPSD_STAGE_SEMANTIC_API=1` for the hosted router,
`OPSD_STAGE_CLASSIFIER=<pickle>` for a learned one. Selection is by *key presence*, so
a run's actual router is provable from its launch log rather than from its file name.
Any failure of either falls back to the rule router per step, so a bad endpoint
degrades silently instead of stopping a run — and silently is exactly how we once
confused two runs, which is why the launch log is the identity.

## Results, and what each one is worth

Numbers are only meaningful with their protocol attached; the protocols, the
non-comparability rules, and the reading rules are in
[`docs/EVALUATION.md`](docs/EVALUATION.md). In short:

**End-to-end, SWE-bench Verified, openhands scaffold, 500/500 clean rows**
(protocol `t07_ctx131072_steps150_stepout16384`). SWE-Master-4B-SFT scores 22.2% gross
/ 25.4% net of the context wall; its RL continuation scores 35.2% / 38.2%. Paired over
the same 500 instances: Δ = +13.0 pp, exact McNemar p = 3.7e-11. The mechanism is
narrower than the delta suggests — conditional on the agent terminating on its own,
both arms solve ~53%; what RL changed is how often a trajectory gets there at all
(30.2% → 60.2%).

**The scaffold is worth 16 points.** Same model, same 100 instances, same server and
budgets, only `--scaffold` differs: 10.0% under `r2egym` against 26.0% under
`openhands` (paired, exact p = 0.0015). The cause is a tool-name mismatch that the
harness reports as an *empty observation* rather than an error — 3,646 calls to a tool
that does not exist in that scaffold, 100% of them returning no content, 25.6% of all
tool steps silently wasted. **Never compare across scaffolds**, and print the scaffold
next to every number.

**The privileges are not interchangeable** (`analysis/probe/`). Candidate next actions
executed in real containers at sampled decision moments, four privilege conditions,
identical history: LOC privilege helps at localization moments (0.840 vs 0.759 no
privilege, paired Wilcoxon p = 0.029) and REPRO privilege helps at reproduction
moments (0.656 vs 0.540, p = 0.004), while each is at or below baseline on the other
rung. This is the strongest evidence here because the actions are executed, not
scored; it is also 93 and 46 moments measured on the *base* policy, so read it as what
privilege does to a next action, not as what RL eventually learns.

**The behavioral funnel** (`analysis/stage_ladder/`, r2egym scaffold, n ≈ 500 per
system) shows localization recall rising 0.336 → 0.467 and solve 0.096 → 0.156 from
the base model to the routed gs90 checkpoint, with the unrouted whole-answer control
in between on solve and *below the base* on verification. Its rungs are heuristics
over trajectory text — only `solve` is test-verified — and it is a different scaffold
and a different starting policy from the end-to-end anchors above. It does not stack
with them.

**Honest negatives ship too.** The token-level audit in
`analysis/grad_interference/routing_validation.json` does *not* recover the rule labels
from per-token teacher preference (18.9% against a 29.4% shuffle null); the claim the
routing rests on is behavioral, not per-token. Every training run here is a single
seed, so one- or two-point differences are not resolvable and none are claimed.

## What this repository does not contain

* **No datasets and no weights.** `data/README.md` gives the recipe and the four public
  sources; `scripts/data/` rebuilds the parquet datasets.
* **No evaluation or training output.** `output/` and `train_output/` are tens of GB
  per run. The analysis scripts that need them say so and exit on a missing path; the
  aggregates they produce are shipped as JSON so the tables can be recomputed.
* **No private evaluation split.** `stage_capability_eval/evaluator_private/` — gold
  patches, test patches, F2P/P2P names, reference records — is generated locally and
  deliberately absent. The public tree is checked against leakage: `oracle_loc` carries
  line spans and a `patch_sha256`, never patch text.
* **No secrets.** Configs that use the hosted router expect `OPSD_STAGE_API_KEY` from
  the launching shell. Setting it to an empty string in a YAML would override the
  environment and drop the router silently back to its rule fallback.

## Requirements

The standalone tooling (`scripts/`, `analysis/`, `stage_capability_eval/`, `eval/`)
runs on plain CPU Python ≥ 3.10: `pip install -r requirements.txt`. It deliberately
pulls in no accelerator stack. The trainer's own dependencies — torch or torch_npu,
vllm, verl, ray — come from the upstream rllm environment you installed into;
`scripts/setup/` has the Ascend and CUDA environment builds we used.

Training ran on 8× Ascend 910B (4B) and 4× A800 (8B). `docs/TRAINING.md` has the
device-specific knobs, including the two that are not performance settings but
correctness ones, and the memory envelope that made us stop trying 8B on NPU.

## License and attribution

Apache-2.0 ([LICENSE](LICENSE)). [NOTICE](NOTICE) records what is derived from what:
the patches and the generated prompt templates are derived works of
[SWE-Master](https://github.com/RUCAIBox/SWE-Master) (which vendors rllm and R2E-Gym)
and [verl](https://github.com/verl-project/verl); everything else is ours.

Work this builds on, all cited in the files that depend on them: SWE-Master
(arXiv:2602.03411), R2E-Gym (arXiv:2504.07164), OpenHands (arXiv:2407.16741), ARPO
(arXiv:2507.19849), and the fork-suppression result replicated in
`analysis/fork_suppression/` (arXiv:2607.05184).
