# Training

A run is one YAML file. `training/configs/README.md` is the catalogue — which config
is which experiment, what each resumes from, and the released ↔ internal name mapping.
This document is the part that is the *same* across all of them: how a run is launched,
how the length and memory numbers are derived, and which knobs are load-bearing.

```bash
cp training/configs/site.env.example training/configs/site.env && $EDITOR …   # once
python3 scripts/tools/render_config.py training/configs/smoke_stage_routing.yaml -o /tmp/run.yaml
bash <rllm>/examples/swe/run_npu_train_from_yaml.sh --dry-run /tmp/run.yaml   # inspect
bash <rllm>/examples/swe/run_npu_train_from_yaml.sh /tmp/run.yaml
```

Run `smoke_stage_routing.yaml` first on any new machine. It is `n=2 × bs=2` and reaches
the first `update_actor` in minutes, which is where most environment problems surface.

## The ladder

Three legs over one dataset, each granting privilege to one rung:

| leg | `OPSD_PRIVILEGED_STAGES` | steps | offset | checkpoint |
|---|---|---|---|---|
| 1 | `causal_localization` | 30 | 0 | `global_step_30` |
| 2 | `localized_issue_reproduction` | 20 | 30 | `global_step_50` |
| 3 | `patch_generation` | 20 | 50 | `global_step_70` |

Each leg is a separate launch that resumes the previous leg's checkpoint:

```bash
python3 scripts/tools/render_config.py training/configs/ladder_4b_openhands_leg2_repro.yaml \
    --set RESUME_CHECKPOINT=$WS/train_output/<leg1-run>/global_step_30 -o /tmp/leg2.yaml
```

`TRAINER_STAGE_GLOBAL_STEP_OFFSET` is what makes the logged `global_step` absolute
across legs, so leg 2's first step is 31 and not 1. The trainer enforces the budget
itself (`fit_agent` exits once `global_steps > offset + TRAINER_TOTAL_TRAINING_STEPS`),
the entrypoint sets `trainer.save_freq=10` so every leg boundary lands on a checkpoint,
and the LR schedule is constant — a short leg does not reshape it.

The 70-step total is why `baseline_grpo_4b_openhands.yaml` runs 70 steps: same data,
same scaffold, same gradient budget, differing only in where privilege is applied.

## The length envelope

Five numbers that must move together. Changing one alone either truncates trajectories
silently or runs the engine out of memory:

```
MAX_PROMPT_LENGTH            12288   system + user prompt
MAX_RESPONSE_LENGTH          38912   everything the policy generates, whole episode
ROLLOUT_MAX_MODEL_LEN        51200   = 12288 + 38912   vLLM's context window
PPO_MAX_TOKEN_LEN_PER_GPU    51200   = ROLLOUT_MAX_MODEL_LEN
LOGPROB_MAX_TOKEN_LEN_PER_GPU 51200  = ROLLOUT_MAX_MODEL_LEN
```

The invariant: `ROLLOUT_MAX_MODEL_LEN >= MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH`, and
the two per-GPU token budgets equal it. (The sweagent legs set it to the sum *plus*
`ROLLOUT_CONTEXT_WINDOW_SAFETY_MARGIN`, which is the same rule with slack.) Three more
knobs are derived and must be changed with `MAX_RESPONSE_LENGTH`, not independently:
`AGENT_MAX_RESPONSE_TOKENS_FOR_TRAINING`, `OPSD_MAX_RESPONSE_TOKENS_FOR_TRAINING`,
`ROLLOUT_MAX_NUM_BATCHED_TOKENS`.

`MAX_RESPONSE_LENGTH` is a **cumulative episode budget**, not a per-turn one. An 88-step
agentic trajectory spends it across all turns, so a budget that looks generous per turn
is not. Per-turn caps are separate: `ROLLOUT_PER_STEP_MAX_TOKENS` (8192) bounds one
assistant turn, and it must be large enough for the longest single edit the model will
emit — a `str_replace` truncated mid-call fails to apply, and the episode then fails for
a reason that has nothing to do with the policy. This is the same failure the evaluation
side hits with a 2048-token per-step cap; see `docs/EVALUATION.md`.

Trajectories that exceed `MAX_RESPONSE_LENGTH` are not discarded: a *solved* overlong
rollout keeps `SOFT_OVERLONG_REWARD_FACTOR` (0.5) of its reward, so full credit does not
reinforce runaway length, while unsolved overlong rollouts score 0 as usual.

## Memory on a colocated node

Colocated means the policy, the reference/teacher and the vLLM engine share the same
cards. `ROLLOUT_GPU_MEMORY_UTILIZATION` caps only vLLM's **KV cache reservation**, not
its total residency, so the nominal share and the real one differ a lot — this is why
the configs sit at 0.18–0.32 rather than the 0.85 a serving-only deployment would use.
Between rollout and update, `ROLLOUT_ENABLE_SLEEP_MODE` + `ROLLOUT_FREE_CACHE_ENGINE`
release the cache so the actor can take the space back.

Two numbers push it over on the OPSD configs specifically: the teacher's top-k forward
pass (one extra forward over the whole batch) and one teacher-context slot per
privileged stage. **An OPSD config cannot inherit a GRPO config's utilization.** We
re-used `0.32` from the GRPO baseline on a staged run and the teacher forward died with
`EL0004` every time; `0.18` with `ACTOR_ENTROPY_CHUNK_SIZE=128` is what it takes at that
batch shape. The 6-slot frontier in `ablation_process_graph_privilege.yaml` needs more
again (it runs at 0.30 and still OOMed an update at step 12 on a 0.32 ancestor).

When you need headroom, in this order:

1. `ACTOR_ENTROPY_CHUNK_SIZE` 512 → 256 → 128 (cheap, no semantic effect)
2. `MODEL_ENABLE_ACTIVATION_OFFLOAD: 'True'`
3. `ROLLOUT_GPU_MEMORY_UTILIZATION` down in steps of 0.02–0.04
4. `ROLLOUT_MAX_NUM_SEQS` down (12 is the floor we have run)

**Do not** reach for these two:

* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — incompatible with vLLM's sleep
  mode. `CuMemAllocator` asserts during engine init and the run dies at startup.
* `ROLLOUT_ENFORCE_EAGER: 'True'` — it does free memory, by disabling graph capture, and
  it costs enough rollout throughput that the run is no longer the same experiment. Fix
  the budget instead.

## Device-specific

**Ascend (NPU).** `RLLM_DISABLE_FSDP_SYNC_MODULE_STATES` is not a performance knob and
has opposite correct values by model: at 4B (tied embeddings) the broadcast corrupts
weights and it must be `'1'`; at 8B (untied, meta-initialised) the broadcast is required
and `'0'` is correct — with it disabled the model emits garbage from the first token.
Both 8B configs here set `'0'`. `HCCL_OP_EXPANSION_MODE=AIV` and
`TASK_QUEUE_ENABLE=1` are the standard Ascend settings; `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`
raises vLLM's 300 s default, which otherwise reports a neighbour's CPU load spike as
`EngineDead`.

**8B on NPU: don't.** We tried and stopped. A colocated 8B policy plus reference plus KV
cache did not fit on one 8-card node: at 64 concurrent sequences the engine pinned ~61 GB
per card and still held ~45 GB after sleep, against ~2 GB of raw training state. The
problem is the memory envelope, not the algorithm. Both 8B configs here are CUDA
(`train_entrypoint_gpu.sh`, `run_gpu_train_from_yaml.sh`).

**Ray startup.** `RAY_agent_register_timeout_ms` / `RAY_raylet_start_wait_time_s` are set
high in every config. On a container whose PID 1 never reaps children, the per-node
metrics agent takes ~30 s to start while the raylet waits ~15 s for its port file, and
the node simply never registers ("node timed out during startup"). The proper fix is a
container restart with `--init`; these two knobs are what make a run possible without it.

**`/dev/shm`.** After any violent death (`kill -9` of a training tree, a crashed engine),
check `df /dev/shm` before relaunching. Orphaned segments fill it, and the next engine
boot dies silently with no useful error.

## Privilege and routing

The knobs that define the method, as opposed to the hardware:

| knob | meaning |
|---|---|
| `OPSD_ENABLE` | staged privileged distillation on |
| `OPSD_PRIVILEGED_STAGES` | which rung gets privilege this leg (absent = all three) |
| `OPSD_STAGE_ROUTING` | per-step routing on; each step's tokens are scored under its stage's teacher |
| `OPSD_STAGE_BACKTRACK` | N consecutive exploration steps move the phase back to LOCATE (0 = forward-only) |
| `OPSD_LOSS_COEF` | weight on the reverse-KL term (0.15 across the ladder) |
| `OPSD_POLICY_LOSS_MODE` | `pg_plus_opsd` = GRPO + distillation; `opd` = distillation only |
| `OPSD_TEACHER_SOURCE` | `ref_policy` (in-process), `rollout`, or an external served teacher |
| `OPD_END_TO_END_PROMPT` | student sees the unified agentic prompt, identical to eval |
| `SWE_SCAFFOLD` | which tools exist in the container — see below |

The router is rule-based by default (a state machine over the action text). The
alternatives are selected by *presence*: an `OPSD_STAGE_CLASSIFIER` key switches to a
learned classifier, `OPSD_STAGE_SEMANTIC_API=1` to a hosted LLM router. A run's actual
router is therefore provable from its launch log rather than from its file name, which
matters because these were confused once already.

`OPSD_STAGE_API_KEY` is deliberately **absent** from the semantic-router configs: export
it in the launching shell. A key set to an empty string in YAML would override the
environment and drop the router silently back to its rule fallback.

## Scaffold is not a free variable

`SWE_SCAFFOLD` decides which tools exist inside the container. A model trained against
one scaffold emits tool calls the others do not implement, and the harness reports that
as an empty observation rather than as an error — so the failure is invisible in the
logs and shows up only as a low score. Train and evaluate a checkpoint under the same
scaffold, and say which one next to every number. `docs/EVALUATION.md` has the measured
size of the effect.

`OPD_PROMPT_TEMPLATE_DIR` must match `SWE_SCAFFOLD`; the shipped trees are
`rllm/opd/prompt_templates_openhands` and `…_sweagent` (`prompt_templates` is the
original r2egym set).

## What to watch

The entrypoint echoes every resolved value at startup. That block, not the file name, is
a run's identity — configs get edited and re-used.

| symptom | what it usually is |
|---|---|
| `resp_len` ≈ 12, `grad_norm` = 0 | the Docker daemon died; the rollouts are empty, the policy is fine |
| `EngineDead`, `shm_broadcast TimeoutError` | neighbour CPU load starving the engine worker, not an engine bug |
| throughput 0 but `/v1/models` answers | Ascend graph-replay OOM outside the KV pool; probe the engine, don't trust the health endpoint |
| `reverse_kl` ≈ 1.4 instead of ≈ 0.02 | expected when teacher ≠ student (base-student / SFT-teacher arm) |
| rising `len_clip` with flat reward | the episode budget is binding; see the length envelope above |

`ENABLE_ROLLOUT_DATA_DUMP=1` writes per-step rollout dumps, which is what
`analysis/stage_ladder/` and the router gold set are built from.
