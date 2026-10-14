# Integration: grafting SWE-Ladder onto the upstream stack

SWE-Ladder is an **overlay, not a fork**. The training stack it runs on
(`rllm` + `verl` + `R2E-Gym`) is three upstream repositories totalling well over
half a million lines. Re-publishing them with our changes stirred in would make
it impossible to see what we actually did — and would silently fork us off
upstream's maintenance.

So the split is:

| what | how it ships |
|---|---|
| code we wrote | normal files: `swe_ladder/`, `training/`, `analysis/`, `eval/`, `scripts/` |
| upstream files we had to change | git patches in `patches/` |
| putting the two together | `install.sh` |

The upside is that `git apply --stat patches/*.patch` is a complete, honest
answer to "what did you change upstream?". The cost is one install step.

## Upstream checkouts

```bash
WS=~/swe-ladder-workspace

# rllm + R2E-Gym (one repository: SWE-Master vendors both)
git clone https://github.com/RUCAIBox/SWE-Master "$WS"
git -C "$WS" checkout fb864a7d397b2c287ba6e38152da2228efc98e03

# verl is a separate repository nested inside rllm
git clone https://github.com/verl-project/verl "$WS/DeepSWE_RL/rllm/verl"
git -C "$WS/DeepSWE_RL/rllm/verl" checkout d4399a673ae60748271af009d38e9a3dca86086b
```

Resulting layout — the one `install.sh` and the launchers assume:

```
$WS/
├── DeepSWE_RL/rllm/            # the RL trainer and agent loop
│   ├── rllm/                   #   `rllm.opd` is grafted in here
│   ├── examples/swe/           #   our launchers and configs land here
│   └── verl/                   # verl, its own git repo
└── R2E-Gym/src/r2egym/         # agent scaffold + Docker sandbox
```

The two pins are the commits the patches were cut against. `git apply` is
content-addressed, so other commits may well work; `install.sh` reports a
HEAD/pin mismatch as advisory and tries anyway.

## Install

```bash
bash integration/install.sh --workspace "$WS"            # patch + install
bash integration/install.sh --workspace "$WS" --check    # dry run, writes nothing
bash integration/install.sh --workspace "$WS" --mode link
```

It does two things:

1. **Applies the three patches** to `rllm`, `R2E-Gym` and `verl`.
2. **Installs our files**: `swe_ladder/` → `<rllm>/rllm/opd` (so the training
   code imports it as `rllm.opd.*`), and `training/` → `<rllm>/examples/swe/`.
   That destination is not cosmetic: every launcher derives the workspace root
   from its own location (`${SCRIPT_DIR}/../../../..`), so `run_npu_train_from_yaml.sh`
   only finds the workspace when it sits in `examples/swe/`.

`--mode link` symlinks both trees back to this repository instead of copying, so
edits here take effect without re-installing — the mode to use while developing.
`--mode copy` (default) is the one to use for a run you want to be able to
reproduce later.

Other flags: `--rllm/--r2egym/--verl` to override any individual checkout,
`--skip-patches` to install only our files, `--force` to overwrite files that are
neither ours nor already backed up, `--help` for the full list.

**Re-running is safe.** Patches already present are detected (`git apply
--reverse --check`) and skipped; a tree that drifted from the pin gets one
`--3way` attempt before the script gives up with the exact command to diagnose
the failing hunk. Our installed files are replaced. A copy-mode install drops a
`rllm/opd/.swe_ladder_install` receipt recording source, mode, pins and
timestamp; that receipt is how a later run knows a destination is ours and may be
replaced without `--force`. Upstream files that collide are backed up once to
`*.pre-swe-ladder`.

**Uninstall.**

```bash
rm -rf  "$WS/DeepSWE_RL/rllm/rllm/opd"
git -C  "$WS/DeepSWE_RL/rllm"      apply --reverse integration/patches/rllm-swe-ladder.patch
git -C  "$WS/R2E-Gym"              apply --reverse integration/patches/r2egym-swe-ladder.patch
git -C  "$WS/DeepSWE_RL/rllm/verl" apply --reverse integration/patches/verl-swe-ladder.patch
```

Then delete the `training/` entries from `examples/swe/` (they are either
symlinks or files whose names do not collide with upstream's five), and restore
any `*.pre-swe-ladder` backup.

After installing, `install.sh` runs the stage router — the one component every
experiment depends on — on a four-step fixture and asserts the labels. It is a
real execution, not an import check, and it needs neither torch nor an
accelerator.

## What is in each patch

### `rllm-swe-ladder.patch` — 22 files, +7578 / −916

This is where the method lives. `rllm.opd` (our `swe_ladder/`) holds the stage
router, the privileged-context builders and the loss helpers; this patch is the
plumbing that calls them.

| file | + | what it adds |
|---|---|---|
| `rllm/trainer/verl/agent_ppo_trainer.py` | 3311 | the staged-privilege training loop: per-stage teacher forward, top-k teacher log-prob capture, the OPSD/GRPO objective switch, per-stage reward and KL telemetry |
| `rllm/engine/agent_execution_engine.py` | 1968 | rollout side: stage classification while the trajectory is being generated, privileged-context assembly per rung, step/context budgets, ARPO entropy branching |
| `rllm/trainer/verl/train_agent_ppo.py` | 686 | entrypoint: the `opsd.*` / `agent.*` config surface, trainer-layout selection, run bookkeeping |
| `rllm/trainer/verl/agent_ppo_trainer_pipeline.py` | 580 | the same objective for the disaggregated (pipeline) resource layout |
| `rllm/router/router.py` | 440 | teacher/privilege routing, top-k entropy used by the ARPO branch policy |
| `rllm/environments/swe/swe_arsenal.py` | 405 | environment variant that talks to a remote Docker host |
| `rllm/environments/swe/swe.py` | 256 | Leaf scaffold wiring, the issue-reproduction verifier reward, observation sanitizing |
| `rllm/agents/swe_agent.py` | 211 | stage-aware prompting, edit accounting, submit gating per rung |
| `rllm/agents/system_prompts.py` | (−192) | the prompt set actually used by the three rungs |
| `rllm/environments/swe/issue_reproduction_verifier.py` | 134 | *new*: run `reproduce_issue.py` on the buggy and the gold-patched tree, parse the two exit codes, score "fails before, passes after" |
| `rllm/trainer/verl/layout.py` | 38 | *new*: colocated vs. pipeline resource-pool selection |
| `rllm/environments/swe/observation_sanitizer.py` | 26 | *new*: strip synthetic answer markers out of observations |
| `rllm/{agents,environments,parser}/__init__.py`, `rllm/environments/base/__init__.py`, `rllm/parser/tool_parser/__init__.py` | 184 | registrations for the above |
| `rllm/trainer/config/ppo_trainer.yaml`, `ppo_megatron_trainer.yaml` | 31 | the new config keys, with defaults that reproduce stock behaviour |
| `rllm/environments/swe/swe_sim.py` | 11 | Leaf branch in the sim-reward environment variant |
| `pyproject.toml` | 3 | dependency pins |

### `r2egym-swe-ladder.patch` — 7 files, +443 / −15

Scaffold and sandbox. Two of these are correctness fixes that materially changed
our measurements, and are worth reading even if you never run SWE-Ladder:

- **`agenthub/action/action.py`** — `normalize_action_for_allowed_commands()`.
  A model trained under one scaffold emits that scaffold's tool names (`bash`,
  `str_replace`, `view`). Under a different registered command set those names
  resolve to nothing, the container exits 127, and the agent receives an *empty
  observation* — a silent no-op it then retries. This maps the legacy names onto
  the registered commands (`bash` → `execute_bash`, editor subcommands →
  `file_editor` with `command=`).
- **`agenthub/agent/agent.py`** — the malformed-parameter fix. The model emits
  `<parameter=command=VALUE>` (training format); upstream's repair rewrote only
  the opening token, producing `<parameter=command>VALUE>` with a stray `>` and
  no closing tag, so the regex parser swallowed the following `<parameter=path>`
  into the command value — commands silently no-oped or crashed with
  `path=None`. Replaced with a full normalization. Also two type guards so a
  tool call whose arguments do not decode to an object raises instead of
  corrupting state.
- **`agenthub/environment/env.py`** — dispatch for the five-function Leaf
  protocol (base64 JSON into a stdin-driven runner), kept out of the legacy XML
  command registry so the `r2egym` and `sweagent` paths are bit-for-bit
  unchanged; plus the call into the normalizer above.
- **`agenthub/tools/leaf.py`**, **`agenthub/tools/_leaf_tool_runner.py`** —
  *new*: the Leaf tool schema (`Read`/`Write`/`Edit`/`Glob`/`Bash`) and a
  stdlib-only in-container executor for it.
- **`agenthub/runtime/docker.py`** — skip the blind `git fetch` when the
  requested SWE-smith ref is already present (over a remote Docker socket that
  fetch usually just burns the whole timeout), and an opt-in `R2EGYM_TIMING=1`
  per-command timing line.
- **`agenthub/config/sweagent/edit_non_fn_calling.yaml`** — the prompt now names
  the command that is actually registered (`execute_bash`, not `bash`).

### `verl-swe-ladder.patch` — 39 files, +2745 / −207

Trainer internals. Four separable groups:

1. **The distillation objective** — `workers/actor/dp_actor.py` (+980),
   `workers/config/actor.py`, `trainer/config/{actor/actor.yaml,algorithm.py,ppo_trainer.yaml}`
   and the four regenerated `_generated_*.yaml`. Top-k teacher distillation
   inside the PPO actor loss: top-k gather from logits, Ulysses sequence-parallel
   slicing of the top-k ids, response padding/alignment, an
   activation-offload-safe path that reuses the main forward instead of taking a
   second one, and the SDAR gate. Every new key defaults to off — with the stock
   config this file computes the stock loss.
2. **Ascend / NPU enablement** — `utils/device.py` (device resolution,
   `VERL_DEVICE_NAME`), `utils/attention_utils.py` (unpad/pad without
   flash-attn), `models/transformers/monkey_patch.py`, `utils/fsdp_utils.py`
   (`fsdp_sync_module_states_enabled`, `safe_reshard_fsdp_module`),
   `single_controller/*`, `checkpoint_engine/hccl_checkpoint_engine.py`,
   `model_merger/fsdp_model_merger.py` + `scripts/legacy_model_merger.py` (NPU
   unpickle stubs, so an NPU checkpoint can be merged on a CPU box), and two
   tests.
3. **Colocated weight sync and memory** — `workers/fsdp_workers.py` (+473),
   `workers/engine/fsdp/transformer_impl.py`,
   `rollout/vllm_rollout/bucketed_weight_transfer.py` (*new*: bucketed transfer
   with file-mmap or IPC shared memory, backend chosen by free space),
   `rollout/vllm_rollout/utils.py` and `vllm_async_server.py` (per-device ZMQ
   handles, `VERL_ZMQ_SOCKET_DIR`/`_PREFIX`).
4. **Telemetry and robustness** — `trainer/ppo/metric_utils.py`
   (`_safe_masked_stats`: an all-false response mask crashed metric computation
   and took the run with it), `trainer/ppo/ray_trainer.py`, `utils/tracking.py`
   (wandb init timeout), `utils/profiler/performance.py`, and small fixes in
   `utils/{model,torch_functional,distributed}.py`, `utils/dataset/rl_dataset.py`.

## How the patches were verified

Patches are easy to get subtly wrong, and a wrong one wastes a reader's day. All
three were checked mechanically:

1. **Pre-image audit.** Every `index <old>..<new>` blob hash in each patch was
   compared against the blob at that path in the pinned upstream tree, so the
   patch demonstrably describes *that* commit.
2. **Applies clean.** Each patch applied with `git apply --check` to a pristine
   `git archive` extraction of its pin — in a fresh `/tmp` workspace, not our
   working tree.
3. **Post-image identity.** After applying, every patched file was compared
   byte-for-byte against the live file in our monorepo — the tree the reported
   experiments were run from. rllm 21/22 and R2E-Gym 7/7 are identical; the one
   difference is intentional: an absolute site path inside an `ImportError`
   message in `rllm/environments/swe/swe.py` was replaced with
   `<workspace>/...`.
4. **No dangling references.** `check_install.py` walks every Python file we ship
   or patch, resolves every `rllm.* / verl.* / r2egym.* / swe_ladder.*` import by
   path against the installed tree, and fails on any that does not exist. This
   is what caught the Leaf tool modules and three `rllm` helpers that existed
   only as untracked files in our monorepo — a patch referring to a file the
   release does not ship is the characteristic way an overlay breaks.

```bash
bash integration/install.sh --workspace "$WS"
python3 integration/check_install.py --workspace "$WS"
```

`check_install.py` is static — it never imports anything, so it runs on a plain
CPU Python with no training stack. It exits non-zero on a dangling reference and
prints two kinds of non-failure note: documented optionals (below) and
references that already dangle in pristine upstream (`swe_sim.py` soft-imports
two `r2egym` modules that do not exist at the pin; it is registered through
`safe_import`, so upstream degrades gracefully and so do we).

## Known gaps

- **`rllm/tip_eval`** is a separate internal eval suite that this release does
  not ship. It is imported lazily behind `agent.enable_tip_eval_snapshot`, which
  is off in every config here; turning it on is the only way to reach the
  import. Nothing in the paper depends on it.
- **The Leaf scaffold** (`--scaffold leaf`) is plumbed end to end but is **not**
  the scaffold any reported number was produced under. Every result in this
  repository comes from `r2egym` or `openhands`, and those two are not
  comparable to each other either — see `docs/EVALUATION.md`. Leaf ships because
  the patches would otherwise have dangling references, not as a recommendation.
- **The verl patch carries our Ascend work.** Groups 2 and 3 above exist because
  the runs were done on 910B NPUs. They are inert on CUDA (`VERL_DEVICE_NAME`
  resolution falls through to the stock path), but they are the least
  interesting part of the diff for a GPU reader, and they are the part most
  likely to conflict if you apply to a newer verl.

## Regenerating a patch

If you change an upstream file and want the patch to reflect it, cut the diff
against the pin rather than editing the patch by hand:

```bash
git -C "$WS/DeepSWE_RL/rllm" diff fb864a7d397b2c287ba6e38152da2228efc98e03 -- \
    $(git -C "$WS/DeepSWE_RL/rllm" diff --name-only fb864a7d397b2c287ba6e38152da2228efc98e03) \
    > integration/patches/rllm-swe-ladder.patch
```

For a file that is **new** (no pre-image at the pin), `git diff` against the
commit will not include it unless it is staged; `git add` it first, or generate
the hunk with `git diff --no-index` and fix up the `a/`/`b/` prefixes. Then
re-verify with steps 2–4 above before committing: the patch is the only copy of
those changes in this repository.
