# End-to-end evaluation on SWE-bench Verified

Five scripts, in the order a run uses them:

```
serve_checkpoint.sh       merge a sharded RL checkpoint -> HF, serve it with vLLM
start_vllm_server.sh      serve an already-merged HF directory (base models, SFT ckpts)
ensure_docker_tunnel.sh   only if dockerd is on another machine
run_swebench_verified_eval.sh   one pass over the benchmark through an R2E-Gym scaffold
run_eval_until_complete.sh      repeat until TARGET clean rows exist
```

The protocol itself — temperature, step caps, which scaffold, how a number is read — is
in `docs/EVALUATION.md`. This file is about the scripts.

## A full run

```bash
# 1. serve the policy (Ascend, 8 cards, TP=8)
ACTOR_DIR=$WS/train_output/<run>/global_step_40/actor \
HF_MODEL_DIR=$WS/models/<run>_gs40 \
SERVED_MODEL_NAME=swe-ladder-gs40 \
BACKEND=ascend TP=8 PORT=8127 RLLM_ROOT=$WS/DeepSWE_RL/rllm \
./serve_checkpoint.sh start

# 2. roll out, recovering from infrastructure failures until 500 clean rows exist
SERVED_MODEL_NAME=swe-ladder-gs40 \
R2EGYM_ROOT=$WS/R2E-Gym \
PARQUET_DATASET=$WS/data/swebench_verified/test-00000-of-00001.parquet \
SCAFFOLD=openhands TARGET=500 MAX_WORKERS=16 \
./run_eval_until_complete.sh
```

Each pass writes `<TRAJ_DIR>/<EXP_NAME>.jsonl` (one row per instance, full trajectory)
and `<EXP_NAME>.summary.json` next to it.

## `serve_checkpoint.sh` — checkpoint to endpoint

`merge | start | stop | status | logs`. `merge` runs verl's
`legacy_model_merger.py` from `RLLM_ROOT` over `ACTOR_DIR` and caches the result in
`HF_MODEL_DIR` (reused if already populated, so `start` is cheap on a rerun); `start`
brings up the OpenAI-compatible server. `SERVED_MODEL_NAME` must match what the eval
runner asks for — it requests `hosted_vllm/<SERVED_MODEL_NAME>` and the runner aborts
up front if the endpoint serves something else, rather than evaluating the wrong model.

Serving a plain HF directory (a base model, an SFT checkpoint) needs no merge: use
`start_vllm_server.sh --backend ascend --model-path <dir>`, which takes the same knobs
as flags and handles Ascend toolkit discovery through `training/lib/ascend_runtime.sh`.

## `run_swebench_verified_eval.sh` — one pass

`SCAFFOLD` picks the tool set the model sees inside the container:

| `SCAFFOLD` | editor tool | submit | source |
|---|---|---|---|
| `openhands` | `str_replace_editor` | `submit` | the upstream SWE-Master recipe |
| `r2egym` | `file_editor` + `search` | `finish` | R2E-Gym's minimal ReAct |
| `sweagent` | `str_replace_editor` | `submit` | SWE-agent's prompt set |

**This is not a cosmetic knob and numbers from different scaffolds are not
comparable.** A model trained to emit `str_replace` scores far below its real ability
under `r2egym`, because `file_editor` has no such command: the call fails inside the
container, the observation comes back empty, and the model retries the same edit until
it runs out of steps. Evaluate a checkpoint under the scaffold it was trained with, and
say which one next to every number. `docs/EVALUATION.md` has the measured size of that
gap.

The script refuses to start on a missing `R2EGYM_ROOT`, an unreachable vLLM endpoint, a
wrong served model name, an unreachable dockerd, or a missing Python dependency —
every one of these otherwise shows up as a plausible-looking low score. It also unsets
`*_proxy`: a stale corporate proxy swallows loopback traffic to vLLM and dockerd, and
the result again looks like a bad model.

`PARQUET_DATASET` is the SWE-bench Verified parquet; it is converted to JSON once and
cached at `DATASET_JSON`. `USE_EXISTING=True` (the default) reuses rows already in the
jsonl and executes only the missing instances.

The per-scaffold `USED_YAML` defaults to a `*_stepout16384` variant where one exists.
This matters: the unsuffixed openhands yaml leaves the per-assistant-turn cap at its
2048-token default, which truncates a long `str_replace` mid-edit — the edit then fails
to apply and the instance scores zero for a reason that has nothing to do with the
policy. Override `USED_YAML` only deliberately.

## `run_eval_until_complete.sh` — the recovery loop

On a shared dockerd a fraction of instances die on container or socket errors rather
than on the model. Those rows land in the jsonl with `output_patch` starting `"Error:"`.
Counting them as failures understates the policy; leaving them in place blocks the rerun,
because `use_existing` treats them as done. So each round drops exactly those rows,
stops once clean rows reach `TARGET`, and otherwise reruns — which `use_existing`
narrows to the missing instances.

Dropping rows is the only destructive thing in `eval/`, and it only ever removes
infrastructure failures, never a real model attempt.

The loop also aborts a round early when many new rows arrive and almost none are clean.
That is the signature of a host-side container killer rather than of a hard benchmark,
and hammering it just burns instances. It then backs off `BACKOFF_SECONDS` and waits up
to `SERVER_WAIT_SECONDS` for a server that vanished mid-run.

`MAX_WORKERS` is a property of the dockerd, not of the eval. We measured silent row loss
(`ConnectionReset`, visible only as missing rows) above ~16 concurrent containers on a
shared daemon; 16 is the default for that reason.

## `ensure_docker_tunnel.sh` — remote dockerd only

Needed only when the daemon that runs the benchmark containers is on another machine.
With a local daemon, point `DOCKER_BASE_URL` at it (`http://127.0.0.1:2375`) and ignore
this script.

`start | status | stop | watch`. It forwards `LOCAL_PORT` to the remote daemon (TCP, or a
unix socket via `REMOTE_DOCKER_SOCKET`) and keeps a watchdog on it. The watchdog is not
optional in practice: a plain `ssh -N -L` dies on a network flap and does not come back,
and a dead tunnel does not announce itself — every container start fails as
`NO_CONTAINER`, so evaluation silently scores 0 and training silently degenerates to
near-empty rollouts. Training against a remote daemon wants this running too.

`SSH_KEY` is a path you provide. No key ships in this repository.

## Reading the output

`<EXP_NAME>.summary.json`, from the 4B RL checkpoint under `openhands`:

```json
{
  "avg_steps": 109.394,
  "exit_reasons": {"abs_step_limit": 147, "agent": 301, "llm_query_error": 52},
  "nonempty_patch": 497,
  "num_success": 176,
  "solve_rate": 0.352,
  "total_trajectories": 500
}
```

`solve_rate` is over all rows, so it is only the headline number when
`infra_error_rows` is 0 — which is what the recovery loop exists to achieve. Report
`unique_instances` alongside it: a solve rate over 361 rows is not comparable to one
over 500, and a mid-run partial is not a result. (The run above predates those two
keys, which the script writes now.)

`exit_reasons` is where the interesting failure analysis lives, and the run above is a
good example of why it belongs next to the score. `agent` means the policy called
submit; `abs_step_limit` (147 of 500) means it ran out of steps with the context budget
binding, not that it was wrong; `llm_query_error` (52) is the serving side failing, and
those rows are scored as failures even though the policy never got to answer. A 35.2%
that carries 147 step-limit exits is a different object from a 35.2% where the policy
submitted every time.

The per-stage capability evaluation (LOC / REPRO / PATCH scored separately, with the
gold artifacts held by the scorer) is a different harness: `stage_capability_eval/`.
