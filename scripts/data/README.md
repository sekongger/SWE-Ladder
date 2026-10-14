# Building the training data

Four steps turn staged JSONL records into the parquet files a run reads. Only the
first is dataset-specific; the rest are mechanical.

```
stages/<stage>/{train,valid}.jsonl          # staged records (see ../../data/README.md)
      │
      │  build_staged_verl_dataset.py       # + per-source Docker image metadata
      ▼
$DATA_ROOT/<family>/<stage>/{train,valid}_verl.parquet
      │
      │  intersect_stages.py                # same instances on every rung
      ▼
$DATA_ROOT/<family>_task_intersection/...   # what training/configs/*.yaml point at
```

## 1. `build_staged_verl_dataset.py` — JSONL → verl parquet

One output row per (instance, stage). `extra_info` carries everything the rollout and
the teacher need: the Docker image to run, the student messages, that stage's
privileged evidence (`teacher_pi`) and the pre-rendered `teacher_messages`.

```bash
PYTHONPATH="$WS/DeepSWE_RL/rllm:$WS/R2E-Gym/src" \
python3 scripts/data/build_staged_verl_dataset.py \
    --stages-root stages --stage all --splits train,valid \
    --output-root "$DATA_ROOT/my_family" \
    --swesmith-data-root   "$WS/SWE-smith" \
    --swerebench-data-root "$WS/SWE-rebench" \
    --r2e-gym-data-root    "$WS/R2E-Gym-Subset" \
    --swe-gym-data-root    "$WS/SWE-Gym" \
    --drop-invalid-samples --require-execution-metadata-for-patch
```

The `PYTHONPATH` is not optional: the student system prompt is baked into every row
and is imported from `rllm.agents.system_prompts`, so that the prompt in the dataset
and the prompt the agent uses cannot drift apart. The script fails loudly rather than
falling back to a copy.

**Docker images.** Each upstream dataset names its images differently, so the image is
matched per source by `instance_id` through the `--*-data-root` flags. A row with no
match keeps `--generic-image`, which can run LOC but not REPRO or PATCH — pass
`--require-execution-metadata-for-patch` to drop those rows from the executing stages
instead of silently training on an unrunnable task. `--swe-gym-docker-host` /
`--swe-gym-docker-image-list` let SWE-Gym images be inferred from what a daemon
actually has, which is how the gaps in that source were closed.

Useful narrowing flags: `--repo-filter`, `--source-dataset-filter`, `--task-id-file`,
`--limit`, `--max-student-prompt-chars`, `--max-teacher-prompt-chars`.

## 2. `intersect_stages.py` — same instances on every rung

The builder converts stages independently, so they can end up with slightly different
instance sets. The ladder needs the same population on every rung, otherwise a
leg-to-leg difference could be the task mix rather than the stage:

```bash
python3 scripts/data/intersect_stages.py \
    --input-root  "$DATA_ROOT/my_family" \
    --output-root "$DATA_ROOT/my_family_task_intersection"
```

It writes `manifest.json` with the input rows, the intersection size, what each stage
dropped, the per-source composition and a sha256 per file. The numbers quoted in
`../../data/README.md` are that file's.

## 3. `build_staged_eval_dataset.py` — the per-stage eval split

Builds `<stage>/<split>_eval.jsonl` for `stage_capability_eval/`. An eval record holds
the student messages and a `verifier` block (gold patch, test patch, fail-to-pass ids)
that the **scorer** sees — never the model.

```bash
python3 scripts/data/build_staged_eval_dataset.py \
    --input-root stages/causal_localization \
    --output-root eval_data --stage causal_localization --splits valid
```

## 4. `run_issue_reproduction_verifier.py` — debug one REPRO score

Runs the fail-then-pass check by hand in a container the model already acted in. Use
it when a REPRODUCE reward looks wrong; it is the same code path as the in-training
reward.

```bash
python3 scripts/data/run_issue_reproduction_verifier.py \
    --eval-jsonl eval_data/localized_issue_reproduction/valid_eval.jsonl \
    --container <container-after-rollout> --output /tmp/verifier.json
```

## Smoke test with the shipped samples

`data/samples/stages/` holds three train and two valid instances per stage, in the real
layout, so the pipeline can be exercised end to end without the full dataset:

```bash
PYTHONPATH="$WS/DeepSWE_RL/rllm:$WS/R2E-Gym/src" \
python3 scripts/data/build_staged_verl_dataset.py \
    --stages-root data/samples/stages --stage all --splits train,valid \
    --output-root /tmp/sample_family
python3 scripts/data/intersect_stages.py \
    --input-root /tmp/sample_family --output-root /tmp/sample_family_ti
```

Without `--*-data-root` metadata every row falls back to the generic image, so the
result is good for inspecting row shape and prompts, not for a training run.

## Process graphs (one ablation only)

`process_graph/` builds the P2T-style per-instance process graphs used by
`training/configs/ablation_process_graph_privilege.yaml`. It is not part of the main
pipeline; see `process_graph/README.md`.
