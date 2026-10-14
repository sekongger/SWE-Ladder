# Stage capability evaluation

End-to-end resolve rate says whether a model fixed the bug, not *which rung it
fell off*. This package scores the three rungs of the ladder separately on the
500 SWE-bench Verified tasks, by handing the model the earlier rungs' answers:

| stage | model is given | model must produce | scored by |
| --- | --- | --- | --- |
| `LOC` | issue + buggy repo | files and functions to change | `loc_scorer.py` against the private LOC labels |
| `REPRO_GIVEN_LOC` | issue + buggy repo + oracle LOC | a test that reproduces the issue | `repro.py`: must fail on buggy, pass on fixed |
| `PATCH_GIVEN_LOC` | issue + buggy repo + oracle LOC | a source patch | `patch.py`: F2P must pass, P2P must not regress |

Giving later stages the oracle localization is the point: it removes the
upstream failure as a confound, so a drop between rungs is attributable.

## The public / private split

Everything here is public **except** `evaluator_private/`, which is generated
locally and is not part of this repository. It holds the gold patches, test
patches, F2P/P2P test names, parsed base-source snapshots, LOC labels, and
reference execution records.

Two things in the public tree are nevertheless answer-bearing, and the split
only holds if you respect them:

- `oracle_loc/oracle_loc.jsonl`, and the `oracle_loc` field in
  `manifests/{repro,patch}_given_loc.jsonl`, **are** the LOC answer. They are
  inputs to the REPRO and PATCH stages and must never reach a model being
  scored on LOC.
- `manifests/all_swebench_verified.jsonl` carries `source_patch_files` and
  `non_test_source_files`, derived from the gold patch. They are collection
  bookkeeping, not prompt material.

Neither leaks the fix itself: `oracle_loc` records line spans
(`old_context: [{start, count}]`) and a `patch_sha256`, never patch text.

## Building the task packages

```bash
# 1. Freeze the manifest from the bootstrap dataset, deriving the private records.
python stage_capability_eval/collect_manifest.py \
  --source output/rucaibox_original_scaffold_swebench_verified_dataset.json

# 2. Map each gold patch to files/functions -> evaluator_private/loc_labels.jsonl.
python stage_capability_eval/loc_labels.py

# 3. 25-task development sample (reports/dev_sample.md records its SHA-256).
python stage_capability_eval/build_dev_sample.py --size 25

# 4. Materialize the public REPRO/PATCH packages + private verifier metadata.
python stage_capability_eval/collect_stage_packages.py

# 5. Static checks: schema, no-leak assertions, scorer behaviour.
python stage_capability_eval/preflight.py --limit 25 --dry-run \
  --out /tmp/stage_capability_preflight.jsonl
python -m unittest discover -s stage_capability_eval/tests -p 'test*.py'
```

`manifests/stage_package_hashes.json` pins the SHA-256 of all four stage
packages, so a rebuilt package can be proven identical to the evaluated one.

## Reference preflight (Docker)

Before scoring any model, the harness verifies each task's environment actually
discriminates: the reference tests must **fail** on the buggy image and **pass**
on the fixed one. A task whose reference run is broken cannot score a model.

```bash
python stage_capability_eval/docker_preflight.py \
  --host tcp://127.0.0.1:23750 \
  --manifest stage_capability_eval/manifests/dev_sample.jsonl \
  --workers 2 \
  --out stage_capability_eval/evaluator_private/reference_dev25.jsonl
```

It talks to the Docker HTTP API directly, so no local Docker CLI or daemon
config is needed. For a daemon on another machine, open a tunnel first with
`eval/ensure_docker_tunnel.sh start` and pass the port it reports.

The 25-task audited run (`reports/dev_stage_collection.md`) has 20 valid
references and 5 pending: three Django tasks error in the buggy image, two hit
the 5-minute timeout. Pending tasks are excluded from the denominator rather
than counted as failures — a broken reference is an infrastructure outcome, and
`status_values` in `protocol/protocol.yaml` keeps those distinct from a model
failure so they cannot silently deflate a score.

## Scoring model artifacts

`execution.py` takes model-generated REPRO tests and PATCH submissions and runs
them through injected runners (`r2egym_docker_runner.py` for the real path, test
doubles elsewhere), returning one of the `status_values` per task. The model
side of collection lives in `r2egym_collect.py`.

This package deliberately claims no model score on its own: without collected
artifacts and a clean reference preflight there is nothing to divide by.
