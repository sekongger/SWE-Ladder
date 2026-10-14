# Data

What ships here is small and checked in: the router gold set the validator scores
against, and a handful of staged records so the build pipeline can be run end to end.
The datasets themselves are not shipped — they are tens of GB of derived parquet, and
every byte of them is reproducible from public upstream datasets with
`scripts/data/` (see `scripts/data/README.md`).

```
data/
  stage_router_gold/        119 hand-adjudicated steps + a 23-step demo trajectory
  samples/stages/           3 train + 2 valid instances per stage, real layout
```

## Staged records: the input format

A staged record is one instance at one rung. `student_input` is what the policy may
see; `teacher_pi` is the privileged evidence only the teacher sees:

```json
{
  "task_id": "0b01001001__spectree-64",
  "stage": "causal_localization",
  "student_input": {
    "issue_statement": "[BUG]description for query parameters can not show in swagger ui ...",
    "hints_text": "",
    "codebase_access": {
      "type": "image_repo",
      "repo": "0b01001001/spectree",
      "base_commit": "a091fab020ac26548250c907bae0855273a98778",
      "checkout_repo": "0b01001001/spectree",
      "checkout_ref": "a091fab020ac26548250c907bae0855273a98778"
    }
  },
  "teacher_pi": {
    "gold_changed_non_test_files": ["setup.py", "spectree/utils.py"],
    "gold_hunk_ranges": [{"file": "setup.py", "hunk_index": 0, "old_start": 14, "old_count": 7, "old_end": 20}],
    "old_buggy_lines": [{"file": "setup.py", "hunks": [{"removed": ["..."]}]}]
  }
}
```

The privilege ladder is monotone — L ⊂ R ⊂ P — and lives entirely in `teacher_pi`:

| stage | `teacher_pi` keys | what the teacher is told |
|---|---|---|
| `causal_localization` | `gold_changed_non_test_files`, `gold_hunk_ranges`, `old_buggy_lines` | *where* the bug is |
| `localized_issue_reproduction` | `fail_to_pass_tests` | *which tests* must flip |
| `patch_generation` | `golden_patch` | the reference fix |

`student_input` also grows along the ladder (`localized_buggy_files` at REPRO,
`+ reproduction_tests` at PATCH). Those extra fields belong to the **staged prompt**
formulation, where each rung is a separate task handed the previous rung's output. All
the runs in this repository use the **end-to-end prompt** (`OPD_END_TO_END_PROMPT=1`),
whose student template interpolates `{issue_statement}` and nothing else — which is
exactly invariant I2: the stage changes the teacher's evidence and the token routing,
never the student's text. The fields stay in the records so the staged formulation
remains buildable.

## The dataset family the runs used

Built from four public sources, then intersected so every rung trains on the same
instances (`scripts/data/intersect_stages.py`, numbers from its `manifest.json`):

| split | per-stage input rows | intersection | dropped (LOC / REPRO / PATCH) |
|---|---|---|---|
| train | 1258 / 1150 / 1158 | **1149** | 109 / 1 / 9 |
| valid | 38 / 32 / 34 | **32** | 6 / 0 / 2 |

Train composition: SWE-smith 500, SWE-rebench 431, R2E-Gym 155, SWE-Gym 63. Zero rows
missing a Docker image, zero missing `FAIL_TO_PASS` — both stages that execute code can
run every instance they are given. The valid split (32 instances) is a training monitor,
not a benchmark; scores in the paper come from the protocols in `docs/EVALUATION.md`.

Upstream sources, all public: [SWE-bench/SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith),
[nebius/SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench),
[R2E-Gym/R2E-Gym-Subset](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset),
[SWE-Gym/SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym). The
`--*-data-root` flags take a local clone of each (the builder reads their parquet
files directly; `--swe-gym-data-root` also accepts a `SumanthRH/SWE-Gym-Subset`
layout or a `canonical/tasks.jsonl`).

Two names in older config files are legacy and need no action:

* `..._enriched` — a repair pass that back-filled per-stage privilege into an early
  family whose LOC and REPRO rows had empty `teacher_pi`. A build from staged JSONL
  carries per-stage privilege by construction (0 of 41,557 LOC train rows are missing
  `gold_changed_non_test_files`), so the suffix only records which parquet a run read.
* `swerebench500_r2e200_swegym100_swesmith500` — the sampling *request*, not the result;
  the table above is what the intersection actually yields.

## `stage_router_gold/` — the router's gold set

Ten complete agent trajectories, every step hand-adjudicated into LOCATE / REPRODUCE /
PATCH / VERIFY: **119 steps** (38 LOCATE, 8 REPRODUCE, 51 PATCH, 22 VERIFY), from a
leg-3 patch-only 4B run at global step 61–63. VERIFY maps onto `patch_generation` when
scored — a step that re-runs a test after a successful edit is closing out the patch,
and the gold patch is the right evidence for it.

```bash
python3 analysis/stage_ladder/validate_router_gold.py            # rule v2, the trained config
python3 analysis/stage_ladder/validate_router_gold.py --version v1 --show-errors
python3 analysis/stage_ladder/validate_router_gold.py --expect 98 --demo
```

`demo_trajectory_labeled.json` is a separate 23-step trajectory carrying the router's
own label per step as recorded at adjudication time; `--demo` recomputes them, which is
a regression test on the released classifier. `extract_gold_from_dumps.py` is provenance
only — it reads the multi-GB rollout dumps that produced `gold_119.jsonl` and that this
repository does not ship.

## `samples/stages/` — runnable examples

Three train and two valid instances per stage in the real directory layout, so
`--stages-root data/samples/stages` works directly. See the smoke test in
`scripts/data/README.md`. The instances are public repositories
(`0b01001001/spectree`, `12rambau/sepal_ui`, `David-Wobrock/sqlvalidator`,
`CSCfi/swift-browser-ui`); their issue text and gold patches come from the upstream
datasets above.
