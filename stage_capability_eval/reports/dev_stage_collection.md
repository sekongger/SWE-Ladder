# REPRO/PATCH development collection

The public stage packages have been materialized from the frozen 500-task
manifest and the shared LOC oracle:

- `repro_given_loc.jsonl`: 500 `REPRO_GIVEN_LOC` tasks
- `patch_given_loc.jsonl`: 500 `PATCH_GIVEN_LOC` tasks
- `repro_metadata.jsonl`: 500 private REPRO verifier records
- `patch_verifier_metadata.jsonl`: 500 private PATCH verifier records

The public records contain no source patch, test patch, F2P/P2P test names, or
gold patch hash. The private records contain only evaluator-side references and
must stay outside model-visible mounts.

The injected evaluators pass development fixtures for:

- valid reproduction: buggy test failure plus fixed test pass;
- invalid, timeout, and infrastructure outcomes;
- patch apply failure, F2P failure, P2P regression, and conditional resolution;
- dry-run execution without invoking a runner.

The remote Docker tunnel has now been validated against
`astropy__astropy-12907`: the buggy image fails both F2P tests, the fixed image
passes both F2P tests, and all 13 P2P tests pass. The 25-task development
preflight uses the pinned SWE-bench 4.1.0 parser, reuses dependencies already
present in each image, runs the repository suite once per phase, and appends
fsynced private records under `evaluator_private/`. The audited 25-task run has
20 valid old/fixed references and 5 pending records: three Django records have
reference test errors in the buggy image, and two records hit the five-minute
execution timeout. There are no reference-expectation mismatches. Pending
records are excluded from any denominator until the environment policy is
resolved, so this report does not approve the full 500-task gate.

No model-generated test or source patch has been collected, and no formal model
score is claimed by this report.
