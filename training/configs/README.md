# Training configs

One YAML per run. Each file is the *whole* run definition: the launcher
(`../run_npu_train_from_yaml.sh` / `../run_gpu_train_from_yaml.sh`) reads it,
exports every `env:` entry, and execs the entrypoint named in `runtime:`. There
are no hidden defaults in a shell wrapper — if a knob matters, it is in the
file, and the entrypoint echoes the resolved values at startup so a run's
identity is provable from its log rather than from its file name.

## Render, then launch

Machine-specific paths are `{{PLACEHOLDER}}` slots, so the configs are safe to
read and diff but cannot be launched as-is:

```bash
cp site.env.example site.env && $EDITOR site.env        # once per machine
python3 ../../scripts/tools/render_config.py ladder_4b_openhands_leg1_loc.yaml -o /tmp/run.yaml
bash ../run_npu_train_from_yaml.sh /tmp/run.yaml
```

`render_config.py` picks up `site.env` from this directory automatically.
`--print-placeholders` lists what one config needs; `--set KEY=VALUE` overrides a
single slot without editing the file, which is the intended way to point a
resuming leg at the checkpoint it should continue from:

```bash
python3 ../../scripts/tools/render_config.py ladder_4b_openhands_leg2_repro.yaml \
    --set RESUME_CHECKPOINT=/.../train_output/<leg1-run>/global_step_30 -o /tmp/run.yaml
```

Why a renderer instead of `${VAR}`: the launcher exports each `env:` value
verbatim (`export KEY=<shell-quoted value>`), so a `${VAR}` inside a value would
reach the trainer as six literal characters, not an expansion.

`OPSD_STAGE_API_KEY` is deliberately absent from the configs that use the
semantic router — export it in the launching shell. Setting it to an empty
string in YAML would *override* the environment and silently drop the router
back to its rule-based fallback.

## The runs

`steps` is `TRAINER_TOTAL_TRAINING_STEPS`; "—" means unset, i.e. one epoch of the
dataset. `from` is the checkpoint the run resumes (`TRAINER_STAGE_GLOBAL_STEP_OFFSET`
in parentheses, which is what makes a leg's logged `global_step` absolute).

### Stage ladder (4B, openhands scaffold, 8x NPU)

| config | privileged stage | steps | from |
| --- | --- | --- | --- |
| `ladder_4b_openhands_leg1_loc.yaml` | `causal_localization` | 30 | base |
| `ladder_4b_openhands_leg1_loc_resume_gs20.yaml` | `causal_localization` | 30 | leg 1 `global_step_20` (0) |
| `ladder_4b_openhands_leg2_repro.yaml` | `localized_issue_reproduction` | 20 | leg 1 `global_step_30` (30) |
| `ladder_4b_openhands_leg3_patch.yaml` | `patch_generation` | 20 | leg 2 `global_step_50` (50) |

Three legs, 30 → 20 → 20 steps, so checkpoints land at absolute `global_step`
30 / 50 / 70. The trainer enforces the budget itself (`fit_agent` exits once
`global_steps > offset + total_training_steps`); `save_freq=10` is set by the
entrypoint so every leg boundary is a checkpoint, and the LR schedule is
constant, so a short leg does not reshape it.

### Ladder variants

| config | what it changes | steps | from |
| --- | --- | --- | --- |
| `ladder_4b_openhands_branch_patch_after_loc.yaml` | skips the REPRODUCE rung: LOC → PATCH | 20 | leg 1 `global_step_30` (30) |
| `ladder_4b_openhands_leg1_loc_semantic_router.yaml` | LLM router instead of the rule router | 30 | base |
| `ablation_ladder_reverse_order.yaml` | runs the rungs backwards (LOC after REPRO) | 30 | repro leg `global_step_40` (40) |
| `ladder_4b_sweagent_leg1_loc.yaml` | sweagent scaffold, semantic router, `n=8 x bs=8` | 30 | base |
| `ladder_4b_sweagent_leg2_repro.yaml` | sweagent scaffold, semantic router | 20 | leg 1 `global_step_30` (30) |
| `ladder_8b_cuda_leg1_loc.yaml` | Qwen3-8B on 4x CUDA | — | base |

### Baselines

| config | method | steps |
| --- | --- | --- |
| `baseline_grpo_4b_openhands.yaml` | plain GRPO, same scaffold and data as the ladder | 70 |
| `baseline_grpo_4b_sweagent.yaml` | plain GRPO, sweagent scaffold, `n=8 x bs=8` | 300 |
| `baseline_grpo_8b_cuda.yaml` | plain GRPO, Qwen3-8B, 4x CUDA | — |
| `baseline_arpo_4b.yaml` | ARPO entropy-branching rollouts (arXiv:2507.19849) | — |
| `baseline_sdar_4b.yaml` | gated self-distillation (`OPSD_SDAR_GATE=1`) | — |

`baseline_grpo_4b_openhands.yaml` is the control for the ladder: 70 steps is the
ladder's total, so the two spend the same gradient budget on the same data and
scaffold and differ only in where privilege is applied.

### Ablations

| config | question it answers |
| --- | --- |
| `ablation_privilege_patch_only.yaml` | is privilege on PATCH alone enough? |
| `ablation_no_router_whole_answer.yaml` | does per-step routing beat distilling every token? |
| `ablation_no_router_all_stages.yaml` | privilege on all three stages at once, no routing |
| `ablation_router_learned_classifier.yaml` | learned step classifier instead of the rule router |
| `ablation_process_graph_privilege.yaml` | process graph as the privileged signal |
| `ablation_pure_opd_no_policy_gradient.yaml` | distillation only, no policy gradient (`n=1 x bs=64`) |
| `ablation_external_served_teacher.yaml` | teacher served out-of-process (one echo-logprobs call per sequence) |
| `ablation_base_student_sft_teacher.yaml` | base student, SFT checkpoint as the in-process teacher |

### Smoke

`smoke_stage_routing.yaml` — `n=2 x bs=2`, enough to prove the router fires, the
privileged branch builds its mask, and checkpoints write. Run this first on a
new machine; it reaches the first `update_actor` in minutes rather than hours.

## Scaffold is not a free variable

`SWE_SCAFFOLD` decides which tools exist inside the container
(`openhands` → `str_replace_editor` + `execute_bash` + `submit`; `r2egym` →
`file_editor` + `search` + `execute_bash` + `finish`; `sweagent` →
`str_replace_editor` with SWE-agent prompts). A model trained against one
scaffold emits tool calls the others do not implement, and the harness reports
the resulting failure as an empty observation rather than an error. **Numbers
from different scaffolds are not comparable**, and a checkpoint must be
evaluated with the scaffold it was trained on. See `docs/TRAINING.md`.

## Datasets

`DATA_ROOT` holds the verl parquet datasets; `site.env.example` lists which
config wants which subdirectory. The `_enriched` variant differs only in
`teacher_pi`: it carries localization and reproduction evidence per task, not
just the gold patch, which is what the LOC and REPRO rungs need in order to have
anything privileged to condition on. Running an `_enriched` config against the
plain dataset trains, but the teacher sees an empty privileged context on two of
the three rungs.

## Source names

These files were renamed on release. The mapping to the internal run names, for
anyone matching them against logs or W&B history:

| released | internal |
| --- | --- |
| `ladder_4b_openhands_leg1_loc.yaml` | `opsd_e2e_stage_routing_loc_only_rule_4b_npu8_openhands.yaml` |
| `ladder_4b_openhands_leg1_loc_resume_gs20.yaml` | `..._loc_only_rule_4b_npu8_openhands_resume_gs20.yaml` |
| `ladder_4b_openhands_leg2_repro.yaml` | `opsd_e2e_stage_routing_repro_only_rule_4b_npu8_openhands.yaml` |
| `ladder_4b_openhands_leg3_patch.yaml` | `opsd_e2e_stage_routing_patch_only_rule_4b_npu8_openhands.yaml` |
| `ladder_4b_openhands_branch_patch_after_loc.yaml` | `..._patch_only_from_loc_gs30_rule_4b_npu8_openhands.yaml` |
| `ladder_4b_openhands_leg1_loc_semantic_router.yaml` | `..._loc_only_semantic_openhands_4b_npu8.yaml` |
| `ladder_4b_sweagent_leg1_loc.yaml` | `opsd_grpo_staged_sweagent_lite2000_curriculum_loc_from_base.yaml` |
| `ladder_4b_sweagent_leg2_repro.yaml` | `opsd_grpo_staged_sweagent_lite2000_curriculum_repro_from_loc30.yaml` |
| `ladder_8b_cuda_leg1_loc.yaml` | `..._enriched_loc_only_rule_8b_gpu4_openhands_cxd.yaml` |
| `baseline_grpo_4b_openhands.yaml` | `pure_grpo_baseline_4b_npu8_openhands_cxd.yaml` |
| `baseline_grpo_4b_sweagent.yaml` | `grpo_baseline_sweagent_r2e_gym_lite_first2000_base_toolfix_20261002.yaml` |
| `baseline_grpo_8b_cuda.yaml` | `pure_grpo_end_to_end_qwen3_8b_gpu4_openhands_cxd.yaml` |
| `baseline_arpo_4b.yaml` | `arpo_4b_npu8_r2egym_cxd.yaml` |
| `baseline_sdar_4b.yaml` | `sdar_baseline_e2e_from_base_colocated_8npu.yaml` |
| `ablation_privilege_patch_only.yaml` | `opsd_grpo_end_to_end_stage_routing_enriched_colocated_8npu.yaml` |
| `ablation_no_router_whole_answer.yaml` | `opsd_grpo_end_to_end_NO_routing_wholeanswer_colocated_8npu.yaml` |
| `ablation_no_router_all_stages.yaml` | `..._no_router_combined_privilege_r2egym_from_base_curriculum.yaml` |
| `ablation_router_learned_classifier.yaml` | `..._enriched_full_stage_clf_from_base_colocated_8npu.yaml` |
| `ablation_ladder_reverse_order.yaml` | `..._enriched_loc_reverse_from_reproleg_gs40.yaml` |
| `ablation_process_graph_privilege.yaml` | `opsd_grpo_e2e_stage_routing_pgraph_from_base_colocated_8npu.yaml` |
| `ablation_pure_opd_no_policy_gradient.yaml` | `..._patch_priv_only_pure_opd_n1_b64_colocated_8npu.yaml` |
| `ablation_external_served_teacher.yaml` | `opsd_grpo_e2e_external_teacher_local8npu.yaml` |
| `ablation_base_student_sft_teacher.yaml` | `..._base_student_sft_teacher_local8npu.yaml` |
| `smoke_stage_routing.yaml` | `opsd_stage_routing_smoke.yaml` |
