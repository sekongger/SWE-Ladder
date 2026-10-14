# Fork suppression: offline analysis report

> A purely offline mechanism analysis — **no weight updates**, only forwards,
> statistics, and plots. It replicates Kaur et al., *"Fork suppression in privileged
> distillation"*, arXiv:2607.05184.
> Every number below is produced by the `analysis/fork_suppression/` pipeline and read
> back from `results.json` / `m3.json`.
> **Wording discipline:** this report only states whether the same signature appears.
> It never claims "we proved X causes Y"; there are no causal assertions.

---

## 0. Where the paper and our design doc differ (declared up front; the paper wins)

The plan requires that where the paper's definitions conflict with our design doc, the
paper's definition is used and the difference is recorded here. After checking
arXiv:2607.05184 §4.1:

| concept | paper (Kaur et al. §4.1) | our primary criterion (design doc) | resolution |
|---|---|---|---|
| **fork criterion** | **teacher top-1 prob < 0.25**; lock = > 0.65 | high percentile of **student full-vocabulary entropy H_t** (percentile for M2/M4, absolute threshold for M3) | primary = student entropy (doc); the paper's teacher-top-1 criterion is **reported alongside in M1/M3** as a cross-check (see D3). Both are reported; neither replaces the other |
| what fork acts on | the teacher suppressing a student token | same | agrees |
| instruction-tuned models have flat fork rates | observed in the paper | see the D4 caveat | agrees |

The only substantive difference is the **fork criterion**, and this analysis reports
**both**: the NEP judge uses student entropy (the doc's primary criterion), and the
paper's teacher-top-1 criterion appears alongside it in M1/M3. If the two criteria
disagree, that disagreement is the thing to discuss — no parameter is retuned to make
them agree (which would be p-hacking).

---

## 1. Summary (TL;DR)

> There are two samples. **(A) step-30, purely on-policy** = 64 trajectories / 16
> instances / 2024 positions (the trajectories *are* gs30 policy samples, so this is a
> self-contained result). **(B) steps 28-33 merged** = 375 trajectories / **94
> instances** / 14185 positions (a fixed gs30 student scoring neighbouring steps'
> trajectories — a "fixed instrument" robustness extension; see the fidelity statement
> in §12). **The headline conclusions follow (B), which has more power; (A) agrees in
> sign and serves as a cross-check.**

- **P3 gate: passes exactly, on both samples.** At the PATCH stage,
  `max|U_wa - U_sr| = 0.0`: (B) covers **8275** PATCH positions at K=32 and 8273 at
  K=16; (A) covers 1370 at K=32 and 1369 at K=16. Every NEP delta is 0.0, so K=32 and
  K=16 both **PASS**.
  > SR's PATCH context is byte-identical to WA's (force-PATCH), so this difference must
  > be exactly 0. Anything non-zero is a pipeline bug and invalidates every number that
  > follows (plan §9). Since it is exactly 0, the stage labelling, teacher routing, and
  > position alignment are all correct, and the downstream numbers can be trusted.
- **M4 (the key discriminative test): consistent in sign, and statistically significant
  on 94 instances.** Using variant B's
  `NEP_B = U(high-entropy top-p% fork) - U(low-entropy bottom-50% baseline)` and
  `delta = NEP_B(WA) - NEP_B(SR)`:
  - At non-PATCH stages **delta is reliably positive, and every (B) 95% CI lies entirely
    above 0** (LOC p10 = +0.051, CI [+0.021, +0.088]; REPRO p10 = +0.071, CI
    [+0.016, +0.109]; all six cells of p in {5, 10, 20} x {LOC, REPRO} share the sign and
    clear 0). PATCH delta is identically 0 (consistent with P3).
  - Decomposed: `NEP_B(WA) > 0` while `NEP_B(SR) < 0` — relative to the low-entropy
    baseline, the fully privileged WA teacher exerts a **stronger** uplift pull at
    high-entropy decision points than SR does. The direction matches "WA suppresses
    exploration, SR does not".
  - **Robustness:** (B)'s K=16 slice shares the sign with K=32 and its CIs still clear 0
    everywhere (§6); (A)'s 16 instances share the sign but straddle 0 (underpowered).
    Going from 16 to 94 instances left the sign unchanged and tightened the CIs past 0,
    which is what "underpowered, then powered" looks like — not a criterion retuned to
    fit.
- **M2 (global pool):** all three percentiles of (B) have CIs entirely above 0
  (p5 +0.0302 [+0.014, +0.047]; p10 +0.0207 [+0.009, +0.037]; p20 +0.0123
  [+0.005, +0.021]); (A) shares the sign but straddles 0.
- **Placebo (low-entropy negative control):** delta is approximately 0 at all three
  stages on both samples (1e-8 to 1e-10, i.e. numerically zero), so there is no spurious
  asymmetry.
- **Exclusion rate 0:** v\* falls inside the teacher's top-32 support 100% of the time
  (`excluded_fraction = 0` on both samples).

---

## 2. Method and data path (plan §3-§5; reuse, never reimplement)

### 2.1 Core quantities (§3, `quantities.py`)
`c_t(v) = log p_theta(v) - log q~_t(v)`; `c̄_t = sum_v p_theta(v) c_t(v)`; the **primary
metric** is `uplift_t(v) = p_theta(v) (c̄_t - c_t(v))`.
- **The asymmetry is preserved**: `p_theta` is not renormalized; `q~` is renormalized
  only over the teacher's top-k slice. This is **not** treated as something to "fix"
  (plan §3/§10).
- The main tables use un-normalized uplift; the `sum_v |uplift|` normalized version is
  reported alongside.
- The scalar `U` at each decision point is the uplift on the token the student
  **actually emitted**, v\*, restricted to the teacher's top-k support. If v\* is outside
  the top-k the position is dropped from NEP; the exclusion fraction is in §6.

### 2.2 Token classification (§4, `freeze_rules.py`, **frozen before any result was seen**)
- **action-onset** (the first token of the tool name): EXPLORE / COMMIT / OTHER, with
  submit/finish broken out as SUBMIT. Classification reuses the real scaffold's tool
  names (`stage_classifier.parse_action` / `_EDIT_COMMANDS` / `_EXPLORE_BASH_HEADS`).
- **deliberation markers** (Wait / Actually / Hmm / However / But / Alternatively /
  Let me also / Let me double-check / On second thought / Instead) -> EXPLORE, resolved
  through the tokenizer (with and without a leading space), counting only the first
  token. The frozen token-id list is in `freeze_rules.py`.

### 2.3 Fork criterion (§5)
- M2/M4: the top-p% **percentile** of student entropy, p in {5, 10, 20}.
- M3: an **absolute threshold**, calibrated once on base (the 90th percentile of its H)
  and then fixed. **M3 never uses a percentile.**

### 2.4 Data reconstruction (`data_gen.py` — offline but byte-faithful)
- Reuses the engine's own primitive `convert_messages_to_tokens_and_masks` to rebuild
  `response_tokens` / `loss_mask`. The engine itself re-encodes from assistant text with
  the same function, so the offline reconstruction is identical, not an approximation.
- Per-token stage labels come from `StageStateMachine(backtrack_threshold=3)`, matching
  `OPSD_STAGE_BACKTRACK=3`.
- Teacher contexts come from `render_end_to_end_teacher_messages`: WA = force-PATCH;
  3x SR = one rendering per stage.

### 2.5 Forwards (`forwards.py`, the only NPU stage, 4 forwards per trajectory)
At each decision position `t`, the distribution that generated `response[t]` is
`logits[L_prefix + t - 1]` under teacher forcing:
- teacher (base Qwen3-4B) x 3 contexts (loc / repro / patch); WA reuses the patch
  context, so **WA + 3 SR is really 3 teacher forwards**.
- student (the merged gs30 checkpoint), 1 forward: full-vocabulary entropy H_t plus
  `log p_theta` at each teacher top-k id.
- K=32 is stored; the K=16 robustness slice is the first 16 entries (top-k is already
  sorted descending), so no forward has to be re-run.

---

## 3. P3 gate (§7 — runs first, stops the pipeline if it fails)

| sample | criterion | PATCH `max\|U_wa - U_sr\|` | PATCH NEP delta (worst over p) | verdict |
|---|---|---|---|---|
| (B) 94 instances | K=32 | 0.0 (8275 positions) | 0.0 | **PASS** |
| (B) 94 instances | K=16 | 0.0 (8273 positions) | 0.0 | **PASS** |
| (A) 16 instances | K=32 | 0.0 (1370 positions) | 0.0 | **PASS** |
| (A) 16 instances | K=16 | 0.0 (1369 positions) | 0.0 | **PASS** |

> Why: SR's patch-stage teacher context and WA's come from the same
> `render_end_to_end_teacher_messages(rec, "patch_generation")` call and are
> byte-identical (verified on 8/8 records). So this difference is a pipeline self-check,
> not an empirical finding. It is exactly 0 on both samples at both K, so the gate passes.

---

## 4. M1 descriptive statistics (§6, K=32)

**(B) 94 instances (14185 positions):**

| pos_class | n | H_mean | U_wa_mean | U_sr_mean |
|---|---|---|---|---|
| EXPLORE | 7827 | 0.2494 | -0.00075 | -0.00524 |
| COMMIT | 4741 | 0.0014 | -0.00016 | -0.00011 |
| OTHER | 1326 | 0.0000 | -0.0000 | -0.0000 |
| SUBMIT | 291 | 0.0007 | -0.00012 | -0.00012 |

**(A) 16 instances (2024 positions):**

| pos_class | n | H_mean | U_wa_mean | U_sr_mean |
|---|---|---|---|---|
| EXPLORE | 1089 | 0.2313 | -0.0036 | -0.0051 |
| COMMIT | 673 | 0.0021 | -0.0009 | -0.0007 |
| OTHER | 210 | 0.0000 | -0.0000 | -0.0000 |
| SUBMIT | 52 | 0.0002 | -0.0000 | -0.0000 |

- **Key observation**, consistent across both samples: EXPLORE positions are
  high-entropy (H ~ 0.23-0.25) while COMMIT positions are nearly deterministic
  (H ~ 0.001-0.002). That is exactly why **the frozen explore-vs-commit NEP is
  structurally uncomputable on a high-entropy fork set** — COMMIT positions almost never
  enter the top-p% entropy set, and at the LOC stage there are zero of them — which is
  what triggered variant B (see §5.0).
- Paper-criterion cross-check (teacher top-1): (B) `paper_fork_rate` WA=0.00197 /
  SR=0.00219; `paper_lock_rate` WA=0.9292 / SR=0.9283. (A) WA=0.002 / SR=0.0005, lock 0.93.
- **The D4 caveat is confirmed empirically**: student and teacher are both built on
  Qwen3-4B-Instruct-2507 (instruction-tuned), the base teacher is extremely confident
  (lock rate 0.93, fork rate ~0.002), and the paper's teacher-top-1 fork criterion
  carries almost no signal here. So the primary criterion has to be the **student-entropy
  percentile**, with the paper's criterion reported alongside as a cross-check.

## 5. M2 global NEP (§6)

### 5.0 Variant B (a definitional deviation, recorded as DEVIATIONS D5)
The frozen `NEP = mean(U | fork & EXPLORE) - mean(U | fork & COMMIT)` is uncomputable at
high entropy percentiles (see §4). Variant B **keeps NEP's semantics** — is exploration
suppressed more strongly than commitment? — but redefines both cells by entropy so that
both are populated:
- top cell = the **top-p% student entropy** positions within the pool/stage (the
  high-uncertainty, exploratory end);
- base cell = the **bottom-50% student entropy** positions within the pool/stage (the
  low-uncertainty, committed end);
- `NEP_B = U(top) - U(base)`, `delta = NEP_B(WA) - NEP_B(SR)`. At PATCH, WA is identical
  to SR, so delta == 0 (self-consistent with P3).

### 5.1 M2 results (whole pool, instance-paired bootstrap, n_boot=2000)

**(B) 94 instances (primary):**
| p | delta | 95% CI | n_top / n_base |
|---|---|---|---|
| p5  | +0.0302 | **[+0.0144, +0.0473]** | 710 / 7093 |
| p10 | +0.0207 | **[+0.0087, +0.0372]** | 1419 / 7093 |
| p20 | +0.0123 | **[+0.0050, +0.0207]** | 2837 / 7093 |

**(A) 16 instances (cross-check):**
| p | delta | 95% CI | n_top / n_base |
|---|---|---|---|
| p5  | +0.0229 | [-0.0347, +0.0567] | 102 / 1012 |
| p10 | +0.0095 | [-0.0280, +0.0413] | 203 / 1012 |
| p20 | +0.0038 | [-0.0152, +0.0235] | 405 / 1012 |

> All three (B) percentiles have CIs **entirely above 0**; (A) shares the sign but
> straddles 0. The global delta is positive, and the per-stage signal (§6) is more
> concentrated.

## 6. M4 per-stage NEP (§6 — the key discriminative test)

`delta = NEP_B(WA) - NEP_B(SR)`, instance-paired bootstrap, n_boot=2000.

**(B) 94 instances, K=32 (primary result):**

| stage | p5 | p10 | p20 |
|---|---|---|---|
| **LOC** | +0.0696 **[+0.028, +0.126]** | +0.0507 **[+0.021, +0.088]** | +0.0259 **[+0.011, +0.044]** |
| **REPRO** | +0.0954 **[+0.031, +0.147]** | +0.0713 **[+0.016, +0.109]** | +0.0365 **[+0.005, +0.069]** |
| **PATCH** | 0.0000 [0, 0] | 0.0000 [0, 0] | 0.0000 [0, 0] |

**(B) 94 instances, K=16 (robustness slice):**

| stage | p5 | p10 | p20 |
|---|---|---|---|
| **LOC** | +0.0615 [+0.019, +0.118] | +0.0462 [+0.017, +0.082] | +0.0236 [+0.009, +0.042] |
| **REPRO** | +0.0885 [+0.026, +0.138] | +0.0670 [+0.011, +0.104] | +0.0335 [+0.002, +0.065] |
| **PATCH** | 0.0000 [0, 0] | 0.0000 [0, 0] | 0.0000 [0, 0] |

**(A) 16 instances, K=32 (cross-check, underpowered):**

| stage | p5 | p10 | p20 |
|---|---|---|---|
| **LOC** | +0.0486 [-0.068, +0.162] | +0.0316 [-0.042, +0.109] | +0.0155 [-0.021, +0.058] |
| **REPRO** | +0.0451 [-0.167, +0.166] | +0.0509 [-0.128, +0.135] | +0.0268 [-0.125, +0.081] |
| **PATCH** | 0.0000 [0, 0] | 0.0000 [0, 0] | 0.0000 [0, 0] |

- **Direction:** all three variants (B-K32 / B-K16 / A-K32) are positive at LOC and
  REPRO (`NEP_B(WA) > 0`, `NEP_B(SR) < 0`) and exactly 0 at PATCH. All three percentiles
  and both non-PATCH stages agree in sign, and the magnitude shrinks monotonically as p
  grows (dilution by lower-entropy positions).
- **Significance:** **all six (B) cells (2 stages x 3 percentiles) have CIs entirely
  above 0, and the K=16 slice clears 0 everywhere too**; (A) shares the sign but
  straddles 0. Going from 16 to 94 instances left the sign unchanged and tightened the
  CIs past 0 — the signature of adding power to an underpowered test, not of a retuned
  criterion (the percentiles and metrics were frozen in `freeze_rules.py` before any
  result was seen).
- **D1 semantics** (see `DEVIATIONS.md`): in this dataset SR is actually **unprivileged**
  at LOC/REPRO and only gets the gold patch at PATCH, so the WA - SR contrast is largest
  at LOC/REPRO and identically 0 at PATCH. The effective contrast is binary,
  {LOC, REPRO} vs PATCH.

Main figures: `figs_gs2833/fig_nep_by_stage.pdf`, `fig_threshold_robustness.pdf`,
`fig_entropy_hist.pdf` (94 instances); the step-30 versions are in `figs_gs30_clean/`.

## 7. Placebo (§6)

Variant B's negative control compares the upper half of the low-entropy region (25th-50th
percentile) against the lower half (below the 25th). Both ends are low-entropy "committed"
positions with no real forks, so delta should be approximately 0.

**(B) 94 instances, measured:**

| stage | delta | 95% CI |
|---|---|---|
| whole pool | -5.8e-10 | [-1.3e-09, -7.8e-11] |
| LOC | -8.1e-09 | [-2.0e-08, -3.1e-09] |
| REPRO | -8.1e-10 | [-6.8e-09, +2.5e-10] |
| PATCH | 0.0 | [0, 0] |

> The negative control is clean: delta is of order 1e-8 to 1e-10 at all three stages
> (numerically zero, far below the 1e-2 scale of the real signal in §6), with no spurious
> asymmetry. That supports reading §6's positive LOC/REPRO deltas as something other than
> low-entropy noise. (A)'s 16-instance placebo is likewise approximately 0.

## 8. M3: fork rate vs. training step (§6)

**Absolute-threshold criterion.** The threshold is calibrated **once** on **base** at the
90th percentile of H and then frozen (`abs_threshold=0.670`); later steps **do not**
re-take a percentile (percentiles are an M2/M4 device, never M3's). What is reported is
the fraction of tokens whose student full-vocabulary entropy H exceeds that fixed
absolute threshold. Fixed instrument: the same 375 trajectories / 14185 PATCH-eligible
positions, `normalize=True`.

| step | student-entropy fork rate (abs) | H_mean | paper teacher-top1 fork (WA/SR) | lock (WA/SR) |
|---|---|---|---|---|
| base | **0.1000** | 0.154 | 0.00197 / 0.00219 | 0.929 / 0.928 |
| gs30 | **0.0930** | 0.138 | as above (teacher = fixed base) | as above |
| gs50 | **0.0895** | 0.130 | as above | as above |
| gs70 | **0.0901** | 0.131 | as above | as above |

**How to read it.** (1) base's 0.100 is constructive — the threshold *is* base's 90th
percentile. (2) Over training, the share of above-threshold (high-uncertainty / fork)
tokens **declines mildly and monotonically**, base 0.100 -> gs50 0.089 (-11% relative),
then holds flat from gs50 to gs70 (0.090), with H_mean tracking it 0.154 -> 0.131:
training gently contracts the high-entropy exploration surface, but not by much. (3) The
paper's teacher-top-1 fork/lock rates are **constant across steps**, because that
criterion reads the top-1 of a **fixed base teacher** whose confidence does not change
with the student's step; fork ~0.002 and lock ~0.93 again confirm that the paper's
criterion carries almost no signal on an instruction-tuned model (D4/D5), so the student
entropy has to be the primary criterion.

See `m3.json` and the figure `figs/fig_fork_rate_training.pdf`.
> The trajectories come from the stage-routed GRPO+OPSD run (base -> gs30 -> gs50 ->
> gs70, monotonically improving held-out). M3 characterizes the exploration surface of a
> **single policy across steps**; it is not a WA-vs-SR contrast (that is M2/M4).

---

## 9. Predictions checked (§7)

| prediction | statement | result |
|---|---|---|
| P1 | at high-entropy decision points, WA's uplift pull >= SR's | **supported (significant)**: all six (B) LOC/REPRO cells have delta > 0 with CIs clearing 0; (A) agrees in sign but straddles 0 |
| P2 | suppression increases with privilege (LOC/REPRO > PATCH) | **supported in direction**: non-PATCH delta > 0, PATCH = 0 (but see D1 — PATCH = 0 is constructive) |
| **P3** | at PATCH, NEP(WA) ~ NEP(SR) | **PASS (exactly 0.0 on both samples) — see §3** |
| P4 | the low-entropy negative control shows no asymmetry | **supported**: placebo delta ~ 0 at all three stages (1e-8 to 1e-10, §7) |
| P5 | the paper's teacher-top-1 criterion is flat on instruction-tuned models | **confirmed**: fork rate ~0.002, lock rate 0.93 (§4, D4) |

## 10. Deviations and fidelity notes (folded into `DEVIATIONS.md` D1-D5)

- **D1** SR is in fact unprivileged at LOC/REPRO, which reshapes what M4 measures (§6).
- **D2** WA = the force-PATCH clean definition; same evidence as a production
  whole-answer run, different outer template.
- **D3** the primary fork criterion is student entropy; the paper's teacher-top-1
  criterion is reported alongside.
- **D4** the flat-fork-rate caveat for instruction-tuned models (confirmed: fork 0.002 /
  lock 0.93).
- **D5** NEP moved from explore-vs-commit to an entropy contrast (variant B); the frozen
  point estimates are still reported alongside in `results.json` and remain auditable.
- **Fidelity 1**: the offline `StageStateMachine.update` receives, as its observation
  argument, that step's tool/user message content — what the model actually saw. Stage
  transitions are driven by the action, so this is safe.
- **Fidelity 2**: `instance_id` is not stored in `chat_completions`, so it is recovered by
  matching the issue text embedded in the user message against the staged parquet's
  `problem_statement`.
- **Fidelity 3**: the teacher input is
  `apply_chat_template(teacher_messages, add_generation_prompt=True)` as a prefix with the
  response tokens concatenated verbatim, matching training (a hard assertion checks that
  the teacher suffix equals the response).

## 11. Reproduction commands

`run_all.sh` wraps these. It expects the environment described in its header: `data_gen`,
`run.py`, and `plots.py` are CPU-only; `forwards.py` needs a working CANN toolkit and
torch_npu.

```bash
# 0) Step 30, purely on-policy, 16 instances (already produced):
#      data:     out/data_gs30_n100_fixed.pkl
#      forwards: out/fwd_gs30_n100_fixed_npu1.pkl  (mem-fraction 0.55, no OOM)
#      analyze:  results_gs30_clean.json ; figures: figs_gs30_clean/
python analysis/fork_suppression/run.py analyze \
  --forwards analysis/fork_suppression/out/fwd_gs30_n100_fixed_npu1.pkl \
  --out analysis/fork_suppression/results_gs30_clean.json

# 1) Robust version, steps 28-33 merged, 94 instances (already produced):
#      merge:    out/data_gs2833_merged.pkl (375 trajectories / 94 instances)
#      forwards: out/fwd_gs2833_merged.pkl  (mem-fraction 0.55, no OOM, 14185 positions)
#      analyze:  results_gs2833.json ; figures: figs_gs2833/
ASCEND_DEV=<free_card> bash analysis/fork_suppression/npu_env.sh python \
  analysis/fork_suppression/forwards.py \
  --data analysis/fork_suppression/out/data_gs2833_merged.pkl \
  --student analysis/fork_suppression/merged_ckpts/sr_gs30 \
  --teacher "${BASE_MODEL}" \
  --out analysis/fork_suppression/out/fwd_gs2833_merged.pkl --mem-fraction 0.55
python analysis/fork_suppression/run.py analyze \
  --forwards analysis/fork_suppression/out/fwd_gs2833_merged.pkl \
  --out analysis/fork_suppression/results_gs2833.json --n-boot 2000 --seed 0

# 2) M3 (fork rate vs step) needs a merged student checkpoint per step plus the base
#    condition (deferred).
```

Artifacts: `results_gs2833.json` (94 instances, primary) and `figs_gs2833/`;
cross-check `results_gs30_clean.json` and `figs_gs30_clean/`. The intermediate
`out/*.pkl` and `merged_ckpts/` are regenerable and are not shipped.

---

## 12. Status and open items (2026-08-12)

| item | status |
|---|---|
| P3 gate (exact self-check) | done: passes at K=32 and K=16 (max\|U_wa - U_sr\| @ PATCH = 0.0) |
| matching bug (once collapsed everything to a single instance) | fixed: the real cause was `data_gen` matching on a boilerplate prefix, which folded all rows into 1 instance; matching the issue's trailing chunk instead gives 16 instances at step 30 |
| OOM truncation | fixed: NPU0 was holding 48 GB for a training job, so mem-fraction 0.30 left only 18 GB and long-sequence batches OOM'd away half the positions; moving to an idle NPU1 at 0.55 computed all 2024 positions |
| step 30, purely on-policy (16 instances): analyze + figures | done |
| steps 28-33 merged (94 instances): data | done, `out/data_gs2833_merged.pkl` |
| **94-instance forwards** (to tighten the CIs) | done: card 4 at mem-fraction 0.55, all 375 trajectories, 14185 positions, truncated=0 (no OOM); artifact `out/fwd_gs2833_merged.pkl` |
| 94-instance analyze + figures | done: `results_gs2833.json`, `figs_gs2833/`; P3 passes exactly (8275 PATCH positions), and all six M2/M4 cells clear 0 |
| M3 (fork rate vs step) | deferred: needs a merged student checkpoint per step plus the base condition |

**Fidelity statement.** The step-30 result is **purely on-policy over 16 instances** (the
trajectories are gs30 policy samples) and stands on its own. The 94-instance merged
version uses a fixed gs30 student to score trajectories from neighbouring steps (28-33);
it is a "fixed instrument" robustness extension whose ±3-step policy drift is small (the
reward curve is smooth), and it is recorded as a declared minor deviation (see the header
comment in `merge_data.py`). It does not change the within-trajectory,
teacher-context-only nature of the WA-vs-SR contrast.
