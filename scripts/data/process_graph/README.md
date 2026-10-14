# Process graphs from gold patches (P2T-style, adapted for on-policy OPSD)

Local, optimized re-implementation of the reverse phase of **P2T**
("From Patches to Trajectories: Privileged Process Supervision for SWE Agents",
arXiv:2605.21996), producing per-instance latent process graphs `G*` from
`teacher_pi.golden_patch`, targeted at our on-policy OPSD+GRPO stage-routing
training rather than P2T's offline SFT curation.

## 1. How P2T builds G* (Sec 4.1 + App D of the paper)

- `G* = (V, E)` DAG. Node `v = (statement, type, unlocker)`; the **unlocker** is
  the replayable environment action that discovers the statement, from a fixed
  taxonomy: `VIEW⟨file,lines⟩`, `VIEW problem_statement`, `BASH⟨cmd⟩`,
  `CREATE⟨path,content⟩`, `STR_REPLACE⟨file,old,new⟩`, `THINK⟨text⟩`.
- Node kinds: **contextual facts** (static: readable from repo; dynamic:
  require execution) and **solution milestones**
  (`reproduce_script → issue_analysis → fix_plan → code_edit → validation`).
- Built by a **proposer–critic loop** of two LLM agents (Qwen3-Coder-480B /
  GLM-5) with sandboxed repo tools, ≤6 rounds, 40 tool calls/round; proposer
  adds nodes to close the gap to p* (sufficiency), critic prunes leaky /
  unmotivated nodes (non-leakage). Hard rules: one claim per node (R1); no
  unlocker may reference p* (R2); actions replayable (R3); observations must be
  actually executed tool output (R4); repro/validation nodes must be executed
  with recorded before/after output (R5); `code_edit.old_str` byte-exact (R6).
- **Edges** are a deterministic post-pass: `u→v` iff v's unlocker references an
  entity first surfaced by u's observation, or the canonical artifact order.
- **Establishment** `Est(v, h_t)`: action match + an LLM verifier (restricted
  to the visible history) judges the observations entail the statement.
  Per-step progress `Prog_t` = newly-established frontier nodes / frontier
  size, hard-zeroed on premature (out-of-frontier) establishment.
- Their stats: median 18 nodes; 66% static facts, 17% dynamic facts, ~5-6% each
  artifact class. Ablation: removing the groundedness gate costs −7~8 pts
  Pass@1; G* disclosure lifts a blinded solver from ~30% to ~95% (App C).

## 2. Weaknesses (their own App L + our analysis for on-policy use)

1. **Cost**: two 480B agents × sandboxed tools × ≤6 rounds ≈ 10² GPU-hours per
   1.8k instances — most of it spent *rediscovering* facts our RL dataset
   already stores as verified fields.
2. **LLM-mediated guarantees**: non-leakage and establishment are judged by
   LLMs; no formal certificate. Per-step LLM verification is far too expensive
   online (our scale: 64 rollouts × ~60 steps × 71 RL steps).
3. **Offline-only**: P2T curates SFT data. Under on-policy drift the curated
   distribution goes stale; their own outlook names "process reward model over
   G* for online RL" as the natural extension — exactly our setting.
4. **Sandbox dependency (R4/R5)**: they must execute repro scripts to trust
   them. Our dataset's `FAIL_TO_PASS` are already execution-verified upstream,
   so this entire requirement can be dropped without losing soundness.
5. **Uniform teacher context**: P2T's forward phase steers a *generator*;
   nothing in their pipeline addresses our specific risk — a privileged
   *scoring* teacher assigning high probability to clairvoyant tokens, which
   reverse-KL then distills into the student (ungrounded leaps at eval time).
   Their groundedness gate inspires the fix (KL masking) but is not directly
   reusable as-is.

## 3. What `build_process_graph.py` does differently

**Phase A (implemented, zero LLM calls)** — synthesize the graph backbone
deterministically from dataset fields:

| P2T proposer work (LLM+tools)         | our source                                  |
|---------------------------------------|---------------------------------------------|
| locate changed files/lines            | parse `teacher_pi.golden_patch` (handles swe_smith's reversed a/b headers) |
| synthesize + execute repro script     | `extra_info.FAIL_TO_PASS` (already verified) |
| byte-exact `old_str` for edits (R6)   | hunk old/new text straight from the diff     |
| validation command                    | `FAIL_TO_PASS` / `run_tests`                 |
| issue-derived facts                   | `problem_statement` (error-line heuristic)   |

Node inventory per instance: issue fact, per-file localization fact, per-hunk
pre-fix-code fact, dynamic repro fact, and the milestone chain
repro → analysis → plan → edit(s) → validation. Edges use P2T's own
deterministic rule. Result on our 1149-instance train set: **1149/1149 built,
all DAGs acyclic, median 12 nodes** (r2e 11 / smith 11 / gym 13 / rebench 15;
P2T's median 18 includes LLM-proposed semantic facts).

**Phase B (hook present, not wired)** — one propose+critic LLM call per
instance to (a) replace the two `[template]` artifact statements
(issue_analysis / fix_plan) with real semantic content, (b) add semantic fact
nodes (API behavior, failing invariants) that pure diff parsing cannot see.
This is the only part that genuinely needs an LLM; budget ≈ 1 call/instance
instead of P2T's multi-round tool-agent loop.

## 4. On-policy consumption

**Implemented and wired** (`training/configs/ablation_process_graph_privilege.yaml`):

1. **Milestone-grade teacher context** — `swe_ladder/process_graph_evidence.py`
   renders, per stage, the *relevant subgraph* of G* instead of flat gold text,
   keeping the ladder monotone (LOC → facts only; REPRO → facts + reproduction
   milestones; PATCH → the full chain). Activated by
   `OPD_TEACHER_EVIDENCE_MODE=graph` plus `OPD_PROCESS_GRAPH_PATH`; a missing
   graph falls back to the flat evidence, so old parquets keep working.
2. **Dynamic frontier routing** — `swe_ladder/graph_frontier.py` goes one step
   past the fixed ladder and transplants P2T's forward phase: it tracks the
   established node set `U_t` with *symbolic* `Est` (no LLM judge, so it is
   affordable online) and gives each establishment epoch a teacher context of
   established + available-frontier nodes. Per-token epoch ids ride the existing
   `stage_token_ids` rail, so the trainer renders one teacher variant per epoch
   (default cap 6) instead of one per fixed stage. Enabled with
   `OPSD_GRAPH_FRONTIER=1`.

**Designed, not implemented** — listed so the gap is explicit, not as a claim:

3. **Grounded KL mask** (P2T's groundedness gate, symbolic-only): maintain
   `Obs(h_t)` = entities surfaced in observations so far; for a response step
   whose text references gold entities ∉ `Obs(h_t)`, zero the OPSD KL on that
   step's tokens (GRPO gradient unaffected). Would block distilling clairvoyant
   leaps the student cannot reproduce at eval. No knob for this exists yet.
4. **Milestone process reward**: symbolic `Est(v, h_t)` over frontier nodes → a
   small dense reward added to GRPO. Higher risk (reward hacking); would belong
   in a separate ablation.

Files:
- `build_process_graph.py` — extractor (Phase A + Phase B hook)
- output: `{task_id, instance_id, source_dataset, nodes[], edges[], meta}` JSONL
