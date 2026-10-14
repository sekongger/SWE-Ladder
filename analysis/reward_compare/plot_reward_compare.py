"""Compare per-step training reward: from-base stage-routed vs from-SFT planB.

Two OPSD+GRPO stage-routed runs, same reward definition, plotted as reward_mean vs global_step:
  - from-base : student = base Qwen3-4B-Instruct-2507 (the 2026-07-11 run; 71 steps, all valid)
  - planB     : student = SWE-Master-4B-SFT, 60K rollout / update-cap 32768 (2026-08-05 run)

IMPORTANT (honesty): planB's docker tunnel died at gs29 (response_length/mean collapses to 12.0,
reward flatlines to exactly 0.0 for gs29-71). Those zeros are a dead-collection artifact, NOT
policy reward, so planB is plotted ONLY over its valid prefix (auto-detected as the first step
whose response_length/mean < 100). The death point is annotated.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree
# that holds output/ and the model weights; it defaults to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
# Where the trainer writes per-run metrics (train_entrypoint_npu.sh's default local_dir).
TRAIN_OUTPUT = os.environ.get(
    "TRAIN_OUTPUT_ROOT", os.path.join(WORKSPACE, "DeepSWE_RL/rllm/rllm/train_output")
)
FROMBASE = os.path.join(
    TRAIN_OUTPUT,
    "swe_opsd_grpo_end_to_end_stage_routing",
    "opsd-grpo-end-to-end-stage-routing-qwen3-4b-colocated-8npu-2x4tp_2026-07-11_03-35-51",
)
PLANB = os.path.join(
    TRAIN_OUTPUT,
    "swe_opsd_grpo_e2e_stage_routing_from_sft_planA",
    "opsd-grpo-e2e-stage-routing-from-sft-planA-qwen3-4b-colocated-8npu-2x4tp_2026-08-05_03-21-01",
)
OUTDIR = os.path.join(ROOT, "analysis/reward_compare")


def load_reward(run_dir):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "metrics/reward_curve.jsonl")) if l.strip()]
    steps = np.array([r["global_step"] for r in rows])
    reward = np.array([r["reward_mean"] for r in rows], dtype=float)
    return steps, reward


def valid_cutoff(run_dir):
    """Return the last valid global_step: first step whose response_length/mean < 100 marks a
    dead-collection (docker death) point; everything from there on is dropped. If no such step,
    all steps are valid."""
    tc = os.path.join(run_dir, "metrics/train_curve.jsonl")
    if not os.path.exists(tc):
        return None
    rows = [json.loads(l) for l in open(tc) if l.strip()]
    for r in rows:
        rl = r.get("response_length/mean")
        if rl is not None and rl < 100:
            return r["global_step"] - 1  # last good step before collapse
    return rows[-1]["global_step"] if rows else None


def rolling(y, w=5):
    if len(y) < 2:
        return y
    w = min(w, len(y))
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


def main():
    bs_steps, bs_rew = load_reward(FROMBASE)
    pb_steps, pb_rew = load_reward(PLANB)

    pb_cut = valid_cutoff(PLANB)          # e.g. 28
    death_step = pb_cut + 1 if pb_cut is not None else None
    if pb_cut is not None:
        m = pb_steps <= pb_cut
        pb_steps_v, pb_rew_v = pb_steps[m], pb_rew[m]
    else:
        pb_steps_v, pb_rew_v = pb_steps, pb_rew

    C_BASE, C_PB = "#1f77b4", "#d62728"
    fig, ax = plt.subplots(figsize=(11, 6.2))

    # raw (thin, faded) + rolling-mean (bold)
    ax.plot(bs_steps, bs_rew, color=C_BASE, alpha=0.25, lw=1.0)
    ax.plot(bs_steps, rolling(bs_rew), color=C_BASE, lw=2.4,
            label=f"from-base (student=base Qwen3-4B) · gs1-{bs_steps[-1]} (mean={bs_rew.mean():.3f})")
    ax.plot(pb_steps_v, pb_rew_v, color=C_PB, alpha=0.25, lw=1.0)
    ax.plot(pb_steps_v, rolling(pb_rew_v), color=C_PB, lw=2.4,
            label=f"planB (student=SFT, 60K rollout) · gs1-{pb_cut} valid (mean={pb_rew_v.mean():.3f})")

    if death_step is not None:
        ax.axvline(death_step, color="gray", ls="--", lw=1.3)
        ax.annotate(f"planB docker died @ gs{death_step}\n(response_length→12, reward artifact 0)",
                    xy=(death_step, 0.02), xytext=(death_step + 1.5, 0.33),
                    fontsize=9, color="gray",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.0))

    ax.set_xlabel("global step", fontsize=12)
    ax.set_ylabel("reward_mean (per step, batch=64)", fontsize=12)
    ax.set_title("OPSD + GRPO stage-routed training reward: from-base vs from-SFT (planB)",
                 fontsize=13)
    ax.set_ylim(-0.01, 0.46)
    ax.set_xlim(0, max(bs_steps[-1], pb_steps[-1]) + 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    fig.tight_layout()

    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"reward_compare_base_vs_planB.{ext}"), dpi=150)

    # stdout summary
    print("=== reward comparison ===")
    print(f"from-base: gs1-{bs_steps[-1]}  mean={bs_rew.mean():.4f}  "
          f"last5={bs_rew[-5:].mean():.4f}  max={bs_rew.max():.4f}")
    print(f"planB    : gs1-{pb_cut} (valid)  mean={pb_rew_v.mean():.4f}  "
          f"last5={pb_rew_v[-5:].mean():.4f}  max={pb_rew_v.max():.4f}  "
          f"[docker died @ gs{death_step}]")
    print(f"saved -> {OUTDIR}/reward_compare_base_vs_planB.png / .pdf")


if __name__ == "__main__":
    main()
