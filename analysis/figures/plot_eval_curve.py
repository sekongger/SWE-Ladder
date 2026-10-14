#!/usr/bin/env python3
"""Held-out SWE-bench Verified pass@1 vs. training step.

All points are from the *original_scaffold* evaluation harness (same r2egym
scaffold, unit-test verified). The run names below are fixed; every number is
read live from each eval's `*.summary.json` solve_rate, so the figure stays in
sync if an eval is refreshed or re-run.

Run:
  MPLCONFIGDIR=/tmp/mpl PYTHONNOUSERSITE=1 \
  WORKSPACE_ROOT=/path/to/workspace python analysis/figures/plot_eval_curve.py
"""
import json
import os
import glob
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree that
# holds output/ (the eval harness writes one dir per checkpoint there); it defaults
# to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
OUT = os.environ.get("EVAL_OUTPUT_ROOT", os.path.join(WORKSPACE, "output"))


def rate(dir_glob):
    """Return solve_rate*100 and total from the single matching eval dir."""
    matches = glob.glob(os.path.join(OUT, dir_glob))
    matches = [m for m in matches if not m.endswith("_01")]
    assert len(matches) == 1, (dir_glob, matches)
    sm = glob.glob(os.path.join(matches[0], "*.summary.json"))[0]
    j = json.load(open(sm))
    return round(100 * j["solve_rate"], 2), j.get("total_trajectories")


P = "swebench_verified_"
S = "_original_scaffold_eval"
E2E = "opsd_grpo_end_to_end_stage_routing_qwen3_4b_colocated_8npu_2x4tp"
E2E2 = "opsd_grpo_end_to_end_stage_routing_qwen3_4b_colocated_8npu_2x4tp_epoch2_from_gs70"
GRPO = "pure_grpo_baseline_end_to_end_qwen3_4b_colocated_8npu_2x4tp_20260717"
PATCH = "opsd_grpo_patch_joint_intersection_from_patch_gs70_qwen3_4b_colocated_8npu_2x4tp"

base, _ = rate(f"{P}qwen3_4b_instruct_2507_base{S}")

# ours: epoch1 (gs30..70) then epoch2 from gs70 (gs90..140)
ours_pts = []
for gs in (30, 50, 60, 70):
    r, n = rate(f"{P}{E2E}_20260711_gs{gs}{S}")
    ours_pts.append((gs, r, n))
for gs in (90, 100, 120, 140):
    r, n = rate(f"{P}{E2E2}_20260714_gs{gs}{S}")
    ours_pts.append((gs, r, n))

# pure GRPO: gs30 (healthy) then gs70 (collapsed, only n=50)
grpo_pts = []
for gs in (30, 70):
    r, n = rate(f"{P}{GRPO}_gs{gs}{S}")
    grpo_pts.append((gs, r, n))

# per-stage OPSD (patch stage), plotted against its own step count
patch_pts = []
for gs, tag in [(30, "20260702_gs30_npu"), (60, "20260702_gs60"),
                (80, "20260703_gs80"), (90, "20260627_gs90"), (110, "20260627_gs110")]:
    r, n = rate(f"{P}{PATCH}_{tag}{S}")
    patch_pts.append((gs, r, n))

# prepend the shared base point (step 0) to the two end-to-end RL curves
ours = [(0, base, 500)] + ours_pts
grpo = [(0, base, 500)] + grpo_pts

C_OURS, C_GRPO, C_PATCH = "#1b6ca8", "#c1272d", "#e8890c"

fig, ax = plt.subplots(figsize=(6.6, 4.3))
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm"})

ax.axhline(base, ls="--", lw=1.2, color="0.45", zorder=1)
ax.text(142, base + 0.15, f"Base {base:.1f}%", color="0.35", fontsize=9,
        ha="right", va="bottom")


def plot(ax, pts, color, label, marker):
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    ax.plot(x, y, color=color, lw=2.0, marker=marker, ms=6, label=label, zorder=3)


plot(ax, ours, C_OURS, "SWE-Ladder (ours)", "o")
plot(ax, grpo, C_GRPO, "GRPO", "s")
plot(ax, patch_pts, C_PATCH, "Per-stage OPSD (patch)", "^")

# flag the gs60 point: eval-time env failures (only 291/482 non-empty patches)
ax.plot([60], [8.09], marker="o", ms=8, mfc="white", mec=C_OURS, mew=1.8, zorder=4)
ax.annotate("$\\dagger$ eval-infra\ndropout", xy=(60, 8.09), xytext=(44, 5.0),
            color=C_OURS, fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color=C_OURS, lw=0.9))

# peak + collapse annotations
peak = max(ours_pts, key=lambda p: p[1])
ax.annotate(f"{peak[1]:.1f}%", xy=(peak[0], peak[1]), xytext=(peak[0] - 4, peak[1] + 0.7),
            color=C_OURS, fontsize=10, fontweight="bold")
ax.annotate("collapse\n(n=50)", xy=(70, 0.0), xytext=(78, 1.4), color=C_GRPO, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=C_GRPO, lw=1.0))

# mark epoch-2 boundary for ours
ax.axvline(70, ls=":", lw=1.0, color="0.7", zorder=1)
ax.text(71, 1.0, "epoch 2", color="0.55", fontsize=8, rotation=90, va="bottom")

ax.set_xlabel("Training step")
ax.set_ylabel("SWE-bench Verified pass@1 (%)")
ax.set_ylim(-0.5, 17)
ax.set_xlim(-4, 145)
ax.grid(True, alpha=0.25, lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=10)

fig.tight_layout()
fig.savefig(os.path.join(ROOT, "analysis/figures/fig_eval_curve.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(ROOT, "analysis/figures/fig_eval_curve.png"), bbox_inches="tight", dpi=200)
print("wrote analysis/figures/fig_eval_curve.{pdf,png}")
print("base =", base)
print("ours =", ours_pts)
print("grpo =", grpo_pts)
print("patch=", patch_pts)
