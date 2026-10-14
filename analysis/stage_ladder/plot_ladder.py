"""Stage-capability ladder figure: 4 systems x 4 stage metrics, two panels.

Panel A: matched checkpoint (gs30) -- fair same-training comparison.
Panel B: each method's best available checkpoint.

Reads the aggregate ladder_r2egym.py writes (shipped, so this runs in a plain
checkout) and saves fig_stage_ladder.png next to it.
"""
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Repo root, which holds analysis/ and swe_ladder/.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
S = json.load(open(f"{ROOT}/analysis/stage_ladder/ladder_r2egym_summary.json"))

SURFACE, INK, MUTED, GRID, BASE_C = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

METRICS = [
    ("loc_recall", "Localization\n(gold-file recall)"),
    ("reproduce",  "Reproduction\n(ran failing repro)"),
    ("verify",     "Verification\n(re-ran repro, passed)"),
    ("solve",      "Solve\n(pass@1)"),
]
PANELS = [
    ("Matched training (gs30)", [
        ("Base Qwen3-4B",  "base",         "#9c9a94"),
        ("Pure GRPO",      "grpo_gs30",    "#1baf7a"),
        ("OPSD w/o routing","noroute_gs30", "#eb6834"),
        ("Ours (routing)", "routing_gs30", "#2a78d6"),
    ]),
    ("Best checkpoint per method", [
        ("Base Qwen3-4B",        "base",         "#9c9a94"),
        ("Pure GRPO (gs30)",     "grpo_gs30",    "#1baf7a"),
        ("OPSD w/o routing (gs50)","noroute_gs50", "#eb6834"),
        ("Ours (routing, gs90)", "routing_gs90", "#2a78d6"),
    ]),
]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=200, sharey=True)
fig.patch.set_facecolor(SURFACE)

for ax, (title, systems) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    x = np.arange(len(METRICS))
    nsys = len(systems)
    w = 0.19
    for j, (label, key, color) in enumerate(systems):
        vals = [S[key][mk] * 100 for mk, _ in METRICS]
        pos = x + (j - (nsys - 1) / 2) * (w + 0.015)
        bars = ax.bar(pos, vals, width=w, color=color, zorder=3,
                      edgecolor=SURFACE, linewidth=0.8, label=label)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.8, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6.8, color=INK, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in METRICS], fontsize=8.5, color=INK)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=10)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASE_C)
    ax.set_ylim(0, 72)

axes[0].set_ylabel("% of trajectories", fontsize=10, color=INK)
axes[0].legend(loc="upper right", fontsize=8, frameon=False, handlelength=1.4)
axes[1].legend(loc="upper right", fontsize=8, frameon=False, handlelength=1.4)
fig.suptitle("Stage-wise capability ladder on SWE-bench Verified (500 held-out tasks)",
             fontsize=12, color=INK, x=0.02, ha="left", y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = f"{ROOT}/analysis/stage_ladder/fig_stage_ladder.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("saved", out)
