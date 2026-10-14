"""Experiment 5 figure: teacher-anchoring audit (stage-matched reverse KL per model)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree
# that holds output/ and the model weights; it defaults to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
T = {m: json.load(open(f"{ROOT}/analysis/grad_interference/transfer/{m}.json"))
     for m in ("base", "routing_gs70", "noroute_gs70")}

SURFACE, INK, MUTED, GRID, BASE_C = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
MODELS = [("Base (= teacher's policy)", "base", "#9c9a94"),
          ("OPSD w/o routing (gs70)", "noroute_gs70", "#eb6834"),
          ("Ours routing (gs70)", "routing_gs70", "#2a78d6")]
SEGS = [("loc", "Localize tokens\nvs loc teacher"),
        ("repro", "Reproduce tokens\nvs repro teacher"),
        ("patch", "Patch tokens\nvs patch teacher")]

fig, ax = plt.subplots(figsize=(7.8, 4.0), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
x = np.arange(len(SEGS)); w = 0.25
for j, (lbl, key, color) in enumerate(MODELS):
    vals, errs = [], []
    for s, _ in SEGS:
        xs = [r[f"kl_{s}_vs_{s}"] for r in T[key] if f"kl_{s}_vs_{s}" in r]
        vals.append(np.mean(xs)); errs.append(np.std(xs) / len(xs) ** 0.5)
    pos = x + (j - 1) * (w + 0.02)
    ax.bar(pos, vals, width=w, color=color, edgecolor=SURFACE, linewidth=0.8,
           zorder=3, label=lbl)
    ax.errorbar(pos, vals, yerr=errs, fmt="none", ecolor=INK, elinewidth=1,
                capsize=2.5, zorder=4)
    for p, v in zip(pos, vals):
        ax.text(p, v + 0.008, f"{v:.3f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([l for _, l in SEGS], fontsize=8.8, color=INK)
ax.set_ylabel("reverse KL to the stage-privileged teacher\n(lower = closer)",
              fontsize=9.5, color=INK)
ax.set_title("After 70 RL steps, which student stays anchored to the privileged teacher?",
             fontsize=11.5, color=INK, loc="left", pad=10)
ax.legend(fontsize=8.5, frameon=False)
ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(BASE_C); ax.tick_params(colors=MUTED)
fig.tight_layout()
fig.savefig(f"{ROOT}/analysis/grad_interference/fig_teacher_anchoring.png",
            facecolor=SURFACE, bbox_inches="tight")
print("saved fig_teacher_anchoring.png")
