"""Rigorous anchoring evidence: each run's ON-POLICY reverse KL to ITS OWN teacher,
straight from training telemetry (opsd/reverse_kl_mean). No off-policy replay,
no cross-objective comparison. Both runs: coef 0.15, top-k 16, teacher mass ~1.0."""
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
D = json.load(open(f"{ROOT}/analysis/stage_ladder/onpolicy_kl_curves.json"))
SURFACE, INK, MUTED, GRID, BASE_C = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

fig, ax = plt.subplots(figsize=(7.8, 4.0), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
for key, lbl, color in [
        ("noroute", "OPSD w/o routing: KL to its whole-answer teacher", "#eb6834"),
        ("routing", "Ours: KL to its per-stage routed teachers", "#2a78d6")]:
    xs = sorted(int(k) for k in D[key])
    ys = [D[key][str(x)] for x in xs]
    ax.plot(xs, ys, color=color, lw=1.8, label=lbl)
    # light smoothing overlay (window 5) for readability
    if len(ys) > 7:
        sm = np.convolve(ys, np.ones(5) / 5, mode="valid")
        ax.plot(xs[2:-2], sm, color=color, lw=3, alpha=0.35)
    ax.annotate(f"{ys[-1]:.2f}", xy=(xs[-1], ys[-1]), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=9,
                color=INK, fontweight="bold")

ax.set_xlabel("training step (each point: the run's own rollouts of that step)",
              fontsize=9.5, color=INK)
ax.set_ylabel("on-policy reverse KL to the run's OWN teacher", fontsize=9.5, color=INK)
ax.set_title("Distillation residual during training: whole-answer target is never reached;\n"
             "the stage-routed target stays close throughout", fontsize=11, color=INK,
             loc="left", pad=10)
ax.legend(fontsize=8.5, frameon=False)
ax.grid(axis="y", color=GRID, lw=0.7)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(BASE_C); ax.tick_params(colors=MUTED)
ax.set_ylim(0, 0.75)
fig.tight_layout()
fig.savefig(f"{ROOT}/analysis/stage_ladder/fig_onpolicy_kl.png", facecolor=SURFACE, bbox_inches="tight")
print(f"saved {ROOT}/analysis/stage_ladder/fig_onpolicy_kl.png")
