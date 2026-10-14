"""Figure for the single-step intervention probe (O3)."""
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
H = {k: np.array(v) for k, v in json.load(open(f"{ROOT}/analysis/probe/hit_rates.json")).items()}
SURFACE, INK, MUTED, GRID, BASE_C = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

CONDS = [("none", "No privilege", "#9c9a94"),
         ("loc", "Localization\nprivilege (routed)", "#2a78d6"),
         ("repro", "Reproduction\nprivilege", "#1baf7a"),
         ("patch", "Whole-answer\nprivilege (no routing)", "#eb6834")]

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=200)
fig.patch.set_facecolor(SURFACE)

# panel 1: gold-contact rate
ax = axes[0]; ax.set_facecolor(SURFACE)
xs = np.arange(len(CONDS))
vals = [H[k].mean() * 100 for k, _, _ in CONDS]
errs = [H[k].std() / len(H[k]) ** 0.5 * 100 for k, _, _ in CONDS]
ax.bar(xs, vals, width=0.6, color=[c for _, _, c in CONDS],
       edgecolor=SURFACE, linewidth=0.8, zorder=3)
ax.errorbar(xs, vals, yerr=errs, fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
for x, v in zip(xs, vals):
    ax.text(x, v + 1.6, f"{v:.1f}", ha="center", fontsize=9.5, color=INK, fontweight="bold")
# significance brackets
def bracket(x1, x2, y, label):
    ax.plot([x1, x1, x2, x2], [y, y + 1, y + 1, y], color=INK, lw=0.9)
    ax.text((x1 + x2) / 2, y + 1.4, label, ha="center", fontsize=8, color=INK)
bracket(0, 1, 92.5, "p=0.029")
bracket(1, 3, 96.5, "p=0.035")
ax.set_xticks(xs)
ax.set_xticklabels([l for _, l, _ in CONDS], fontsize=8.3, color=INK)
ax.set_ylabel("gold-file contact rate (%)", fontsize=9.5, color=INK)
ax.set_title("Executing sampled actions at real retrieval moments\n"
             "(93 moments × 8 samples per condition)", fontsize=10.5, color=INK,
             loc="left", pad=10)
ax.set_ylim(0, 104)
ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(BASE_C); ax.tick_params(colors=MUTED)

# panel 2: action-mix shift
ax = axes[1]; ax.set_facecolor(SURFACE)
rows = [json.load(open(f)) for f in __import__("glob").glob(f"{ROOT}/analysis/probe/results/*.json")]
kinds_order = ["search", "view", "edit", "finish"]
w = 0.19
for j, (k, lbl, color) in enumerate(CONDS):
    counts = {}
    tot = 0
    for r in rows:
        for s in r["conditions"][k]["samples"]:
            counts[s["kind"]] = counts.get(s["kind"], 0) + 1; tot += 1
    vals = [counts.get(kk, 0) / tot * 100 for kk in kinds_order]
    pos = np.arange(len(kinds_order)) + (j - 1.5) * (w + 0.015)
    ax.bar(pos, vals, width=w, color=color, edgecolor=SURFACE, linewidth=0.8,
           zorder=3, label=lbl.replace("\n", " "))
    for p, v in zip(pos, vals):
        if v > 1.5:
            ax.text(p, v + 1.0, f"{v:.0f}", ha="center", fontsize=7, color=INK)
ax.set_xticks(np.arange(len(kinds_order)))
ax.set_xticklabels(["search", "view file", "edit", "finish (!)"], fontsize=9, color=INK)
ax.set_ylabel("% of sampled actions", fontsize=9.5, color=INK)
ax.set_title("What the model chooses to do next, per privilege", fontsize=10.5,
             color=INK, loc="left", pad=10)
ax.legend(fontsize=7.5, frameon=False)
ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(BASE_C); ax.tick_params(colors=MUTED)

fig.suptitle("Causal probe: stage-matched privilege at localization moments",
             fontsize=12, color=INK, x=0.02, ha="left", y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{ROOT}/analysis/probe/fig_step_probe.png", facecolor=SURFACE, bbox_inches="tight")
print("saved analysis/probe/fig_step_probe.png")
