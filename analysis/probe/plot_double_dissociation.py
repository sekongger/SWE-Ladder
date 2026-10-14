"""Combined causal-probe figure: double dissociation of stage privileges."""
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
L = {k: np.array(v) for k, v in json.load(open(f"{ROOT}/analysis/probe/hit_rates.json")).items()}
R = {k: np.array(v) for k, v in json.load(open(f"{ROOT}/analysis/probe/repro_hit_rates.json")).items()}
SURFACE, INK, MUTED, GRID, BASE_C = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
CONDS = [("none", "No\nprivilege", "#9c9a94"),
         ("loc", "Localization\nprivilege", "#2a78d6"),
         ("repro", "Reproduction\nprivilege", "#1baf7a"),
         ("patch", "Whole-answer\nprivilege", "#eb6834")]

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), dpi=200, sharey=False)
fig.patch.set_facecolor(SURFACE)

def panel(ax, H, title, ylabel, matched, brackets):
    ax.set_facecolor(SURFACE)
    xs = np.arange(len(CONDS))
    vals = [H[k].mean() * 100 for k, _, _ in CONDS]
    errs = [H[k].std() / len(H[k]) ** 0.5 * 100 for k, _, _ in CONDS]
    cols = [c for _, _, c in CONDS]
    bars = ax.bar(xs, vals, width=0.6, color=cols, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    # highlight the stage-matched bar
    mi = [k for k, _, _ in CONDS].index(matched)
    bars[mi].set_linewidth(2.2); bars[mi].set_edgecolor(INK)
    ax.errorbar(xs, vals, yerr=errs, fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.8, f"{v:.1f}", ha="center", fontsize=9.5, color=INK, fontweight="bold")
    ymax = max(vals) + 14
    for (x1, x2, lbl), dy in zip(brackets, (0, 6)):
        y = ymax + dy
        ax.plot([x1, x1, x2, x2], [y, y + 1.2, y + 1.2, y], color=INK, lw=0.9)
        ax.text((x1 + x2) / 2, y + 1.8, lbl, ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l, _ in CONDS], fontsize=8.3, color=INK)
    ax.set_ylabel(ylabel, fontsize=9.3, color=INK)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    ax.set_ylim(0, ymax + 13)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE_C); ax.tick_params(colors=MUTED)

panel(axes[0], L, "At LOCALIZATION moments (93 moments)",
      "gold-file contact rate (%)", "loc",
      [(0, 1, "p=0.029"), (1, 3, "p=0.035")])
panel(axes[1], R, "At REPRODUCTION moments (46 moments)",
      "failing-repro production rate (%)", "repro",
      [(0, 2, "p=0.004"), (2, 3, "p=0.014")])
fig.suptitle("Double dissociation: each stage privilege helps only in its own phase "
             "(actions executed in real environments)",
             fontsize=11.5, color=INK, x=0.02, ha="left", y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(f"{ROOT}/analysis/probe/fig_double_dissociation.png",
            facecolor=SURFACE, bbox_inches="tight")
print("saved fig_double_dissociation.png")
