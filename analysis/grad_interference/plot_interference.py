"""Gradient-interference experiment, step 4: figures (signal-specificity framing).

Fig 1: cross-stage gradient correlation heatmaps, whole-answer vs routed privilege
       (diagonal grayed; within-stage split-half baseline quoted in the title).
Fig 2: paired violins -- same-tokens agreement between privilege variants, with
       shuffled-privilege controls. Key readout: for REPRO tokens the whole-answer
       teacher agrees LESS with the stage teacher than a wrong-instance control.
Fig 3: per-layer cosine profile for the key pairs.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree
# that holds output/ and the model weights; it defaults to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
GRADS = f"{ROOT}/analysis/grad_interference/grads"
OUTD = f"{ROOT}/analysis/grad_interference"

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#f0efec"

rows = [json.load(open(f)) for f in sorted(glob.glob(f"{GRADS}/*.json"))]
print(f"{len(rows)} trajectories")

def collect(group, a=None, b=None):
    vals = []
    for r in rows:
        for c in r["cosines"]:
            if c["group"] != group: continue
            if a and c["a"] != a: continue
            if b and c["b"] != b: continue
            vals.append(c["cos"])
    return np.array(vals)

split_half = collect("D_splithalf")
sh_mean = split_half.mean() if len(split_half) else float("nan")

# ---------- Fig 1: cross-stage correlation heatmaps ----------
STAGE_LBL = ["Localize", "Reproduce", "Patch"]
PAIRKEY = {("g", 0, 1): ("A_global", "loc_g", "repro_g"),
           ("g", 0, 2): ("A_global", "loc_g", "patch"),
           ("g", 1, 2): ("A_global", "repro_g", "patch"),
           ("r", 0, 1): ("A_routed", "loc_r", "repro_r"),
           ("r", 0, 2): ("A_routed", "loc_r", "patch"),
           ("r", 1, 2): ("A_routed", "repro_r", "patch")}

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=200,
                         gridspec_kw={"wspace": 0.32})
fig.patch.set_facecolor(SURFACE)
for pi, (ax, scheme, title) in enumerate([
        (axes[0], "g", "Whole-answer privilege (no routing)"),
        (axes[1], "r", "Per-stage privilege (routing)")]):
    mat = np.zeros((3, 3)); mask = np.eye(3, dtype=bool)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        v = collect(*PAIRKEY[(scheme, i, j)]).mean()
        mat[i, j] = mat[j, i] = v
    im = ax.imshow(np.ma.masked_array(mat, mask), vmin=-0.10, vmax=0.10, cmap="RdBu_r")
    for i in range(3):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color=NEUTRAL, zorder=2))
        ax.text(i, i, "—", ha="center", va="center", fontsize=11, color=MUTED, zorder=3)
    for i in range(3):
        for j in range(3):
            if i == j: continue
            ax.text(j, i, f"{mat[i,j]:+.3f}", ha="center", va="center", fontsize=10, color=INK)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(STAGE_LBL, fontsize=9)
    if pi == 0:
        ax.set_yticklabels(STAGE_LBL, fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=10.5, color=INK, pad=8)
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cb.set_label("mean gradient cosine", fontsize=9, color=MUTED)
fig.suptitle(f"Cross-stage correlation of the distillation gradient "
             f"(within-stage split-half baseline: {sh_mean:.2f})",
             fontsize=11.5, color=INK, x=0.03, ha="left")
fig.savefig(f"{OUTD}/fig_crossstage_heatmap.png", facecolor=SURFACE, bbox_inches="tight")
print("fig 1 saved")

# ---------- Fig 2: paired violins (signal specificity) ----------
panels = [
    ("Localize tokens\nwhole-answer vs\nstage teacher", collect("B_mismatch", "loc_r", "loc_g"), BLUE),
    ("Localize tokens\nwrong-instance\ncontrol", collect("C_shuffle", "loc_r", "loc_sh"), "#9c9a94"),
    ("Reproduce tokens\nwhole-answer vs\nstage teacher", collect("B_mismatch", "repro_r", "repro_g"), BLUE),
    ("Reproduce tokens\nwrong-instance\ncontrol", collect("C_shuffle", "repro_r", "repro_sh"), "#9c9a94"),
    ("Patch tokens\nwrong-instance\ncontrol", collect("C_shuffle", "patch", "patch_sh"), "#9c9a94"),
    ("Within-stage\nsplit-half\nbaseline", split_half, AQUA),
]
panels = [(l, v, c) for l, v, c in panels if len(v)]
fig, ax = plt.subplots(figsize=(9.6, 4.2), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
parts = ax.violinplot([v for _, v, _ in panels], showmedians=True, widths=0.75)
for pc, (_, _, c) in zip(parts["bodies"], panels):
    pc.set_facecolor(c); pc.set_alpha(0.6)
for k in ("cmedians", "cbars", "cmins", "cmaxes"):
    parts[k].set_color(INK); parts[k].set_linewidth(1.0)
ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
ax.set_xticks(range(1, len(panels) + 1))
ax.set_xticklabels([l for l, _, _ in panels], fontsize=8, color=INK)
ax.set_ylabel("gradient cosine (same tokens)", fontsize=10, color=INK)
ax.set_title("How well does whole-answer privilege approximate the stage-correct "
             "teacher signal?", fontsize=11.5, color=INK, loc="left", pad=10)
for i, (_, v, _) in enumerate(panels):
    ax.text(i + 1, np.median(v) + 0.03, f"{np.median(v):.2f}", ha="center",
            fontsize=8.5, color=INK, fontweight="bold")
ax.grid(axis="y", color=GRID, lw=0.7)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.tick_params(colors=MUTED)
fig.tight_layout()
fig.savefig(f"{OUTD}/fig_specificity_violin.png", facecolor=SURFACE, bbox_inches="tight")
print("fig 2 saved")

# ---------- Fig 3: per-layer profile ----------
def layer_series(group, a, b):
    acc = {}
    for r in rows:
        for c in r["cosines"]:
            if c["group"] == group and c["a"] == a and c["b"] == b:
                for k, v in c["per_layer"].items():
                    acc.setdefault(k, []).append(v)
    xs, ys = [], []
    for k, v in acc.items():
        if k.startswith("layer."):
            xs.append(int(k.split(".")[1])); ys.append(np.mean(v))
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]

fig, ax = plt.subplots(figsize=(7.8, 3.8), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
for (grp, a, b, lbl, color, ls) in [
        ("B_mismatch", "loc_r", "loc_g", "Localize: whole-answer vs stage teacher", BLUE, "-"),
        ("B_mismatch", "repro_r", "repro_g", "Reproduce: whole-answer vs stage teacher", ORANGE, "-"),
        ("C_shuffle", "repro_r", "repro_sh", "Reproduce: wrong-instance control", "#9c9a94", "--"),
        ("D_splithalf", "loc_h1", "loc_h2", "Within-stage split-half baseline", AQUA, ":")]:
    xs, ys = layer_series(grp, a, b)
    if len(xs): ax.plot(xs, ys, color=color, lw=2, ls=ls, label=lbl)
ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
ax.set_xlabel("transformer layer", fontsize=10, color=INK)
ax.set_ylabel("gradient cosine", fontsize=10, color=INK)
ax.set_title("Per-layer agreement of distillation gradients", fontsize=11.5,
             color=INK, loc="left", pad=10)
ax.legend(fontsize=8, frameon=False, loc="best")
ax.grid(axis="y", color=GRID, lw=0.7)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.tick_params(colors=MUTED)
fig.tight_layout()
fig.savefig(f"{OUTD}/fig_layer_profile.png", facecolor=SURFACE, bbox_inches="tight")
print("fig 3 saved")

# ---------- summary ----------
print("\n=== summary (mean ± std) ===")
for grp, a, b in [("A_global","loc_g","repro_g"),("A_global","loc_g","patch"),("A_global","repro_g","patch"),
                  ("A_routed","loc_r","repro_r"),("A_routed","loc_r","patch"),("A_routed","repro_r","patch"),
                  ("B_mismatch","loc_r","loc_g"),("B_mismatch","repro_r","repro_g"),
                  ("C_shuffle","loc_r","loc_sh"),("C_shuffle","repro_r","repro_sh"),("C_shuffle","patch","patch_sh"),
                  ("D_splithalf","loc_h1","loc_h2")]:
    v = collect(grp, a, b)
    if len(v): print(f"{grp:12s} {a:9s} x {b:9s}: {v.mean():+.3f} ± {v.std():.3f}  (n={len(v)})")

# solved/unsolved breakdown for the headline pair
for tag, (grp, a, b) in {"B repro": ("B_mismatch", "repro_r", "repro_g")}.items():
    sv = [c["cos"] for r in rows if r["solved"] for c in r["cosines"]
          if c["group"] == grp and c["a"] == a and c["b"] == b]
    us = [c["cos"] for r in rows if not r["solved"] for c in r["cosines"]
          if c["group"] == grp and c["a"] == a and c["b"] == b]
    print(f"{tag}: solved {np.mean(sv):+.3f} (n={len(sv)})  unsolved {np.mean(us):+.3f} (n={len(us)})")
