#!/usr/bin/env python3
"""Training dynamics: SWE-Ladder (ours) vs. pure GRPO.

Reads the per-step train_curve.jsonl logs of the two end-to-end runs and plots
four panels: policy entropy, gradient norm, response length, and training solve
rate. The message: SWE-Ladder's optimization stays stable (flat entropy, healthy
gradients, preserved-and-growing length, rising reward) exactly where pure GRPO
collapses (entropy -> 0, gradient norm -> 0, length runaway, reward decays).

Run:
  MPLCONFIGDIR=/tmp/mpl python analysis/figures/plot_dynamics.py
"""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree that
# holds the training outputs and model weights; it defaults to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
TR = os.environ.get("TRAIN_OUTPUT_ROOT", os.path.join(WORKSPACE, "DeepSWE_RL/rllm/rllm/train_output"))

# --- our method: epoch-1 from scratch (steps 1..71) then epoch-2 from gs70 (71..142)
OURS_E1 = os.path.join(
    TR,
    "swe_opsd_grpo_end_to_end_stage_routing",
    "opsd-grpo-end-to-end-stage-routing-qwen3-4b-colocated-8npu-2x4tp_2026-07-11_03-35-51",
    "metrics/train_curve.jsonl",
)
OURS_E2 = os.path.join(
    TR,
    "swe_opsd_grpo_end_to_end_stage_routing",
    "opsd-grpo-end-to-end-stage-routing-qwen3-4b-colocated-8npu-2x4tp-epoch2-from-gs70_2026-07-14_05-15-15",
    "metrics/train_curve.jsonl",
)
# --- pure GRPO baseline (steps 1..71, collapses)
GRPO = os.path.join(
    TR,
    "swe_pure_grpo_baseline_end_to_end",
    "pure-grpo-baseline-end-to-end-qwen3-4b-colocated-8npu-2x4tp_2026-07-17_12-48-44",
    "metrics/train_curve.jsonl",
)


def load(fp):
    rows = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def series(rows, key, alt=None):
    xs, ys = [], []
    for d in rows:
        v = d.get(key, d.get(alt) if alt else None)
        if v is not None:
            xs.append(d["global_step"])
            ys.append(v)
    return np.array(xs, float), np.array(ys, float)


def ema(y, span=5):
    if len(y) == 0:
        return y
    a = 2.0 / (span + 1.0)
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = a * y[i] + (1 - a) * out[i - 1]
    return out


# stitch our two epochs: epoch1 for steps < 71, epoch2 for steps >= 71
ours = load(OURS_E1) + [d for d in load(OURS_E2) if d["global_step"] > 71]
grpo = load(GRPO)

C_OURS = "#1b6ca8"   # deep blue
C_GRPO = "#c1272d"   # red

PANELS = [
    ("actor/entropy", None, "Policy entropy", 1.0, False),
    ("actor/grad_norm", None, "Gradient norm", 1.0, False),
    ("response_length/mean", None, "Response length (k tokens)", 1e-3, False),
    ("critic/score/mean", "reward_mean", "Training solve rate", 1.0, True),
]

plt.rcParams.update({
    "font.size": 11,
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.15))

for ax, (key, alt, ylabel, scale, smooth) in zip(axes, PANELS):
    for rows, color, lab in [(ours, C_OURS, "SWE-Ladder (ours)"), (grpo, C_GRPO, "GRPO")]:
        x, y = series(rows, key, alt)
        if len(x) == 0:
            continue
        y = y * scale
        ys = ema(y, span=5) if smooth else y
        # raw faint
        ax.plot(x, y, color=color, alpha=0.18, lw=1.0)
        ax.plot(x, ys if smooth else y, color=color, lw=1.9, label=lab)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.margins(x=0.02)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# annotate the collapse on entropy + grad-norm panels
axes[0].annotate("entropy collapse", xy=(71, 0.005), xytext=(40, 0.018),
                 color=C_GRPO, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=C_GRPO, lw=1.0))
axes[1].annotate("grad norm $\\to 0$", xy=(69, 0.05), xytext=(34, 1.55),
                 color=C_GRPO, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=C_GRPO, lw=1.0))
axes[2].annotate("length runaway", xy=(71, 24.4), xytext=(18, 26.5),
                 color=C_GRPO, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=C_GRPO, lw=1.0))

handles = [
    Line2D([0], [0], color=C_OURS, lw=2.2, label="SWE-Ladder (ours)"),
    Line2D([0], [0], color=C_GRPO, lw=2.2, label="GRPO"),
]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.06), fontsize=11)

fig.tight_layout(rect=[0, 0, 1, 0.97])
out_pdf = os.path.join(ROOT, "analysis/figures/fig_dynamics.pdf")
out_png = os.path.join(ROOT, "analysis/figures/fig_dynamics.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print("wrote", out_pdf)
print("wrote", out_png)

# --- print caption-ready stats
def endpoints(rows, key, alt=None):
    x, y = series(rows, key, alt)
    return (int(x[0]), round(float(y[0]), 4), int(x[-1]), round(float(y[-1]), 4),
            round(float(np.min(y)), 4), round(float(np.max(y)), 4))

print("\n--- caption stats ---")
for name, rows in [("OURS", ours), ("GRPO", grpo)]:
    print(name,
          "steps", rows[0]["global_step"], "..", rows[-1]["global_step"])
    for k in ["actor/entropy", "actor/grad_norm", "response_length/mean", "critic/score/mean"]:
        print("   ", k, endpoints(rows, k, "reward_mean"))
