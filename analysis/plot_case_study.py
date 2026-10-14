"""Mechanistic case-study figure: per-step teacher-divergence curves over one
real solved trajectory, with rule-stage bands.

Divergence of teacher s at step t = mean over the step's assistant tokens of
(lp_student - lp_s)  -- how strongly privilege s wants to deviate from what the
policy actually did there. If routing is meaningful, each teacher's divergence
should peak inside its own stage band.
"""
import importlib.util, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree
# that holds output/ and the model weights; it defaults to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[1])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
DATA = f"{ROOT}/analysis/grad_interference/data"
ALP = f"{ROOT}/analysis/grad_interference/actual_lp"
EVAL_JSONL = f"{WORKSPACE}/output/swebench_verified_qwen3_4b_instruct_2507_base_original_scaffold_eval/swebench_verified_qwen3_4b_instruct_2507_base_original_scaffold_eval.jsonl"

sys.path.insert(0, f"{ROOT}/analysis/grad_interference")
import prep_data as pd
import os
from pathlib import Path
spec = importlib.util.spec_from_file_location("sc", f"{ROOT}/swe_ladder/stage_classifier.py")
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)

IID = sys.argv[1] if len(sys.argv) > 1 else "django__django-7530"

DEC = json.JSONDecoder()
def find_rec(path, iid):
    for line in open(path):
        line = line.strip(); idx = 0
        while idx < len(line):
            try: obj, end = DEC.raw_decode(line, idx)
            except json.JSONDecodeError: break
            idx = end
            ds = obj["ds"]; ds = json.loads(ds) if isinstance(ds, str) else ds
            if ds["instance_id"] == iid: return obj, ds
    raise KeyError(iid)

rec, ds = find_rec(EVAL_JSONL, IID)
z = np.load(f"{DATA}/{IID}.npz"); alp = np.load(f"{ALP}/{IID}.npz")
resp_start = int(z["resp_start"])

msgs = pd.build_messages(rec, pd.student_user(ds.get("problem_statement") or ""))
_, spans = pd.tokenize_with_spans(msgs)
steps = rec["trajectory_steps"]
pairs = [(f"{s.get('thought') or ''}\n\n{s.get('action') or ''}",
          str(s.get("observation") or "")) for s in steps]
labels = sc.classify_trajectory(pairs, backtrack_threshold=3)

D = {p: [] for p in ("loc", "repro", "patch")}
for si in range(len(steps)):
    a0, a1 = spans[2 + 2 * si]
    t0, t1 = a0 - resp_start, a1 - resp_start
    for p in D:
        D[p].append(float((alp["lp_student"][t0:t1] - alp[f"lp_{p}"][t0:t1]).mean()))

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
STAGE_BG = {sc.LOCALIZATION: "#2a78d6", sc.REPRODUCTION: "#1baf7a", sc.PATCH: "#eda100"}
CURVES = [("loc", "Localization teacher", "#2a78d6"),
          ("repro", "Reproduction teacher", "#1baf7a"),
          ("patch", "Patch teacher", "#eda100")]

fig, ax = plt.subplots(figsize=(9.2, 3.9), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
xs = np.arange(len(steps))
# stage bands
prev, start = labels[0], 0
for i in range(1, len(labels) + 1):
    if i == len(labels) or labels[i] != prev:
        ax.axvspan(start - 0.5, i - 0.5, color=STAGE_BG[prev], alpha=0.10, zorder=0)
        prev = labels[i] if i < len(labels) else prev; start = i
for p, lbl, c in CURVES:
    ax.plot(xs, D[p], color=c, lw=2, marker="o", ms=4, label=lbl, zorder=3)
ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
# annotate stage bands at top
seen = set()
prev, start = labels[0], 0
for i in range(1, len(labels) + 1):
    if i == len(labels) or labels[i] != prev:
        name = {sc.LOCALIZATION: "localize", sc.REPRODUCTION: "reproduce",
                sc.PATCH: "patch"}[prev]
        if name not in seen:
            ax.text((start + i - 1) / 2, ax.get_ylim()[1] * 0.02 + max(max(v) for v in D.values()) * 1.02,
                    name, ha="center", fontsize=9, color=STAGE_BG[prev], fontweight="bold")
            seen.add(name)
        if i < len(labels): prev, start = labels[i], i
ax.set_xlabel("agent step", fontsize=10, color=INK)
ax.set_ylabel("teacher divergence from policy\n(nats/token, higher = more to say)",
              fontsize=9, color=INK)
ax.set_title(f"Per-step divergence of each privileged teacher — {IID} (solved)",
             fontsize=11, color=INK, loc="left", pad=12)
ax.legend(fontsize=8.5, frameon=False, loc="upper right")
ax.grid(axis="y", color=GRID, lw=0.7)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.tick_params(colors=MUTED)
fig.tight_layout()
out = f"{ROOT}/analysis/fig_case_study_{IID}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("saved", out)
print("labels:", [l.split("_")[0] for l in labels])
