"""Aggregate step-probe results: per-condition gold-contact rates + paired tests."""
import glob, json
import numpy as np
import os
from pathlib import Path

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree
# that holds output/ and the model weights; it defaults to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
CONDS = ("none", "loc", "repro", "patch")

rows = [json.load(open(f)) for f in sorted(glob.glob(f"{ROOT}/analysis/probe/results/*.json"))]
print(f"probe moments: {len(rows)}")

# per-moment hit rates
H = {c: np.array([r["conditions"][c]["hit_rate"] for r in rows]) for c in CONDS}
print("\n=== gold-contact rate (mean over moments; 8 samples each) ===")
for c in CONDS:
    print(f"  {c:6s}: {H[c].mean():.3f}")

try:
    from scipy import stats
    print("\n=== paired wilcoxon ===")
    for a, b, alt in [("loc", "none", "greater"), ("loc", "patch", "greater"),
                      ("loc", "repro", "greater"), ("patch", "none", "greater"),
                      ("repro", "none", "less")]:
        d = H[a] - H[b]
        nz = d[d != 0]
        if len(nz) < 5:
            print(f"  {a} vs {b}: too few non-ties"); continue
        p = stats.wilcoxon(H[a], H[b], alternative=alt).pvalue
        print(f"  {a:5s} vs {b:5s} ({alt:7s}): mean diff {d.mean():+.3f}  p={p:.4f}")
except ImportError:
    print("(scipy unavailable here — run with env python for tests)")

# behavioral side-effect: action-kind mix per condition
print("\n=== action-kind distribution per condition ===")
for c in CONDS:
    kinds = {}
    tot = 0
    for r in rows:
        for s in r["conditions"][c]["samples"]:
            kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1; tot += 1
    mix = {k: round(v / tot, 3) for k, v in sorted(kinds.items(), key=lambda x: -x[1])}
    print(f"  {c:6s}: {mix}")

json.dump({c: H[c].tolist() for c in CONDS},
          open(f"{ROOT}/analysis/probe/hit_rates.json", "w"))
print("\nwrote hit_rates.json")
