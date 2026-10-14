"""Merge several data_gen pickles (one per training step) into one trajectory pool.

Rationale (fidelity note, -> REPORT §10): each step file dumps a GRPO group of ~16 distinct
instances x 4 rollouts. Step 30 alone (fully on-policy: trajectories from the gs30 policy) yields
16 instances -- enough for a by-instance bootstrap but modest. To widen the instance pool we may
aggregate ADJACENT steps (e.g. 28-33). We keep ONE fixed student (gs30) as the measurement
instrument for the subsequent forwards pass; scoring adjacent-step trajectories with the gs30
student is coherent (the uplift/entropy are defined w.r.t. a fixed reference policy), with the
only cost being that off-step trajectories were not sampled from exactly gs30. Policy drift over
+/-3 steps is small (smooth reward curve), so this is a documented minor deviation, not a
confound of the WA-vs-SR contrast (which is within-trajectory, teacher-context-only).

Instance disjointness across steps is NOT guaranteed by construction, so instance_id remains the
bootstrap unit: if the same instance recurs in two steps its rows simply pool under one id.
"""

from __future__ import annotations

import argparse
import pickle
import sys

# unpickle stand-ins (data_gen pickles its dataclasses under __main__)
import __main__ as _main


class ReconTrajectory:  # noqa: D401 - stand-in
    pass


class Position:  # noqa: D401 - stand-in
    pass


for _n, _c in (("ReconTrajectory", ReconTrajectory), ("Position", Position)):
    if not hasattr(_main, _n):
        setattr(_main, _n, _c)


def main():
    ap = argparse.ArgumentParser(description="Merge data_gen pickles into one trajectory pool.")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="data_gen pickles")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    trajs = []
    metas = []
    for p in args.inputs:
        with open(p, "rb") as f:
            d = pickle.load(f)
        trajs.extend(d["trajectories"])
        metas.append({"src": p, **d.get("meta", {})})

    matched = [t for t in trajs if getattr(t, "matched", False)]
    inst = sorted({getattr(t, "instance_id", None) for t in matched})
    meta = {
        "merged_from": metas,
        "n_trajectories": len(trajs),
        "n_matched": len(matched),
        "n_distinct_instances": len([i for i in inst if i]),
    }
    with open(args.out, "wb") as f:
        pickle.dump({"meta": meta, "trajectories": trajs}, f)
    print(f"merged {len(args.inputs)} pickles -> {len(trajs)} trajs, "
          f"{meta['n_distinct_instances']} distinct instances", file=sys.stderr)
    print(meta)


if __name__ == "__main__":
    main()
