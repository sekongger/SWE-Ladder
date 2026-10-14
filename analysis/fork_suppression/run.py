"""§6/§9 driver: build the metrics table from forwards output, run M1-M4 + placebo,
enforce the P3 gate FIRST, and assemble results.json.

Execution order (plan §9):
  P3 gate (must pass) -> M1 descriptive -> M2 (whole-pool NEP) -> M4 (by-stage NEP, key test)
  -> placebo (bottom-50% entropy) -> M3 (fork-rate vs step, separate forwards per ckpt).

P3 gate (plan §7, the built-in bug detector): at the PATCH stage the SR teacher context IS the
WA context (both = force-PATCH render, identical bytes), so NEP(WA)-NEP(SR) must be ~0 at PATCH.
A non-trivial PATCH delta means the pipeline mislabels stages, misroutes teachers, or misaligns
positions -- every other number is then untrustworthy. We STOP and report if it fails.

This module is CPU-only (numpy); it consumes the pickle produced by forwards.py.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

from forwards import compute_uplift_table  # type: ignore  # noqa: E402
import metrics as M  # type: ignore  # noqa: E402

STAGES = ["causal_localization", "localized_issue_reproduction", "patch_generation"]
PERCENTILES = (5, 10, 20)
P3_EPS = 1e-6  # PATCH NEP(WA)-NEP(SR) must be numerically ~0 (identical teacher context)


def _load_rows(forwards_pickle):
    with open(forwards_pickle, "rb") as f:
        d = pickle.load(f)
    return d["rows"], d["meta"]


def _table(rows, k, normalize):
    return compute_uplift_table(rows, k=k, normalize_teacher_topk=normalize)


# ---------------------------------------------------------------------------
# P3 gate
# ---------------------------------------------------------------------------
def p3_gate(table, percentiles=PERCENTILES):
    """Return (passed, detail). At PATCH stage WA==SR context => delta must be ~0.

    We also directly check the raw per-position identity: U_wa == U_sr for PATCH positions
    (up to float tolerance), which is a stronger statement than NEP(delta)~0.
    """
    stage = np.asarray(table["stage"])
    patch = stage == "patch_generation"
    U_wa = np.asarray(table["U_wa"], float)
    U_sr = np.asarray(table["U_sr"], float)
    both = patch & np.isfinite(U_wa) & np.isfinite(U_sr)
    n = int(both.sum())
    max_abs_diff = float(np.max(np.abs(U_wa[both] - U_sr[both]))) if n else float("nan")

    nep_delta = {}
    worst = 0.0
    for p in percentiles:
        sel = M.within_stage_fork_selector(p, "patch_generation")
        res = M.nep_wa_sr(table, sel(table))
        d = res["delta"]
        nep_delta[f"p{p}"] = res
        if np.isfinite(d):
            worst = max(worst, abs(d))

    passed = (np.isfinite(max_abs_diff) and max_abs_diff <= 1e-4) and (worst <= 1e-4)
    return passed, {
        "n_patch_positions_in_support_both": n,
        "max_abs_U_wa_minus_U_sr_patch": max_abs_diff,
        "worst_abs_nep_delta_patch": worst,
        "nep_delta_patch_by_percentile": nep_delta,
        "eps": P3_EPS,
        "note": "SR patch context == WA (force-PATCH) render by construction; delta must be ~0.",
    }


# ---------------------------------------------------------------------------
# full analysis for one condition (one forwards pickle)
# ---------------------------------------------------------------------------
def analyze_condition(forwards_pickle, k_primary=32, k_robust=16,
                      n_boot=2000, seed=0, normalize=True):
    rows, meta = _load_rows(forwards_pickle)
    out = {"forwards_meta": meta, "k_primary": k_primary, "k_robust": k_robust,
           "normalize_teacher_topk": normalize, "n_boot": n_boot, "seed": seed}

    for kname, k in (("k32", k_primary), ("k16", k_robust)):
        table = _table(rows, k, normalize)
        n_support_wa = int(np.isfinite(np.asarray(table["U_wa"], float)).sum())
        n_support_sr = int(np.isfinite(np.asarray(table["U_sr"], float)).sum())
        n_total = int(len(table["U_wa"]))

        passed, p3 = p3_gate(table)
        block = {
            "n_positions": n_total,
            "n_in_support_wa": n_support_wa, "n_in_support_sr": n_support_sr,
            "excluded_fraction_wa": 1 - n_support_wa / n_total if n_total else float("nan"),
            "excluded_fraction_sr": 1 - n_support_sr / n_total if n_total else float("nan"),
            "P3_gate_passed": bool(passed),
            "P3_gate": p3,
        }
        if passed:
            block["M1"] = M.m1_descriptive(table)
            # PRIMARY metric = variant B (user-authorized 2026-08-12): entropy-contrast NEP,
            # U(top-p% entropy fork) − U(bottom-50% entropy baseline), within pool/stage.
            # The frozen explore-vs-commit NEP is structurally uncomputable at high entropy
            # (COMMIT positions never enter the high-entropy fork set); see metrics.py header.
            block["nep_variant"] = (
                "B_entropy_contrast: NEP = U(top-p% entropy) - U(bottom-50% entropy), "
                "within pool(M2)/stage(M4); replaces frozen EXPLORE-vs-COMMIT cells because "
                "COMMIT is near-deterministic and never reaches the high-entropy fork.")
            block["M2"] = M.m2_nep_contrast(table, PERCENTILES, n_boot=n_boot, seed=seed)
            block["M4"] = M.m4_nep_by_stage_contrast(table, STAGES, PERCENTILES, n_boot=n_boot, seed=seed)
            block["placebo"] = M.placebo_contrast(table, STAGES, n_boot=n_boot, seed=seed)
            # frozen explore-vs-commit NEP kept on record (point estimates only; mostly NaN by
            # construction) so the deviation is fully auditable.
            block["M2_frozen_explore_vs_commit"] = {
                f"p{p}": M.nep_wa_sr(table, M.whole_pool_fork_selector(p)(table)) for p in PERCENTILES}
            block["M4_frozen_explore_vs_commit"] = {
                stg: {f"p{p}": M.nep_wa_sr(table, M.within_stage_fork_selector(p, stg)(table))
                      for p in PERCENTILES} for stg in STAGES}
        else:
            block["ABORTED"] = "P3 gate failed; downstream metrics suppressed (plan §9)."
        out[kname] = block

    return out


# ---------------------------------------------------------------------------
# M3: fork-rate vs training step (needs one forwards pickle per ckpt)
# ---------------------------------------------------------------------------
def m3_fork_rate(step_to_pickle: dict, base_key="base", percentile=90, normalize=True):
    """Absolute-threshold fork rate vs step. Threshold calibrated ONCE on base (its `percentile`th
    student-entropy), then FIXED across steps (plan §5: M3 uses absolute, never percentile)."""
    tables = {}
    for step, pkl in step_to_pickle.items():
        rows, _ = _load_rows(pkl)
        tables[step] = _table(rows, 32, normalize)

    if base_key not in tables:
        raise ValueError(f"M3 needs a '{base_key}' condition to calibrate the absolute threshold")
    base_H = np.asarray(tables[base_key]["H"], float)
    abs_thr = float(np.percentile(base_H, percentile))

    out = {"abs_threshold": abs_thr, "calibrated_on": base_key,
           "calibration_percentile": percentile, "by_step": {}}
    for step, table in tables.items():
        H = np.asarray(table["H"], float)
        fr = M.fork_rate_absolute(H, abs_thr)
        # paper cross-check fork/lock rates
        m1 = M.m1_descriptive(table)
        out["by_step"][str(step)] = {
            "student_entropy_fork_rate_abs": fr,
            "paper_fork_rate_wa": m1.get("paper_fork_rate_wa"),
            "paper_fork_rate_sr": m1.get("paper_fork_rate_sr"),
            "paper_lock_rate_wa": m1.get("paper_lock_rate_wa"),
            "paper_lock_rate_sr": m1.get("paper_lock_rate_sr"),
            "H_mean": float(H.mean()) if H.size else float("nan"),
            "n_positions": int(H.size),
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fork-suppression metrics driver (CPU).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="M1-M4 + placebo + P3 gate for one condition")
    a.add_argument("--forwards", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--n-boot", type=int, default=2000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--no-normalize", action="store_true")

    g = sub.add_parser("p3gate", help="quick P3 gate check only")
    g.add_argument("--forwards", required=True)
    g.add_argument("--no-normalize", action="store_true")

    m = sub.add_parser("m3", help="fork-rate vs step; --step base=path --step 30=path ...")
    m.add_argument("--step", action="append", required=True, metavar="KEY=PATH")
    m.add_argument("--base-key", default="base")
    m.add_argument("--percentile", type=int, default=90)
    m.add_argument("--out", required=True)
    m.add_argument("--no-normalize", action="store_true")

    args = ap.parse_args()

    if args.cmd == "analyze":
        res = analyze_condition(args.forwards, n_boot=args.n_boot, seed=args.seed,
                                normalize=not args.no_normalize)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, default=str)
        passed = res["k32"]["P3_gate_passed"]
        print(f"P3 gate (k32): {'PASS' if passed else 'FAIL'}")
        print(json.dumps(res["k32"]["P3_gate"], indent=2, default=str))
        if not passed:
            print("\n*** P3 GATE FAILED — downstream numbers are NOT trustworthy (plan §9). ***")
            sys.exit(2)

    elif args.cmd == "p3gate":
        rows, meta = _load_rows(args.forwards)
        table = _table(rows, 32, not args.no_normalize)
        passed, detail = p3_gate(table)
        print(f"P3 gate: {'PASS' if passed else 'FAIL'}")
        print(json.dumps(detail, indent=2, default=str))
        sys.exit(0 if passed else 2)

    elif args.cmd == "m3":
        step_map = {}
        for kv in args.step:
            key, path = kv.split("=", 1)
            step_map[key] = path
        res = m3_fork_rate(step_map, base_key=args.base_key,
                           percentile=args.percentile, normalize=not args.no_normalize)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
