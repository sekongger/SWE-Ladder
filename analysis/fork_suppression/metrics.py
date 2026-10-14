"""§6 metrics for the fork-suppression analysis: M1-M4 + placebo.

All metrics operate on a flat "positions" table (one row per classified decision
point) with these columns (1-D arrays of equal length N):

    instance_id   : str/int   grouping key for by-instance bootstrap
    pos_class     : str       EXPLORE | COMMIT | SUBMIT | OTHER  (freeze_rules)
    stage         : str       causal_localization | localized_issue_reproduction | patch_generation
    H             : float      student full-vocab entropy at this position
    U_wa          : float      per-position uplift under WA teacher (nan if v* out of top-k)
    U_sr          : float      per-position uplift under SR teacher (nan if v* out of top-k)
    teacher_top1_wa / teacher_top1_sr : float  (for paper fork/lock cross-check)

FROZEN protocol (see freeze_rules.py):
  - fork = student-entropy top-p% within the CURRENT POOL (M2: whole table;
    M4: within each stage). Percentile, not absolute (M3 alone uses absolute).
  - NEP = mean(U | fork & EXPLORE) - mean(U | fork & COMMIT).
  - bootstrap resamples INSTANCES, never tokens (plan §10).
  - placebo re-runs M2/M4 on the BOTTOM 50% entropy positions; expect no asymmetry.
"""

from __future__ import annotations

import numpy as np

from freeze_rules import (  # type: ignore
    EXPLORE, COMMIT,
    paper_teacher_fork_lock,
    student_fork_mask_percentile,
)


# ---------------------------------------------------------------------------
# NEP core
# ---------------------------------------------------------------------------
def _nep(U, pos_class, fork_mask):
    """NEP = mean(U | fork & EXPLORE) - mean(U | fork & COMMIT), ignoring nan U."""
    U = np.asarray(U, dtype=float)
    pc = np.asarray(pos_class)
    fm = np.asarray(fork_mask, dtype=bool)
    expl = fm & (pc == EXPLORE) & np.isfinite(U)
    comm = fm & (pc == COMMIT) & np.isfinite(U)
    if expl.sum() == 0 or comm.sum() == 0:
        return np.nan, int(expl.sum()), int(comm.sum())
    return float(U[expl].mean() - U[comm].mean()), int(expl.sum()), int(comm.sum())


def nep_wa_sr(table, fork_mask):
    """Return dict with NEP(WA), NEP(SR), and delta = NEP(WA)-NEP(SR) on given fork_mask."""
    nep_wa, ne, nc = _nep(table["U_wa"], table["pos_class"], fork_mask)
    nep_sr, _, _ = _nep(table["U_sr"], table["pos_class"], fork_mask)
    delta = nep_wa - nep_sr if (np.isfinite(nep_wa) and np.isfinite(nep_sr)) else np.nan
    return {"nep_wa": nep_wa, "nep_sr": nep_sr, "delta": delta,
            "n_explore": ne, "n_commit": nc}


# ---------------------------------------------------------------------------
# by-instance paired bootstrap
# ---------------------------------------------------------------------------
def paired_bootstrap_by_instance(table, fork_selector, n_boot=2000, seed=0,
                                 statistic="delta"):
    """Percentile CI for a NEP statistic, resampling INSTANCES with replacement.

    fork_selector(sub_table) -> boolean fork mask for that resampled pool. Passing the
    selector (not a precomputed mask) lets the entropy percentile be recomputed on each
    resampled pool, which is the correct paired-bootstrap behavior.

    Returns dict: point (on full data), mean, ci_lo, ci_hi (2.5/97.5), n_boot_valid.
    """
    rng = np.random.default_rng(seed)
    inst = np.asarray(table["instance_id"])
    uniq = np.unique(inst)
    # index lists per instance (built once)
    idx_by_inst = {u: np.where(inst == u)[0] for u in uniq}

    def stat_on(rows):
        sub = {k: np.asarray(v)[rows] for k, v in table.items()}
        fm = fork_selector(sub)
        res = nep_wa_sr(sub, fm)
        return res[statistic]

    point = stat_on(np.arange(len(inst)))
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=uniq.size, replace=True)
        rows = np.concatenate([idx_by_inst[u] for u in chosen])
        val = stat_on(rows)
        if np.isfinite(val):
            boots.append(val)
    boots = np.asarray(boots, dtype=float)
    if boots.size == 0:
        return {"point": point, "mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_boot_valid": 0}
    return {"point": float(point), "mean": float(boots.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)),
            "n_boot_valid": int(boots.size)}


# ---------------------------------------------------------------------------
# fork selectors (frozen)
# ---------------------------------------------------------------------------
def whole_pool_fork_selector(p):
    return lambda sub: student_fork_mask_percentile(sub["H"], p)


def within_stage_fork_selector(p, stage):
    def sel(sub):
        in_stage = np.asarray(sub["stage"]) == stage
        mask = np.zeros(len(in_stage), dtype=bool)
        if in_stage.sum() > 0:
            sub_mask = student_fork_mask_percentile(np.asarray(sub["H"])[in_stage], p)
            mask[np.where(in_stage)[0]] = sub_mask
        return mask
    return sel


def bottom50_selector(_p=None):
    """Placebo: bottom 50% entropy positions (low-entropy, non-fork)."""
    def sel(sub):
        h = np.asarray(sub["H"], dtype=float)
        if h.size == 0:
            return np.zeros(0, dtype=bool)
        med = np.percentile(h, 50)
        return h <= med
    return sel


# ---------------------------------------------------------------------------
# M1 descriptive
# ---------------------------------------------------------------------------
def m1_descriptive(table):
    pc = np.asarray(table["pos_class"])
    H = np.asarray(table["H"], dtype=float)
    out = {"n_total": int(len(H)), "by_class": {}}
    for cls in np.unique(pc):
        m = pc == cls
        out["by_class"][str(cls)] = {
            "n": int(m.sum()),
            "H_mean": float(H[m].mean()) if m.any() else np.nan,
            "U_wa_mean": float(np.nanmean(np.asarray(table["U_wa"])[m])) if m.any() else np.nan,
            "U_sr_mean": float(np.nanmean(np.asarray(table["U_sr"])[m])) if m.any() else np.nan,
        }
    # paper cross-check fork/lock rates (both teachers)
    for tag in ("wa", "sr"):
        f, l = paper_teacher_fork_lock(table[f"teacher_top1_{tag}"])
        out[f"paper_fork_rate_{tag}"] = float(np.mean(f)) if len(f) else np.nan
        out[f"paper_lock_rate_{tag}"] = float(np.mean(l)) if len(l) else np.nan
    return out


# ---------------------------------------------------------------------------
# M2 / M4 / placebo drivers
# ---------------------------------------------------------------------------
def m2_nep(table, percentiles, n_boot=2000, seed=0):
    """NEP(WA) vs NEP(SR) on whole-pool forks, per percentile, with by-instance CI on delta."""
    out = {}
    for p in percentiles:
        sel = whole_pool_fork_selector(p)
        pt = nep_wa_sr(table, sel(table))
        bs = paired_bootstrap_by_instance(table, sel, n_boot=n_boot, seed=seed, statistic="delta")
        out[f"p{p}"] = {**pt, "delta_ci": bs}
    return out


def m4_nep_by_stage(table, stages, percentiles, n_boot=2000, seed=0):
    """The KEY discriminative test: NEP by stage. Mismatch -> harm concentrated in
    LOC/REPRO; too-strong teacher -> uniform across stages."""
    out = {}
    for stage in stages:
        out[stage] = {}
        for p in percentiles:
            sel = within_stage_fork_selector(p, stage)
            pt = nep_wa_sr(table, sel(table))
            bs = paired_bootstrap_by_instance(table, sel, n_boot=n_boot, seed=seed, statistic="delta")
            out[stage][f"p{p}"] = {**pt, "delta_ci": bs}
    return out


def placebo(table, stages, percentile_for_report=10, n_boot=2000, seed=0):
    """Re-run M2 & M4 on BOTTOM 50% entropy positions; expect NO asymmetry (delta~0)."""
    sel = bottom50_selector()
    m2 = {**nep_wa_sr(table, sel(table)),
          "delta_ci": paired_bootstrap_by_instance(table, sel, n_boot=n_boot, seed=seed, statistic="delta")}
    m4 = {}
    for stage in stages:
        def stage_sel(sub, _stage=stage):
            in_stage = np.asarray(sub["stage"]) == _stage
            mask = bottom50_selector()(sub)
            return mask & in_stage
        m4[stage] = {**nep_wa_sr(table, stage_sel(table)),
                     "delta_ci": paired_bootstrap_by_instance(table, stage_sel, n_boot=n_boot, seed=seed, statistic="delta")}
    return {"m2_placebo": m2, "m4_placebo": m4}


# ---------------------------------------------------------------------------
# VARIANT B (user-authorized deviation 2026-08-12): entropy-contrast NEP.
#
# The frozen NEP (mean U over EXPLORE-fork − mean U over COMMIT-fork) is structurally
# uncomputable at high-entropy percentiles: COMMIT positions are near-deterministic
# (H_mean≈0.002) so they never enter the top-p% entropy fork set (loc stage has 0 COMMIT
# at all). Variant B keeps NEP's "is exploration suppressed more than commitment" semantics
# but redefines the two cells by ENTROPY, so both are populated:
#   top  cell = top-p% student-entropy positions (the high-uncertainty "fork" / exploration end)
#   base cell = bottom-`base_pct`% student-entropy positions (the low-uncertainty "commitment" end)
# both taken WITHIN the current pool (M2) or WITHIN a stage (M4). NEP_B = U(top) − U(base);
# delta = NEP_B(WA) − NEP_B(SR). At PATCH, WA≡SR by construction ⇒ delta==0 (self-consistency).
# ---------------------------------------------------------------------------
def _nep_contrast(U, top_mask, base_mask):
    """NEP_B = mean(U | top-entropy) − mean(U | bottom-entropy), ignoring nan U."""
    U = np.asarray(U, dtype=float)
    t = np.asarray(top_mask, dtype=bool) & np.isfinite(U)
    b = np.asarray(base_mask, dtype=bool) & np.isfinite(U)
    if t.sum() == 0 or b.sum() == 0:
        return np.nan, int(t.sum()), int(b.sum())
    return float(U[t].mean() - U[b].mean()), int(t.sum()), int(b.sum())


def nep_wa_sr_contrast(table, top_mask, base_mask):
    nw, nt, nb = _nep_contrast(table["U_wa"], top_mask, base_mask)
    ns, _, _ = _nep_contrast(table["U_sr"], top_mask, base_mask)
    delta = nw - ns if (np.isfinite(nw) and np.isfinite(ns)) else np.nan
    return {"nep_wa": nw, "nep_sr": ns, "delta": delta, "n_top": nt, "n_base": nb}


def contrast_selector_whole(p, base_pct=50):
    def sel(sub):
        H = np.asarray(sub["H"], float)
        if H.size == 0:
            return np.zeros(0, bool), np.zeros(0, bool)
        top = student_fork_mask_percentile(H, p)
        base = H <= np.percentile(H, base_pct)
        return top, base
    return sel


def contrast_selector_stage(p, stage, base_pct=50):
    def sel(sub):
        st = np.asarray(sub["stage"])
        in_stage = st == stage
        n = len(st)
        top = np.zeros(n, bool)
        base = np.zeros(n, bool)
        if in_stage.sum() > 0:
            idx = np.where(in_stage)[0]
            H = np.asarray(sub["H"], float)[in_stage]
            top[idx] = student_fork_mask_percentile(H, p)
            base[idx] = H <= np.percentile(H, base_pct)
        return top, base
    return sel


def paired_bootstrap_contrast(table, sel, n_boot=2000, seed=0):
    """By-instance percentile CI on delta for a two-mask (top,base) selector. Mirrors
    paired_bootstrap_by_instance but the selector returns (top_mask, base_mask)."""
    rng = np.random.default_rng(seed)
    inst = np.asarray(table["instance_id"])
    uniq = np.unique(inst)
    idx_by_inst = {u: np.where(inst == u)[0] for u in uniq}

    def stat_on(rows):
        sub = {k: np.asarray(v)[rows] for k, v in table.items()}
        top, base = sel(sub)
        return nep_wa_sr_contrast(sub, top, base)["delta"]

    point = stat_on(np.arange(len(inst)))
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=uniq.size, replace=True)
        rows = np.concatenate([idx_by_inst[u] for u in chosen])
        val = stat_on(rows)
        if np.isfinite(val):
            boots.append(val)
    boots = np.asarray(boots, dtype=float)
    if boots.size == 0:
        return {"point": point, "mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_boot_valid": 0}
    return {"point": float(point), "mean": float(boots.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)),
            "n_boot_valid": int(boots.size)}


def m2_nep_contrast(table, percentiles, n_boot=2000, seed=0, base_pct=50):
    out = {}
    for p in percentiles:
        sel = contrast_selector_whole(p, base_pct)
        top, base = sel(table)
        pt = nep_wa_sr_contrast(table, top, base)
        bs = paired_bootstrap_contrast(table, sel, n_boot=n_boot, seed=seed)
        out[f"p{p}"] = {**pt, "delta_ci": bs}
    return out


def m4_nep_by_stage_contrast(table, stages, percentiles, n_boot=2000, seed=0, base_pct=50):
    out = {}
    for stage in stages:
        out[stage] = {}
        for p in percentiles:
            sel = contrast_selector_stage(p, stage, base_pct)
            top, base = sel(table)
            pt = nep_wa_sr_contrast(table, top, base)
            bs = paired_bootstrap_contrast(table, sel, n_boot=n_boot, seed=seed)
            out[stage][f"p{p}"] = {**pt, "delta_ci": bs}
    return out


def placebo_contrast(table, stages, n_boot=2000, seed=0):
    """Negative control for variant B: WITHIN the low-entropy region (bottom 50%), contrast its
    upper half (25-50th pct) against its lower half (bottom 25th). Both cells are low-entropy
    'commitment' positions with no genuine fork, so the WA-vs-SR delta should be ~0."""
    def sel_whole(sub):
        H = np.asarray(sub["H"], float)
        if H.size == 0:
            return np.zeros(0, bool), np.zeros(0, bool)
        q25, q50 = np.percentile(H, 25), np.percentile(H, 50)
        return (H > q25) & (H <= q50), H <= q25

    m2 = {**nep_wa_sr_contrast(table, *sel_whole(table)),
          "delta_ci": paired_bootstrap_contrast(table, sel_whole, n_boot=n_boot, seed=seed)}
    m4 = {}
    for stage in stages:
        def sel_stage(sub, _stage=stage):
            st = np.asarray(sub["stage"])
            in_stage = st == _stage
            n = len(st)
            top = np.zeros(n, bool)
            base = np.zeros(n, bool)
            if in_stage.sum() > 0:
                idx = np.where(in_stage)[0]
                H = np.asarray(sub["H"], float)[in_stage]
                q25, q50 = np.percentile(H, 25), np.percentile(H, 50)
                top[idx] = (H > q25) & (H <= q50)
                base[idx] = H <= q25
            return top, base
        m4[stage] = {**nep_wa_sr_contrast(table, *sel_stage(table)),
                     "delta_ci": paired_bootstrap_contrast(table, sel_stage, n_boot=n_boot, seed=seed)}
    return {"m2_placebo": m2, "m4_placebo": m4}


# ---------------------------------------------------------------------------
# M3 fork-rate (absolute threshold) — used by run.py per checkpoint
# ---------------------------------------------------------------------------
def fork_rate_absolute(entropies, abs_threshold):
    h = np.asarray(entropies, dtype=float)
    if h.size == 0:
        return np.nan
    return float(np.mean(h >= abs_threshold))
