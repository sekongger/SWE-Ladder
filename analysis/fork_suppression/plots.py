"""§8 figures for the fork-suppression report. Pure matplotlib, PDF out.

Four figures:
  fig_nep_by_stage.pdf        (MAIN) NEP(WA) vs NEP(SR) per stage, with by-instance CIs;
                              PATCH bars must coincide (P3). Reads results.json M4 block.
  fig_fork_rate_training.pdf  fork-rate vs training step (student-entropy abs threshold +
                              paper teacher-top1 cross-check). Reads m3.json.
  fig_entropy_hist.pdf        student full-vocab entropy H by pos_class (EXPLORE/COMMIT),
                              with the M2 percentile thresholds marked. Reads a forwards pickle.
  fig_threshold_robustness.pdf NEP delta (WA-SR) vs percentile p in {5,10,20}, per stage.
                              Reads results.json M4 block.

All figures are descriptive; captions must avoid causal language (plan §10).
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
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STAGES = ["causal_localization", "localized_issue_reproduction", "patch_generation"]
STAGE_SHORT = {"causal_localization": "LOC", "localized_issue_reproduction": "REPRO",
               "patch_generation": "PATCH"}
PERCENTILES = (5, 10, 20)


def _err_from_ci(point, ci):
    lo = ci.get("ci_lo", np.nan)
    hi = ci.get("ci_hi", np.nan)
    if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(point)):
        return np.nan, np.nan
    return max(0.0, point - lo), max(0.0, hi - point)


# ---------------------------------------------------------------------------
def fig_nep_by_stage(results, out_path, percentile=10, kname="k32"):
    """MAIN figure: grouped bars NEP(WA)/NEP(SR) per stage at a chosen percentile."""
    m4 = results[kname]["M4"]
    labels, wa, sr, wa_err, sr_err, deltas = [], [], [], [], [], []
    for stg in STAGES:
        cell = m4[stg][f"p{percentile}"]
        labels.append(STAGE_SHORT[stg])
        wa.append(cell["nep_wa"]); sr.append(cell["nep_sr"])
        deltas.append(cell["delta"])
        # CI is on delta; show as an annotation, and use symmetric bars w/o err if only delta CI
        wa_err.append(np.nan); sr_err.append(np.nan)

    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - w/2, wa, w, label="WA (whole-answer / full-privilege)", color="#c44e52")
    ax.bar(x + w/2, sr, w, label="SR (stage-routed)", color="#4c72b0")
    ax.axhline(0, color="k", lw=0.7)
    for i, d in enumerate(deltas):
        cell = m4[STAGES[i]][f"p{percentile}"]["delta_ci"]
        lo, hi = cell.get("ci_lo", np.nan), cell.get("ci_hi", np.nan)
        txt = f"Δ={d:.2e}" if np.isfinite(d) else "Δ=NA"
        if np.isfinite(lo) and np.isfinite(hi):
            txt += f"\n[{lo:.1e},{hi:.1e}]"
        ax.annotate(txt, (x[i], max(wa[i], sr[i])), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("NEP$_B$ = U(top-p% entropy fork) − U(bottom-50% entropy baseline)")
    ax.set_title(f"NEP$_B$ by stage (fork = student-entropy top-{percentile}%, {kname})\n"
                 "PATCH: SR context ≡ WA by construction (P3 self-test ⇒ Δ=0)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def fig_threshold_robustness(results, out_path, kname="k32"):
    """NEP delta (WA-SR) vs percentile per stage, with CIs."""
    m4 = results[kname]["M4"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for stg in STAGES:
        ys, los, his = [], [], []
        for p in PERCENTILES:
            cell = m4[stg][f"p{p}"]
            ys.append(cell["delta"])
            ci = cell["delta_ci"]
            los.append(ci.get("ci_lo", np.nan)); his.append(ci.get("ci_hi", np.nan))
        ys = np.array(ys, float)
        yerr = np.vstack([ys - np.array(los), np.array(his) - ys])
        yerr = np.clip(yerr, 0, None)
        ax.errorbar(PERCENTILES, ys, yerr=yerr, marker="o", capsize=3, label=STAGE_SHORT[stg])
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xlabel("fork percentile p (top-p% student entropy)")
    ax.set_ylabel("NEP$_B$(WA) − NEP$_B$(SR)")
    ax.set_title(f"Threshold robustness of the WA−SR NEP$_B$ gap ({kname})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def fig_fork_rate_training(m3, out_path):
    by_step = m3["by_step"]
    def _num(k):
        try:
            return int(k)
        except ValueError:
            return -1
    steps = sorted(by_step.keys(), key=_num)
    xs = [_num(s) if _num(s) >= 0 else 0 for s in steps]
    stu = [by_step[s]["student_entropy_fork_rate_abs"] for s in steps]
    pf_wa = [by_step[s]["paper_fork_rate_wa"] for s in steps]
    pf_sr = [by_step[s]["paper_fork_rate_sr"] for s in steps]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(xs, stu, marker="o", label=f"student-entropy fork rate (abs thr={m3['abs_threshold']:.2f})")
    ax.plot(xs, pf_wa, marker="s", ls="--", label="paper fork rate (WA teacher top1<0.25)")
    ax.plot(xs, pf_sr, marker="^", ls="--", label="paper fork rate (SR teacher top1<0.25)")
    ax.set_xlabel("training step"); ax.set_ylabel("fork rate")
    ax.set_title("Fork rate vs training step\n(instruction-model fork rates are flat by nature — see caveat D4)")
    ax.set_xticks(xs); ax.set_xticklabels(steps)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def fig_entropy_hist(forwards_pickle, out_path, percentiles=PERCENTILES):
    from forwards import compute_uplift_table  # local import (numpy only)
    with open(forwards_pickle, "rb") as f:
        d = pickle.load(f)
    table = compute_uplift_table(d["rows"], k=32, normalize_teacher_topk=True)
    H = np.asarray(table["H"], float)
    pc = np.asarray(table["pos_class"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.linspace(float(np.nanmin(H)), float(np.nanmax(H)), 40) if H.size else 40
    for cls, color in (("EXPLORE", "#55a868"), ("COMMIT", "#c44e52"), ("OTHER", "#8172b3"),
                       ("SUBMIT", "#ccb974")):
        m = pc == cls
        if m.sum() > 0:
            ax.hist(H[m], bins=bins, alpha=0.5, label=f"{cls} (n={int(m.sum())})", color=color)
    for p in percentiles:
        thr = float(np.percentile(H, 100 - p)) if H.size else np.nan
        ax.axvline(thr, ls="--", lw=0.9, color="k")
        ax.text(thr, ax.get_ylim()[1]*0.9, f"top-{p}%", rotation=90, fontsize=7, va="top")
    ax.set_xlabel("student full-vocab entropy H_t (nats)")
    ax.set_ylabel("count")
    ax.set_title("Decision-point entropy by position class (fork thresholds marked)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Render fork-suppression figures.")
    ap.add_argument("--results", help="analyze results.json (for nep_by_stage + robustness)")
    ap.add_argument("--m3", help="m3.json (for fork_rate_training)")
    ap.add_argument("--forwards", help="a forwards pickle (for entropy_hist)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--percentile", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.results:
        with open(args.results) as f:
            results = json.load(f)
        kname = "k32" if "k32" in results and results["k32"].get("P3_gate_passed") else "k32"
        if results.get(kname, {}).get("M4"):
            fig_nep_by_stage(results, os.path.join(args.outdir, "fig_nep_by_stage.pdf"),
                             percentile=args.percentile, kname=kname)
            fig_threshold_robustness(results, os.path.join(args.outdir, "fig_threshold_robustness.pdf"),
                                     kname=kname)
            print("wrote fig_nep_by_stage.pdf, fig_threshold_robustness.pdf")
        else:
            print("results has no M4 (P3 gate failed?) — skipping NEP figures")
    if args.m3:
        with open(args.m3) as f:
            m3 = json.load(f)
        fig_fork_rate_training(m3, os.path.join(args.outdir, "fig_fork_rate_training.pdf"))
        print("wrote fig_fork_rate_training.pdf")
    if args.forwards:
        fig_entropy_hist(args.forwards, os.path.join(args.outdir, "fig_entropy_hist.pdf"))
        print("wrote fig_entropy_hist.pdf")


if __name__ == "__main__":
    main()
