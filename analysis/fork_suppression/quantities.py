"""§3 core quantities for the fork-suppression analysis.

Everything is derived from the SAME two top-k tensors the OPSD loss consumes
(swe_ladder/topk_loss.py::teacher_topk_reverse_kl), so the numbers here are the loss's
own internals, not a re-derivation:

    student_topk_log_probs[t,v]  = student FULL-VOCAB log p_theta at teacher top-k id v
    teacher_topk_log_probs[t,v]  = teacher FULL-VOCAB log-prob at the same id v

Definitions (doc §3), with the deliberate asymmetry PRESERVED:
    q~_t(v)      = softmax over the top-k slice of teacher log-probs   (RENORMALIZED)
    p_theta(v)   = exp(student_topk_log_probs)                          (NOT renormalized)
    c_t(v)       = log p_theta(v) - log q~_t(v)          = log_ratio
    c_bar_t      = sum_v p_theta(v) c_t(v)               = per_token_kl (the OPSD reverse-KL)
    uplift_t(v)  = p_theta(v) * (c_bar_t - c_t(v))       <-- MAIN metric

We report BOTH the unnormalized uplift (main table) and the sum|uplift|-normalized
version. We do NOT "fix" the p/q asymmetry (plan §3, §10).

============================================================================
FROZEN metric-definition decision (before any result; see freeze_rules.py header):
The per-POSITION scalar U used by NEP (M2/M4) is the uplift evaluated at the
token the student ACTUALLY emitted at that decision point (v* = emitted id),
restricted to teacher top-k support. If v* is not among the teacher's top-k ids,
the OPSD loss puts ~0 weight there, so U is undefined and the position is EXCLUDED
from NEP; the excluded fraction is reported. Rationale: "does the teacher suppress
the exploratory choice the student made" is a statement about v*, and the loss only
reaches the teacher top-k support.
============================================================================
"""

from __future__ import annotations

import numpy as np

NEG_BIG = -1e30


def _safe(a):
    a = np.asarray(a, dtype=np.float64)
    return np.where(np.isfinite(a), a, NEG_BIG)


def per_token_quantities(student_topk_log_probs, teacher_topk_log_probs,
                         normalize_teacher_topk: bool = True):
    """Return dict of per-token / per-(token,v) arrays mirroring teacher_topk_reverse_kl.

    Shapes: inputs [T, k]. Returns:
        log_ratio   [T,k]  = c_t(v)
        c_bar       [T]     = c_bar_t = per_token_kl (reverse KL per token)
        student_probs [T,k] = p_theta(v)  (unnormalized full-vocab prob at top-k ids)
        uplift      [T,k]  = p_theta(v)*(c_bar_t - c_t(v))
        uplift_norm [T,k]  = uplift / sum_v|uplift|  (per token; 0 if denom 0)
        teacher_topk_mass [T] = sum_v exp(teacher_log_prob(v))  (coverage diagnostic)
    """
    s = _safe(student_topk_log_probs)
    q = _safe(teacher_topk_log_probs)
    finite = np.isfinite(np.asarray(student_topk_log_probs, dtype=np.float64)) & \
             np.isfinite(np.asarray(teacher_topk_log_probs, dtype=np.float64))

    if normalize_teacher_topk:
        # log_softmax over the top-k slice (RENORMALIZE teacher)
        m = q.max(axis=-1, keepdims=True)
        lse = m.squeeze(-1) + np.log(np.exp(q - m).sum(axis=-1))
        teacher_log_for_kl = q - lse[:, None]
    else:
        teacher_log_for_kl = q
    teacher_topk_mass = np.exp(q).sum(axis=-1)

    student_probs = np.where(finite, np.exp(s), 0.0)          # p_theta, NOT renormalized
    log_ratio = np.where(finite, s - teacher_log_for_kl, 0.0)  # c_t(v)
    c_bar = (student_probs * log_ratio).sum(axis=-1)           # c_bar_t = per_token_kl

    uplift = student_probs * (c_bar[:, None] - log_ratio)      # MAIN
    denom = np.abs(uplift).sum(axis=-1)
    uplift_norm = np.where(denom[:, None] > 0, uplift / np.where(denom[:, None] > 0, denom[:, None], 1.0), 0.0)

    return {
        "log_ratio": log_ratio,
        "c_bar": c_bar,
        "student_probs": student_probs,
        "uplift": uplift,
        "uplift_norm": uplift_norm,
        "teacher_topk_mass": teacher_topk_mass,
    }


def position_uplift_on_emitted(uplift, teacher_topk_token_ids, emitted_token_ids):
    """FROZEN U: uplift on the emitted token v*, restricted to teacher top-k support.

    Args:
        uplift              [T,k]   from per_token_quantities
        teacher_topk_token_ids [T,k] the teacher top-k vocab ids at each position
        emitted_token_ids   [T]     the token the student actually emitted at t
    Returns:
        U            [T]  uplift on v* (nan where v* not in teacher top-k)
        in_support   [T]  bool, whether v* was in the teacher top-k
    """
    uplift = np.asarray(uplift, dtype=np.float64)
    ids = np.asarray(teacher_topk_token_ids)
    emit = np.asarray(emitted_token_ids)
    T, k = uplift.shape
    U = np.full(T, np.nan, dtype=np.float64)
    in_support = np.zeros(T, dtype=bool)
    for t in range(T):
        hit = np.where(ids[t] == emit[t])[0]
        if hit.size:
            U[t] = uplift[t, hit[0]]
            in_support[t] = True
    return U, in_support


def paper_teacher_top1_prob(teacher_topk_log_probs, normalize_teacher_topk: bool = True):
    """Teacher top-1 probability per token (for the paper's fork/lock cross-check).

    Uses the renormalized top-k distribution when normalize_teacher_topk (matches how
    the loss forms q~); the top-1 of the renormalized slice. Returns [T]."""
    q = _safe(teacher_topk_log_probs)
    if normalize_teacher_topk:
        m = q.max(axis=-1, keepdims=True)
        lse = m.squeeze(-1) + np.log(np.exp(q - m).sum(axis=-1))
        p = np.exp(q - lse[:, None])
    else:
        p = np.exp(q)
    return p.max(axis=-1)
