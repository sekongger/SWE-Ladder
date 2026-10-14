"""§ forwards: the ONLY NPU stage. Produces the metrics-ready per-position top-k tensors.

Given a data_gen pickle (reconstructed trajectories + rendered teacher contexts), for each
classified decision position t we compute EXACTLY the two top-k tensors the OPSD loss consumes
(swe_ladder/topk_loss.py and the agent_ppo_trainer teacher-forward), so downstream quantities are the
loss's own internals rather than a re-derivation:

  student_topk_log_probs[t,v] = student FULL-VOCAB log p_theta at teacher-top-k id v
  teacher_topk_log_probs[t,v] = teacher FULL-VOCAB log-prob (un-renormalized) at id v
  topk_ids[t,v]               = teacher top-k vocab ids
  H_t                         = student FULL-VOCAB entropy (fork signal, doc primary)

Teacher-forward assembly (faithful to agent_ppo_trainer.py:~2275-2376, dp_actor.py:1024):
  teacher_input = apply_chat_template(teacher_messages_c, add_generation_prompt=True)  ++  response_tokens
  (the generation prompt appends "<|im_start|>assistant\n"; our reconstructed response begins with
   the assistant content, so the concat reproduces training's prefix+response byte-for-byte, and the
   trainer's hard assert teacher_suffix == response holds by construction.)
  prompt_length_c = len(teacher_prefix_c);  the distribution that GENERATED response token t is read
  from logits at position (prompt_length_c + t - 1).

Per-token stage routing (SR):  SR teacher at position t == render for stage_ids[t] (clamp [0,2]).
  Frozen WA definition == force-PATCH render for ALL t  ==  SR's own "patch" context.
  => at PATCH-stage positions SR teacher IS the WA teacher (identical bytes) => P3 exact self-test.
  => we run only 3 teacher hidden passes (loc/repro/patch); WA reuses the patch pass everywhere.

Efficiency: we never materialize full logits [L, vocab]. We run model.model(...) to get last
hidden states, then compute logits ONLY at the needed rows via hidden_row @ lm_head.weight.T.
We store K=32 (opd_topk_k default); K=16 robustness is a pure top-16 slice (topk is sorted).

ENV (must be set by the caller / run.py):
  source /usr/local/Ascend/ascend-toolkit/set_env.sh ; PYTHONNOUSERSITE=1 ; ascend-rl-pack python
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import asdict
from typing import Any, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_RLLM = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "DeepSWE_RL", "rllm")
for p in (_HERE, _RLLM):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from quantities import per_token_quantities, position_uplift_on_emitted, paper_teacher_top1_prob  # type: ignore  # noqa: E402

# data_gen.py is run as a script, so its ReconTrajectory/Position dataclasses are pickled under
# module "__main__". They are plain dataclasses (no __slots__/custom __reduce__), so unpickling
# only needs a same-named class in __main__ whose __dict__ we can populate. Register lightweight
# stand-ins here (avoids importing data_gen's heavy rllm dependency chain into the NPU stage).
import __main__ as _main  # noqa: E402


class ReconTrajectory:  # unpickle stand-in; attributes are set from pickled __dict__
    pass


class Position:  # unpickle stand-in
    pass


for _n, _c in (("ReconTrajectory", ReconTrajectory), ("Position", Position)):
    if not hasattr(_main, _n):
        setattr(_main, _n, _c)

# stage id -> render key used in data_gen.teacher_messages_by_key
_STAGE_KEY = {0: "causal_localization", 1: "localized_issue_reproduction", 2: "patch_generation"}
WA_KEY = "patch_generation"  # frozen: WA == force-PATCH == SR patch context
DEFAULT_K = 32


# ---------------------------------------------------------------------------
# model loading (NPU)
# ---------------------------------------------------------------------------
def _import_torch_npu():
    try:
        import torch_npu  # noqa: F401
        return True
    except Exception as e:  # pragma: no cover
        print(f"[forwards] torch_npu import failed ({e}); falling back to available device", file=sys.stderr)
        return False


def _device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_model(model_path: str, device, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model.eval().to(device)
    return model


def _lm_head_weight(model) -> torch.Tensor:
    # tie_word_embeddings=True for Qwen3-4B: lm_head shares embed weight
    if getattr(model, "lm_head", None) is not None and model.lm_head.weight is not None:
        return model.lm_head.weight
    return model.get_input_embeddings().weight


@torch.no_grad()
def _last_hidden(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return [L, H] last hidden state for a single [1, L] sequence (no full-logit materialization)."""
    base = model.model if hasattr(model, "model") else model.base_model
    out = base(input_ids=input_ids, use_cache=False, return_dict=True)
    hs = out.last_hidden_state  # [1, L, H]
    # Qwen3 applies a final norm inside model.model already; last_hidden_state is post-norm.
    return hs[0]


@torch.no_grad()
def _logits_rows(hidden_rows: torch.Tensor, lm_head_w: torch.Tensor, logit_scale: float = 1.0) -> torch.Tensor:
    """hidden_rows [n, H] -> full-vocab logits [n, V] in fp32."""
    h = hidden_rows.to(lm_head_w.dtype)
    logits = torch.matmul(h, lm_head_w.t())
    if logit_scale != 1.0:
        logits = logits * logit_scale
    return logits.float()


# ---------------------------------------------------------------------------
# per-trajectory forwards
# ---------------------------------------------------------------------------
def _tokenize_teacher_prefix(tokenizer, teacher_messages) -> list[int]:
    ids = tokenizer.apply_chat_template(teacher_messages, add_generation_prompt=True, tokenize=True)
    return list(ids)


@torch.no_grad()
def _teacher_topk_at_positions(model, lm_head_w, tokenizer, teacher_messages,
                               response_tokens, want_positions, k, device,
                               max_seq_len: Optional[int]):
    """Run one teacher context; return {t: (ids[k] int64, logq[k] fp32)} for t in want_positions.

    logq = full-vocab log-prob at the top-k ids (NOT renormalized; renorm happens in quantities).
    """
    if not want_positions:
        return {}, 0
    prefix = _tokenize_teacher_prefix(tokenizer, teacher_messages)
    Lpre = len(prefix)
    full = prefix + list(response_tokens)
    truncated = 0
    if max_seq_len is not None and len(full) > max_seq_len:
        full = full[:max_seq_len]
        truncated = 1
    input_ids = torch.tensor([full], dtype=torch.long, device=device)
    hidden = _last_hidden(model, input_ids)  # [L, H]
    L = hidden.shape[0]
    # generating position for response index t is (Lpre + t - 1)
    rows, valid_t = [], []
    for t in want_positions:
        p = Lpre + t - 1
        if 0 <= p < L:
            rows.append(p)
            valid_t.append(t)
    if not rows:
        return {}, truncated
    logits = _logits_rows(hidden[rows], lm_head_w)  # [n, V] fp32
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    logprobs = logits - lse  # full-vocab log-probs
    topk = torch.topk(logits, k=k, dim=-1)  # sorted desc
    top_ids = topk.indices  # [n, k]
    top_logq = torch.gather(logprobs, -1, top_ids)  # [n, k]
    out = {}
    for i, t in enumerate(valid_t):
        out[t] = (top_ids[i].to("cpu").numpy().astype(np.int64),
                  top_logq[i].to("cpu").numpy().astype(np.float64))
    return out, truncated


@torch.no_grad()
def _student_at_positions(model, lm_head_w, prompt_tokens, response_tokens,
                          want_positions, device, max_seq_len: Optional[int]):
    """Return {t: (full_logprob_row_fn, H_t)} — but to save memory we return per-t (logprob_row fp32 cpu, H_t)."""
    full = list(prompt_tokens) + list(response_tokens)
    Lpre = len(prompt_tokens)
    truncated = 0
    if max_seq_len is not None and len(full) > max_seq_len:
        full = full[:max_seq_len]
        truncated = 1
    input_ids = torch.tensor([full], dtype=torch.long, device=device)
    hidden = _last_hidden(model, input_ids)
    L = hidden.shape[0]
    rows, valid_t = [], []
    for t in want_positions:
        p = Lpre + t - 1
        if 0 <= p < L:
            rows.append(p)
            valid_t.append(t)
    if not rows:
        return {}, truncated
    logits = _logits_rows(hidden[rows], lm_head_w)  # [n, V] fp32
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    logprobs = logits - lse  # [n, V]
    probs = logits.softmax(dim=-1)
    H = -(probs * torch.log(probs.clamp_min(1e-30))).sum(dim=-1)  # [n]
    out = {}
    for i, t in enumerate(valid_t):
        out[t] = (logprobs[i].to("cpu"), float(H[i].item()))
    return out, truncated


def process_trajectory(traj, student_model, student_lm, teacher_model, teacher_lm,
                       tokenizer, k, device, normalize_teacher_topk, max_seq_len):
    """Return list of per-position result dicts (metrics-ready) for one trajectory."""
    positions = traj.positions
    if not positions:
        return [], {"trunc_student": 0, "trunc_teacher": 0}
    response_tokens = traj.response_tokens
    prompt_tokens = traj.prompt_tokens
    tmsgs = traj.teacher_messages_by_key
    if not tmsgs:
        return [], {"trunc_student": 0, "trunc_teacher": 0}

    all_t = sorted({p.t for p in positions})
    stage_of = {p.t: max(0, min(2, int(p.stage_id))) for p in positions}

    # ---- teacher passes: patch (WA + SR-patch), loc, repro ----
    trunc_teacher = 0
    # WA / patch context: need ALL positions
    patch_res, tr = _teacher_topk_at_positions(
        teacher_model, teacher_lm, tokenizer, tmsgs[WA_KEY],
        response_tokens, all_t, k, device, max_seq_len); trunc_teacher += tr
    # loc context: only positions whose stage==0
    loc_t = [t for t in all_t if stage_of[t] == 0]
    loc_res, tr = _teacher_topk_at_positions(
        teacher_model, teacher_lm, tokenizer, tmsgs[_STAGE_KEY[0]],
        response_tokens, loc_t, k, device, max_seq_len); trunc_teacher += tr
    # repro context: only positions whose stage==1
    repro_t = [t for t in all_t if stage_of[t] == 1]
    repro_res, tr = _teacher_topk_at_positions(
        teacher_model, teacher_lm, tokenizer, tmsgs[_STAGE_KEY[1]],
        response_tokens, repro_t, k, device, max_seq_len); trunc_teacher += tr
    sr_by_stage = {0: loc_res, 1: repro_res, 2: patch_res}

    # ---- student pass: once ----
    stu_res, trunc_student = _student_at_positions(
        student_model, student_lm, prompt_tokens, response_tokens, all_t, device, max_seq_len)

    results = []
    for p in positions:
        t = p.t
        if t not in patch_res or t not in stu_res:
            continue
        stg = stage_of[t]
        wa_ids, wa_logq = patch_res[t]                    # WA == patch context
        sr_pack = sr_by_stage[stg].get(t)
        if sr_pack is None:
            continue
        sr_ids, sr_logq = sr_pack
        stu_logprob_row, H_t = stu_res[t]                 # [V] fp32 cpu, scalar

        # student full-vocab log p_theta at each teacher's top-k ids
        stu_wa = stu_logprob_row[torch.as_tensor(wa_ids, dtype=torch.long)].numpy().astype(np.float64)
        stu_sr = stu_logprob_row[torch.as_tensor(sr_ids, dtype=torch.long)].numpy().astype(np.float64)

        results.append({
            "instance_id": traj.instance_id,
            "source_dataset": traj.source_dataset,
            "traj_score": traj.traj_score,
            "pos_class": p.pos_class,
            "kind": p.kind,
            "stage": _STAGE_KEY[stg],
            "stage_id": stg,
            "t": int(t),
            "emitted_token_id": int(p.emitted_token_id),
            "H": float(H_t),
            # raw K=32 tensors (K=16 robustness = slice [:16]); dtype float64/int64
            "wa_topk_ids": wa_ids, "wa_teacher_logq": wa_logq, "wa_student_logp": stu_wa,
            "sr_topk_ids": sr_ids, "sr_teacher_logq": sr_logq, "sr_student_logp": stu_sr,
        })
    return results, {"trunc_student": trunc_student, "trunc_teacher": trunc_teacher}


# ---------------------------------------------------------------------------
# U/table computation from raw topk (kept here so run.py can re-slice K)
# ---------------------------------------------------------------------------
def compute_uplift_table(rows, k, normalize_teacher_topk=True):
    """From raw per-position topk rows -> flat table metrics.py consumes.

    U = uplift on the emitted token v*, restricted to teacher top-k (nan if out-of-support).
    Slicing to k<=32 uses the top-k prefix (topk is sorted desc) then renormalizes teacher inside
    per_token_quantities. Returns dict of parallel arrays.
    """
    cols = {c: [] for c in ("instance_id", "pos_class", "stage", "stage_id", "kind",
                            "H", "U_wa", "U_sr", "teacher_top1_wa", "teacher_top1_sr",
                            "in_support_wa", "in_support_sr", "traj_score", "source_dataset",
                            "emitted_token_id", "t")}
    for r in rows:
        for tag in ("wa", "sr"):
            ids = np.asarray(r[f"{tag}_topk_ids"])[:k][None, :]
            tq = np.asarray(r[f"{tag}_teacher_logq"])[:k][None, :]
            sp = np.asarray(r[f"{tag}_student_logp"])[:k][None, :]
            q = per_token_quantities(sp, tq, normalize_teacher_topk=normalize_teacher_topk)
            U, insup = position_uplift_on_emitted(q["uplift"], ids, np.array([r["emitted_token_id"]]))
            top1 = paper_teacher_top1_prob(tq, normalize_teacher_topk=normalize_teacher_topk)
            cols[f"U_{tag}"].append(float(U[0]))
            cols[f"in_support_{tag}"].append(bool(insup[0]))
            cols[f"teacher_top1_{tag}"].append(float(top1[0]))
        cols["instance_id"].append(r["instance_id"])
        cols["pos_class"].append(r["pos_class"])
        cols["stage"].append(r["stage"])
        cols["stage_id"].append(r["stage_id"])
        cols["kind"].append(r["kind"])
        cols["H"].append(r["H"])
        cols["traj_score"].append(r["traj_score"])
        cols["source_dataset"].append(r["source_dataset"])
        cols["emitted_token_id"].append(r["emitted_token_id"])
        cols["t"].append(r["t"])
    return {c: np.asarray(v) for c, v in cols.items()}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(data_pickle: str, student_path: str, teacher_path: str, out_path: str,
        k: int = DEFAULT_K, normalize_teacher_topk: bool = True,
        max_seq_len: Optional[int] = None, limit: Optional[int] = None,
        dtype: str = "bfloat16", mem_fraction: Optional[float] = None):
    _import_torch_npu()
    device = _device()
    # Hard-cap this process's HBM so a forward spike can never OOM a co-resident training job
    # sharing the same physical NPU (see analysis notes: these 910B3 cards are usually busy).
    if mem_fraction is not None and device.type == "npu":
        try:
            torch.npu.set_per_process_memory_fraction(float(mem_fraction), device.index or 0)
            free, total = torch.npu.mem_get_info(device.index or 0)
            print(f"[forwards] mem cap={mem_fraction} free={free/1e9:.1f}GB total={total/1e9:.1f}GB", flush=True)
        except Exception as e:  # pragma: no cover
            print(f"[forwards] mem cap/query failed ({e})", file=sys.stderr, flush=True)
    print(f"[forwards] device={device} k={k} normalize_teacher_topk={normalize_teacher_topk}", flush=True)

    with open(data_pickle, "rb") as f:
        data = pickle.load(f)
    trajs = data["trajectories"]
    if limit is not None:
        trajs = trajs[:limit]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)

    td = getattr(torch, dtype)
    print(f"[forwards] loading student={student_path}", flush=True)
    student_model = load_model(student_path, device, dtype=td)
    student_lm = _lm_head_weight(student_model)
    print(f"[forwards] loading teacher={teacher_path}", flush=True)
    teacher_model = load_model(teacher_path, device, dtype=td)
    teacher_lm = _lm_head_weight(teacher_model)

    all_rows, trunc = [], {"trunc_student": 0, "trunc_teacher": 0}
    n_ok = 0
    for i, traj in enumerate(trajs):
        if not traj.matched or not traj.teacher_messages_by_key:
            continue
        try:
            rows, tr = process_trajectory(
                traj, student_model, student_lm, teacher_model, teacher_lm,
                tokenizer, k, device, normalize_teacher_topk, max_seq_len)
        except Exception as e:  # pragma: no cover
            print(f"[forwards] traj {i} ({traj.instance_id}) failed: {e}", file=sys.stderr, flush=True)
            continue
        all_rows.extend(rows)
        trunc["trunc_student"] += tr["trunc_student"]
        trunc["trunc_teacher"] += tr["trunc_teacher"]
        n_ok += 1
        if (i + 1) % 10 == 0:
            print(f"[forwards] {i+1}/{len(trajs)} trajectories, {len(all_rows)} positions", flush=True)

    meta = {
        "data_pickle": data_pickle, "student_path": student_path, "teacher_path": teacher_path,
        "k": k, "normalize_teacher_topk": normalize_teacher_topk, "max_seq_len": max_seq_len,
        "n_trajectories_processed": n_ok, "n_positions": len(all_rows),
        "truncated_student_forwards": trunc["trunc_student"],
        "truncated_teacher_forwards": trunc["trunc_teacher"],
    }
    with open(out_path, "wb") as f:
        pickle.dump({"meta": meta, "rows": all_rows}, f)
    print(json.dumps(meta, indent=2, default=str), flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NPU forwards: per-position student/teacher top-k for fork analysis.")
    ap.add_argument("--data", required=True, help="data_gen pickle")
    ap.add_argument("--student", required=True, help="merged student HF ckpt dir")
    ap.add_argument("--teacher", required=True, help="base Qwen3-4B HF dir (WA + all SR share these weights)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--no-normalize-teacher-topk", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--mem-fraction", type=float, default=0.30,
                    help="cap this process's NPU HBM fraction so it can't OOM a co-resident job")
    args = ap.parse_args()

    run(args.data, args.student, args.teacher, args.out, k=args.k,
        normalize_teacher_topk=not args.no_normalize_teacher_topk,
        max_seq_len=args.max_seq_len, limit=args.limit, dtype=args.dtype,
        mem_fraction=args.mem_fraction)
