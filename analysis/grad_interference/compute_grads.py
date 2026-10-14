"""Gradient-interference experiment, step 3: student gradients + cosine geometry.

For each trajectory, compute the student (base model) OPSD reverse-KL gradient for
each (stage segment, teacher variant) pair, then pairwise cosines:

  gradients per trajectory (8 backwards):
    g_loc_r   = loss on LOC tokens,   teacher = loc-privileged     (routed)
    g_repro_r = loss on REPRO tokens, teacher = repro-privileged   (routed)
    g_patch   = loss on PATCH tokens, teacher = patch-privileged   (routed == global)
    g_loc_g   = loss on LOC tokens,   teacher = patch-privileged   (global/whole-answer)
    g_repro_g = loss on REPRO tokens, teacher = patch-privileged   (global)
    g_loc_sh / g_repro_sh / g_patch_sh = shuffled-privilege controls

  reported cosines (full-model + per-layer):
    measurement A (cross-stage interference):
      global scheme: (loc_g, repro_g), (loc_g, patch), (repro_g, patch)
      routed scheme: (loc_r, repro_r), (loc_r, patch), (repro_r, patch)
    measurement B (signal mismatch): (loc_r, loc_g), (repro_r, repro_g)
    controls: routed vs shuffled per stage; split-half within-stage baseline
              (loc under routed teacher, two interleaved halves)

Loss = teacher_topk_reverse_kl (the exact training loss, imported from rllm).
Gradient params: all model parameters; per-layer buckets on CPU bf16.

Run (one worker per NPU):
  ASCEND_RT_VISIBLE_DEVICES=<i> python compute_grads.py --shard <i> --num-shards <n>
"""
import argparse, glob, importlib.util, json, os, sys, time
import numpy as np
import torch
from pathlib import Path

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree
# that holds output/ and the model weights; it defaults to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
DATA = f"{ROOT}/analysis/grad_interference/data"
TEACH = f"{ROOT}/analysis/grad_interference/teacher"
OUT = f"{ROOT}/analysis/grad_interference/grads"
MODEL_DIR = f"{WORKSPACE}/hf_models/Qwen/Qwen3-4B-Instruct-2507"
K = 16

spec = importlib.util.spec_from_file_location("topk_loss", f"{ROOT}/swe_ladder/topk_loss.py")
tkl = importlib.util.module_from_spec(spec); spec.loader.exec_module(tkl)

def layer_key(pname):
    if pname.startswith("model.layers."):
        return "layer." + pname.split(".")[2]
    if "embed" in pname: return "embed"
    if pname.startswith("lm_head"): return "lm_head"
    return "other"

def collect_grad(model):
    """-> {layer_key: flat bf16 cpu tensor}; zeros model grads afterwards."""
    buckets = {}
    for n, p in model.named_parameters():
        if p.grad is None: continue
        buckets.setdefault(layer_key(n), []).append(p.grad.detach().to(torch.float32).flatten())
    out = {k: torch.cat(v).to(torch.bfloat16).cpu() for k, v in buckets.items()}
    model.zero_grad(set_to_none=True)
    return out

def cos_full_and_layers(a, b):
    dot_t, na_t, nb_t = 0.0, 0.0, 0.0
    per_layer = {}
    for k in a:
        if k not in b: continue
        x = a[k].to(torch.float32); y = b[k].to(torch.float32)
        d = float((x * y).sum()); nx = float((x * x).sum()); ny = float((y * y).sum())
        per_layer[k] = d / max((nx ** 0.5) * (ny ** 0.5), 1e-30)
        dot_t += d; na_t += nx; nb_t += ny
    return dot_t / max((na_t ** 0.5) * (nb_t ** 0.5), 1e-30), per_layer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split-half", action="store_true", help="also split-half baseline")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    import torch_npu  # noqa
    device = "npu:0"
    from transformers import AutoModelForCausalLM, AutoConfig
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # grads; qwen3 has no dropout so this stays deterministic
    lm_head = model.get_output_embeddings()
    base = model.model  # Qwen3Model: last_hidden_state is already final-norm'ed,
                        # i.e. exactly what feeds lm_head; full [T,V] logits never materialized

    manifest = json.load(open(f"{DATA}/manifest.json"))
    if args.limit: manifest = manifest[:args.limit]
    manifest = [m for i, m in enumerate(manifest) if i % args.num_shards == args.shard]

    for m in manifest:
        iid = m["instance_id"]
        out_path = f"{OUT}/{iid}.json"
        if os.path.exists(out_path): continue
        z = np.load(f"{DATA}/{iid}.npz")
        ids = z["ids"]; resp_start = int(z["resp_start"])
        stage_ids = z["stage_ids"]; assist = z["assist_mask"]
        T = len(ids)
        resp_len = T - resp_start

        # per-(variant) teacher tensors over response positions
        teachers = {}
        for v in ("loc", "repro", "patch", "loc_sh", "repro_sh", "patch_sh"):
            f = f"{TEACH}/{iid}.{v}.npz"
            if not os.path.exists(f): teachers = None; break
            tz = np.load(f)
            teachers[v] = (tz["topk_ids"], tz["topk_lp"], tz["scored"])
        if teachers is None:
            print(f"skip {iid}: missing teacher scores", flush=True); continue

        # stage masks over response region, assistant tokens only
        seg = {}
        for name, sid in (("loc", 0), ("repro", 1), ("patch", 2)):
            msk = ((stage_ids == sid) & (assist == 1)).astype(np.float32)
            seg[name] = msk[resp_start:]

        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        def backward_for(seg_mask_np, variant):
            """One forward+backward; returns per-layer grad buckets."""
            t_ids, t_lp, scored = teachers[variant]
            mask_np = seg_mask_np * scored.astype(np.float32)
            if mask_np.sum() < 16: return None
            out = base(input_ids=input_ids, use_cache=False)
            h = out.last_hidden_state[0]           # [T, H], post final-norm
            del out
            # logprob of position t uses hidden at t-1
            pos = np.nonzero(mask_np)[0] + resp_start          # sequence positions
            hidx = torch.tensor(pos - 1, dtype=torch.long, device=device)
            hsel = h[hidx]                                     # [n, H]
            tk = torch.tensor(t_ids[mask_np > 0], dtype=torch.long, device=device)   # [n, K]
            tlp = torch.tensor(t_lp[mask_np > 0], dtype=torch.float32, device=device)
            valid = tk >= 0
            tk = tk.clamp(min=0)
            # student top-k logprobs via chunked logZ (reuse training helper)
            slp = tkl.gather_topk_log_probs_from_hidden_chunked(
                hidden_states=hsel.unsqueeze(0),
                lm_head_weight=lm_head.weight,
                topk_token_ids=tk.unsqueeze(0),
                loss_mask=torch.ones(1, hsel.shape[0], device=device),
                lm_head_bias=getattr(lm_head, "bias", None),
                vocab_chunk_size=8192,
                temperature=1.0)
            tlp = torch.where(valid, tlp, torch.full_like(tlp, float("-inf")))
            loss, _ = tkl.teacher_topk_reverse_kl(
                student_topk_log_probs=slp,
                teacher_topk_log_probs=tlp.unsqueeze(0),
                loss_mask=torch.ones(1, hsel.shape[0], device=device),
                normalize_teacher_topk=True)
            loss.backward()
            del h, hsel, slp, tlp, tk
            torch.npu.empty_cache()
            return collect_grad(model)

        t0 = time.time()
        grads = {}
        jobs = [
            ("loc_r",   seg["loc"],   "loc"),
            ("repro_r", seg["repro"], "repro"),
            ("patch",   seg["patch"], "patch"),
            ("loc_g",   seg["loc"],   "patch"),
            ("repro_g", seg["repro"], "patch"),
            ("loc_sh",  seg["loc"],   "loc_sh"),
            ("repro_sh",seg["repro"], "repro_sh"),
            ("patch_sh",seg["patch"], "patch_sh"),
        ]
        if args.split_half:
            runs = np.nonzero(seg["loc"])[0]
            half = np.zeros_like(seg["loc"]); half[runs[::2]] = 1
            other = np.zeros_like(seg["loc"]); other[runs[1::2]] = 1
            jobs += [("loc_h1", half, "loc"), ("loc_h2", other, "loc")]
        ok = True
        for tag, msk, var in jobs:
            tj = time.time()
            g = backward_for(msk, var)
            print(f"  {iid} {tag}: {time.time()-tj:.0f}s", flush=True)
            if g is None and tag in ("loc_r", "repro_r", "patch"):
                ok = False; break
            if g is not None: grads[tag] = g
        if not ok:
            print(f"skip {iid}: segment too small after scoring", flush=True); continue

        pairs = [
            ("A_global", "loc_g", "repro_g"), ("A_global", "loc_g", "patch"), ("A_global", "repro_g", "patch"),
            ("A_routed", "loc_r", "repro_r"), ("A_routed", "loc_r", "patch"), ("A_routed", "repro_r", "patch"),
            ("B_mismatch", "loc_r", "loc_g"), ("B_mismatch", "repro_r", "repro_g"),
            ("C_shuffle", "loc_r", "loc_sh"), ("C_shuffle", "repro_r", "repro_sh"), ("C_shuffle", "patch", "patch_sh"),
        ]
        if "loc_h1" in grads and "loc_h2" in grads:
            pairs.append(("D_splithalf", "loc_h1", "loc_h2"))
        res = {"instance_id": iid, "solved": m["solved"], "cosines": []}
        tj = time.time()
        for group, a, b in pairs:
            if a not in grads or b not in grads: continue
            c, per_layer = cos_full_and_layers(grads[a], grads[b])
            res["cosines"].append({"group": group, "a": a, "b": b, "cos": c, "per_layer": per_layer})
        print(f"  {iid} cosines: {time.time()-tj:.0f}s", flush=True)
        json.dump(res, open(out_path, "w"))
        print(f"{iid}: {len(res['cosines'])} pairs in {time.time()-t0:.0f}s", flush=True)
        del grads
        torch.npu.empty_cache()

if __name__ == "__main__":
    main()
