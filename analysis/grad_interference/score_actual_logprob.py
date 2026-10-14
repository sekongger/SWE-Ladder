"""Routing-validation prerequisite: exact per-position logprob of the ACTUAL
next token under 4 contexts (student/unprivileged, loc, repro, patch teachers).

Unlike the top-16 files, this gathers log p(x_t) exactly for every response
position -- no top-k truncation, so positions where teachers disagree with the
trajectory (the informative ones) are not censored.

Output: actual_lp/{iid}.npz with keys lp_student / lp_loc / lp_repro / lp_patch,
each [resp_len] float32.

Run: ASCEND_RT_VISIBLE_DEVICES=<d> python score_actual_logprob.py --shard i --num-shards n
"""
import argparse, json, os, time
import numpy as np
import torch
from pathlib import Path

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree
# that holds output/ and the model weights; it defaults to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
DATA = f"{ROOT}/analysis/grad_interference/data"
OUT = f"{ROOT}/analysis/grad_interference/actual_lp"
MODEL_DIR = f"{WORKSPACE}/hf_models/Qwen/Qwen3-4B-Instruct-2507"
POS_CHUNK = 4096

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    import torch_npu  # noqa
    device = "npu:0"
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True).to(device).eval()
    base = model.model
    Wt = model.get_output_embeddings().weight.detach().to(torch.float32).t().contiguous()

    manifest = json.load(open(f"{DATA}/manifest.json"))
    manifest = [m for i, m in enumerate(manifest) if i % args.num_shards == args.shard]

    for m in manifest:
        iid = m["instance_id"]
        out_path = f"{OUT}/{iid}.npz"
        if os.path.exists(out_path): continue
        z = np.load(f"{DATA}/{iid}.npz")
        ids = z["ids"]; resp_start = int(z["resp_start"])
        resp = ids[resp_start:]
        payload = {}
        t0 = time.time()
        for tag, prefix in (("student", ids[:resp_start]),
                            ("loc", z["prefix_loc"]), ("repro", z["prefix_repro"]),
                            ("patch", z["prefix_patch"])):
            full = np.concatenate([prefix, resp]).astype(np.int64)
            T = len(full); rstart = len(prefix)
            with torch.no_grad():
                inp = torch.from_numpy(full).unsqueeze(0).to(device)
                h = base(input_ids=inp, use_cache=False).last_hidden_state[0]
                hsel = h[rstart - 1:T - 1]                       # [resp, H]
                tgt = torch.from_numpy(resp.astype(np.int64)).to(device)
                n = hsel.shape[0]
                out = torch.empty(n, dtype=torch.float32)
                for s in range(0, n, POS_CHUNK):
                    e = min(s + POS_CHUNK, n)
                    logits = hsel[s:e].to(torch.float32) @ Wt
                    lp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                    out[s:e] = lp.gather(-1, tgt[s:e, None]).squeeze(-1).cpu()
                    del logits, lp
                payload[f"lp_{tag}"] = out.numpy()
                del h, hsel, inp
                torch.npu.empty_cache()
        np.savez_compressed(out_path, **payload)
        print(f"{iid}: resp={len(resp)} 4 contexts in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
