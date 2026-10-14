"""Gradient-interference experiment, step 2 (torch edition): teacher top-k scoring.

vllm-ascend's prompt_logprobs crashes with chunked prefill, so score the teacher
directly with a no-grad torch forward of the SAME base model:
  for each (trajectory, context variant): forward prefix+response once, then for
  every response position compute full-vocab logits in position-chunks and keep
  the top-K (ids, logprobs).

Output format identical to score_teacher.py:
  teacher/{iid}.{variant}.npz : topk_ids [resp,K] int32, topk_lp [resp,K] f32,
                                scored [resp] int8 (all 1 here)

Run one worker per NPU:
  ASCEND_RT_VISIBLE_DEVICES=<d> python score_teacher_torch.py --shard i --num-shards n
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
OUT = f"{ROOT}/analysis/grad_interference/teacher"
MODEL_DIR = f"{WORKSPACE}/hf_models/Qwen/Qwen3-4B-Instruct-2507"
K = 16
POS_CHUNK = 2048  # positions per logits chunk: 2048 x 151k x 4B ~= 1.2 GB

VARIANTS = ("loc", "repro", "patch", "loc_sh", "repro_sh", "patch_sh")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    import torch_npu  # noqa
    device = "npu:0"
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True).to(device).eval()
    base = model.model
    lm_head = model.get_output_embeddings()
    Wt = lm_head.weight.detach().to(torch.float32).t().contiguous()  # [H, V] fp32, cast once

    manifest = json.load(open(f"{DATA}/manifest.json"))
    if args.limit: manifest = manifest[:args.limit]
    manifest = [m for i, m in enumerate(manifest) if i % args.num_shards == args.shard]

    for m in manifest:
        iid = m["instance_id"]
        z = np.load(f"{DATA}/{iid}.npz")
        ids = z["ids"]; resp_start = int(z["resp_start"])
        resp = ids[resp_start:]
        for v in VARIANTS:
            out_path = f"{OUT}/{iid}.{v}.npz"
            if os.path.exists(out_path): continue
            t0 = time.time()
            prefix = z[f"prefix_{v}"]
            full = np.concatenate([prefix, resp]).astype(np.int64)
            T = len(full); rstart = len(prefix)
            with torch.no_grad():
                inp = torch.from_numpy(full).unsqueeze(0).to(device)
                h = base(input_ids=inp, use_cache=False).last_hidden_state[0]  # [T,H]
                # logprob of token at pos t comes from hidden t-1
                hsel = h[rstart - 1:T - 1]                                     # [resp,H]
                n = hsel.shape[0]
                tid_out = torch.empty((n, K), dtype=torch.int32)
                tlp_out = torch.empty((n, K), dtype=torch.float32)
                for s in range(0, n, POS_CHUNK):
                    e = min(s + POS_CHUNK, n)
                    logits = hsel[s:e].to(torch.float32) @ Wt
                    logz = torch.logsumexp(logits, dim=-1, keepdim=True)
                    lp = logits - logz
                    top = torch.topk(lp, K, dim=-1)
                    tid_out[s:e] = top.indices.to(torch.int32).cpu()
                    tlp_out[s:e] = top.values.to(torch.float32).cpu()
                    del logits, logz, lp, top
                del h, hsel, inp
                torch.npu.empty_cache()
            np.savez_compressed(out_path,
                                topk_ids=tid_out.numpy(), topk_lp=tlp_out.numpy(),
                                scored=np.ones(len(resp), dtype=np.int8))
            print(f"{iid} {v}: T={T} resp={len(resp)} in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
