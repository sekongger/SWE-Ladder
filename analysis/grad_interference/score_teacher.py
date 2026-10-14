"""Gradient-interference experiment, step 2: teacher top-k scoring via vLLM.

For each trajectory and each teacher context variant (loc/repro/patch + shuffled),
send prompt = prefix_variant_ids + response_ids to /v1/completions with
prompt_logprobs=K, and store top-K (token_id, logprob) for every RESPONSE position.

Saves per (traj, variant): analysis/grad_interference/teacher/{iid}.{variant}.npz
  topk_ids  [resp_len, K] int32   (-1 pad)
  topk_lp   [resp_len, K] float32 (-inf pad)
  scored    [resp_len]    int8    (1 if position has >=1 entry)

Run: python score_teacher.py [--variants loc,repro,patch,loc_sh,repro_sh,patch_sh]
"""
import argparse, glob, json, os, sys, time
import numpy as np
import requests
from pathlib import Path

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree
# that holds output/ and the model weights; it defaults to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
DATA = f"{ROOT}/analysis/grad_interference/data"
OUT = f"{ROOT}/analysis/grad_interference/teacher"
URL = "http://127.0.0.1:8131/v1/completions"
MODEL = "qwen3-4b-base-teacher"
K = 16

def score(prompt_ids, resp_len):
    body = {
        "model": MODEL,
        "prompt": prompt_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": K,
    }
    r = requests.post(URL, json=body, timeout=1800)
    r.raise_for_status()
    out = r.json()
    plp = out["choices"][0].get("prompt_logprobs")
    if plp is None:
        raise RuntimeError("server returned no prompt_logprobs")
    n = len(prompt_ids)
    start = n - resp_len
    ids = np.full((resp_len, K), -1, dtype=np.int32)
    lps = np.full((resp_len, K), -np.inf, dtype=np.float32)
    scored = np.zeros(resp_len, dtype=np.int8)
    # prompt_logprobs[i] corresponds to prompt token i (None at i=0)
    for pos in range(start, n):
        ent = plp[pos]
        if not ent: continue
        j = 0
        # ent: {token_id_str: {"logprob": .., "rank": ..}}
        for tid_s, info in sorted(ent.items(), key=lambda kv: kv[1]["logprob"], reverse=True):
            if j >= K: break
            ids[pos - start, j] = int(tid_s)
            lps[pos - start, j] = float(info["logprob"])
            j += 1
        if j: scored[pos - start] = 1
    return ids, lps, scored

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="loc,repro,patch,loc_sh,repro_sh,patch_sh")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    variants = args.variants.split(",")
    os.makedirs(OUT, exist_ok=True)
    manifest = json.load(open(f"{DATA}/manifest.json"))
    if args.limit: manifest = manifest[:args.limit]
    t0 = time.time()
    done = 0
    for m in manifest:
        iid = m["instance_id"]
        z = np.load(f"{DATA}/{iid}.npz")
        ids = z["ids"]; resp_start = int(z["resp_start"])
        resp = ids[resp_start:].tolist()
        for v in variants:
            out_path = f"{OUT}/{iid}.{v}.npz"
            if os.path.exists(out_path): continue
            prefix = z[f"prefix_{v}"].tolist()
            tids, tlps, scored = score(prefix + resp, len(resp))
            np.savez_compressed(out_path, topk_ids=tids, topk_lp=tlps, scored=scored)
            done += 1
            el = time.time() - t0
            print(f"[{el:7.0f}s] {iid} {v}: resp={len(resp)} scored={int(scored.sum())}", flush=True)
    print(f"finished, {done} scorings in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
