"""Experiment 5: knowledge-transfer audit.

Question: did stage routing actually transfer STAGE-APPROPRIATE privileged
knowledge into the student?

For each model M in {base, routing_gs70, noroute_gs70} (student prompt, NO
privilege) and each stage segment s of the 48 base trajectories, compute the
truncated reverse KL (the training loss, k=16) between M and each privileged
teacher context t in {loc, repro, patch}:

    KL_s,t(M) = mean over stage-s assistant tokens of
                sum_i p_M(i) * (log p_M(i) - log p_teacher_t(i))

Discriminating prediction:
  - routing-trained M should be closer to the STAGE-MATCHED teacher within each
    stage (low KL_loc,loc / KL_repro,repro),
  - no-routing M (whole-answer distillation) should be closer to the PATCH
    teacher everywhere, including on loc/repro tokens.

Teacher top-16 scores already exist (base model + privilege = the actual OPSD
teacher of these from-base runs). Forward-only; run on one NPU:
  ASCEND_RT_VISIBLE_DEVICES=<d> python compute_transfer.py --model <name>
"""
import argparse, importlib.util, json, os, time
import numpy as np
import torch
from pathlib import Path

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree
# that holds output/ and the model weights; it defaults to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
DATA = f"{ROOT}/analysis/grad_interference/data"
TEACH = f"{ROOT}/analysis/grad_interference/teacher"
OUT = f"{ROOT}/analysis/grad_interference/transfer"

MODELS = {
    "base": f"{WORKSPACE}/hf_models/Qwen/Qwen3-4B-Instruct-2507",
    "routing_gs70": f"{WORKSPACE}/output/hf_models/swe_opsd_grpo_end_to_end_stage_routing_qwen3_4b_colocated_8npu_2x4tp_20260711_gs70_original_scaffold",
    "noroute_gs70": f"{WORKSPACE}/output/hf_models/swe_opsd_grpo_e2e_no_routing_wholeanswer_patch_resume_gs10_qwen3_4b_colocated_8npu_2x4tp_20260724_gs70_original_scaffold",
}
K = 16

spec = importlib.util.spec_from_file_location("topk_loss", f"{ROOT}/swe_ladder/topk_loss.py")
tkl = importlib.util.module_from_spec(spec); spec.loader.exec_module(tkl)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out_path = f"{OUT}/{args.model}.json"

    import torch_npu  # noqa
    device = "npu:0"
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[args.model], torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True).to(device).eval()
    base = model.model
    lm_head = model.get_output_embeddings()

    manifest = json.load(open(f"{DATA}/manifest.json"))
    results = []
    for m in manifest:
        iid = m["instance_id"]
        z = np.load(f"{DATA}/{iid}.npz")
        ids = z["ids"]; resp_start = int(z["resp_start"])
        stage_ids = z["stage_ids"][resp_start:]; assist = z["assist_mask"][resp_start:]
        T = len(ids)
        with torch.no_grad():
            inp = torch.from_numpy(ids.astype(np.int64)).unsqueeze(0).to(device)
            h = base(input_ids=inp, use_cache=False).last_hidden_state[0]
            row = {"instance_id": iid, "solved": m["solved"]}
            for seg_name, sid in (("loc", 0), ("repro", 1), ("patch", 2)):
                msk = (stage_ids == sid) & (assist == 1)
                if msk.sum() < 16:
                    continue
                pos = np.nonzero(msk)[0] + resp_start
                hsel = h[torch.tensor(pos - 1, dtype=torch.long, device=device)]
                for t_name in ("loc", "repro", "patch"):
                    tz = np.load(f"{TEACH}/{iid}.{t_name}.npz")
                    tk = torch.tensor(tz["topk_ids"][msk], dtype=torch.long, device=device)
                    tlp = torch.tensor(tz["topk_lp"][msk], dtype=torch.float32, device=device)
                    valid = tk >= 0
                    slp = tkl.gather_topk_log_probs_from_hidden_chunked(
                        hidden_states=hsel.unsqueeze(0),
                        lm_head_weight=lm_head.weight,
                        topk_token_ids=tk.clamp(min=0).unsqueeze(0),
                        loss_mask=torch.ones(1, hsel.shape[0], device=device),
                        lm_head_bias=getattr(lm_head, "bias", None),
                        vocab_chunk_size=8192, temperature=1.0)
                    tlp = torch.where(valid, tlp, torch.full_like(tlp, float("-inf")))
                    kl, _ = tkl.teacher_topk_reverse_kl(
                        student_topk_log_probs=slp,
                        teacher_topk_log_probs=tlp.unsqueeze(0),
                        loss_mask=torch.ones(1, hsel.shape[0], device=device),
                        normalize_teacher_topk=True)
                    row[f"kl_{seg_name}_vs_{t_name}"] = float(kl)
            del h, inp
            torch.npu.empty_cache()
        results.append(row)
        print(f"{iid}: " + " ".join(f"{k}={v:.3f}" for k, v in row.items()
                                    if k.startswith("kl_")), flush=True)
    json.dump(results, open(out_path, "w"))
    print(f"wrote {out_path}")

if __name__ == "__main__":
    main()
