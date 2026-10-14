"""Experiment 7b: intervention probe at REPRODUCTION moments.

At real repro moments (first REPRODUCTION-labeled steps of base trajectories):
restore env by replaying the prefix, then under 4 privilege conditions sample N
candidate actions. For candidates that CREATE a script: execute the create, run
`python <file>` with a timeout, score "exhibits the bug" (failure signal or
nonzero exit), then rm the created file to restore state. Other candidates score 0
for repro-success (recorded by kind).

Metric per condition: repro-success rate = fraction of samples that produced a
runnable script exhibiting the bug.

Run (cwd=R2E-Gym-rucaibox):
  R2EGYM_DOCKER_HOST=tcp://127.0.0.1:23750 python repro_probe.py --shard i --num-shards n
"""
import argparse, importlib.util, json, os, re, sys, time, traceback

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree
# that holds output/ and the model weights; it defaults to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
DATA = f"{ROOT}/analysis/grad_interference/data"
EVAL_JSONL = f"{WORKSPACE}/output/swebench_verified_qwen3_4b_instruct_2507_base_original_scaffold_eval/swebench_verified_qwen3_4b_instruct_2507_base_original_scaffold_eval.jsonl"
DS_JSON = f"{WORKSPACE}/output/rucaibox_original_scaffold_swebench_verified_dataset.json"
OUT_DIR = f"{ROOT}/analysis/probe/repro_results"
VLLM = "http://127.0.0.1:8132/v1/chat/completions"
MODEL = "qwen3-4b-base-probe"
N_SAMPLES = 6
TEMP = 1.0
MAX_PREFIX_STEPS = 14

sys.path.insert(0, f"{ROOT}/analysis/grad_interference")
import prep_data as pd
spec = importlib.util.spec_from_file_location("sc", f"{ROOT}/swe_ladder/stage_classifier.py")
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)

import requests
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.action import Action
from pathlib import Path

TOOLS = ["./src/r2egym/agenthub/tools/r2egym/file_editor.py",
         "./src/r2egym/agenthub/tools/search.py",
         "./src/r2egym/agenthub/tools/r2egym/execute_bash.py",
         "./src/r2egym/agenthub/tools/finish.py"]

FAIL_RE = re.compile(r"Traceback \(most recent call last\)|AssertionError|Error:|error:|"
                     r"Exception|FAILED|assert(ion)? failed", re.I)

DEC = json.JSONDecoder()
def iter_records(path):
    for line in open(path):
        line = line.strip(); idx = 0
        while idx < len(line):
            try: obj, end = DEC.raw_decode(line, idx)
            except json.JSONDecodeError: break
            yield obj; idx = end
            while idx < len(line) and line[idx] in " \t": idx += 1

def sample_actions(messages, n):
    body = dict(model=MODEL, messages=messages, n=n, temperature=TEMP,
                max_tokens=2048, stop=["</function>"])
    r = requests.post(VLLM, json=body, timeout=600)
    r.raise_for_status()
    outs = []
    for ch in r.json()["choices"]:
        txt = ch["message"]["content"] or ""
        if "</function>" not in txt: txt += "</function>"
        m = re.search(r"<function=.*?</function>", txt, re.S)
        outs.append(m.group(0) if m else "")
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    manifest = json.load(open(f"{DATA}/manifest.json"))
    wanted = {m["instance_id"] for m in manifest}
    ds_rows = {r["instance_id"]: r for r in json.load(open(DS_JSON))}
    recs = {}
    for r in iter_records(EVAL_JSONL):
        ds = r["ds"]; ds = json.loads(ds) if isinstance(ds, str) else ds
        if ds["instance_id"] in wanted: recs[ds["instance_id"]] = (r, ds)

    # moments: first REPRODUCTION-labeled step, prefix <= MAX_PREFIX_STEPS
    moments = []
    for iid, (rec, ds) in sorted(recs.items()):
        steps = rec["trajectory_steps"]
        pairs = [(f"{s.get('thought') or ''}\n\n{s.get('action') or ''}",
                  str(s.get("observation") or "")) for s in steps]
        labels = sc.classify_trajectory(pairs, backtrack_threshold=3)
        for si, lab in enumerate(labels):
            if lab == sc.REPRODUCTION and si <= MAX_PREFIX_STEPS:
                moments.append((iid, si)); break
    moments = [m for i, m in enumerate(moments) if i % args.num_shards == args.shard]
    if args.limit: moments = moments[:args.limit]
    print(f"repro moments in shard: {len(moments)}", flush=True)

    for iid, si in moments:
        out_path = f"{OUT_DIR}/{iid}.step{si}.json"
        if os.path.exists(out_path): continue
        rec, ds = recs[iid]
        row = ds_rows.get(iid)
        if row is None: continue
        issue = ds.get("problem_statement") or ""
        tp = pd.build_teacher_pi(ds)
        env = None
        try:
            t0 = time.time()
            env = RepoEnv(EnvArgs(ds=row), ip="127.0.0.1", backend="docker", verbose=False)
            env.add_commands(TOOLS)
            for s in rec["trajectory_steps"][:si]:
                a = Action.from_string(s.get("action") or "")
                if not a.function_name: continue
                env.step(a, timeout=90)
            conds = {
                "none":  pd.student_user(issue),
                "loc":   pd.teacher_user(issue, sc.LOCALIZATION, tp),
                "repro": pd.teacher_user(issue, sc.REPRODUCTION, tp),
                "patch": pd.teacher_user(issue, sc.PATCH, tp),
            }
            result = dict(instance_id=iid, step=si, conditions={})
            for cname, user1 in conds.items():
                msgs = pd.build_messages(rec, user1)[: 2 + 2 * si]
                cands = sample_actions(msgs, N_SAMPLES)
                samples = []
                for a_str in cands:
                    fn, params = sc.parse_action(a_str or "")
                    fn = fn or ""
                    cmd = (params.get("command") or "").strip().lower()
                    path = (params.get("path") or "").strip()
                    is_create = (fn in ("file_editor", "str_replace_editor") and cmd == "create") \
                                or fn == "create"
                    if not is_create or not path.endswith(".py"):
                        samples.append(dict(kind=fn + ":" + cmd, success=False)); continue
                    try:
                        env.step(Action.from_string(a_str), timeout=60)
                        run_act = (f"<function=execute_bash>\n<parameter=cmd>"
                                   f"timeout 30 python {path}</parameter>\n</function>")
                        obs, _, _, _ = env.step(Action.from_string(run_act), timeout=60)
                        text = str(obs)
                        exhibits = bool(FAIL_RE.search(text)) or "Exit code: 1" in text \
                                   or "Exit code: 2" in text
                        samples.append(dict(kind="create+run", success=bool(exhibits)))
                    except Exception:
                        samples.append(dict(kind="create_err", success=False))
                    finally:
                        try:
                            env.step(Action.from_string(
                                f"<function=execute_bash>\n<parameter=cmd>rm -f {path}"
                                f"</parameter>\n</function>"), timeout=30)
                        except Exception: pass
                succ = sum(s["success"] for s in samples)
                result["conditions"][cname] = dict(
                    samples=samples, success_rate=succ / len(samples) if samples else 0.0)
            json.dump(result, open(out_path, "w"))
            sr = {c: f"{v['success_rate']:.2f}" for c, v in result["conditions"].items()}
            print(f"{iid} step{si}: {sr} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"ERROR {iid} step{si}: {e}", flush=True)
            traceback.print_exc()
        finally:
            if env is not None:
                try: env.close()
                except Exception: pass

if __name__ == "__main__":
    main()
