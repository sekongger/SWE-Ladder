"""Experiment 7 (O3): single-step intervention probe at retrieval moments.

For sampled "the model is retrieving" moments (search/view steps inside the
localization phase of real base trajectories):
  1. restore env state by replaying the recorded prefix actions in a fresh
     container of the instance's SWE-bench Verified image;
  2. under 4 conditions (student / +loc / +repro / +patch privilege in the user
     prompt; identical history otherwise) sample N candidate next actions from
     the base model;
  3. EXECUTE read-only candidates (search / file_editor view) in the env and
     score gold contact:
       search -> any gold (non-test) file path appears in the results
       view   -> the opened path is a gold file
       edit   -> path match counts as gold contact (not executed)
       bash/finish/other -> no contact
  4. per-moment per-condition hit rate -> paired comparison across conditions.

Run (cwd MUST be R2E-Gym-rucaibox for relative tool paths):
  R2EGYM_DOCKER_HOST=tcp://127.0.0.1:23750 PYTHONPATH=<repo>/R2E-Gym-rucaibox/src \
  python step_probe.py --shard i --num-shards n
"""
import argparse, importlib.util, json, os, re, sys, time, traceback

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree
# that holds output/ and the model weights; it defaults to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
DATA = f"{ROOT}/analysis/grad_interference/data"
EVAL_JSONL = f"{WORKSPACE}/output/swebench_verified_qwen3_4b_instruct_2507_base_original_scaffold_eval/swebench_verified_qwen3_4b_instruct_2507_base_original_scaffold_eval.jsonl"
DS_JSON = f"{WORKSPACE}/output/rucaibox_original_scaffold_swebench_verified_dataset.json"
OUT_DIR = f"{ROOT}/analysis/probe/results"
VLLM = "http://127.0.0.1:8132/v1/chat/completions"
MODEL = "qwen3-4b-base-probe"
N_SAMPLES = 8
TEMP = 1.0
MAX_PREFIX_STEPS = 6          # keep replay cheap; loc moments are early anyway
MOMENTS_PER_INSTANCE = 2

sys.path.insert(0, f"{ROOT}/analysis/grad_interference")
import prep_data as pd            # student_user / teacher_user / build_messages / build_teacher_pi
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

DEC = json.JSONDecoder()
def iter_records(path):
    for line in open(path):
        line = line.strip(); idx = 0
        while idx < len(line):
            try: obj, end = DEC.raw_decode(line, idx)
            except json.JSONDecodeError: break
            yield obj; idx = end
            while idx < len(line) and line[idx] in " \t": idx += 1

def norm(p):
    p = (p or "").strip()
    if p.startswith("/testbed/"): p = p[len("/testbed/"):]
    return p.lstrip("./")

def gold_contact(action_str, obs_text, gold_files):
    """(contact: bool, kind: str). obs_text=None for non-executed candidates."""
    fn, params = sc.parse_action(action_str)
    fn = fn or ""
    path = norm(params.get("path") or params.get("file_path") or "")
    cmd = (params.get("command") or "").strip().lower()
    if fn in ("file_editor", "str_replace_editor"):
        if cmd in ("view",) or not cmd:
            return (path in gold_files, "view")
        if cmd in ("str_replace", "insert", "create"):
            return (path in gold_files, "edit")
        return (False, "other")
    if fn in ("search", "grep"):
        if obs_text is None: return (False, "search")
        return (any(g in obs_text or os.path.basename(g) in obs_text
                    for g in gold_files), "search")
    return (False, fn or "none")

def is_readonly(action_str):
    fn, params = sc.parse_action(action_str)
    fn = fn or ""
    cmd = (params.get("command") or "").strip().lower()
    return fn in ("search", "grep") or (fn in ("file_editor", "str_replace_editor")
                                        and (cmd == "view" or not cmd))

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

    # pick probe moments: search/view steps in loc phase, step_idx <= MAX_PREFIX_STEPS
    moments = []
    for iid, (rec, ds) in sorted(recs.items()):
        steps = rec["trajectory_steps"]
        pairs = [(f"{s.get('thought') or ''}\n\n{s.get('action') or ''}",
                  str(s.get("observation") or "")) for s in steps]
        labels = sc.classify_trajectory(pairs, backtrack_threshold=3)
        picked = 0
        for si, (s, lab) in enumerate(zip(steps, labels)):
            if si > MAX_PREFIX_STEPS or picked >= MOMENTS_PER_INSTANCE: break
            if lab != sc.LOCALIZATION: continue
            fn, params = sc.parse_action(s.get("action") or "")
            cmd = (params.get("command") or "").strip().lower()
            if fn in ("search", "grep") or (fn in ("file_editor", "str_replace_editor")
                                            and cmd == "view"):
                moments.append((iid, si)); picked += 1
    moments = [m for i, m in enumerate(moments) if i % args.num_shards == args.shard]
    if args.limit: moments = moments[:args.limit]
    print(f"probe moments in shard: {len(moments)}", flush=True)

    for iid, si in moments:
        out_path = f"{OUT_DIR}/{iid}.step{si}.json"
        if os.path.exists(out_path): continue
        rec, ds = recs[iid]
        row = ds_rows.get(iid)
        if row is None: print(f"skip {iid}: no ds row", flush=True); continue
        issue = ds.get("problem_statement") or ""
        tp = pd.build_teacher_pi(ds)
        gold_files = set(tp["gold_changed_non_test_files"])
        if not gold_files: print(f"skip {iid}: no gold files", flush=True); continue

        env = None
        try:
            t0 = time.time()
            env = RepoEnv(EnvArgs(ds=row), ip="127.0.0.1", backend="docker", verbose=False)
            env.add_commands(TOOLS)
            # replay prefix
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
            result = dict(instance_id=iid, step=si, gold_files=sorted(gold_files),
                          orig_action=(rec["trajectory_steps"][si].get("action") or "")[:200],
                          conditions={})
            obs_cache = {}
            for cname, user1 in conds.items():
                msgs = pd.build_messages(rec, user1)[: 2 + 2 * si]
                cands = sample_actions(msgs, N_SAMPLES)
                samples = []
                for a_str in cands:
                    if not a_str:
                        samples.append(dict(kind="none", contact=False)); continue
                    obs_text = None
                    if is_readonly(a_str):
                        key = a_str
                        if key not in obs_cache:
                            try:
                                obs, _, _, _ = env.step(Action.from_string(a_str), timeout=60)
                                obs_cache[key] = str(obs)
                            except Exception:
                                obs_cache[key] = ""
                        obs_text = obs_cache[key]
                    contact, kind = gold_contact(a_str, obs_text, gold_files)
                    samples.append(dict(kind=kind, contact=bool(contact)))
                hits = sum(s["contact"] for s in samples)
                result["conditions"][cname] = dict(
                    samples=samples, hit_rate=hits / len(samples) if samples else 0.0)
            json.dump(result, open(out_path, "w"))
            hr = {c: f"{v['hit_rate']:.2f}" for c, v in result["conditions"].items()}
            print(f"{iid} step{si}: {hr} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"ERROR {iid} step{si}: {e}", flush=True)
            traceback.print_exc()
        finally:
            if env is not None:
                try: env.close()
                except Exception: pass

if __name__ == "__main__":
    main()
