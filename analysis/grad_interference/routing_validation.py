"""Experiment 6: is the rule-based stage routing actually right?

Inputs: data/{iid}.npz (tokens, stage labels, spans rebuilt here), actual_lp/{iid}.npz
(exact log p(x_t) under student/loc/repro/patch contexts), base eval jsonl (thoughts).

V1  privilege GAIN by ACTION TYPE (classifier-free): gain_s(t) = lp_s(t) - lp_student(t),
    averaged over assistant tokens grouped by the enclosing action's observable type.
    Directly answers: "at search/view tokens, is loc privilege really the best teacher?"
V2  token-level best-teacher vs rule label confusion (on informative tokens), with a
    label-shuffle null.
V3  boundary-shift sensitivity: stage-matched mean gain when step labels are shifted
    by k steps.
V4  thought-intent (regex on the model's own thought text) vs rule label agreement;
    also dumps a 100-step sample TSV for human labeling.
"""
import glob, importlib.util, json, os, re, sys
import numpy as np

# Repo root, which holds analysis/ and swe_ladder/. WORKSPACE_ROOT is the tree
# that holds output/ and the model weights; it defaults to the repo root.
ROOT = os.environ.get("SWE_LADDER_ROOT") or str(Path(__file__).resolve().parents[2])
WORKSPACE = os.environ.get("WORKSPACE_ROOT", ROOT)
DATA = f"{ROOT}/analysis/grad_interference/data"
ALP = f"{ROOT}/analysis/grad_interference/actual_lp"
EVAL_JSONL = f"{WORKSPACE}/output/swebench_verified_qwen3_4b_instruct_2507_base_original_scaffold_eval/swebench_verified_qwen3_4b_instruct_2507_base_original_scaffold_eval.jsonl"
TOKENIZER_DIR = f"{WORKSPACE}/hf_models/Qwen/Qwen3-4B-Instruct-2507"
OUT = f"{ROOT}/analysis/grad_interference/routing_validation.json"

spec = importlib.util.spec_from_file_location("sc", f"{ROOT}/swe_ladder/stage_classifier.py")
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
sys.path.insert(0, f"{ROOT}/analysis/grad_interference")
import prep_data as pd  # reuse build_messages / tokenize_with_spans / student_user
from pathlib import Path

DEC = json.JSONDecoder()
def iter_records(path):
    for line in open(path):
        line = line.strip(); idx = 0
        while idx < len(line):
            try: obj, end = DEC.raw_decode(line, idx)
            except json.JSONDecodeError: break
            yield obj; idx = end
            while idx < len(line) and line[idx] in " \t": idx += 1

def action_type(fn, params):
    if fn in ("search", "grep"): return "search"
    if fn in ("file_editor", "str_replace_editor"):
        cmd = (params.get("command") or "").strip().lower()
        path = params.get("path") or params.get("file_path") or ""
        if cmd in ("str_replace", "insert", "create"):
            if sc.is_repro_path(path): return "edit_repro"
            if sc.is_test_path(path): return "edit_test"
            return "edit_source"
        return "view"
    if fn == "execute_bash":
        cmd = params.get("cmd") or params.get("command") or ""
        script = sc._bash_script_arg(cmd) if cmd else None
        if script and sc.is_repro_path(script): return "bash_repro"
        return "bash_other"
    if fn == "finish": return "finish"
    return "other"

INTENT = {
    "causal_localization": re.compile(
        r"\b(locate|locating|find (?:the|where)|search(?:ing)? for|identify(?:ing)? (?:the|where)|"
        r"look(?:ing)? (?:for|at) the (?:bug|issue|cause|relevant)|understand(?:ing)? the (?:issue|code|structure)|explore)\b", re.I),
    "localized_issue_reproduction": re.compile(
        r"\b(reproduc\w*|confirm(?:ing)? the (?:issue|bug|error)|verify(?:ing)? the (?:issue|bug|error)|"
        r"write .{0,30}(?:script|test) to (?:reproduce|confirm|verify|demonstrate))\b", re.I),
    "patch_generation": re.compile(
        r"\b(fix(?:ing)?|patch(?:ing)?|modify(?:ing)?|implement(?:ing)? (?:the|a) (?:fix|change)|"
        r"apply(?:ing)? the (?:fix|change)|correct(?:ing)? the)\b", re.I),
}

def main():
    manifest = json.load(open(f"{DATA}/manifest.json"))
    wanted = {m["instance_id"]: m for m in manifest}
    recs = {}
    for r in iter_records(EVAL_JSONL):
        ds = r["ds"]; ds = json.loads(ds) if isinstance(ds, str) else ds
        if ds["instance_id"] in wanted: recs[ds["instance_id"]] = r
    print(f"matched {len(recs)} trajectories")

    # accumulators
    gain_by_atype = {}      # atype -> {priv: [token gains]}
    gain_by_atype_solved = {}
    conf = np.zeros((3, 3)); conf_null = np.zeros((3, 3))
    shift_gain = {k: [] for k in (-4, -2, -1, 0, 1, 2, 4)}
    v4 = dict(agree=0, total=0, matrix=np.zeros((3, 3)))
    sample_rows = []
    step_records = []
    rng = np.random.default_rng(0)
    S2I = {sc.LOCALIZATION: 0, sc.REPRODUCTION: 1, sc.PATCH: 2}
    PRIVS = ("loc", "repro", "patch")

    for iid, m in wanted.items():
        alp_f = f"{ALP}/{iid}.npz"
        if not os.path.exists(alp_f): continue
        rec = recs[iid]
        z = np.load(f"{DATA}/{iid}.npz")
        alp = np.load(alp_f)
        ids = z["ids"]; resp_start = int(z["resp_start"])
        gains = np.stack([alp[f"lp_{p}"] - alp["lp_student"] for p in PRIVS])  # [3, resp]
        solved = bool(rec.get("reward"))

        msgs = pd.build_messages(rec, pd.student_user(
            (json.loads(rec["ds"]) if isinstance(rec["ds"], str) else rec["ds"]).get("problem_statement") or ""))
        _, spans = pd.tokenize_with_spans(msgs)
        steps = rec["trajectory_steps"]
        pairs = [(f"{s.get('thought') or ''}\n\n{s.get('action') or ''}",
                  str(s.get("observation") or "")) for s in steps]
        labels = sc.classify_trajectory(pairs, backtrack_threshold=3)

        # per-step assistant token ranges (response-relative)
        step_tok = []
        for si in range(len(steps)):
            a0, a1 = spans[2 + 2 * si]
            step_tok.append((a0 - resp_start, a1 - resp_start))

        # V1 + V2 + V3
        n_steps = len(steps)
        for si, s in enumerate(steps):
            fn, params = sc.parse_action(s.get("action") or "")
            at = action_type(fn or "", params)
            t0, t1 = step_tok[si]
            if t1 <= t0: continue
            g = gains[:, t0:t1].mean(axis=1)          # [3] mean gain of this step's tokens
            step_records.append(dict(iid=iid, step=si, atype=at,
                                     label=labels[si], solved=solved,
                                     g_loc=float(g[0]), g_repro=float(g[1]),
                                     g_patch=float(g[2])))
            gain_by_atype.setdefault(at, {p: [] for p in PRIVS})
            for pi, p in enumerate(PRIVS): gain_by_atype[at][p].append(float(g[pi]))
            if solved:
                gain_by_atype_solved.setdefault(at, {p: [] for p in PRIVS})
                for pi, p in enumerate(PRIVS): gain_by_atype_solved[at][p].append(float(g[pi]))
            # V2 on informative steps
            lab_i = S2I[labels[si]]
            if g.max() - g.min() > 0.02:
                conf[lab_i, int(g.argmax())] += 1
                conf_null[int(rng.integers(3)), int(g.argmax())] += 1
            # V3
            for k in shift_gain:
                sj = min(max(si + k, 0), n_steps - 1)
                shift_gain[k].append(float(g[S2I[labels[sj]]]))
            # V4
            th = s.get("thought") or ""
            hits = [st for st, rx in INTENT.items() if rx.search(th)]
            if len(hits) == 1:
                v4["total"] += 1
                v4["matrix"][S2I[labels[si]], S2I[hits[0]]] += 1
                v4["agree"] += (hits[0] == labels[si])
                if len(sample_rows) < 100 and rng.random() < 0.15:
                    sample_rows.append((iid, si, labels[si], hits[0],
                                        th.replace("\t", " ").replace("\n", " ")[:220]))

    res = {}
    print("\n=== V1: mean privilege gain (nats/token) by ACTION TYPE ===")
    print(f"{'action type':13s} {'n':>5s} {'loc':>8s} {'repro':>8s} {'patch':>8s}   best")
    for at in sorted(gain_by_atype, key=lambda a: -len(gain_by_atype[a]["loc"])):
        d = gain_by_atype[at]; n = len(d["loc"])
        means = [np.mean(d[p]) for p in PRIVS]
        res.setdefault("v1", {})[at] = dict(n=n, **{p: float(np.mean(d[p])) for p in PRIVS})
        print(f"{at:13s} {n:5d} " + "".join(f"{v:8.4f}" for v in means) +
              f"   {PRIVS[int(np.argmax(means))]}")

    print("\n=== V2: rule label vs best-gain teacher (informative steps, spread>0.02) ===")
    rn = conf / conf.sum(axis=1, keepdims=True).clip(min=1)
    nn = conf_null / conf_null.sum(axis=1, keepdims=True).clip(min=1)
    for i, lab in enumerate(("loc", "repro", "patch")):
        print(f"label={lab:6s}: best-fit " +
              " ".join(f"{p}={rn[i, j]:.2f}" for j, p in enumerate(PRIVS)) +
              f"   (null diag={nn[i, i]:.2f})")
    diag = float(np.trace(conf) / conf.sum())
    diag_null = float(np.trace(conf_null) / conf_null.sum())
    print(f"overall diagonal: {diag:.3f} vs shuffle-null {diag_null:.3f}  (n={int(conf.sum())})")
    res["v2"] = dict(confusion=conf.tolist(), diag=diag, diag_null=diag_null)

    print("\n=== V3: stage-matched gain vs boundary shift (steps) ===")
    for k in sorted(shift_gain):
        res.setdefault("v3", {})[str(k)] = float(np.mean(shift_gain[k]))
        print(f"shift {k:+d}: {np.mean(shift_gain[k]):.4f}")

    print("\n=== V4: thought-intent vs rule label ===")
    print(f"steps with unambiguous intent: {v4['total']}  agreement={v4['agree']/max(v4['total'],1):.3f}")
    print("matrix (rows=rule, cols=intent):")
    print(v4["matrix"])
    res["v4"] = dict(total=v4["total"], agree=v4["agree"],
                     matrix=v4["matrix"].tolist())

    with open(f"{ROOT}/analysis/grad_interference/human_label_sample.tsv", "w") as f:
        f.write("instance_id\tstep\trule_label\tintent_label\tthought_excerpt\tyour_label\n")
        for row in sample_rows:
            f.write("\t".join(map(str, row)) + "\t\n")
    json.dump(res, open(OUT, "w"), indent=1)
    with open(f"{ROOT}/analysis/grad_interference/routing_steps.jsonl", "w") as f:
        for r in step_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {OUT}, routing_steps.jsonl ({len(step_records)} steps), "
          f"human_label_sample.tsv ({len(sample_rows)} rows)")

if __name__ == "__main__":
    main()
