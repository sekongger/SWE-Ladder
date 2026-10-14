"""Build P2T-style latent process graphs (G*) from gold patches — locally, cheaply.

Reference: "From Patches to Trajectories: Privileged Process Supervision for
SWE Agents" (arXiv:2605.21996), Sec 4.1 + App D. P2T distills the reference
patch p* into a DAG G* = (V, E) of contextual-fact nodes and solution-milestone
nodes, each carrying an *unlocker* (the replayable environment action that
discovers it), via a proposer-critic loop of two 480B-LLM agents with sandboxed
repo tools (40 tool calls/round, <=6 rounds, ~10^2 GPU-hours per 1.8k instances).

Our optimization, exploiting what P2T's proposer must rediscover but our RL
dataset already carries as verified structured fields:

  P2T proposer work                         ->  our deterministic source
  -----------------------------------------     ----------------------------------
  locate changed files/lines (repo tools)   ->  parse teacher_pi.golden_patch
  synthesize + EXECUTE a repro script (R5)  ->  extra_info.FAIL_TO_PASS (dataset-
                                                verified fail->pass tests; no
                                                sandbox execution needed)
  byte-exact edit old_str (R6)              ->  hunk old/new text from the diff
  validation command                        ->  extra_info.run_tests / FAIL_TO_PASS
  issue-derived facts                       ->  extra_info.problem_statement

So Phase A below synthesizes the graph BACKBONE with zero LLM calls: static-fact
nodes (files/hunks/pre-fix code), a dynamic repro-fact node, and the milestone
chain reproduce -> analysis -> fix_plan -> code_edit(s) -> validation, then wires
prerequisite edges with P2T's own deterministic rule (edge u->v iff v's unlocker
references an entity first surfaced by u, plus the canonical artifact order).
Phase B (--llm-endpoint) optionally enriches analysis/fix_plan statements and adds
semantic fact nodes (API behavior / failing invariants) with a single
propose+critic call per instance — the only part that genuinely needs an LLM.

Output: JSONL, one graph per instance:
  {task_id, instance_id, source_dataset, nodes: [{id, type, statement,
   unlocker: {action, args}, entities, exec_required}], edges: [[u, v]], meta}

Usage:
  python build_process_graph.py \
    --parquet .../patch_generation/train_verl.parquet \
    --output graphs_train.jsonl [--limit 5] [--llm-endpoint http://...]
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------- diff parsing

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
# swe_smith golden patches were produced by reversing the bug-introducing diff,
# which leaves the headers as "--- b/path" / "+++ a/path"; accept either order.
FILE_RE = re.compile(r"^(?:---|\+\+\+) (?:[ab]/)?(.+)$")

TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing)(/|_)|(_test|test_)[^/]*\.py$")


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    old_lines: list = field(default_factory=list)   # removed + context (pre-fix)
    new_lines: list = field(default_factory=list)   # added + context (post-fix)
    removed: list = field(default_factory=list)     # (abs_old_lineno, text)
    added: list = field(default_factory=list)       # text


def parse_unified_diff(diff_text):
    """-> {file_path: [Hunk, ...]} for non-test files."""
    files, cur_file, cur_hunk, old_ln = {}, None, None, 0
    for raw in diff_text.splitlines():
        if raw.startswith("diff --git"):
            cur_file, cur_hunk = None, None
            continue
        if raw.startswith(("---", "+++")):
            m = FILE_RE.match(raw)
            if m and m.group(1) != "/dev/null":
                cur_file = m.group(1)
            continue
        if raw.startswith("@@"):
            m = HUNK_RE.match(raw)
            if not (m and cur_file):
                cur_hunk = None
                continue
            cur_hunk = Hunk(int(m.group(1)), int(m.group(2) or 1),
                            int(m.group(3)), int(m.group(4) or 1))
            old_ln = cur_hunk.old_start
            files.setdefault(cur_file, []).append(cur_hunk)
            continue
        if cur_hunk is None:
            continue
        if raw.startswith("-"):
            cur_hunk.old_lines.append(raw[1:])
            cur_hunk.removed.append((old_ln, raw[1:]))
            old_ln += 1
        elif raw.startswith("+"):
            cur_hunk.new_lines.append(raw[1:])
            cur_hunk.added.append(raw[1:])
        elif raw.startswith(" ") or raw == "":
            cur_hunk.old_lines.append(raw[1:])
            cur_hunk.new_lines.append(raw[1:])
            old_ln += 1
    return {f: hs for f, hs in files.items() if not TEST_PATH_RE.search(f)}


IDENT_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")


def hunk_symbols(hunk):
    """def/class names appearing in a hunk's pre- or post-fix text."""
    syms = set()
    for line in hunk.old_lines + hunk.new_lines:
        m = IDENT_RE.match(line)
        if m:
            syms.add(m.group(1))
    return sorted(syms)


def first_issue_excerpt(problem_statement, max_chars=400):
    """A deterministic issue-derived fact: prefer an error/traceback line,
    else the first non-empty paragraph."""
    for line in problem_statement.splitlines():
        s = line.strip()
        if re.search(r"(Error|Exception|Traceback|assert|FAILED|raise )", s):
            return s[:max_chars]
    for para in re.split(r"\n\s*\n", problem_statement):
        s = " ".join(para.split())
        if s:
            return s[:max_chars]
    return problem_statement[:max_chars]


# ---------------------------------------------------------------- graph build

def snip(lines, n=3):
    txt = "\n".join(lines[:n])
    return txt if len(txt) <= 240 else txt[:240] + "…"


def build_graph(ei):
    """extra_info dict -> graph dict (Phase A, deterministic)."""
    tp = ei.get("teacher_pi") or {}
    if isinstance(tp, str):
        try:
            tp = json.loads(tp)
        except json.JSONDecodeError:
            tp = {}
    gold = tp.get("golden_patch") or ei.get("patch") or ""
    files = parse_unified_diff(gold)
    if not files:
        return None

    f2p = ei.get("FAIL_TO_PASS") or []
    if isinstance(f2p, str):
        try:
            f2p = json.loads(f2p)
        except json.JSONDecodeError:
            f2p = [f2p]

    nodes, edges = [], []

    def add(ntype, statement, action, args, entities, exec_required=False):
        nid = f"n{len(nodes)}"
        nodes.append({
            "id": nid, "type": ntype, "statement": statement,
            "unlocker": {"action": action, "args": args},
            "entities": sorted(set(entities)), "exec_required": exec_required,
        })
        return nid

    # -- issue fact (VIEW problem_statement)
    excerpt = first_issue_excerpt(ei.get("problem_statement") or "")
    n_issue = add("fact", f"The issue reports: {excerpt}",
                  "VIEW", {"target": "problem_statement"}, [])

    # -- static facts: one localization node per file, one pre-fix-code node per hunk
    file_nodes, hunk_nodes = {}, []
    for fpath, hunks in files.items():
        lo = min(h.old_start for h in hunks)
        hi = max(h.old_start + h.old_count - 1 for h in hunks)
        syms = sorted({s for h in hunks for s in hunk_symbols(h)})
        stmt = (f"The code governing this issue lives in {fpath} "
                f"(pre-fix lines ~{lo}-{hi}"
                + (f", around {', '.join(syms[:4])}" if syms else "") + ").")
        file_nodes[fpath] = add(
            "fact", stmt, "VIEW", {"file": fpath, "lines": [lo, hi]},
            [fpath] + syms)
        for k, h in enumerate(hunks):
            if not h.removed:
                continue
            stmt = (f"The pre-fix code at {fpath}:{h.old_start}-"
                    f"{h.old_start + h.old_count - 1} reads:\n"
                    f"{snip([t for _, t in h.removed])}")
            hunk_nodes.append((fpath, k, add(
                "fact", stmt, "VIEW",
                {"file": fpath, "lines": [h.old_start,
                                          h.old_start + h.old_count - 1]},
                [fpath] + hunk_symbols(h))))

    # -- dynamic repro fact + reproduce milestone (dataset-verified, no sandbox)
    n_repro = None
    if f2p:
        tests_s = " ".join(f2p[:3]) + (" …" if len(f2p) > 3 else "")
        n_repro_fact = add(
            "fact",
            f"Pre-fix, the following test(s) FAIL: {tests_s}",
            "BASH", {"command": f"python -m pytest {' '.join(f2p[:3])}"},
            [t.split("::")[0] for t in f2p[:3]], exec_required=True)
        n_repro = add(
            "reproduce_script",
            "A runnable reproduction exists: executing the failing test(s) "
            "(or an equivalent minimal script) demonstrates the buggy behavior.",
            "BASH", {"command": f"python -m pytest {' '.join(f2p[:3])}"},
            [t.split("::")[0] for t in f2p[:3]], exec_required=True)
        edges.append([n_repro_fact, n_repro])
        edges.append([n_issue, n_repro_fact])

    # -- analysis + fix plan (deterministic templates; --llm-endpoint upgrades)
    all_files = sorted(files)
    n_analysis = add(
        "issue_analysis",
        "Root cause: the pre-fix logic in "
        + ", ".join(f"{f} (hunks at "
                    + ", ".join(str(h.old_start) for h in files[f]) + ")"
                    for f in all_files)
        + " produces the reported behavior. [template — enrich with --llm-endpoint]",
        "THINK", {}, all_files)
    n_plan = add(
        "fix_plan",
        "Plan: edit " + "; ".join(
            f"{f}:{h.old_start} ({len(h.removed)} line(s) replaced by "
            f"{len(h.added)})" for f in all_files for h in files[f])
        + ". [template — enrich with --llm-endpoint]",
        "THINK", {}, all_files)

    # -- code edits (byte-exact from the diff; P2T rule R6 is free here)
    edit_nodes = []
    for fpath, hunks in files.items():
        for h in hunks:
            old = "\n".join(h.old_lines)
            new = "\n".join(h.new_lines)
            edit_nodes.append(add(
                "code_edit",
                f"Apply the fix at {fpath}:{h.old_start}.",
                "STR_REPLACE",
                {"file": fpath, "old_str": old, "new_str": new},
                [fpath] + hunk_symbols(h)))

    # -- validation
    n_val = add(
        "validation",
        "Post-fix, the previously failing test(s) pass"
        + (" and the regression suite stays green." if f2p else "."),
        "BASH",
        {"command": f"python -m pytest {' '.join(f2p[:3])}" if f2p
         else "python -m pytest"},
        [t.split("::")[0] for t in f2p[:3]], exec_required=True)

    # -- prerequisite edges (P2T's deterministic post-pass):
    # (1) canonical artifact order repro -> analysis -> plan -> edit -> validation
    prev = n_repro or n_issue
    for nid in [n_analysis, n_plan]:
        edges.append([prev, nid])
        prev = nid
    for e in edit_nodes:
        edges.append([n_plan, e])
        edges.append([e, n_val])
    # (2) entity-reference wiring: a fact that first surfaces an entity precedes
    # any artifact whose unlocker references it
    for fpath, fnode in file_nodes.items():
        edges.append([n_issue, fnode])
        for hp, _, hnode in hunk_nodes:
            if hp == fpath:
                edges.append([fnode, hnode])
        for e in edit_nodes:
            enode = nodes[int(e[1:])]
            if fpath in enode["entities"]:
                edges.append([fnode, e])
    for hp, k, hnode in hunk_nodes:
        edges.append([hnode, n_analysis])

    edges = sorted({(u, v) for u, v in edges if u != v})
    return {
        "task_id": ei.get("task_id"),
        "instance_id": ei.get("instance_id"),
        "source_dataset": ei.get("source_dataset"),
        "nodes": nodes,
        "edges": [list(e) for e in edges],
        "meta": {
            "n_nodes": len(nodes),
            "n_static_facts": sum(1 for n in nodes if n["type"] == "fact"
                                  and not n["exec_required"]),
            "n_dynamic_facts": sum(1 for n in nodes if n["type"] == "fact"
                                   and n["exec_required"]),
            "n_artifacts": sum(1 for n in nodes if n["type"] != "fact"),
            "gold_files": sorted(files),
        },
    }


def check_dag(graph):
    """Kahn's algorithm; returns True iff acyclic."""
    ids = {n["id"] for n in graph["nodes"]}
    indeg = {i: 0 for i in ids}
    out = {i: [] for i in ids}
    for u, v in graph["edges"]:
        indeg[v] += 1
        out[u].append(v)
    q = [i for i in ids if indeg[i] == 0]
    seen = 0
    while q:
        u = q.pop()
        seen += 1
        for v in out[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return seen == len(ids)


# ------------------------------------------------- Phase B: LLM enrichment

ENRICH_PROMPT = """You are helping build a "latent process graph" for a software bug-fix task, \
in the style of P2T (privileged process supervision). You see PRIVILEGED information \
(the gold patch); your outputs will guide a teacher model, but must be phrased so a \
NON-privileged developer could plausibly discover them from the repository and issue alone.

## Issue
{issue}

## Gold patch (PRIVILEGED - never reference its existence in your outputs)
{patch}

## Failing tests (verified to fail pre-fix, pass post-fix)
{tests}

## Already-known facts (from deterministic analysis)
{seeded}

## Your tasks
1. "issue_analysis": 2-4 sentences stating the ROOT CAUSE, as a developer would after
   inspecting the code above. Reference concrete files/functions/behaviors.
2. "fix_plan": 2-4 sentences describing WHAT to change and WHY (files/symbols/logic),
   without quoting the literal post-fix code.
3. "semantic_facts": 0-5 additional prerequisite facts a solver must discover that the
   seeded facts do NOT cover: API contracts, failing invariants, runtime behaviors,
   cross-file relationships. Each has:
   - "statement": ONE claim, self-contained;
   - "kind": "static" (discoverable by reading/grep) or "dynamic" (requires running code);
   - "unlocker_action": "VIEW" (static) or "BASH" (dynamic);
   - "unlocker_args": {{"file": "...", "lines": [a, b]}} for VIEW, or {{"command": "..."}} for BASH.

## Hard rules (violations make the output unusable)
- R-leak: no statement or unlocker may mention the patch, "the fix", or post-fix code.
- R-one-claim: one claim per fact; split compounds.
- R-ground: every entity you reference must exist in the issue, seeded facts, or patch context shown above.
- Output STRICT JSON with keys: issue_analysis, fix_plan, semantic_facts. No markdown fences."""


def llm_call(endpoint, key, model, prompt, timeout=150, pace=None, attempts=4):
    """One paced, retrying call. The routify gateway wraps rate-limit and
    upstream errors in HTTP-200 bodies (routify_response.success=false, no
    content) — treat those as retriable failures, with exponential backoff."""
    import time
    import urllib.error
    import urllib.request
    # thinking disabled: ds.deepseek-v4-pro burns hidden reasoning tokens out
    # of max_tokens without returning them -- with it on, long prompts exhaust
    # the budget and yield EMPTY text (stop_reason=max_tokens, text="").
    body = json.dumps({
        "model": model, "max_tokens": 4000, "temperature": 0.4,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    last = None
    for k in range(attempts):
        if pace:
            pace()
        try:
            req = urllib.request.Request(
                endpoint.rstrip("/") + "/v1/messages", data=body,
                headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read())
            rr = resp.get("routify_response") or {}
            if rr and rr.get("success") is False:
                raise RuntimeError(f"gateway: {rr.get('error_message')}"[:150])
            text = "".join(b.get("text", "")
                           for b in resp.get("content", []))
            if not text.strip():
                raise RuntimeError(
                    f"empty content (stop={resp.get('stop_reason')})")
            return text
        except (RuntimeError, urllib.error.URLError, TimeoutError,
                json.JSONDecodeError) as e:
            last = e
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"llm_call failed after {attempts}: {last}")


def parse_enrichment(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        out = json.loads(t)
    except json.JSONDecodeError:
        # tolerate prose around the JSON: take the outermost {...} span
        i, j = t.find("{"), t.rfind("}")
        if i < 0 or j <= i:
            raise
        out = json.loads(t[i:j + 1])
    assert isinstance(out.get("issue_analysis"), str) and out["issue_analysis"]
    assert isinstance(out.get("fix_plan"), str) and out["fix_plan"]
    facts = out.get("semantic_facts") or []
    assert isinstance(facts, list)
    LEAK = re.compile(r"\b(patch|the fix|post-fix|golden|reference solution)\b", re.I)
    clean = []
    for f in facts[:5]:
        if not isinstance(f, dict) or not f.get("statement"):
            continue
        act = f.get("unlocker_action")
        if act not in ("VIEW", "BASH"):
            continue
        if LEAK.search(f["statement"]):        # critic-lite: drop leaky nodes
            continue
        clean.append(f)
    return out["issue_analysis"], out["fix_plan"], clean


def enrich_graph(g, ei, cfg):
    """Phase B: one propose+critic-lite call; mutates and returns g."""
    seeded = "\n".join(f"- {n['statement'][:200]}" for n in g["nodes"]
                       if n["type"] == "fact")
    tp = ei.get("teacher_pi") or {}
    if isinstance(tp, str):
        try:
            tp = json.loads(tp)
        except json.JSONDecodeError:
            tp = {}
    f2p = ei.get("FAIL_TO_PASS") or []
    if isinstance(f2p, str):
        try:
            f2p = json.loads(f2p)
        except json.JSONDecodeError:
            f2p = [f2p]
    prompt = ENRICH_PROMPT.format(
        issue=(ei.get("problem_statement") or "")[:4000],
        patch=(tp.get("golden_patch") or ei.get("patch") or "")[:8000],
        tests="\n".join(f2p[:5]) or "(none listed)",
        seeded=seeded[:3000])

    pace = cfg.get("pace")
    text = llm_call(cfg["endpoint"], cfg["key"], cfg["model"], prompt,
                    pace=pace)
    try:
        analysis, plan, facts = parse_enrichment(text)
    except (json.JSONDecodeError, AssertionError):
        text = llm_call(cfg["endpoint"], cfg["key"], cfg["model"],
                        prompt + "\n\nREMINDER: output STRICT JSON only.",
                        pace=pace)
        analysis, plan, facts = parse_enrichment(text)

    id2node = {n["id"]: n for n in g["nodes"]}
    n_analysis = n_issue = None
    for n in g["nodes"]:
        if n["type"] == "issue_analysis":
            n["statement"] = analysis
            n_analysis = n["id"]
        elif n["type"] == "fix_plan":
            n["statement"] = plan
        elif n["type"] == "fact" and n["unlocker"]["args"].get("target") == "problem_statement":
            n_issue = n["id"]

    for f in facts:
        nid = f"n{len(g['nodes'])}"
        ents = re.findall(r"[\w./]+\.py|\b[A-Za-z_]\w*(?=\()", f["statement"])[:8]
        g["nodes"].append({
            "id": nid, "type": "fact", "statement": f["statement"],
            "unlocker": {"action": f["unlocker_action"],
                         "args": f.get("unlocker_args") or {}},
            "entities": sorted(set(ents)),
            "exec_required": f.get("kind") == "dynamic",
            "llm_proposed": True,
        })
        if n_issue:
            g["edges"].append([n_issue, nid])
        if n_analysis:
            g["edges"].append([nid, n_analysis])
    g["edges"] = sorted({(u, v) for u, v in g["edges"] if u != v})
    g["edges"] = [list(e) for e in g["edges"]]
    g["meta"]["n_nodes"] = len(g["nodes"])
    g["meta"]["llm_enriched"] = True
    return g


# ------------------------------------------------------------------- driver

def main():
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--llm-endpoint", default=None,
                    help="Anthropic-protocol endpoint for Phase-B enrichment, "
                         "e.g. https://routify-pub.alibaba-inc.com/protocol/anthropic. "
                         "Key from env ROUTIFY_API_KEY. Omitted -> Phase A only.")
    ap.add_argument("--llm-model", default="ds.deepseek-v4-pro")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true",
                    help="skip instance_ids already present in --output")
    args = ap.parse_args()

    cfg = None
    if args.llm_endpoint:
        key = os.environ.get("ROUTIFY_API_KEY")
        assert key, "set ROUTIFY_API_KEY for --llm-endpoint"
        cfg = {"endpoint": args.llm_endpoint, "key": key, "model": args.llm_model}

    import pyarrow.parquet as pq
    t = pq.read_table(args.parquet)
    n = t.num_rows if not args.limit else min(args.limit, t.num_rows)

    done = set()
    mode = "w"
    if args.resume and os.path.exists(args.output):
        keep = []
        for line in open(args.output):
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                continue
            # with LLM enabled, only successfully-enriched rows count as done;
            # Phase-A-degraded rows are re-attempted
            if cfg and not g.get("meta", {}).get("llm_enriched"):
                continue
            done.add(g["instance_id"])
            keep.append(line)
        with open(args.output, "w") as f:
            f.writelines(keep)
        mode = "a"

    rows = []
    for i in range(n):
        ei = t.column("extra_info")[i].as_py()
        if isinstance(ei, str):
            ei = json.loads(ei)
        if ei.get("instance_id") in done:
            continue
        rows.append(ei)

    lock = threading.Lock()
    # stay under the endpoint's 60 rpm: min spacing between request starts,
    # applied to EVERY call (including parse-retries and backoff retries)
    rate_lock, last_start = threading.Lock(), [0.0]

    def paced():
        import time
        with rate_lock:
            wait = max(0.0, last_start[0] + 1.5 - time.monotonic())
            if wait:
                time.sleep(wait)
            last_start[0] = time.monotonic()

    if cfg:
        cfg["pace"] = paced

    ok = bad = llm_fail = 0
    stats = []
    fo = open(args.output, mode)

    def work(ei):
        nonlocal ok, bad, llm_fail
        g = build_graph(ei)
        if g is None:
            with lock:
                bad += 1
            return
        assert check_dag(g), f"cycle in graph for {g['instance_id']}"
        if cfg:
            try:
                g = enrich_graph(g, ei, cfg)
            except Exception as e:      # keep the Phase-A graph on LLM failure
                g["meta"]["llm_enriched"] = False
                g["meta"]["llm_error"] = f"{type(e).__name__}: {e}"[:200]
                with lock:
                    llm_fail += 1
        assert check_dag(g), f"cycle after enrich for {g['instance_id']}"
        with lock:
            fo.write(json.dumps(g, ensure_ascii=False) + "\n")
            fo.flush()
            ok += 1
            stats.append(g["meta"]["n_nodes"])
            if ok % 50 == 0:
                print(f"  progress: {ok}/{len(rows)} (llm_fail={llm_fail})",
                      flush=True)

    if cfg and args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, rows))
    else:
        for ei in rows:
            work(ei)
    fo.close()

    stats.sort()
    med = stats[len(stats) // 2] if stats else 0
    print(f"graphs: {ok} written, {bad} skipped, {llm_fail} LLM-degraded"
          f" (Phase-A fallback)")
    if stats:
        print(f"nodes/graph: min={stats[0]} median={med} max={stats[-1]} "
              f"(P2T reports median 18 with LLM-proposed semantic facts)")


if __name__ == "__main__":
    main()
