"""Process-graph privileged evidence for OPSD stage routing (P2T-style G*).

Replaces the flat per-stage privileged blocks (gold files / gold tests /
golden patch) with stage-scoped subgraphs of the per-instance latent process
graph built by scripts/data/process_graph/build_process_graph.py.

## Privilege visibility ladder (WHAT the teacher sees, PER STAGE)

The ladder is monotone (L ⊂ R ⊂ P) and follows P2T's canonical milestone
order (reproduce -> analysis -> plan -> edit -> validation): a stage's
teacher sees the graph up to (and including) that stage's own milestone
frontier, and NEVER a downstream milestone.

  causal_localization        -> FACT nodes only (issue fact, file/hunk
                                localization facts, semantic facts), each with
                                its unlocker ("how a non-privileged developer
                                would discover this"). No milestones: the
                                localization teacher guides WHERE/WHAT to look
                                at, not the solution.
  localized_issue_reproduction -> facts + reproduction milestones (failing
                                tests / reproduce_script nodes). No analysis /
                                plan / edit / validation.
  patch_generation           -> the full graph: facts + reproduction +
                                issue_analysis + fix_plan + code_edit
                                (byte-exact old/new) + validation. Privilege
                                parity with the old golden_patch block, but
                                structured as the reasoning chain that
                                MOTIVATES the fix.

Activation: OPD_TEACHER_EVIDENCE_MODE=graph plus a graph attached to
teacher_pi["process_graph"] (injected at entry render time from the JSONL at
OPD_PROCESS_GRAPH_PATH, keyed by task_id or instance_id). Anything missing ->
callers fall back to the flat evidence, so old parquets keep working.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

_STAGE_LOCALIZATION = "causal_localization"
_STAGE_REPRODUCTION = "localized_issue_reproduction"
_STAGE_PATCH = "patch_generation"

_REPRO_TYPES = {"reproduce_script"}
_PATCH_ONLY_TYPES = {"issue_analysis", "fix_plan", "code_edit", "validation"}

_graphs_lock = threading.Lock()
_graphs_cache: dict[str, dict[str, Any]] | None = None
_graphs_path_loaded: str | None = None


def graph_mode_enabled() -> bool:
    return os.environ.get("OPD_TEACHER_EVIDENCE_MODE", "flat").strip().lower() == "graph"


def _load_graphs() -> dict[str, dict[str, Any]]:
    """Load and cache the graphs JSONL, indexed by task_id AND instance_id."""
    global _graphs_cache, _graphs_path_loaded
    path = os.environ.get("OPD_PROCESS_GRAPH_PATH", "")
    with _graphs_lock:
        if _graphs_cache is not None and _graphs_path_loaded == path:
            return _graphs_cache
        index: dict[str, dict[str, Any]] = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        g = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for key in (g.get("task_id"), g.get("instance_id")):
                        if key:
                            index[str(key)] = g
        _graphs_cache = index
        _graphs_path_loaded = path
        return index


def lookup_graph(*keys: Any) -> dict[str, Any] | None:
    graphs = _load_graphs()
    for k in keys:
        if k and str(k) in graphs:
            return graphs[str(k)]
    return None


def inject_graph_into_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Attach the instance's process graph to every teacher_pi carried by the
    entry (top-level and privileged_context), so the trainer's per-stage
    teacher variants see it downstream. No-op unless graph mode is on and a
    graph exists for this instance."""
    if not graph_mode_enabled():
        return entry
    graph = lookup_graph(entry.get("task_id"), entry.get("instance_id"))
    if graph is None:
        return entry
    for holder in (entry, entry.get("raw_record"), entry.get("privileged_context")):
        if not isinstance(holder, dict):
            continue
        tp = holder.get("teacher_pi")
        if isinstance(tp, dict):
            tp["process_graph"] = graph
        elif isinstance(tp, str):
            try:
                tp_d = json.loads(tp)
            except json.JSONDecodeError:
                continue
            if isinstance(tp_d, dict):
                tp_d["process_graph"] = graph
                holder["teacher_pi"] = tp_d
    return entry


def _topo_order(graph: dict[str, Any]) -> list[str]:
    ids = [n["id"] for n in graph.get("nodes", [])]
    pos = {i: k for k, i in enumerate(ids)}
    indeg = {i: 0 for i in ids}
    out: dict[str, list[str]] = {i: [] for i in ids}
    for u, v in graph.get("edges", []):
        if u in indeg and v in indeg:
            indeg[v] += 1
            out[u].append(v)
    ready = sorted([i for i in ids if indeg[i] == 0], key=pos.get)
    order = []
    while ready:
        u = ready.pop(0)
        order.append(u)
        for v in out[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
        ready.sort(key=pos.get)
    return order if len(order) == len(ids) else ids


def _describe_unlocker(unlocker: dict[str, Any]) -> str:
    action = str(unlocker.get("action", ""))
    args = unlocker.get("args") or {}
    if action == "VIEW":
        if args.get("target") == "problem_statement":
            return "read the issue statement"
        lines = args.get("lines")
        loc = f"{args.get('file', '?')}" + (f" lines {lines[0]}-{lines[1]}" if isinstance(lines, list) and len(lines) == 2 else "")
        return f"view {loc}"
    if action == "BASH":
        return f"run: {args.get('command', '?')}"
    if action == "STR_REPLACE":
        return f"edit {args.get('file', '?')}"
    if action == "CREATE":
        return f"create {args.get('path', '?')}"
    if action == "THINK":
        return "reason from the facts above"
    return action or "?"


def _node_visible(node: dict[str, Any], stage: str) -> bool:
    ntype = str(node.get("type", ""))
    if ntype == "fact":
        return True
    if ntype in _REPRO_TYPES:
        return stage in (_STAGE_REPRODUCTION, _STAGE_PATCH)
    if ntype in _PATCH_ONLY_TYPES:
        return stage == _STAGE_PATCH
    return stage == _STAGE_PATCH  # unknown milestone types: most-privileged only


def graph_evidence_for_stage(graph: dict[str, Any], stage: str) -> dict[str, Any] | None:
    """Render the stage-visible subgraph as the teacher evidence dict that
    fills {teacher_evidence_json} in the end-to-end teacher template."""
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    if not nodes:
        return None
    order = _topo_order(graph)
    facts, milestones = [], []
    for nid in order:
        n = nodes.get(nid)
        if n is None or not _node_visible(n, stage):
            continue
        item: dict[str, Any] = {
            "claim": n.get("statement", ""),
            "how_a_developer_discovers_this": _describe_unlocker(n.get("unlocker") or {}),
        }
        if n.get("type") == "fact":
            facts.append(item)
        else:
            item["milestone_type"] = n.get("type")
            unlocker = n.get("unlocker") or {}
            if n.get("type") == "code_edit":
                args = unlocker.get("args") or {}
                item["edit"] = {
                    "file": args.get("file", ""),
                    "old_str": args.get("old_str", ""),
                    "new_str": args.get("new_str", ""),
                }
            milestones.append(item)
    if not facts and not milestones:
        return None
    evidence: dict[str, Any] = {
        "verified_process_knowledge": {
            "contextual_facts_in_discovery_order": facts,
        }
    }
    if milestones:
        evidence["verified_process_knowledge"]["solution_milestones_in_order"] = milestones
    return evidence


def maybe_graph_evidence(stage: str, teacher_pi: dict[str, Any]) -> dict[str, Any] | None:
    """Entry point for staged_dataset._verified_repository_evidence: returns
    the graph evidence, or None to fall back to the flat evidence."""
    if not graph_mode_enabled():
        return None
    graph = teacher_pi.get("process_graph")
    if not isinstance(graph, dict):
        return None
    try:
        return graph_evidence_for_stage(graph, stage)
    except Exception:
        return None
