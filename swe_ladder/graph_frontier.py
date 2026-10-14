"""Dynamic frontier tracking over the latent process graph (P2T-faithful).

P2T's forward phase does NOT classify the trajectory into stages: it maintains
the established node set U_t = {v : Est(v, h_t)} and always targets the
*available frontier* A_t = {v not in U_t : Pred(v) subset of U_t}. Privileged
guidance unlocks with the trajectory's actual progress.

This module transplants that to on-policy OPSD token-level distillation:

- `FrontierTracker` consumes the student's (action_text, observation) step
  stream and maintains U via SYMBOLIC establishment checks (P2T uses an LLM
  verifier per step; that is unaffordable online, so we match unlocker actions
  and node entities/snippets against the accumulated visible history).
- Whenever U grows, a new *epoch* begins. Tokens generated within an epoch
  share one teacher context = established nodes (compact, "already surfaced")
  + current frontier (detailed, with unlockers, "discover next"). Nodes whose
  prerequisites are unmet are NEVER shown — milestones (analysis/plan/edit/
  validation) unlock only after their prerequisite facts, recovering the old
  static ladder as a special case but per-trajectory and adaptive.
- Epoch count is capped (default 6): after the cap, later establishment events
  keep updating U but stay in the last context slot. The per-token epoch ids
  ride the existing stage-routing rail (`stage_token_ids`), and the trainer
  renders one teacher variant per epoch instead of one per fixed stage.

Enable with OPSD_GRAPH_FRONTIER=1 (requires OPSD_STAGE_ROUTING=1 rails and
OPD_TEACHER_EVIDENCE_MODE=graph). OPSD_GRAPH_FRONTIER_CONTEXTS caps slots.
"""

from __future__ import annotations

import os
import re
from typing import Any

_MILESTONE_TYPES = ("reproduce_script", "issue_analysis", "fix_plan", "code_edit", "validation")
_FAIL_MARKERS = ("FAILED", "Error", "error", "Traceback", "assert", "FAIL")
# Infra files that gold patches (esp. swe_gym) bundle alongside the real fix:
# CI workflows, dependency pins, packaging. Their fact/edit nodes are noise in
# the teacher context and their edits over-strictify milestone prerequisites
# (a validation gated on an untouched requirements.txt edit never unlocks).
_INFRA_FILE_RE = re.compile(
    r"(^|/)\.github/|(^|/)requirements[^/]*\.txt$|(^|/)setup\.(py|cfg)$|"
    r"\.ya?ml$|\.cfg$|\.ini$|\.toml$|(^|/)Makefile$|(^|/)Dockerfile$"
)


def _node_file(n: dict[str, Any]) -> str:
    args = (n.get("unlocker") or {}).get("args") or {}
    return str(args.get("file") or args.get("path") or "")


def frontier_enabled() -> bool:
    return os.environ.get("OPSD_GRAPH_FRONTIER", "0") == "1"


def frontier_max_contexts() -> int:
    try:
        return max(2, int(os.environ.get("OPSD_GRAPH_FRONTIER_CONTEXTS", "6")))
    except ValueError:
        return 6


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _sig_lines(lines: list[str], min_len: int = 12) -> list[str]:
    """Distinctive line fragments for matching hunk content in observations."""
    out = []
    for ln in lines:
        s = _norm(ln)
        if len(s) >= min_len:
            out.append(s)
    return out[:6]


class FrontierTracker:
    """Symbolic Est() over a process graph, driven by the step stream."""

    def __init__(self, graph: dict[str, Any], max_contexts: int | None = None):
        nodes_all = {n["id"]: n for n in graph.get("nodes", [])}
        # Drop infra-file fact/edit nodes (CI, requirements, packaging) UNLESS
        # the fix itself lives only in infra files (then keep everything).
        infra_ids = {
            nid for nid, n in nodes_all.items()
            if n.get("type") in ("fact", "code_edit")
            and _node_file(n) and _INFRA_FILE_RE.search(_node_file(n))
        }
        remaining_edits = [
            nid for nid, n in nodes_all.items()
            if n.get("type") == "code_edit" and nid not in infra_ids
        ]
        if not remaining_edits:
            infra_ids = set()
        self.nodes = {nid: n for nid, n in nodes_all.items() if nid not in infra_ids}
        self.preds: dict[str, set[str]] = {i: set() for i in self.nodes}
        for u, v in graph.get("edges", []):
            if u in self.nodes and v in self.nodes:
                self.preds[v].add(u)
        self.established: set[str] = set()
        self.max_contexts = max_contexts or frontier_max_contexts()
        self.epoch = 0
        self.snapshots: list[dict[str, Any]] = []  # evidence at each epoch's START
        self._history_blob = ""  # accumulated visible text (bounded)
        self._had_source_edit = False
        self._had_test_run_after_edit = False
        # issue fact is established from step 0 (the issue is in the prompt)
        for nid, n in self.nodes.items():
            if (n.get("unlocker") or {}).get("args", {}).get("target") == "problem_statement":
                self.established.add(nid)
        # precompute matching signatures per node
        self._sigs: dict[str, dict[str, Any]] = {}
        for nid, n in self.nodes.items():
            u = n.get("unlocker") or {}
            args = u.get("args") or {}
            stmt = n.get("statement") or ""
            snippet_lines = []
            m = re.search(r"reads:\n(.*)$", stmt, re.S)
            if m:
                snippet_lines = _sig_lines(m.group(1).splitlines())
            self._sigs[nid] = {
                "action": u.get("action"),
                "file": args.get("file") or args.get("path") or "",
                "command": args.get("command") or "",
                "old_str_sig": _sig_lines((args.get("old_str") or "").splitlines()),
                "new_str_sig": _sig_lines((args.get("new_str") or "").splitlines()),
                "snippet": snippet_lines,
                "entities": [e for e in (n.get("entities") or []) if len(str(e)) > 3],
                "type": n.get("type"),
            }
        self.snapshots.append(self.snapshot())  # epoch-0 evidence (initial frontier)

    # ---------------------------------------------------------- establishment

    def _effective_established(self) -> set[str]:
        """Progress overrides missed symbolic matches: once the student has
        begun editing source, remaining FACT prerequisites no longer gate
        milestone availability (P2T's Est is LLM-judged and would have caught
        real discovery; our symbolic proxy under-establishes). The un-matched
        facts stay in the frontier display — truthful 'still discoverable'."""
        if not self._had_source_edit:
            return self.established
        return self.established | {
            nid for nid, n in self.nodes.items() if n.get("type") == "fact"
        }

    def frontier(self) -> set[str]:
        eff = self._effective_established()
        out = set()
        for nid in self.nodes:
            if nid in self.established:
                continue
            if self.nodes[nid].get("type") == "validation":
                # Relaxed prerequisite: validating is available as soon as ANY
                # code edit is established (requiring ALL edits deadlocks when
                # the gold patch bundles edits the student never needs).
                if any(
                    self.nodes[p].get("type") == "code_edit" and p in self.established
                    for p in self.preds[nid]
                ):
                    out.add(nid)
                continue
            if self.preds[nid] <= eff:
                out.add(nid)
        return out

    def _est_check(self, nid: str, step_text: str, blob: str) -> bool:
        sig = self._sigs[nid]
        ntype = sig["type"]
        f = sig["file"]
        if ntype == "fact":
            if sig["action"] == "VIEW":
                if sig["snippet"]:
                    # pre-fix-code fact: its content must have surfaced
                    return any(s in blob for s in sig["snippet"])
                # file-localization fact: the file must have been touched/seen
                return bool(f) and f in blob
            if sig["action"] == "BASH":
                # dynamic fact (failing tests): a run touching the test file
                # with a failure marker in the visible history. Match on the
                # node's file entities (test file paths), not the exact test id.
                targets = [t for t in ([f] + [str(e) for e in sig["entities"]]) if t]
                return any(t in blob for t in targets) and any(m in blob for m in _FAIL_MARKERS)
            return bool(sig["entities"]) and all(str(e) in blob for e in sig["entities"][:2])
        if ntype == "reproduce_script":
            targets = [t for t in ([f] + [str(e) for e in sig["entities"]]) if t]
            ran_something = ("python" in blob or "pytest" in blob)
            hit = (not targets) or any(t in blob for t in targets) or "repro" in blob
            return ran_something and any(m in blob for m in _FAIL_MARKERS) and hit
        if ntype == "issue_analysis":
            # symbolically: analysis is "done" once every prerequisite fact is
            # established AND the student has begun editing source
            return self._had_source_edit
        if ntype == "fix_plan":
            return self._had_source_edit
        if ntype == "code_edit":
            # the edit's old content was replaced: old_str fragment appeared in
            # an edit action on that file (str_replace echo/confirmation)
            if not f or f not in blob:
                return False
            return any(s in blob for s in sig["old_str_sig"][:3]) or any(
                s in blob for s in sig["new_str_sig"][:3])
        if ntype == "validation":
            return self._had_test_run_after_edit
        return False

    def update(self, action_text: str, observation: str) -> int:
        """Consume one completed step; returns the epoch id for the NEXT step's
        tokens. Establishment uses only prerequisite-satisfied nodes (frontier),
        mirroring P2T's dependency-respecting coverage."""
        step_text = _norm(action_text)[:4000] + "\n" + _norm(observation)[:6000]
        self._history_blob = (self._history_blob + "\n" + step_text)[-200_000:]
        edit_before = self._had_source_edit
        if "str_replace" in step_text or "file has been edited" in step_text or "edit_file" in step_text:
            self._had_source_edit = True
        if self._had_source_edit and ("pytest" in step_text or "python" in step_text):
            self._had_test_run_after_edit = True
        edit_transition = self._had_source_edit and not edit_before

        gained = False
        # establishment restricted to the frontier (dependency-respecting);
        # iterate until fixpoint so a step that unlocks a chain (fact -> plan)
        # establishes all of it in one pass
        while True:
            newly = [nid for nid in self.frontier()
                     if self._est_check(nid, step_text, self._history_blob)]
            if not newly:
                break
            self.established.update(newly)
            gained = True
        # First source edit switches context even without a node match: it flips
        # the progress override, exposing milestone guidance in the frontier.
        gained = gained or edit_transition
        if gained and self.epoch < self.max_contexts - 1:
            self.epoch += 1
            self.snapshots.append(self.snapshot())
        return self.epoch

    def snapshot(self) -> dict[str, Any]:
        """Evidence dict for the CURRENT state: established (compact) +
        frontier (detailed with unlockers). Never shows prerequisite-unmet
        nodes. This fills {teacher_evidence_json} for this epoch's variant."""
        from .process_graph_evidence import _describe_unlocker

        est_items, frontier_items = [], []
        frontier_now = self.frontier()
        for nid in self.nodes:
            n = self.nodes[nid]
            if nid in self.established:
                est_items.append(n.get("statement", "")[:300])
            elif nid in frontier_now:
                item: dict[str, Any] = {
                    "claim": n.get("statement", ""),
                    "how_to_establish_next": _describe_unlocker(n.get("unlocker") or {}),
                }
                if n.get("type") in _MILESTONE_TYPES:
                    item["milestone_type"] = n.get("type")
                    if n.get("type") == "code_edit":
                        args = (n.get("unlocker") or {}).get("args") or {}
                        item["edit"] = {
                            "file": args.get("file", ""),
                            "old_str": args.get("old_str", ""),
                            "new_str": args.get("new_str", ""),
                        }
                frontier_items.append(item)
        return {
            "verified_process_knowledge": {
                "already_established_by_your_investigation": est_items,
                "available_next_discoveries": frontier_items,
            }
        }
