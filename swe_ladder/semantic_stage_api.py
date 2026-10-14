"""Semantic subtask labeling for OPSD stage routing via a small hosted model.

Replaces/augments the action-signature rule router (stage_classifier.StageStateMachine)
with a purpose-based 4-class subtask label (LOCATE/REPRODUCE/PATCH/VERIFY) obtained from
a small LLM (default qwen-flash) over a compact window (last 2 steps + current thought/
action/observation head). Rubric: subtask_prompt_v3.md (privilege-purpose framing, rules
R1-R7), measured at 112/119 = 94.1% against the 10-trajectory human-adjudicated gold set
shipped as data/stage_router_gold/gold_119.jsonl (rubric frozen 2026-09-28). That score
needs the hosted gateway and so cannot be re-run offline; the rule router's reproducible
baselines on the same gold are 113/119 (v1) and 98/119 (v2) via
analysis/stage_ladder/validate_router_gold.py. The training runs used v2.

Subtask -> stage mapping for privileged routing:
  LOCATE    -> causal_localization
  REPRODUCE -> localized_issue_reproduction
  PATCH     -> patch_generation
  VERIFY    -> patch_generation   (post-successful-edit verification/closure; the gold-patch
                                   teacher is the appropriate evidence for verifying a fix)

Failure policy: any exception/timeout/bad-label returns None and the CALLER falls back to
the rule state machine, so training never blocks on the gateway. Calls are meant to be
made from a worker thread (asyncio.to_thread) - they are blocking HTTP.
"""
from __future__ import annotations

import os
import re

import requests

LOCALIZATION = "causal_localization"
REPRODUCTION = "localized_issue_reproduction"
PATCH = "patch_generation"

SUBTASK2STAGE = {
    "LOCATE": LOCALIZATION,
    "REPRODUCE": REPRODUCTION,
    "PATCH": PATCH,
    "VERIFY": PATCH,
}

_LT = "<"
_FUNC_SPLIT_RE = re.compile(_LT + r"function=")
_LABEL_RE = re.compile(r"LOCATE|REPRODUCE|PATCH|VERIFY")
_WS_RE = re.compile(r"\s+")

_DEFAULT_PROMPT = os.path.join(os.path.dirname(__file__), "subtask_prompt_v3.md")
_prompt_cache: dict[str, str] = {}


def load_prompt(path: str | None = None) -> str:
    p = path or os.environ.get("OPSD_STAGE_API_PROMPT_FILE") or _DEFAULT_PROMPT
    if p not in _prompt_cache:
        with open(p) as f:
            _prompt_cache[p] = f.read()
    return _prompt_cache[p]


def _scrub(s: str, n: int) -> str:
    return _WS_RE.sub(" ", (s or ""))[:n]


def build_window(messages, assistant_message, observation) -> str:
    """Window byte-identical to the harness the 119-step gold score was measured with:
    '[prev {j}] <scrubbed action 150> -> <raw obs 80>' x2 + '[current {i}] thought/action/obs'.
    Changing the window invalidates that measurement."""
    idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i] is assistant_message:
            idx = i
            break
    if idx is None:
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content") == assistant_message.get("content"):
                idx = i
                break
    if idx is None:
        return ""
    step_i = sum(1 for m in messages[: idx + 1] if isinstance(m, dict) and m.get("role") == "assistant") - 1
    asst_idx = [k for k, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant"]
    pos = asst_idx.index(idx) if idx in asst_idx else step_i
    parts = []
    for j in (pos - 2, pos - 1):
        if j < 0:
            continue
        a = messages[asst_idx[j]]
        o = messages[asst_idx[j] + 1] if asst_idx[j] + 1 < len(messages) else {}
        parts.append(f"[prev {j}] {_scrub(str(a.get('content', '')), 150)} -> {str(o.get('content', ''))[:80]}")
    cur = str(assistant_message.get("content", ""))
    thought = _scrub(_FUNC_SPLIT_RE.split(cur)[0], 500)
    act = _scrub(cur, 220)
    parts.append(f"[current {pos}] thought: {thought}\naction: {act}\nobservation: {_scrub(str(observation), 200)}\nLabel:")
    return "\n".join(parts)


def label_via_api(window: str, model: str | None = None, base: str | None = None,
                  key: str | None = None, timeout: float = 8.0) -> str | None:
    """Return one of LOCATE/REPRODUCE/PATCH/VERIFY, or None on any failure."""
    if not window:
        return None
    model = model or os.environ.get("OPSD_STAGE_API_MODEL", "qwen-flash")
    base = base or os.environ.get("OPSD_STAGE_API_BASE", "https://dashscope.aliyuncs.com/apps/anthropic")
    key = key or os.environ.get("OPSD_STAGE_API_KEY", "")
    if not key:
        return None
    prompt = load_prompt() + "\n" + window
    for attempt in (1, 2):
        try:
            r = requests.post(
                base.rstrip("/") + "/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": model, "max_tokens": 6, "temperature": 0.0,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=timeout,
            )
            if r.status_code == 429 and attempt == 1:
                import time as _t
                _t.sleep(1.0)
                continue
            r.raise_for_status()
            txt = r.json()["content"][0]["text"].strip().upper()
            m = _LABEL_RE.search(txt)
            return m.group(0) if m else None
        except Exception:
            if attempt == 1:
                continue
            return None
    return None


def to_stage(subtask: str | None) -> str | None:
    return SUBTASK2STAGE.get(subtask or "")
