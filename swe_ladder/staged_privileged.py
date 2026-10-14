"""Per-instance, per-stage privileged-context provider for end-to-end OPSD.

For the end-to-end OPSD baseline a single trajectory spans all three stages, so the
OPD teacher must score each step's tokens under the *stage-appropriate* privileged
information. The three staged datasets already contain the same instances (task
intersection) with stage-specific `teacher_messages` pre-built in `extra_info`. This
module builds a lookup

    instance_id -> {stage: teacher_messages}

by reading the three staged parquet files once, so per-step routing is just a dict
lookup (no teacher model, no re-derivation).
"""

from __future__ import annotations

import json
import os
from typing import Any

from .stage_classifier import LOCALIZATION, REPRODUCTION, PATCH

STAGE_DIRS = {
    LOCALIZATION: "causal_localization",
    REPRODUCTION: "localized_issue_reproduction",
    PATCH: "patch_generation",
}


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return {}
    return {}


def _extract(extra_info: Any):
    """Return (instance_id, stage, teacher_messages) from an extra_info record."""
    ei = _as_dict(extra_info)
    iid = ei.get("instance_id")
    stage = ei.get("stage")
    tm = ei.get("teacher_messages")
    # teacher_messages may be a numpy array of dicts; normalize to a plain list
    if tm is not None and not isinstance(tm, list):
        try:
            tm = list(tm)
        except Exception:
            tm = None
    return iid, stage, tm


def load_staged_teacher_messages(base_dir: str, split: str = "train") -> dict[str, dict[str, Any]]:
    """Build {instance_id: {stage: teacher_messages}} from the three staged parquets.

    Uses the canonical stage keys (causal_localization / localized_issue_reproduction /
    patch_generation). Missing stages for an instance are simply absent from its dict.
    """
    import pandas as pd

    lookup: dict[str, dict[str, Any]] = {}
    for stage_key, subdir in STAGE_DIRS.items():
        path = os.path.join(base_dir, subdir, f"{split}_verl.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path, columns=["extra_info"])
        for extra_info in df["extra_info"].tolist():
            iid, stage, tm = _extract(extra_info)
            if not iid or not tm:
                continue
            lookup.setdefault(iid, {})[stage_key] = tm
    return lookup


def get_stage_teacher_messages(
    lookup: dict[str, dict[str, Any]],
    instance_id: str,
    stage: str,
    fallback_stage: str = PATCH,
) -> Any | None:
    """Look up teacher_messages for (instance_id, stage), falling back to another stage
    (default patch) then to any available stage. Returns None if nothing is found."""
    per_stage = lookup.get(instance_id)
    if not per_stage:
        return None
    if stage in per_stage:
        return per_stage[stage]
    if fallback_stage in per_stage:
        return per_stage[fallback_stage]
    # last resort: any available stage, in canonical order
    for s in (PATCH, REPRODUCTION, LOCALIZATION):
        if s in per_stage:
            return per_stage[s]
    return None
