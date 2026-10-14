"""§4/§ data reconstruction for the fork-suppression analysis (OFFLINE, faithful).

We REUSE the training pipeline's own primitives so the reconstructed tensors are
byte-identical to what the engine produced at rollout time (NOT an approximation):

  * response tokens/masks         -> rllm.agents.utils.convert_messages_to_tokens_and_masks
                                     (the engine itself re-encodes the assistant message text
                                      via this same fn to build response_tokens; it does NOT
                                      keep vLLM's raw sampled ids -- see
                                      agent_execution_engine.py:1078. So re-encoding offline
                                      reproduces the exact ids.)
  * per-token stage labels        -> swe_ladder.stage_classifier.StageStateMachine
                                     (backtrack_threshold matches the run's OPSD_STAGE_BACKTRACK)
  * step segmentation             -> rllm.agents.utils.get_recent_assistant_user_messages logic
  * chat template / parser        -> rllm.parser.chat_template.parser.ChatTemplateParser
  * teacher contexts (WA + 3 SR)  -> swe_ladder.runtime_prompt.render_end_to_end_teacher_messages

Engine step order faithfully replicated (agent_execution_engine.py:1047-1069, 1074-1121):
  for each step i: stage_i = STAGE_ORDER[stage_sm.update(assistant_i, observation_i)];
  response += assistant_tokens(mask=1) + env_tokens(mask=0); stage_ids += [stage_i]*(both).

The per-token analysis position t is the index (in the flat response) of an emitted token
whose *generating* distribution we probe. In teacher-forcing over [prefix]+[response], the
distribution that emitted response[t] is read from logits[len(prefix)+t-1]; forwards.py owns
that offset. data_gen only records the response index t, its pos_class, stage, emitted id.

Minor documented fidelity notes (-> REPORT §10 / DEVIATIONS):
  * observation string fed to StageStateMachine.update offline == the step's tool/user message
    content(s) (the text the model actually saw); the engine passed str(env observation), which
    is the same text pre-agent-wrapping. Stage transitions are action-driven, so this is safe.
  * instance_id is not stored in chat_completions; we recover it by matching the raw github
    issue embedded in the rollout's user message against the staged parquet's rendered `prompt`
    column (see instance_id recovery below). The earlier "problem_statement prefix" match was a
    bug: the user message wraps the real issue behind a long SHARED boilerplate (the
    "[SYSTEM] ... BEGIN FUNCTION" tool preamble, byte-identical across all instances) and the
    parquet's `problem_statement` field is itself the full staged patch-prompt (same boilerplate
    + trailing teacher-privilege), so prefix keys collided on boilerplate and the loose
    containment fallback collapsed every rollout onto one arbitrary instance (n_unique=1 ->
    degenerate by-instance bootstrap). The fix isolates the real issue (anchor
    "Now consider the following Github issue:" -> last <github_issue> .. </github_issue>) and
    matches a distinctive middle chunk against the parquet `prompt` text, which is unique per
    instance even across swe_smith mutations of the same repo.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

# --- reuse training primitives (do NOT reimplement) ---
# swe_ladder comes from this repo; rllm.agents / rllm.parser come from the rllm checkout
# the run trained with (RLLM_ROOT), because the point is to re-encode exactly as the
# engine does, not to approximate it.
_ROOT = os.environ.get("SWE_LADDER_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
for _p in (_ROOT, os.environ.get("RLLM_ROOT", "")):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from rllm.agents.utils import (  # noqa: E402
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)
from rllm.parser.chat_template.parser import ChatTemplateParser  # noqa: E402
from swe_ladder.stage_classifier import (  # noqa: E402
    STAGE_ORDER,
    LOCALIZATION,
    StageStateMachine,
)
from swe_ladder.runtime_prompt import render_end_to_end_teacher_messages  # noqa: E402

from freeze_rules import (  # type: ignore  # noqa: E402
    FUNCTION_ONSET_LITERAL,
    classify_action_onset,
    find_marker_token_positions,
)

# stage id -> stage name (for SR rendering); STAGE_ORDER maps name->id
_ID_TO_STAGE = {v: k for k, v in STAGE_ORDER.items()}
# The WA (whole-answer) teacher = force-PATCH clean render for ALL tokens (frozen decision).
WA_STAGE_NAME = "patch_generation"


# ---------------------------------------------------------------------------
# instance_id recovery: real github issue (rollout user msg) -> staged parquet `prompt` text
# ---------------------------------------------------------------------------
_ISSUE_ANCHOR = "Now consider the following Github issue:"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def extract_issue_text(user_content: str) -> str:
    """Isolate the REAL github issue from a rollout user message.

    The message is [shared boilerplate + tool preamble] ++ (anchor) ++ <github_issue> REAL
    </github_issue> ++ trailing instructions (which themselves mention <github_issue>). We anchor
    on the "Now consider the following Github issue:" line, then take the FIRST <github_issue>
    block after it (a rfind would land on the trailing-instructions mention). Some renders end the
    block with <END WRITING> before the close tag, so we also trim that.
    """
    c = user_content or ""
    a = c.find(_ISSUE_ANCHOR)
    seg = c[a:] if a >= 0 else c
    s = seg.find("<github_issue>")
    if s >= 0:
        seg = seg[s + len("<github_issue>"):]
    e = seg.find("</github_issue>")
    if e >= 0:
        seg = seg[:e]
    w = seg.find("<END WRITING>")
    if w >= 0:
        seg = seg[:w]
    return seg


def _prompt_text(prompt) -> str:
    """Flatten a parquet `prompt` cell (list[{role,content}] or str) to one string."""
    if isinstance(prompt, str):
        return prompt
    parts = []
    for m in (prompt or []):
        if isinstance(m, dict):
            c = m.get("content", "")
            if isinstance(c, list):
                c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
            parts.append(str(c))
    return "\n".join(parts)


def build_record_index(parquet_path: str) -> list:
    """Return [(normalized_prompt_text, raw_record), ...] from a staged parquet.

    raw_record carries {task_id, stage, student_input, teacher_pi} (+ instance_id/source_dataset)
    for teacher rendering. We match against the rendered `prompt` column (the exact text fed to
    the policy at rollout time), which embeds the real issue verbatim and is unique per instance.
    """
    import pyarrow.parquet as pq

    tbl = pq.read_table(parquet_path, columns=["prompt", "extra_info"])
    prompts = tbl.column("prompt").to_pylist()
    eis = tbl.column("extra_info").to_pylist()
    index: list = []
    for p, ei in zip(prompts, eis):
        e = json.loads(ei) if isinstance(ei, str) else (ei or {})
        rec = dict(e.get("raw_record") or {})
        rec.setdefault("instance_id", e.get("instance_id"))
        rec.setdefault("source_dataset", e.get("source_dataset"))
        index.append((_normalize(_prompt_text(p)), rec))
    return index


def match_record(user_content: str, index: list) -> Optional[dict]:
    """Match a rollout to its staged record by containment of the real issue in the parquet prompt.

    Uses a distinctive middle chunk of the normalized issue (chars 80:580) to skip the generic
    title/instruction head that swe_smith mutations of the same repo share. Falls back to full-issue
    containment on ties. Returns None if unmatched or genuinely ambiguous.
    """
    issue = _normalize(extract_issue_text(user_content))
    if len(issue) < 50:
        return None
    chunk = issue[80:580] if len(issue) > 200 else issue
    hits = [rec for ptext, rec in index if chunk and chunk in ptext]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        full = [rec for ptext, rec in index if issue in ptext]
        return full[0] if len(full) == 1 else None  # ambiguous -> drop (do not fabricate a match)
    # no chunk hit: last-resort full-issue containment
    for ptext, rec in index:
        if issue and issue in ptext:
            return rec
    return None


# ---------------------------------------------------------------------------
# tokenizer / parser
# ---------------------------------------------------------------------------
def load_tokenizer_and_parser(model_path: str, disable_thinking: bool = False):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    parser = ChatTemplateParser.get_parser(tok, disable_thinking=disable_thinking)
    return tok, parser


# ---------------------------------------------------------------------------
# trajectory reconstruction
# ---------------------------------------------------------------------------
@dataclass
class Position:
    t: int                 # index into the flat response token sequence
    pos_class: str         # EXPLORE | COMMIT | SUBMIT | OTHER (action-onset) or EXPLORE (marker)
    kind: str              # "action_onset" | "marker"
    stage_id: int          # 0/1/2 (-1 if unlabeled)
    emitted_token_id: int  # response_tokens[t]


@dataclass
class ReconTrajectory:
    instance_id: Optional[str]
    source_dataset: Optional[str]
    traj_score: Optional[float]
    prompt_messages: list           # [system, user, ...] up to first assistant
    prompt_tokens: list             # ids
    response_tokens: list           # ids (assistant + env interleaved, per step)
    response_masks: list            # 1 = assistant (opsd loss position), 0 = env
    stage_ids: list                 # per response token, 0/1/2 (-1 if disabled)
    positions: list                 # list[Position]
    teacher_messages_by_key: dict   # {"wa","causal_localization","localized_issue_reproduction","patch_generation"} -> messages
    n_steps: int
    matched: bool
    extra: dict = field(default_factory=dict)


def _split_prompt_and_steps(messages: list):
    """Return (prompt_messages, steps) where steps = list of (assistant_msg, [env_msgs])."""
    # prompt = leading system/user messages before the first assistant message
    first_assistant = next((i for i, m in enumerate(messages) if m.get("role") == "assistant"), len(messages))
    prompt_messages = messages[:first_assistant]
    rest = messages[first_assistant:]
    steps = []
    i = 0
    while i < len(rest):
        m = rest[i]
        if m.get("role") == "assistant":
            env = []
            j = i + 1
            while j < len(rest) and rest[j].get("role") in ("user", "tool"):
                env.append(rest[j])
                j += 1
            steps.append((m, env))
            i = j
        else:
            i += 1
    return prompt_messages, steps


def _tool_name_onset_token_index(assistant_text_stripped: str, token_ids: list, tokenizer) -> Optional[int]:
    """Index (within this assistant segment's token_ids) of the tool-name's FIRST token.

    Located via the FUNCTION_ONSET_LITERAL char offset and the tokenizer's offset mapping on
    the SAME text convert_messages_to_tokens_and_masks encoded (assistant_token already stripped),
    so offsets align 1:1 with token_ids.
    """
    ci = assistant_text_stripped.find(FUNCTION_ONSET_LITERAL)
    if ci < 0:
        return None
    name_char = ci + len(FUNCTION_ONSET_LITERAL)
    enc = tokenizer(assistant_text_stripped, add_special_tokens=False, return_offsets_mapping=True)
    # sanity: same tokenization the engine used
    if list(enc["input_ids"]) != list(token_ids):
        # offsets still usable for locating the char; fall back to enc's ids for indexing
        offsets = enc["offset_mapping"]
    else:
        offsets = enc["offset_mapping"]
    for idx, (a, b) in enumerate(offsets):
        if a <= name_char < b or (a == b == 0 and idx == name_char):  # normal case: char inside token span
            return idx
        if a <= name_char < b:
            return idx
    # find first token starting at/after name_char
    for idx, (a, b) in enumerate(offsets):
        if a >= name_char:
            return idx
    return None


def _assistant_stripped_text(assistant_msg: dict, parser: ChatTemplateParser) -> str:
    txt = parser.parse([assistant_msg], add_generation_prompt=False, is_first_msg=False)
    if txt.startswith(parser.assistant_token):
        txt = txt.replace(parser.assistant_token, "", 1)
    return txt


def reconstruct_trajectory(record_msgs: list, tokenizer, parser, *,
                           stage_backtrack: int = 3,
                           traj_score: Optional[float] = None,
                           rec: Optional[dict] = None) -> ReconTrajectory:
    prompt_messages, steps = _split_prompt_and_steps(record_msgs)
    prompt_tokens, _ = convert_messages_to_tokens_and_masks(
        prompt_messages, tokenizer=tokenizer, parser=parser,
        contains_first_msg=True, contains_generation_msg=True,
    )

    stage_sm = StageStateMachine(backtrack_threshold=stage_backtrack)
    current_stage_id = STAGE_ORDER[LOCALIZATION]

    response_tokens: list[int] = []
    response_masks: list[int] = []
    stage_ids: list[int] = []
    positions: list[Position] = []

    for (assistant_msg, env_msgs) in steps:
        assistant_content = str(assistant_msg.get("content", ""))
        observation = "\n".join(str(e.get("content", "")) for e in env_msgs)
        # stage advance BEFORE token append (engine order)
        current_stage_id = STAGE_ORDER[stage_sm.update(assistant_content, observation)]

        a_tokens, a_masks = convert_messages_to_tokens_and_masks(
            [assistant_msg], tokenizer=tokenizer, parser=parser,
            contains_first_msg=False, contains_generation_msg=False,
        )
        e_tokens, e_masks = ([], [])
        if env_msgs:
            e_tokens, e_masks = convert_messages_to_tokens_and_masks(
                env_msgs, tokenizer=tokenizer, parser=parser,
                contains_first_msg=False, contains_generation_msg=True,
            )

        seg_start = len(response_tokens)  # global index of this assistant segment's first token

        # ---- action-onset position (tool-name first token) ----
        pos_class = classify_action_onset(assistant_content)
        a_text = _assistant_stripped_text(assistant_msg, parser)
        onset_local = _tool_name_onset_token_index(a_text, a_tokens, tokenizer)
        if onset_local is not None and onset_local < len(a_tokens):
            t = seg_start + onset_local
            positions.append(Position(t=t, pos_class=pos_class, kind="action_onset",
                                       stage_id=current_stage_id, emitted_token_id=a_tokens[onset_local]))

        # ---- deliberation markers (EXPLORE), anchored at first token ----
        for mloc in find_marker_token_positions(a_tokens):
            t = seg_start + mloc
            positions.append(Position(t=t, pos_class="EXPLORE", kind="marker",
                                       stage_id=current_stage_id, emitted_token_id=a_tokens[mloc]))

        # append segment (assistant then env), stage id for both
        response_tokens.extend(a_tokens)
        response_masks.extend(a_masks)
        response_tokens.extend(e_tokens)
        response_masks.extend(e_masks)
        stage_ids.extend([current_stage_id] * (len(a_tokens) + len(e_tokens)))

    # teacher contexts: WA (force-PATCH) + 3 SR stage renders (built from THIS traj's record)
    teacher_by_key: dict[str, Any] = {}
    if rec is not None:
        render_rec = {"student_input": rec.get("student_input"), "teacher_pi": rec.get("teacher_pi"),
                      "task_id": rec.get("task_id"), "stage": rec.get("stage")}
        teacher_by_key["wa"] = render_end_to_end_teacher_messages(render_rec, WA_STAGE_NAME)
        for sid in (0, 1, 2):
            sname = _ID_TO_STAGE[sid]
            teacher_by_key[sname] = render_end_to_end_teacher_messages(render_rec, sname)

    return ReconTrajectory(
        instance_id=(rec or {}).get("instance_id"),
        source_dataset=(rec or {}).get("source_dataset"),
        traj_score=traj_score,
        prompt_messages=prompt_messages,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        response_masks=response_masks,
        stage_ids=stage_ids,
        positions=positions,
        teacher_messages_by_key=teacher_by_key,
        n_steps=len(steps),
        matched=(rec is not None),
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def generate(chat_completions_path: str, parquet_path: str, model_path: str,
             out_path: str, limit: Optional[int] = None, stage_backtrack: int = 3,
             require_match: bool = True, disable_thinking: bool = False) -> dict:
    tok, parser = load_tokenizer_and_parser(model_path, disable_thinking=disable_thinking)
    index = build_record_index(parquet_path)

    trajs: list[ReconTrajectory] = []
    n_read = n_matched = 0
    with open(chat_completions_path) as f:
        for line in f:
            if limit is not None and n_read >= limit:
                break
            line = line.strip()
            if not line:
                continue
            n_read += 1
            obj = json.loads(line)
            msgs = obj.get("chat_completion") or obj.get("messages") or []
            if not msgs:
                continue
            user_msg = next((m for m in msgs if m.get("role") == "user"), None)
            rec = match_record(user_msg.get("content", "") if user_msg else "", index)
            if rec is None and require_match:
                continue
            if rec is not None:
                n_matched += 1
            rt = reconstruct_trajectory(
                msgs, tok, parser, stage_backtrack=stage_backtrack,
                traj_score=obj.get("traj_score"), rec=rec,
            )
            trajs.append(rt)

    meta = {
        "chat_completions_path": chat_completions_path,
        "parquet_path": parquet_path,
        "model_path": model_path,
        "n_read": n_read,
        "n_matched": n_matched,
        "n_trajectories": len(trajs),
        "stage_backtrack": stage_backtrack,
        "wa_stage_name": WA_STAGE_NAME,
    }
    with open(out_path, "wb") as f:
        pickle.dump({"meta": meta, "trajectories": trajs}, f)
    return meta


def _summarize(trajs: list[ReconTrajectory]) -> dict:
    from collections import Counter
    cls = Counter()
    stg = Counter()
    n_pos = 0
    for rt in trajs:
        for p in rt.positions:
            n_pos += 1
            cls[(p.kind, p.pos_class)] += 1
            stg[p.stage_id] += 1
    return {"n_positions": n_pos, "by_kind_class": {f"{k}:{c}": v for (k, c), v in cls.items()},
            "by_stage": dict(stg)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Reconstruct fixed trajectories for fork-suppression forwards.")
    ap.add_argument("--chat-completions", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--model-path", required=True, help="tokenizer source (student ckpt or base; tokenizer identical)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stage-backtrack", type=int, default=3)
    ap.add_argument("--allow-unmatched", action="store_true")
    ap.add_argument("--disable-thinking", action="store_true")
    args = ap.parse_args()

    meta = generate(
        args.chat_completions, args.parquet, args.model_path, args.out,
        limit=args.limit, stage_backtrack=args.stage_backtrack,
        require_match=not args.allow_unmatched, disable_thinking=args.disable_thinking,
    )
    with open(args.out, "rb") as f:
        data = pickle.load(f)
    print(json.dumps({**meta, **_summarize(data["trajectories"])}, indent=2, default=str))
