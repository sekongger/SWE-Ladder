from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests
import torch


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "None":
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def extract_next_token_top_logprobs(payload: dict[str, Any]) -> dict[str, Any]:
    choice = (payload.get("choices") or [{}])[0]
    logprobs = choice.get("logprobs") or {}
    tokens = list(logprobs.get("tokens") or [])
    token_logprobs = list(logprobs.get("token_logprobs") or [])
    top_logprobs = list(logprobs.get("top_logprobs") or [])
    return {
        "generated_token": tokens[0] if tokens else choice.get("text"),
        "generated_token_logprob": token_logprobs[0] if token_logprobs else None,
        "top_logprobs": top_logprobs[0] if top_logprobs else {},
    }


@dataclass
class TeacherTopKClient:
    """OpenAI-compatible client that returns per-position top-k teacher log-probs.

    Two scoring strategies, selected by ``use_echo_single_call`` (default: echo):

    * ``_score_via_echo`` (DEFAULT, efficient): a SINGLE ``/completions`` call per
      sample. We send ``prompt = prompt_token_ids + response_token_ids`` (raw token
      ids, so no detokenize/retokenize drift) with ``echo=True`` + ``logprobs=k``.
      vLLM then returns the top-k distribution at EVERY position of the echoed
      sequence in one forward pass (``choices[0].logprobs.top_logprobs``), and we
      slice out the response positions. This mirrors the single-forward
      ``compute_logprobs`` pattern used by Tinker / TRL-GKD and replaces the old
      O(L) per-token request loop with O(1) requests per sample.
      REQUIRES a same-family tokenizer teacher (token ids are shared) -- which is
      our standing constraint.

    * ``_score_per_token`` (LEGACY fallback): one 1-token ``/completions`` request
      per response token. Correct but ~L× slower. Selected when
      ``OPSD_TEACHER_PER_TOKEN=1`` or ``use_echo_single_call=False`` (e.g. for a
      server that does not support prompt-echo logprobs).
    """

    base_url: str
    model_name: str
    api_key: str | None = None
    timeout_sec: float = 120.0
    use_echo_single_call: bool | None = None

    def __post_init__(self) -> None:
        if self.use_echo_single_call is None:
            # Escape hatch for endpoints without echo/prompt-logprob support.
            self.use_echo_single_call = (
                os.environ.get("OPSD_TEACHER_PER_TOKEN", "0").strip().lower()
                not in ("1", "true", "yes")
            )

    # ------------------------------------------------------------------ requests
    def _completions(
        self,
        *,
        prompt: Any,
        topk_k: int,
        echo: bool,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """POST /completions. ``prompt`` may be a token-id list, a list of token-id
        lists (batched), or text."""
        response = requests.post(
            f"{self.base_url.rstrip('/')}/completions",
            headers=_headers(self.api_key),
            json={
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "echo": echo,
                "logprobs": topk_k,
                "temperature": temperature,
            },
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        return response.json()

    def request_next_token_top_logprobs(self, *, prompt_text: str, topk_k: int) -> dict[str, Any]:
        """LEGACY: single 1-token completion for one next-token distribution."""
        return self._completions(
            prompt=prompt_text, topk_k=topk_k, echo=False, max_tokens=1
        )

    # ------------------------------------------------------------------ scoring
    def score_response_token_ids(
        self,
        *,
        tokenizer: Any,
        prompt_token_ids: list[int],
        response_token_ids: list[int],
        topk_k: int,
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """Return one row per response token with its teacher top-k distribution.

        Signature is unchanged from the legacy client so callers need no change.
        ``tokenizer`` is only used by the legacy per-token path (kept for compat).
        """
        if self.use_echo_single_call:
            return self._score_via_echo(
                prompt_token_ids=prompt_token_ids,
                response_token_ids=response_token_ids,
                topk_k=topk_k,
                max_tokens=max_tokens,
            )
        return self._score_per_token(
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            response_token_ids=response_token_ids,
            topk_k=topk_k,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _rows_from_top_logprobs(
        top_logprobs: list[dict[str, float] | None],
        *,
        prompt_len: int,
        target_ids: list[int],
    ) -> list[dict[str, Any]]:
        """Slice the response positions out of an echoed top_logprobs array.

        ``top_logprobs[pos]`` is the teacher's next-token distribution CONDITIONED
        on all tokens before ``pos``. For response token ``r`` at absolute position
        ``prompt_len + r`` that is exactly the distribution predicting it given
        ``prompt + response[:r]`` -- the reverse-KL target.
        """
        rows: list[dict[str, Any]] = []
        n = len(top_logprobs)
        for r, tid in enumerate(target_ids):
            pos = prompt_len + r
            tl = top_logprobs[pos] if 0 <= pos < n else None
            rows.append(
                {
                    "target_token_index": r,
                    "target_token_id": int(tid),
                    "top_logprobs": tl or {},
                }
            )
        return rows

    def _score_via_echo(
        self,
        *,
        prompt_token_ids: list[int],
        response_token_ids: list[int],
        topk_k: int,
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        target_ids = (
            response_token_ids[:max_tokens] if max_tokens and max_tokens > 0 else list(response_token_ids)
        )
        if not target_ids:
            return []
        full_ids = list(prompt_token_ids) + list(target_ids)
        # echo=True returns logprobs for the whole prompt; max_tokens=1 (not 0) is
        # accepted across vLLM versions and the single generated tail position is
        # simply ignored.
        payload = self._completions(prompt=full_ids, topk_k=topk_k, echo=True, max_tokens=1)
        choice = (payload.get("choices") or [{}])[0]
        top_logprobs = list((choice.get("logprobs") or {}).get("top_logprobs") or [])
        return self._rows_from_top_logprobs(
            top_logprobs, prompt_len=len(prompt_token_ids), target_ids=target_ids
        )

    def score_response_token_ids_batched(
        self,
        *,
        prompt_token_ids_list: list[list[int]],
        response_token_ids_list: list[list[int]],
        topk_k: int,
        max_tokens: int,
    ) -> list[list[dict[str, Any]]]:
        """One /completions call for a whole micro-batch (vLLM accepts a list of
        token-id prompts and returns one choice per prompt, ordered by ``index``).

        Not wired into the trainer loop yet; provided for callers that want to cut
        request count from O(batch) to O(1). Falls back to per-sample echo calls if
        the legacy per-token mode is active.
        """
        if not self.use_echo_single_call:
            return [
                self._score_per_token(
                    tokenizer=None,
                    prompt_token_ids=p,
                    response_token_ids=r,
                    topk_k=topk_k,
                    max_tokens=max_tokens,
                )
                for p, r in zip(prompt_token_ids_list, response_token_ids_list)
            ]
        full_prompts: list[list[int]] = []
        prompt_lens: list[int] = []
        targets: list[list[int]] = []
        for p, r in zip(prompt_token_ids_list, response_token_ids_list):
            t = r[:max_tokens] if max_tokens and max_tokens > 0 else list(r)
            full_prompts.append(list(p) + list(t))
            prompt_lens.append(len(p))
            targets.append(t)
        payload = self._completions(prompt=full_prompts, topk_k=topk_k, echo=True, max_tokens=1)
        choices = payload.get("choices") or []
        by_index: dict[int, dict[str, Any]] = {}
        for i, c in enumerate(choices):
            by_index[int(c.get("index", i))] = c
        out: list[list[dict[str, Any]]] = []
        for i, (plen, t) in enumerate(zip(prompt_lens, targets)):
            choice = by_index.get(i, {})
            top_logprobs = list((choice.get("logprobs") or {}).get("top_logprobs") or [])
            out.append(self._rows_from_top_logprobs(top_logprobs, prompt_len=plen, target_ids=t))
        return out

    def _score_per_token(
        self,
        *,
        tokenizer: Any,
        prompt_token_ids: list[int],
        response_token_ids: list[int],
        topk_k: int,
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """LEGACY: one request per response token (O(L) HTTP calls)."""
        rows: list[dict[str, Any]] = []
        target_ids = response_token_ids[:max_tokens] if max_tokens > 0 else response_token_ids
        prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
        for offset, target_id in enumerate(target_ids):
            prefix_text = prompt_text + tokenizer.decode(target_ids[:offset], skip_special_tokens=False)
            payload = self.request_next_token_top_logprobs(prompt_text=prefix_text, topk_k=topk_k)
            row = extract_next_token_top_logprobs(payload)
            row.update(
                {
                    "target_token_index": offset,
                    "target_token_id": int(target_id),
                    "target_token": tokenizer.decode([int(target_id)], skip_special_tokens=False),
                }
            )
            rows.append(row)
        return rows


def top_logprobs_to_tensors(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_response_length: int,
    topk_k: int,
    train_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_ids = torch.full((max_response_length, topk_k), int(getattr(tokenizer, "pad_token_id", 0) or 0), dtype=torch.long)
    log_probs = torch.full((max_response_length, topk_k), float("-inf"), dtype=torch.float32)
    loss_mask = torch.zeros(max_response_length, dtype=torch.float32)

    for default_pos, row in enumerate(rows[:max_response_length]):
        pos = int(row.get("target_token_index", default_pos))
        if pos < 0 or pos >= max_response_length:
            continue
        col = 0
        top_logprobs = row.get("top_logprobs") or {}
        for token_text, logprob in top_logprobs.items():
            encoded = tokenizer.encode(token_text, add_special_tokens=False)
            if len(encoded) != 1:
                continue
            token_ids[pos, col] = int(encoded[0])
            log_probs[pos, col] = float(logprob)
            col += 1
            if col >= topk_k:
                break
        if col > 0:
            loss_mask[pos] = 1.0

    if train_mask is not None:
        loss_mask = loss_mask * train_mask.to(dtype=torch.float32)[:max_response_length].cpu()
    return token_ids, log_probs, loss_mask
