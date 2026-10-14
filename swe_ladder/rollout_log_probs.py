"""Safe reuse of sampled-token rollout log probabilities for RLSD."""

from __future__ import annotations

import math
from collections.abc import Sequence


def align_sampled_log_probs(
    generated_token_ids: Sequence[int] | None,
    sampled_log_probs: Sequence[float] | None,
    assistant_token_ids: Sequence[int],
) -> list[float] | None:
    """Return sampled logprobs only when rollout and training tokens match exactly."""

    if generated_token_ids is None or sampled_log_probs is None:
        return None
    if list(generated_token_ids) != list(assistant_token_ids):
        return None
    if len(sampled_log_probs) != len(assistant_token_ids):
        return None
    aligned = [float(log_prob) for log_prob in sampled_log_probs]
    if not all(math.isfinite(log_prob) for log_prob in aligned):
        return None
    return aligned
