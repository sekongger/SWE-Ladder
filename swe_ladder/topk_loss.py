from __future__ import annotations

from typing import Any

import torch


def gather_topk_log_probs_from_hidden_chunked(
    *,
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    topk_token_ids: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    lm_head_bias: torch.Tensor | None = None,
    vocab_chunk_size: int = 8192,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute top-k log-probs from hidden states without materializing full logits.

    This is mathematically equivalent to ``logits.gather(-1, topk_ids) -
    logsumexp(logits)`` where ``logits = hidden @ lm_head_weight.T + bias``.
    The normalizer is computed over vocabulary chunks so the peak tensor is
    ``[active_tokens, vocab_chunk_size]`` instead of ``[all_tokens, vocab]``.
    Inactive ``loss_mask`` positions are returned as zeros and do not participate
    in the chunked computation.
    """

    if hidden_states.dim() not in (2, 3):
        raise ValueError(
            f"hidden_states must be [tokens, hidden] or [batch, seq, hidden], got {tuple(hidden_states.shape)}"
        )
    if topk_token_ids.shape[:-1] != hidden_states.shape[:-1]:
        raise ValueError(
            "topk_token_ids prefix must match hidden_states prefix, "
            f"got {tuple(topk_token_ids.shape[:-1])} and {tuple(hidden_states.shape[:-1])}"
        )
    if lm_head_weight.dim() != 2:
        raise ValueError(f"lm_head_weight must be [vocab, hidden], got {tuple(lm_head_weight.shape)}")
    if lm_head_weight.shape[1] != hidden_states.shape[-1]:
        raise ValueError(
            f"lm_head hidden dim {lm_head_weight.shape[1]} does not match hidden dim {hidden_states.shape[-1]}"
        )
    if loss_mask is not None and loss_mask.shape != hidden_states.shape[:-1]:
        raise ValueError(
            f"loss_mask shape {tuple(loss_mask.shape)} must match hidden prefix {tuple(hidden_states.shape[:-1])}"
        )
    if vocab_chunk_size <= 0:
        raise ValueError(f"vocab_chunk_size must be positive, got {vocab_chunk_size}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    original_shape = topk_token_ids.shape
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_topk = topk_token_ids.reshape(-1, topk_token_ids.shape[-1])
    if loss_mask is None:
        active = torch.ones(flat_hidden.shape[0], dtype=torch.bool, device=flat_hidden.device)
    else:
        active = loss_mask.reshape(-1).to(device=flat_hidden.device) > 0

    output = flat_hidden.new_zeros((flat_hidden.shape[0], flat_topk.shape[-1]))
    if not bool(active.any()):
        # In Ulysses sequence parallel, a rank can own no OPSD-supervised
        # positions after slicing. Still return a graph-connected zero so the
        # rank can participate in backward without "does not require grad".
        return (output + flat_hidden.sum() * 0.0).reshape(original_shape)

    active_hidden = flat_hidden[active]
    active_topk = flat_topk[active].to(device=flat_hidden.device).clamp(0, lm_head_weight.shape[0] - 1)
    weight = lm_head_weight.to(device=flat_hidden.device)
    bias = lm_head_bias.to(device=flat_hidden.device) if lm_head_bias is not None else None

    topk_weight = weight[active_topk]
    topk_logits = torch.einsum("ah,akh->ak", active_hidden, topk_weight)
    if bias is not None:
        topk_logits = topk_logits + bias[active_topk]
    topk_logits = topk_logits / temperature

    log_z = None
    vocab_size = weight.shape[0]
    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab_size)
        logits_chunk = active_hidden @ weight[start:end].transpose(0, 1)
        if bias is not None:
            logits_chunk = logits_chunk + bias[start:end]
        logits_chunk = logits_chunk / temperature
        chunk_log_z = torch.logsumexp(logits_chunk.float(), dim=-1)
        log_z = chunk_log_z if log_z is None else torch.logaddexp(log_z, chunk_log_z)

    active_log_probs = topk_logits - log_z.to(dtype=topk_logits.dtype).unsqueeze(-1)
    output[active] = active_log_probs.to(dtype=output.dtype)
    return output.reshape(original_shape)


def teacher_topk_reverse_kl(
    *,
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    normalize_teacher_topk: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute stop-gradient truncated reverse KL over teacher top-k tokens.

    Args:
        student_topk_log_probs: Student log-probabilities gathered at teacher top-k ids,
            shape ``[batch, response_len, k]``.
        teacher_topk_log_probs: Teacher log-probabilities for the same ids,
            shape ``[batch, response_len, k]``.
        loss_mask: Token mask, shape ``[batch, response_len]``.
        normalize_teacher_topk: If true, renormalize teacher log-probability mass
            inside the top-k slice before forming the detached log-ratio. The
            outer student probability remains the full-vocabulary probability
            gathered at teacher top-k ids.
    """

    if student_topk_log_probs.shape != teacher_topk_log_probs.shape:
        raise ValueError(
            "student_topk_log_probs and teacher_topk_log_probs must have the same shape, "
            f"got {tuple(student_topk_log_probs.shape)} and {tuple(teacher_topk_log_probs.shape)}"
        )
    if student_topk_log_probs.dim() != 3:
        raise ValueError(f"top-k log-prob tensors must be 3D [batch, response_len, k], got {student_topk_log_probs.dim()}D")
    if loss_mask.shape != student_topk_log_probs.shape[:2]:
        raise ValueError(f"loss_mask shape {tuple(loss_mask.shape)} must match top-k prefix {tuple(student_topk_log_probs.shape[:2])}")

    finite_teacher = torch.isfinite(teacher_topk_log_probs)
    finite_student = torch.isfinite(student_topk_log_probs)
    valid_topk = finite_teacher & finite_student
    safe_teacher_log_probs = torch.where(
        finite_teacher,
        teacher_topk_log_probs.detach(),
        torch.full_like(teacher_topk_log_probs, -1e30),
    )
    safe_student_log_probs = torch.where(
        finite_student,
        student_topk_log_probs,
        torch.full_like(student_topk_log_probs, -1e30),
    )

    if normalize_teacher_topk:
        teacher_log_probs_for_kl = torch.log_softmax(safe_teacher_log_probs, dim=-1)
        teacher_topk_mass = torch.exp(safe_teacher_log_probs).sum(dim=-1)
    else:
        teacher_log_probs_for_kl = safe_teacher_log_probs
        teacher_topk_mass = torch.exp(safe_teacher_log_probs).sum(dim=-1)

    student_probs = torch.where(
        valid_topk,
        torch.exp(safe_student_log_probs),
        torch.zeros_like(safe_student_log_probs),
    )
    log_ratio = safe_student_log_probs.detach() - teacher_log_probs_for_kl.detach()
    log_ratio = torch.where(valid_topk, log_ratio, torch.zeros_like(log_ratio))
    per_token_kl = (student_probs * log_ratio).sum(dim=-1)
    mask = loss_mask.to(dtype=per_token_kl.dtype, device=per_token_kl.device)
    active_tokens = mask.sum()
    if active_tokens <= 0:
        zero = per_token_kl.sum() * 0.0
        return zero, {
            "opsd/active_tokens": active_tokens.detach(),
            "opsd/teacher_topk_mass": zero.detach(),
            "opsd/reverse_kl_mean": zero.detach(),
        }

    loss = (per_token_kl * mask).sum() / active_tokens.clamp_min(eps)
    teacher_mass = (teacher_topk_mass * mask).sum() / active_tokens.clamp_min(eps)
    return loss, {
        "opsd/active_tokens": active_tokens.detach(),
        "opsd/teacher_topk_mass": teacher_mass.detach(),
        "opsd/reverse_kl_mean": loss.detach(),
    }


def tensor_metrics_to_floats(metrics: dict[str, torch.Tensor | float | int]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            result[key] = float(value.detach().float().cpu().item())
        else:
            result[key] = float(value)
    return result


def get_opsd_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if getter is not None:
        return getter(key, default)
    return getattr(config, key, default)
