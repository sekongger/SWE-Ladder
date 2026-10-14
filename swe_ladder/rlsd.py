"""RLSD token-level advantage modulation for privileged self-distillation."""

from __future__ import annotations

from typing import Any

import torch


def compute_rlsd_dynamic_batch_token_limit(
    *,
    default_max_token_len: int,
    configured_max_packed_context_tokens: int = 0,
    padded_sequence_len: int,
) -> int:
    """Apply an optional logical-context cap to RLSD actor dynamic batching."""

    default_limit = int(default_max_token_len)
    configured_limit = int(configured_max_packed_context_tokens or 0)
    sequence_len = int(padded_sequence_len)
    if configured_limit <= 0:
        return default_limit
    if configured_limit < sequence_len:
        raise ValueError(
            "RLSD actor max packed context tokens must cover one padded sequence: "
            f"configured={configured_limit}, padded_sequence_len={sequence_len}"
        )
    return min(default_limit, configured_limit)


def compute_rlsd_lambda_schedule(
    *,
    global_step: int,
    lambda_default: float = 1.0,
    lambda_init: float | None = None,
    lambda_final: float | None = None,
    decay_steps: int = 0,
) -> float:
    """Return the RLSD mixing coefficient for the current actor update."""

    initial = float(lambda_default if lambda_init is None else lambda_init)
    final = float(initial if lambda_final is None else lambda_final)
    decay_steps = int(decay_steps or 0)
    if not 0.0 <= initial <= 1.0:
        raise ValueError(f"RLSD initial lambda must be in [0, 1], got {initial}")
    if not 0.0 <= final <= 1.0:
        raise ValueError(f"RLSD final lambda must be in [0, 1], got {final}")
    if decay_steps < 0:
        raise ValueError(f"RLSD lambda decay steps must be non-negative, got {decay_steps}")
    if decay_steps == 0:
        return initial

    completed_updates = max(int(global_step) - 1, 0)
    progress = min(float(completed_updates) / float(decay_steps), 1.0)
    return initial + (final - initial) * progress


def _masked_stat(values: torch.Tensor, mask: torch.Tensor, op: str) -> float:
    active = values[mask]
    if active.numel() == 0:
        return 0.0
    if op == "mean":
        return float(active.mean().item())
    if op == "std":
        return float(active.std(unbiased=False).item())
    if op == "max":
        return float(active.max().item())
    if op == "min":
        return float(active.min().item())
    raise ValueError(f"Unknown statistic: {op}")


def compute_rlsd_token_advantage(
    sequence_advantage: torch.Tensor,
    teacher_sampled_logp: torch.Tensor,
    student_prior_sampled_logp: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    lambda_rlsd: float = 1.0,
    weight_clip_eps: float = 0.2,
    delta_clip: float = 5.0,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Modulate sequence-level GRPO advantage using detached teacher evidence."""

    if not 0.0 <= lambda_rlsd <= 1.0:
        raise ValueError(f"lambda_rlsd must be in [0, 1], got {lambda_rlsd}")
    if not 0.0 <= weight_clip_eps < 1.0:
        raise ValueError(f"weight_clip_eps must be in [0, 1), got {weight_clip_eps}")
    if delta_clip < 0.0:
        raise ValueError(f"delta_clip must be non-negative, got {delta_clip}")
    if teacher_sampled_logp.shape != student_prior_sampled_logp.shape:
        raise ValueError("teacher and student prior sampled log-probabilities must have identical shapes")
    if response_mask.shape != teacher_sampled_logp.shape:
        raise ValueError("response_mask shape must match sampled log-probability shapes")
    if sequence_advantage.ndim == 1:
        sequence_advantage = sequence_advantage.unsqueeze(-1)
    if sequence_advantage.ndim != 2 or sequence_advantage.shape[0] != response_mask.shape[0]:
        raise ValueError("sequence_advantage must have shape [batch] or [batch, 1]")

    with torch.no_grad():
        seq_adv = sequence_advantage.detach().to(device=teacher_sampled_logp.device, dtype=teacher_sampled_logp.dtype)
        mask = response_mask.detach().to(device=teacher_sampled_logp.device, dtype=teacher_sampled_logp.dtype)
        active = mask > 0
        teacher_logp = teacher_sampled_logp.detach()
        student_logp = student_prior_sampled_logp.detach()
        delta = teacher_logp - student_logp
        delta = torch.nan_to_num(delta, nan=0.0, posinf=delta_clip, neginf=-delta_clip)
        delta = delta.clamp(min=-delta_clip, max=delta_clip)
        sign_adv = torch.sign(seq_adv)
        raw_weight = torch.exp(sign_adv * delta)
        clipped_weight = raw_weight.clamp(min=1.0 - weight_clip_eps, max=1.0 + weight_clip_eps)
        mixed_weight = (1.0 - lambda_rlsd) + lambda_rlsd * clipped_weight
        token_advantage = seq_adv * mixed_weight * mask

        expanded_sign = sign_adv.expand_as(token_advantage)
        nonzero = active & (expanded_sign != 0)
        sign_flip_count = float(
            (torch.sign(token_advantage[nonzero]) != expanded_sign[nonzero]).sum().item()
        )
        assert not teacher_logp.requires_grad
        assert not student_logp.requires_grad
        assert not token_advantage.requires_grad
        assert sign_flip_count == 0.0

        if not return_stats:
            return token_advantage

        positive = active & (expanded_sign > 0)
        negative = active & (expanded_sign < 0)
        stats = {
            "rlsd_delta_mean": _masked_stat(delta, active, "mean"),
            "rlsd_delta_std": _masked_stat(delta, active, "std"),
            "rlsd_delta_max": _masked_stat(delta, active, "max"),
            "rlsd_delta_min": _masked_stat(delta, active, "min"),
            "rlsd_weight_mean": _masked_stat(clipped_weight, active, "mean"),
            "rlsd_weight_std": _masked_stat(clipped_weight, active, "std"),
            "rlsd_weight_clip_fraction": _masked_stat((raw_weight != clipped_weight).float(), active, "mean"),
            "rlsd_positive_adv_weight_mean": _masked_stat(clipped_weight, positive, "mean"),
            "rlsd_negative_adv_weight_mean": _masked_stat(clipped_weight, negative, "mean"),
            "token_advantage_mean": _masked_stat(token_advantage, active, "mean"),
            "token_advantage_abs_mean": _masked_stat(token_advantage.abs(), active, "mean"),
            "sign_flip_count": sign_flip_count,
            "teacher_sampled_logp_mean": _masked_stat(teacher_logp, active, "mean"),
            "student_prior_sampled_logp_mean": _masked_stat(student_logp, active, "mean"),
            "response_mask_token_count": float(active.sum().item()),
        }
        return token_advantage, stats
