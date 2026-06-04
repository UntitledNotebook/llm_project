from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def gather_token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return log p(input_ids[t+1] | input_ids[:t+1]) for each non-initial position.

    Args:
        logits: [batch, seq_len, vocab]
        input_ids: [batch, seq_len]
    Returns:
        [batch, seq_len - 1]
    """
    logits = logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    *,
    loss_type: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    grouped = rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if loss_type == "grpo":
        std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
        return (centered / std).view(-1)
    if loss_type == "dr_grpo":
        return centered.view(-1)


def _aggregate_token_loss(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_type: str,
    max_completion_length: int | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    mask = mask.to(values.dtype)
    if loss_type == "grpo":
        return (values * mask).sum() / mask.sum().clamp_min(eps)
    if loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("max_completion_length is required when loss_type='dr_grpo'")
        denom = values.new_tensor(float(values.size(0) * max_completion_length))
        return (values * mask).sum() / denom.clamp_min(eps)


@dataclass
class GRPOLossOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl: torch.Tensor
    mean_ratio: torch.Tensor


def grpo_loss(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
    beta_kl: float,
    loss_type: str = "grpo",
    max_completion_length: int | None = None,
    sampling_logprobs: torch.Tensor | None = None,
    importance_sampling: bool = False,
    importance_sampling_cap: float = 3.0,
) -> GRPOLossOutput:
    """Token-level clipped GRPO/PPO-style objective.

    vLLM rollouts can be slightly off-policy relative to the training model. When
    ``importance_sampling`` is enabled, ``sampling_logprobs`` must contain the sampled-token
    logprobs from the rollout engine aligned to ``completion_mask[:, 1:]``. The correction is a
    simple token-level truncated importance weight, detached from the policy gradient.
    """
    current_logprobs = gather_token_logprobs(logits, input_ids)
    shifted_mask = completion_mask[:, 1:].to(current_logprobs.dtype)
    advantages = advantages.to(current_logprobs.dtype).view(-1, 1)

    log_ratio = current_logprobs - old_logprobs.detach()
    ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

    importance_weights = torch.ones_like(ratio)
    if importance_sampling:
        if sampling_logprobs is None:
            raise ValueError("sampling_logprobs is required when importance_sampling=True")
        raw_importance_weights = torch.exp(
            torch.clamp(old_logprobs.detach() - sampling_logprobs.detach(), min=-20.0, max=20.0)
        )
        importance_weights = torch.clamp(raw_importance_weights, max=float(importance_sampling_cap))
        importance_weights = importance_weights.detach()

    loss_unclipped = importance_weights * (-advantages * ratio)
    loss_clipped = importance_weights * (-advantages * clipped_ratio)
    policy_loss_per_token = torch.maximum(loss_unclipped, loss_clipped)
    policy_loss = _aggregate_token_loss(
        policy_loss_per_token,
        shifted_mask,
        loss_type=loss_type,
        max_completion_length=max_completion_length,
    )

    # Non-negative unbiased-ish KL estimator used in several RLHF implementations:
    # exp(log p_ref - log p_policy) - (log p_ref - log p_policy) - 1.
    logp_diff = ref_logprobs.detach() - current_logprobs
    kl_per_token = torch.exp(torch.clamp(logp_diff, min=-20.0, max=20.0)) - logp_diff - 1.0
    kl_loss = _aggregate_token_loss(
        kl_per_token,
        shifted_mask,
        loss_type=loss_type,
        max_completion_length=max_completion_length,
    )
    kl = masked_mean(kl_per_token, shifted_mask)

    loss = policy_loss + float(beta_kl) * kl_loss
    mean_ratio = masked_mean(ratio, shifted_mask)
    return GRPOLossOutput(
        loss=loss,
        policy_loss=policy_loss.detach(),
        kl=kl.detach(),
        mean_ratio=mean_ratio.detach(),
    )
