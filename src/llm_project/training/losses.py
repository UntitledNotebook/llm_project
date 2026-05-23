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
) -> GRPOLossOutput:
    """Token-level clipped GRPO/PPO-style objective.

    For a prompt group, rewards are normalized to advantages outside this function. This loss then
    shifts the current policy toward completions with positive group-relative advantage and away
    from completions with negative advantage, with a reference-model KL penalty.
    """
    current_logprobs = gather_token_logprobs(logits, input_ids)
    shifted_mask = completion_mask[:, 1:].to(current_logprobs.dtype)
    advantages = advantages.to(current_logprobs.dtype).view(-1, 1)

    log_ratio = current_logprobs - old_logprobs.detach()
    ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

    loss_unclipped = -advantages * ratio
    loss_clipped = -advantages * clipped_ratio
    policy_loss_per_token = torch.maximum(loss_unclipped, loss_clipped)
    policy_loss = masked_mean(policy_loss_per_token, shifted_mask)

    # Non-negative unbiased-ish KL estimator used in several RLHF implementations:
    # exp(log p_ref - log p_policy) - (log p_ref - log p_policy) - 1.
    logp_diff = ref_logprobs.detach() - current_logprobs
    kl_per_token = torch.exp(torch.clamp(logp_diff, min=-20.0, max=20.0)) - logp_diff - 1.0
    kl = masked_mean(kl_per_token, shifted_mask)

    loss = policy_loss + float(beta_kl) * kl
    mean_ratio = masked_mean(ratio, shifted_mask)
    return GRPOLossOutput(loss=loss, policy_loss=policy_loss.detach(), kl=kl.detach(), mean_ratio=mean_ratio.detach())
