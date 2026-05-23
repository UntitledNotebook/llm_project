import torch

from llm_project.training.losses import grpo_loss


def test_grpo_loss_runs():
    batch, seq, vocab = 2, 5, 11
    input_ids = torch.randint(0, vocab, (batch, seq))
    logits = torch.randn(batch, seq, vocab)
    completion_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
    old_logprobs = torch.zeros(batch, seq - 1)
    ref_logprobs = torch.zeros(batch, seq - 1)
    advantages = torch.tensor([1.0, -1.0])
    out = grpo_loss(
        logits=logits,
        input_ids=input_ids,
        completion_mask=completion_mask,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        advantages=advantages,
        clip_range=0.2,
        beta_kl=0.02,
    )
    assert torch.isfinite(out.loss)
