from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import deepspeed
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from llm_project.config import load_config, save_config, to_plain_dict
from llm_project.data.gsm8k_dataset import GSM8KPromptDataset, gsm8k_collate, load_gsm8k_raw
from llm_project.distributed import (
    barrier,
    global_rank,
    is_main_process,
    local_rank,
    maybe_init_distributed,
    print_rank0,
    world_size,
)
from llm_project.evaluation.gsm8k_eval import evaluate_gsm8k_model
from llm_project.logging_utils import JsonlLogger
from llm_project.models import enable_training_mode, freeze_model, load_causal_lm, load_tokenizer
from llm_project.seed import set_seed
from llm_project.training.checkpointing import save_hf_checkpoint
from llm_project.training.generation import generate_completions
from llm_project.training.losses import gather_token_logprobs, grpo_loss
from llm_project.training.rewards import GSM8KReward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for Choice A Part 2")
    parser.add_argument("--config", type=str, required=True, help="Path to configs/grpo.yaml")
    parser.add_argument("--ds_config", type=str, required=True, help="Path to DeepSpeed JSON config")
    parser.add_argument("--local_rank", type=int, default=-1, help="Injected by DeepSpeed launcher")
    return parser.parse_args()


def compute_group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return ((grouped - mean) / std).view(-1)


@torch.no_grad()
def compute_reference_logprobs(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return gather_token_logprobs(logits, input_ids)


def main() -> None:
    args = parse_args()
    maybe_init_distributed()
    device = torch.device("cuda", local_rank())
    cfg = load_config(args.config)
    set_seed(int(cfg.seed), rank_offset=global_rank())

    output_dir = Path(cfg.train.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(to_plain_dict(cfg), output_dir / "resolved_grpo_config.yaml")
        shutil.copyfile(args.ds_config, output_dir / "deepspeed_config.json")
    barrier()

    model_name = cfg.model.init_from_sft_checkpoint or cfg.model.name_or_path
    tokenizer = load_tokenizer(
        model_name,
        trust_remote_code=bool(cfg.model.trust_remote_code),
        padding_side="left",
    )

    raw = load_gsm8k_raw(
        dataset_name=cfg.dataset.name,
        config_name=cfg.dataset.config_name,
        split=cfg.dataset.train_split,
        max_samples=cfg.dataset.max_train_samples,
    )
    train_dataset = GSM8KPromptDataset(raw)
    sampler = DistributedSampler(
        train_dataset, num_replicas=world_size(), rank=global_rank(), shuffle=True, seed=int(cfg.seed)
    ) if world_size() > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.train.per_device_prompt_batch_size),
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=gsm8k_collate,
        num_workers=2,
        pin_memory=True,
    )

    policy_model = load_causal_lm(
        model_name,
        trust_remote_code=bool(cfg.model.trust_remote_code),
        dtype=cfg.model.dtype,
        attn_implementation=cfg.model.attn_implementation,
    )
    enable_training_mode(policy_model, bool(cfg.model.gradient_checkpointing))

    reference_name = cfg.model.reference_model_name_or_path or cfg.model.name_or_path
    reference_model = load_causal_lm(
        reference_name,
        trust_remote_code=bool(cfg.model.trust_remote_code),
        dtype=cfg.model.dtype,
        attn_implementation=cfg.model.attn_implementation,
    ).to(device)
    freeze_model(reference_model)

    optimizer = torch.optim.AdamW(
        policy_model.parameters(), lr=float(cfg.train.learning_rate), weight_decay=float(cfg.train.weight_decay)
    )
    grad_accum = 1
    try:
        with open(args.ds_config, "r", encoding="utf-8") as f:
            grad_accum = int(json.load(f).get("gradient_accumulation_steps", 1))
    except Exception:
        pass
    total_update_steps = math.ceil(len(train_loader) / max(1, grad_accum)) * int(cfg.train.num_train_epochs)
    warmup_steps = int(float(cfg.train.warmup_ratio) * total_update_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=policy_model,
        model_parameters=policy_model.parameters(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=args.ds_config,
    )

    reward_fn = GSM8KReward(
        correct_answer_reward=float(cfg.reward.correct_answer_reward),
        wrong_answer_reward=float(cfg.reward.wrong_answer_reward),
        format_reward=float(cfg.reward.format_reward),
        require_final_answer_marker=bool(cfg.reward.require_final_answer_marker),
    )
    logger = JsonlLogger(output_dir / "logs" / "grpo_metrics.jsonl", enabled=is_main_process())
    global_step = 0

    print_rank0(
        f"GRPO start: train_examples={len(train_dataset)} group_size={cfg.rollout.group_size} "
        f"world_size={world_size()} total_update_steps≈{total_update_steps}"
    )

    for epoch in range(int(cfg.train.num_train_epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)
        progress = tqdm(train_loader, disable=not is_main_process(), desc=f"GRPO epoch {epoch}")
        for batch in progress:
            prompts: list[str] = batch["prompt"]
            refs: list[str | None] = batch["reference_answer"]
            group_size = int(cfg.rollout.group_size)

            rollout = generate_completions(
                model_engine.module,
                tokenizer,
                prompts,
                max_prompt_length=int(cfg.dataset.max_prompt_length),
                max_new_tokens=int(cfg.rollout.max_new_tokens),
                group_size=group_size,
                temperature=float(cfg.rollout.temperature),
                top_p=float(cfg.rollout.top_p),
                do_sample=bool(cfg.rollout.do_sample),
            )
            input_ids = rollout["input_ids"].to(device)
            attention_mask = rollout["attention_mask"].to(device)
            completion_mask = rollout["completion_mask"].to(device)
            completion_texts: list[str] = rollout["completion_texts"]

            repeated_refs = [ref for ref in refs for _ in range(group_size)]
            reward_results = [reward_fn(text, ref) for text, ref in zip(completion_texts, repeated_refs)]
            rewards = torch.tensor([item.reward for item in reward_results], dtype=torch.float32, device=device)
            advantages = compute_group_advantages(rewards, group_size=group_size)

            with torch.no_grad():
                old_outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask)
                old_logprobs = gather_token_logprobs(old_outputs.logits, input_ids).detach()
                ref_logprobs = compute_reference_logprobs(reference_model, input_ids, attention_mask).detach()

            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask)
            loss_out = grpo_loss(
                logits=outputs.logits,
                input_ids=input_ids,
                completion_mask=completion_mask,
                old_logprobs=old_logprobs,
                ref_logprobs=ref_logprobs,
                advantages=advantages,
                clip_range=float(cfg.train.clip_range),
                beta_kl=float(cfg.train.beta_kl),
            )
            model_engine.backward(loss_out.loss)
            model_engine.step()

            if model_engine.is_gradient_accumulation_boundary():
                global_step += 1
                avg_reward = float(rewards.mean().item())
                group_acc = sum(int(item.correct) for item in reward_results) / max(1, len(reward_results))
                avg_len = float(completion_mask[:, 1:].sum(dim=1).float().mean().item())
                if global_step % int(cfg.train.logging_steps) == 0:
                    lr = scheduler.get_last_lr()[0] if scheduler is not None else float(cfg.train.learning_rate)
                    logger.write(
                        {
                            "phase": "train",
                            "epoch": epoch,
                            "step": global_step,
                            "loss": float(loss_out.loss.detach().float().item()),
                            "policy_loss": float(loss_out.policy_loss.float().item()),
                            "kl": float(loss_out.kl.float().item()),
                            "mean_ratio": float(loss_out.mean_ratio.float().item()),
                            "reward": avg_reward,
                            "rollout_accuracy": group_acc,
                            "avg_completion_tokens": avg_len,
                            "lr": lr,
                        }
                    )
                    progress.set_postfix(loss=f"{float(loss_out.loss):.4f}", reward=f"{avg_reward:.3f}")

                if int(cfg.train.eval_steps) > 0 and global_step % int(cfg.train.eval_steps) == 0:
                    # Test-set accuracy curve during RL training. Rank 0 evaluates its local full model replica.
                    if is_main_process():
                        metrics = evaluate_gsm8k_model(
                            model_engine.module,
                            tokenizer,
                            dataset_name=cfg.dataset.name,
                            config_name=cfg.dataset.config_name,
                            split=cfg.dataset.test_split,
                            max_samples=cfg.dataset.max_eval_samples,
                            batch_size=1,
                            max_new_tokens=int(cfg.rollout.max_new_tokens),
                            temperature=0.0,
                            output_path=output_dir / "eval" / f"gsm8k_step{global_step}.json",
                        )
                        logger.write({"phase": "eval", "epoch": epoch, "step": global_step, **metrics})
                        print_rank0(f"GRPO eval step={global_step}: {metrics}")
                    barrier()
    if bool(cfg.train.save_hf_at_end):
        save_hf_checkpoint(model_engine, tokenizer, output_dir / "hf")
    print_rank0("GRPO finished.")


if __name__ == "__main__":
    main()
