from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
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
from llm_project.models import enable_training_mode, freeze_model, load_causal_lm, load_tokenizer
from llm_project.rollout import create_rollout_backend
from llm_project.seed import set_seed
from llm_project.training.checkpointing import save_hf_checkpoint
from llm_project.training.losses import compute_group_advantages, gather_token_logprobs, grpo_loss
from llm_project.training.rewards import GSM8KReward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for Choice A Part 2/3")
    parser.add_argument("--config", type=str, required=True, help="Path to configs/grpo.yaml")
    parser.add_argument("--ds_config", type=str, required=True, help="Path to DeepSpeed JSON config")
    parser.add_argument("--local_rank", type=int, default=-1, help="Injected by DeepSpeed launcher")
    parser.add_argument("--run_name", type=str, default=None, help="Override wandb run name")
    return parser.parse_args()


@torch.no_grad()
def compute_reference_logprobs(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return gather_token_logprobs(logits, input_ids)


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reduce_speed_metrics(
    rollout_time_sec: float,
    step_time_sec: float,
    sync_weights_time_sec: float,
    rollout_tokens: float,
    device: torch.device,
) -> tuple[float, float, float, float]:
    time_stats = torch.tensor(
        [rollout_time_sec, step_time_sec, sync_weights_time_sec],
        dtype=torch.float64,
        device=device,
    )
    token_stats = torch.tensor([rollout_tokens], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(time_stats, op=dist.ReduceOp.MAX)
        dist.all_reduce(token_stats, op=dist.ReduceOp.SUM)
    rollout_time_sec = float(time_stats[0].item())
    step_time_sec = float(time_stats[1].item())
    sync_weights_time_sec = float(time_stats[2].item())
    rollout_tokens = float(token_stats[0].item())
    rollout_tokens_per_sec = rollout_tokens / max(rollout_time_sec, 1e-9)
    return rollout_time_sec, step_time_sec, sync_weights_time_sec, rollout_tokens_per_sec


def main() -> None:
    args = parse_args()
    maybe_init_distributed()
    device = torch.device("cuda", local_rank())
    cfg = load_config(args.config)
    max_completion_length = int(cfg.rollout.max_new_tokens)
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
    prompt_builder = cfg.dataset.get("prompt_builder")
    train_dataset = GSM8KPromptDataset(raw, prompt_builder=prompt_builder)
    sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size(),
            rank=global_rank(),
            shuffle=True,
            seed=int(cfg.seed),
        )
        if world_size() > 1
        else None
    )
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
    enable_training_mode(policy_model)

    reference_name = cfg.model.reference_model_name_or_path or cfg.model.name_or_path
    reference_model = load_causal_lm(
        reference_name,
        trust_remote_code=bool(cfg.model.trust_remote_code),
        dtype=cfg.model.dtype,
        attn_implementation=cfg.model.attn_implementation,
    ).to(device)
    freeze_model(reference_model)

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
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

    rollout_backend = create_rollout_backend(
        cfg=cfg,
        model_name_or_path=model_name,
        tokenizer=tokenizer,
        model_engine=model_engine,
        device=device,
    )

    reward_fn = GSM8KReward(
        correct_answer_reward=float(cfg.reward.correct_answer_reward),
        wrong_answer_reward=float(cfg.reward.wrong_answer_reward),
        format_reward=float(cfg.reward.format_reward),
        require_final_answer_marker=bool(cfg.reward.require_final_answer_marker),
    )
    if is_main_process():
        import wandb

        wandb_config = to_plain_dict(cfg)
        wandb.init(project="llm-course-project", name=args.run_name or cfg.run_name, config=wandb_config)

    global_step = 0
    update_rollout_time_sec = 0.0
    update_step_time_sec = 0.0
    update_sync_weights_time_sec = rollout_backend.sync_weights(model_engine, step=0)
    update_rollout_tokens = 0.0

    print_rank0(
        f"GRPO start: train_examples={len(train_dataset)} group_size={cfg.rollout.group_size} "
        f"loss_type={cfg.train.loss_type} rollout_backend={rollout_backend.name} "
        f"sync_method={rollout_backend.sync_method} prompt_builder={train_dataset.prompt_builder} "
        f"world_size={world_size()} total_update_steps≈{total_update_steps}"
    )

    for epoch in range(int(cfg.train.num_train_epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)
        progress = tqdm(train_loader, disable=not is_main_process(), desc=f"GRPO epoch {epoch}")
        for batch in progress:
            prompts: list[str] = batch["prompt"]
            refs: list[str | None] = batch["answer"]
            group_size = int(cfg.rollout.group_size)

            sync_cuda(device)
            step_start = time.perf_counter()
            rollout_start = time.perf_counter()
            rollout = rollout_backend.generate(prompts).to(device)
            sync_cuda(device)
            rollout_time_sec = time.perf_counter() - rollout_start
            input_ids = rollout.input_ids
            attention_mask = rollout.attention_mask
            completion_mask = rollout.completion_mask
            completion_texts = rollout.completion_texts
            rollout_tokens = float(completion_mask.sum().item())

            repeated_refs = [ref for ref in refs for _ in range(group_size)]
            reward_results = [reward_fn(text, ref) for text, ref in zip(completion_texts, repeated_refs)]
            rewards = torch.tensor([item.reward for item in reward_results], dtype=torch.float32, device=device)
            advantages = compute_group_advantages(rewards, group_size=group_size, loss_type=cfg.train.loss_type)

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
                loss_type=cfg.train.loss_type,
                max_completion_length=max_completion_length,
                sampling_logprobs=rollout.sampling_logprobs,
                importance_sampling=bool(rollout_backend.importance_sampling),
                importance_sampling_cap=float(rollout_backend.importance_sampling_cap),
            )
            model_engine.backward(loss_out.loss)
            model_engine.step()
            sync_cuda(device)
            step_time_sec = time.perf_counter() - step_start
            update_rollout_time_sec += rollout_time_sec
            update_step_time_sec += step_time_sec
            update_rollout_tokens += rollout_tokens

            if model_engine.is_gradient_accumulation_boundary():
                global_step += 1
                update_sync_weights_time_sec += rollout_backend.sync_weights(model_engine, step=global_step)
                avg_reward = float(rewards.mean().item())
                group_acc = sum(int(item.correct) for item in reward_results) / max(1, len(reward_results))
                avg_len = float(completion_mask[:, 1:].sum(dim=1).float().mean().item())
                if global_step % int(cfg.train.logging_steps) == 0:
                    lr = scheduler.get_last_lr()[0] if scheduler is not None else float(cfg.train.learning_rate)
                    (
                        logged_rollout_time_sec,
                        logged_step_time_sec,
                        logged_sync_weights_time_sec,
                        logged_rollout_tokens_per_sec,
                    ) = reduce_speed_metrics(
                        update_rollout_time_sec,
                        update_step_time_sec,
                        update_sync_weights_time_sec,
                        update_rollout_tokens,
                        device,
                    )
                    if is_main_process():
                        log_payload: dict[str, Any] = {
                            "epoch": epoch,
                            "grpo/train/loss": float(loss_out.loss.detach().float().item()),
                            "grpo/train/policy_loss": float(loss_out.policy_loss.float().item()),
                            "grpo/train/kl": float(loss_out.kl.float().item()),
                            "grpo/train/mean_ratio": float(loss_out.mean_ratio.float().item()),
                            "grpo/train/reward": avg_reward,
                            "grpo/train/rollout_accuracy": group_acc,
                            "grpo/train/avg_completion_tokens": avg_len,
                            "grpo/train/lr": lr,
                            "grpo/rollout/backend": rollout_backend.name,
                            "grpo/rollout/sync_method": rollout_backend.sync_method,
                            "grpo/speed/rollout_time_sec": logged_rollout_time_sec,
                            "grpo/speed/step_time_sec": logged_step_time_sec,
                            "grpo/speed/sync_weights_time_sec": logged_sync_weights_time_sec,
                            "grpo/speed/rollout_tokens_per_sec": logged_rollout_tokens_per_sec,
                        }
                        wandb.log(log_payload, step=global_step)
                        print_rank0(
                            "GRPO debug "
                            f"step={global_step} backend={rollout_backend.name} "
                            f"sync_method={rollout_backend.sync_method} "
                            f"rollout_tokens_per_sec={logged_rollout_tokens_per_sec:.2f} "
                            f"rollout_time_sec={logged_rollout_time_sec:.2f} "
                            f"step_time_sec={logged_step_time_sec:.2f} "
                            f"sync_weights_time_sec={logged_sync_weights_time_sec:.2f} "
                            f"reward={avg_reward:.4f}"
                        )
                    progress.set_postfix(loss=f"{float(loss_out.loss):.4f}", reward=f"{avg_reward:.3f}")
                update_rollout_time_sec = 0.0
                update_step_time_sec = 0.0
                update_sync_weights_time_sec = 0.0
                update_rollout_tokens = 0.0
        if bool(cfg.train.get("save_hf_each_epoch", False)):
            save_hf_checkpoint(model_engine, tokenizer, output_dir / f"hf_epoch_{epoch + 1:03d}")
    if bool(cfg.train.save_hf_at_end):
        save_hf_checkpoint(model_engine, tokenizer, output_dir / "hf")
    rollout_backend.close()
    if is_main_process():
        wandb.finish()
    print_rank0("GRPO finished.")


if __name__ == "__main__":
    main()
