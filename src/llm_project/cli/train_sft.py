from __future__ import annotations

import argparse
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
from llm_project.data.sft_dataset import (
    SFTDataCollator,
    SFTDataset,
    load_numina_gsm8k_sft_raw,
    train_validation_split,
)
from llm_project.distributed import (
    all_reduce_sum,
    barrier,
    global_rank,
    is_main_process,
    local_rank,
    maybe_init_distributed,
    print_rank0,
    world_size,
)
from llm_project.models import enable_training_mode, load_causal_lm, load_tokenizer
from llm_project.seed import set_seed
from llm_project.training.checkpointing import save_hf_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT training for Choice A Part 1")
    parser.add_argument("--config", type=str, required=True, help="Path to configs/templates/sft.yaml")
    parser.add_argument("--ds_config", type=str, required=True, help="Path to DeepSpeed JSON config")
    parser.add_argument("--local_rank", type=int, default=-1, help="Injected by DeepSpeed launcher")
    parser.add_argument("--run_name", type=str, default=None, help="Override wandb run name")
    return parser.parse_args()


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model_engine: Any, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    model_engine.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = torch.tensor(0.0, device=device)
    for batch in dataloader:
        batch = move_to_device(batch, device)
        outputs = model_engine(**batch)
        labels = batch["labels"]
        n_tokens = (labels != -100).sum().float()
        total_loss += outputs.loss.float() * n_tokens
        total_tokens += n_tokens
    total_loss = all_reduce_sum(total_loss)
    total_tokens = all_reduce_sum(total_tokens)
    model_engine.train()
    loss = (total_loss / total_tokens.clamp_min(1.0)).item()
    return {"val_loss": loss, "val_ppl": float(math.exp(min(loss, 20.0)))}


def main() -> None:
    args = parse_args()
    maybe_init_distributed()
    device = torch.device("cuda", local_rank())
    cfg = load_config(args.config)
    set_seed(int(cfg.seed), rank_offset=global_rank())

    output_dir = Path(cfg.train.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(to_plain_dict(cfg), output_dir / "resolved_sft_config.yaml")
        shutil.copyfile(args.ds_config, output_dir / "deepspeed_config.json")
    barrier()

    tokenizer = load_tokenizer(
        cfg.model.name_or_path,
        trust_remote_code=bool(cfg.model.trust_remote_code),
        padding_side="right",
    )
    raw = load_numina_gsm8k_sft_raw(max_samples=cfg.dataset.max_train_samples)
    train_raw, val_raw = train_validation_split(raw, cfg.dataset.validation_size, int(cfg.seed))
    if cfg.dataset.max_eval_samples is not None:
        val_raw = val_raw.select(range(min(int(cfg.dataset.max_eval_samples), len(val_raw))))
    prompt_builder = cfg.dataset.get("prompt_builder")

    train_dataset = SFTDataset(
        train_raw,
        tokenizer,
        max_seq_length=cfg.dataset.max_seq_length,
        prompt_builder=prompt_builder,
    )
    val_dataset = SFTDataset(
        val_raw,
        tokenizer,
        max_seq_length=cfg.dataset.max_seq_length,
        prompt_builder=prompt_builder,
    )
    collator = SFTDataCollator(tokenizer)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size(), rank=global_rank(), shuffle=True, seed=int(cfg.seed)
    ) if world_size() > 1 else None
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size(), rank=global_rank(), shuffle=False
    ) if world_size() > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.train.per_device_train_batch_size),
        sampler=train_sampler,
        shuffle=train_sampler is None,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.train.per_device_eval_batch_size),
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    model = load_causal_lm(
        cfg.model.name_or_path,
        trust_remote_code=bool(cfg.model.trust_remote_code),
        dtype=cfg.model.dtype,
        attn_implementation=cfg.model.attn_implementation,
    )
    enable_training_mode(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.train.learning_rate), weight_decay=float(cfg.train.weight_decay)
    )
    grad_accum = 1
    try:
        import json

        with open(args.ds_config, "r", encoding="utf-8") as f:
            grad_accum = int(json.load(f).get("gradient_accumulation_steps", 1))
    except Exception:
        grad_accum = 1
    total_update_steps = math.ceil(len(train_loader) / max(1, grad_accum)) * int(cfg.train.num_train_epochs)
    warmup_steps = int(float(cfg.train.warmup_ratio) * total_update_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=args.ds_config,
    )

    if is_main_process():
        import wandb

        wandb.init(project="llm-course-project", name=args.run_name or cfg.run_name, config=to_plain_dict(cfg))
    global_step = 0
    print_rank0(
        f"SFT start: train_examples={len(train_dataset)} val_examples={len(val_dataset)} "
        f"prompt_builder={train_dataset.prompt_builder} "
        f"world_size={world_size()} total_update_steps≈{total_update_steps}"
    )

    for epoch in range(int(cfg.train.num_train_epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        loss_total = 0.0
        loss_count = 0
        progress = tqdm(train_loader, disable=not is_main_process(), desc=f"SFT epoch {epoch}")
        for micro_step, batch in enumerate(progress, start=1):
            batch = move_to_device(batch, device)
            outputs = model_engine(**batch)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()
            loss_total += float(loss.detach().float().item())
            loss_count += 1

            if model_engine.is_gradient_accumulation_boundary():
                global_step += 1
                if global_step % int(cfg.train.logging_steps) == 0:
                    lr = (
                        scheduler.get_last_lr()[0]
                        if scheduler is not None
                        else float(cfg.train.learning_rate)
                    )
                    avg_loss = loss_total / max(1, loss_count)
                    if is_main_process():
                        wandb.log(
                            {
                                "epoch": epoch,
                                "sft/train/loss": avg_loss,
                                "sft/train/lr": lr,
                            },
                            step=global_step,
                        )
                    progress.set_postfix(loss=f"{avg_loss:.4f}", step=global_step)
                    loss_total = 0.0
                    loss_count = 0

                if int(cfg.train.eval_steps) > 0 and global_step % int(cfg.train.eval_steps) == 0:
                    metrics = evaluate(model_engine, val_loader, device)
                    if is_main_process():
                        wandb.log(
                            {
                                "epoch": epoch,
                                "sft/eval/val_loss": metrics["val_loss"],
                                "sft/eval/val_ppl": metrics["val_ppl"],
                            },
                            step=global_step,
                        )
                    print_rank0(f"eval step={global_step}: {metrics}")
        if bool(cfg.train.get("save_hf_each_epoch", False)):
            save_hf_checkpoint(model_engine, tokenizer, output_dir / f"hf_epoch_{epoch + 1:03d}")
    metrics = evaluate(model_engine, val_loader, device)
    if is_main_process():
        wandb.log(
            {
                "epoch": int(cfg.train.num_train_epochs),
                "sft/eval/val_loss": metrics["val_loss"],
                "sft/eval/val_ppl": metrics["val_ppl"],
            },
            step=global_step,
        )
    if bool(cfg.train.save_hf_at_end):
        save_hf_checkpoint(model_engine, tokenizer, output_dir / "hf")
    if is_main_process():
        wandb.finish()
    print_rank0("SFT finished.")


if __name__ == "__main__":
    main()
