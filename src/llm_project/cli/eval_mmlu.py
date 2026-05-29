from __future__ import annotations

import argparse
import json

import torch

from llm_project.config import load_config, to_plain_dict
from llm_project.evaluation.mmlu_eval import evaluate_mmlu_model
from llm_project.models import load_causal_lm, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on MMLU using generated final answers"
    )
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override model.name_or_path")
    parser.add_argument(
        "--subjects", type=str, default=None, help="all or comma-separated MMLU subjects"
    )
    parser.add_argument("--output", type=str, default=None, help="Override mmlu.output_path")
    parser.add_argument("--max_samples_per_subject", type=int, default=None)
    parser.add_argument(
        "--attn", type=str, default=None, help="Override attn implementation, e.g. eager"
    )
    parser.add_argument(
        "--stage", type=str, default="custom", help="Metric stage name, e.g. base, sft, grpo"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_name = args.model or cfg.model.name_or_path
    import wandb

    wandb_config = to_plain_dict(cfg)
    wandb_config["eval_stage"] = args.stage
    wandb_config["model"]["name_or_path"] = model_name
    wandb.init(project="llm-course-project", name=f"{args.stage}_mmlu_eval", config=wandb_config)
    tokenizer = load_tokenizer(
        model_name, trust_remote_code=bool(cfg.model.trust_remote_code), padding_side="left"
    )
    model = load_causal_lm(
        model_name,
        trust_remote_code=bool(cfg.model.trust_remote_code),
        dtype=cfg.model.dtype,
        attn_implementation=args.attn or cfg.model.attn_implementation,
    ).cuda()
    metrics = evaluate_mmlu_model(
        model,
        tokenizer,
        dataset_name=cfg.mmlu.dataset_name,
        subjects=args.subjects or cfg.mmlu.subjects,
        split=cfg.mmlu.split,
        max_samples_per_subject=(
            args.max_samples_per_subject
            if args.max_samples_per_subject is not None
            else cfg.mmlu.max_samples_per_subject
        ),
        batch_size=int(cfg.mmlu.get("batch_size", 1)),
        max_new_tokens=int(cfg.mmlu.get("max_new_tokens", 512)),
        temperature=float(cfg.mmlu.get("temperature", 0.0)),
        top_p=float(cfg.mmlu.get("top_p", 1.0)),
        output_path=args.output or cfg.mmlu.output_path,
    )
    wandb_metrics = {
        f"eval/{args.stage}/mmlu_accuracy": metrics["accuracy"],
        f"eval/{args.stage}/mmlu_correct": metrics["correct"],
        f"eval/{args.stage}/mmlu_total": metrics["total"],
    }
    for subject, subject_metrics in metrics["per_subject"].items():
        wandb_metrics[f"eval/{args.stage}/mmlu/{subject}_accuracy"] = subject_metrics["accuracy"]
    wandb.log(wandb_metrics)
    wandb.finish()
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_subject"}, indent=2))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
