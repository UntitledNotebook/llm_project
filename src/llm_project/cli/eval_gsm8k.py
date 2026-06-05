from __future__ import annotations

import argparse
import json
import os

from llm_project.config import load_config, to_plain_dict
from llm_project.evaluation.gsm8k_eval import evaluate_gsm8k_model
from llm_project.evaluation.vllm_utils import load_vllm_llm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8K with vLLM")
    parser.add_argument("--config", type=str, default="configs/templates/eval.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override model.name_or_path")
    parser.add_argument("--output", type=str, default=None, help="Override gsm8k.output_path")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None, help="Set wandb run name")
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
    wandb.init(
        project="llm-course-project",
        entity=os.environ["WANDB_ENTITY"],
        name=args.run_name or cfg.get("run_name"),
        config=wandb_config,
    )
    llm = load_vllm_llm(model_name, cfg)
    metrics = evaluate_gsm8k_model(
        llm,
        dataset_name=cfg.gsm8k.dataset_name,
        config_name=cfg.gsm8k.config_name,
        split=cfg.gsm8k.split,
        max_samples=args.max_samples if args.max_samples is not None else cfg.gsm8k.max_samples,
        batch_size=int(cfg.gsm8k.batch_size),
        max_new_tokens=int(cfg.gsm8k.max_new_tokens),
        temperature=float(cfg.gsm8k.temperature),
        top_p=float(cfg.gsm8k.top_p),
        prompt_builder=cfg.gsm8k.get("prompt_builder"),
        output_path=args.output or cfg.gsm8k.output_path,
    )
    wandb.log(
        {
            f"eval/{args.stage}/gsm8k_accuracy": metrics["accuracy"],
            f"eval/{args.stage}/gsm8k_correct": metrics["correct"],
            f"eval/{args.stage}/gsm8k_total": metrics["total"],
        }
    )
    wandb.finish()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
