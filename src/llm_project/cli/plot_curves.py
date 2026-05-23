from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def read_jsonl(path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def plot_sft(path: str | Path, output: str | Path) -> None:
    df = read_jsonl(path)
    plt.figure()
    train = df[df["phase"] == "train"]
    if not train.empty:
        plt.plot(train["step"], train["loss"], label="train loss")
    eval_df = df[df["phase"] == "eval"]
    if not eval_df.empty:
        plt.plot(eval_df["step"], eval_df["val_loss"], label="validation loss")
    plt.xlabel("Update step")
    plt.ylabel("Loss")
    plt.title("SFT train / validation curve")
    plt.legend()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def plot_grpo(path: str | Path, output: str | Path) -> None:
    df = read_jsonl(path)
    train = df[df["phase"] == "train"]
    plt.figure()
    if not train.empty:
        plt.plot(train["step"], train["reward"], label="mean reward")
    eval_df = df[df["phase"] == "eval"]
    if not eval_df.empty and "accuracy" in eval_df:
        plt.plot(eval_df["step"], eval_df["accuracy"], label="test accuracy")
    plt.xlabel("Update step")
    plt.ylabel("Value")
    plt.title("GRPO reward and test accuracy curve")
    plt.legend()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot report curves from JSONL logs")
    parser.add_argument("--sft", type=str, default=None, help="Path to sft_metrics.jsonl")
    parser.add_argument("--grpo", type=str, default=None, help="Path to grpo_metrics.jsonl")
    parser.add_argument("--out_dir", type=str, default="outputs/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.sft:
        plot_sft(args.sft, out_dir / "sft_train_validation_curve.png")
        print(f"Saved {out_dir / 'sft_train_validation_curve.png'}")
    if args.grpo:
        plot_grpo(args.grpo, out_dir / "grpo_reward_accuracy_curve.png")
        print(f"Saved {out_dir / 'grpo_reward_accuracy_curve.png'}")


if __name__ == "__main__":
    main()
