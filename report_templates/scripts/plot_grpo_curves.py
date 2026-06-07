"""Fetch selected GRPO W&B metrics and render report figures.

Run from the repository root:

    WANDB_ENTITY=zcm25-cvhw .venv/bin/python report_templates/scripts/plot_grpo_curves.py

Use cached CSVs after the first successful fetch:

    .venv/bin/python report_templates/scripts/plot_grpo_curves.py --use-cache
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


REPORT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = REPORT_DIR / "experiment_inventory.md"
DEFAULT_FIGURES_DIR = REPORT_DIR / "figures"
DEFAULT_DATA_DIR = DEFAULT_FIGURES_DIR / "data"
DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "zcm25-cvhw")
DEFAULT_PROJECT = "llm-course-project"

TRAIN_METRICS = ("grpo/train/reward", "grpo/train/kl")
REWARD_EMA_ALPHA = 1.0 - 0.99
SELECTED_RUNS = (
    (
        "grpo_hf_3epoch_grpo",
        "HF, GRPO loss",
        "hf_grpo",
    ),
    (
        "grpo_hf_3epoch_dr_grpo",
        "HF, Dr. GRPO loss",
        "hf_dr_grpo",
    ),
    (
        "grpo_vllm_colocate_3epoch_grpo_256",
        "vLLM colocate, GRPO loss, 256 tokens",
        "vllm_colocate_grpo_256",
    ),
)
RUN_ORDER = tuple(run_name for run_name, _, _ in SELECTED_RUNS)
LABEL_BY_RUN = {run_name: label for run_name, label, _ in SELECTED_RUNS}
KEY_BY_RUN = {run_name: key for run_name, _, key in SELECTED_RUNS}


@dataclass(frozen=True)
class TrainRunSpec:
    run_name: str
    run_label: str
    run_key: str
    run_id: str | None


def clean_markdown_cell(cell: str) -> str:
    return cell.strip().strip("`").strip()


def parse_percent(cell: str) -> float:
    value = clean_markdown_cell(cell).rstrip("%").strip()
    return float(value) / 100.0


def parse_inventory(path: Path) -> tuple[list[TrainRunSpec], pd.DataFrame]:
    text = path.read_text(encoding="utf-8")
    train_ids: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("| `grpo_"):
            continue
        cells = [clean_markdown_cell(cell) for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and cells[0] in RUN_ORDER and re.fullmatch(r"[a-z0-9]{8}", cells[3]):
            train_ids[cells[0]] = cells[3]

    train_runs = [
        TrainRunSpec(
            run_name=run_name,
            run_label=LABEL_BY_RUN[run_name],
            run_key=KEY_BY_RUN[run_name],
            run_id=train_ids.get(run_name),
        )
        for run_name in RUN_ORDER
    ]

    eval_rows: list[dict[str, object]] = []
    in_grpo_evals = False
    for line in text.splitlines():
        if line.startswith("## GRPO evals"):
            in_grpo_evals = True
            continue
        if in_grpo_evals and line.startswith("## "):
            break
        if not in_grpo_evals or not line.startswith("| `grpo_"):
            continue

        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        train_run_name = clean_markdown_cell(cells[0])
        if train_run_name not in RUN_ORDER:
            continue
        epoch = int(clean_markdown_cell(cells[1]))
        gsm8k = parse_percent(cells[-2])
        mmlu = parse_percent(cells[-1])
        eval_rows.extend(
            [
                {
                    "run_name": train_run_name,
                    "run_label": LABEL_BY_RUN[train_run_name],
                    "run_key": KEY_BY_RUN[train_run_name],
                    "checkpoint_epoch": epoch,
                    "metric_label": "GSM8K",
                    "accuracy": gsm8k,
                },
                {
                    "run_name": train_run_name,
                    "run_label": LABEL_BY_RUN[train_run_name],
                    "run_key": KEY_BY_RUN[train_run_name],
                    "checkpoint_epoch": epoch,
                    "metric_label": "MMLU",
                    "accuracy": mmlu,
                },
            ]
        )

    missing_ids = [spec.run_name for spec in train_runs if spec.run_id is None]
    if missing_ids:
        raise ValueError("Missing W&B train ids in inventory: " + ", ".join(missing_ids))
    if not eval_rows:
        raise ValueError(f"No selected GRPO eval rows found in {path}")
    return train_runs, pd.DataFrame(eval_rows)


def iter_metric_history(run: object, metric: str) -> Iterable[dict[str, object]]:
    for row in run.scan_history(keys=["_step", "epoch", metric], page_size=1000):
        value = row.get(metric)
        if value is not None:
            yield {"step": row.get("_step"), "epoch": row.get("epoch"), "metric": metric, "value": value}


def fetch_training_history(api: object, train_runs: list[TrainRunSpec], *, entity: str, project: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in train_runs:
        run = api.run(f"{entity}/{project}/{spec.run_id}")
        for metric in TRAIN_METRICS:
            for history_row in iter_metric_history(run, metric):
                rows.append(
                    {
                        **history_row,
                        "run_name": spec.run_name,
                        "run_label": spec.run_label,
                        "run_key": spec.run_key,
                        "run_id": getattr(run, "id", spec.run_id),
                    }
                )
    if not rows:
        raise RuntimeError("No GRPO training history rows were fetched from W&B.")
    df = pd.DataFrame(rows)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["step", "value"])


def fetch_data(*, inventory: Path, entity: str, project: str, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    import wandb

    train_runs, eval_df = parse_inventory(inventory)
    api = wandb.Api()
    training_df = fetch_training_history(api, train_runs, entity=entity, project=project)

    data_dir.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(data_dir / "grpo_training_history.csv", index=False)
    eval_df.to_csv(data_dir / "grpo_epoch_eval.csv", index=False)
    metadata = {
        "entity": entity,
        "project": project,
        "inventory": str(inventory),
        "train_runs": [spec.__dict__ for spec in train_runs],
        "train_metrics": list(TRAIN_METRICS),
    }
    (data_dir / "grpo_wandb_fetch_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return training_df, eval_df


def load_cached_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_path = data_dir / "grpo_training_history.csv"
    eval_path = data_dir / "grpo_epoch_eval.csv"
    if not training_path.exists() or not eval_path.exists():
        raise FileNotFoundError(f"Cached data not found under {data_dir}; run without --use-cache first.")
    return pd.read_csv(training_path), pd.read_csv(eval_path)


def save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(figures_dir / f"{stem}{suffix}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def setup_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def ema_099(series: pd.Series) -> pd.Series:
    return series.ewm(alpha=REWARD_EMA_ALPHA, adjust=False).mean()


def plot_training_metric(
    training_df: pd.DataFrame,
    figures_dir: Path,
    metric: str,
    title: str,
    ylabel: str,
    stem: str,
    *,
    value_transform: object | None = None,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for run_name in RUN_ORDER:
        part = training_df[(training_df["run_name"] == run_name) & (training_df["metric"] == metric)].copy()
        part = part.dropna(subset=["step", "value"]).sort_values("step")
        values = part["value"]
        if value_transform is not None:
            values = value_transform(values)
        ax.plot(part["step"], values, linewidth=1.6, label=LABEL_BY_RUN[run_name])
    if log_y:
        ax.set_yscale("log")
    setup_axis(ax, title, "Training step", ylabel)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
    save_figure(fig, figures_dir, stem)


def plot_eval_metric(eval_df: pd.DataFrame, figures_dir: Path, metric_label: str, title: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for run_name in RUN_ORDER:
        part = eval_df[(eval_df["run_name"] == run_name) & (eval_df["metric_label"] == metric_label)].copy()
        part["accuracy_pct"] = 100.0 * pd.to_numeric(part["accuracy"], errors="coerce")
        part = part.dropna(subset=["accuracy_pct"]).sort_values("checkpoint_epoch")
        ax.plot(part["checkpoint_epoch"], part["accuracy_pct"], marker="o", label=LABEL_BY_RUN[run_name])
    setup_axis(ax, title, "Checkpoint epoch", "Accuracy (%)")
    ax.set_xticks([1, 2, 3])
    save_figure(fig, figures_dir, stem)


def plot_figures(training_df: pd.DataFrame, eval_df: pd.DataFrame, figures_dir: Path) -> None:
    plot_training_metric(
        training_df,
        figures_dir,
        "grpo/train/reward",
        "GRPO training reward (EMA 0.99)",
        "Reward",
        "grpo_reward",
        value_transform=ema_099,
    )
    plot_training_metric(training_df, figures_dir, "grpo/train/kl", "GRPO training KL", "KL", "grpo_kl", log_y=True)
    plot_eval_metric(eval_df, figures_dir, "GSM8K", "GRPO GSM8K accuracy", "grpo_gsm8k_eval")
    plot_eval_metric(eval_df, figures_dir, "MMLU", "GRPO MMLU accuracy", "grpo_mmlu_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--use-cache", action="store_true", help="Render from cached CSVs instead of querying W&B.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_cache:
        training_df, eval_df = load_cached_data(args.data_dir)
    else:
        training_df, eval_df = fetch_data(inventory=args.inventory, entity=args.entity, project=args.project, data_dir=args.data_dir)
    plot_figures(training_df, eval_df, args.figures_dir)
    print(f"Wrote figures to {args.figures_dir}")
    print(f"Wrote cached W&B data to {args.data_dir}")


if __name__ == "__main__":
    main()
