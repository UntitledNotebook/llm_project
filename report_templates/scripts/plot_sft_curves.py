"""Fetch SFT W&B metrics and render plain report figures.

Run from the repository root:

    WANDB_ENTITY=zcm25-cvhw .venv/bin/python report_templates/scripts/plot_sft_curves.py
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

SFT_TRAIN_RUN_NAMES = (
    "sft_qwen25_1p5b_numina_gsm8k_simple_3epoch",
    "sft_qwen25_1p5b_numina_gsm8k_step_by_step_3epoch",
)
TRAIN_METRICS = ("sft/train/loss", "sft/eval/val_loss", "sft/eval/val_ppl")
EVAL_METRIC_BY_RUN_SUFFIX = {
    "gsm8k_step_by_step": ("GSM8K step-by-step", "eval/sft/gsm8k_accuracy"),
    "gsm8k_simple": ("GSM8K simple", "eval/sft/gsm8k_accuracy"),
    "mmlu": ("MMLU", "eval/sft/mmlu_accuracy"),
}
PROMPT_LABELS = {"simple": "Simple SFT", "step_by_step": "Step-by-step SFT"}
PROMPT_ORDER = ("simple", "step_by_step")


@dataclass(frozen=True)
class TrainRunSpec:
    prompt: str
    run_name: str
    run_id: str | None


@dataclass(frozen=True)
class EvalRunSpec:
    prompt: str
    checkpoint_epoch: int
    metric_label: str
    wandb_metric: str
    run_name: str


def clean_markdown_cell(cell: str) -> str:
    return cell.strip().strip("`").strip()


def prompt_from_train_run_name(run_name: str) -> str:
    if "_step_by_step_" in run_name:
        return "step_by_step"
    if "_simple_" in run_name:
        return "simple"
    raise ValueError(f"Cannot infer SFT prompt from run name: {run_name}")


def parse_inventory(path: Path) -> tuple[list[TrainRunSpec], list[EvalRunSpec]]:
    text = path.read_text(encoding="utf-8")
    train_ids: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("| `sft_qwen25_1p5b_numina_gsm8k_"):
            continue
        cells = [clean_markdown_cell(cell) for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and cells[0] in SFT_TRAIN_RUN_NAMES:
            train_ids[cells[0]] = cells[3]

    train_runs = [
        TrainRunSpec(
            prompt=prompt_from_train_run_name(run_name),
            run_name=run_name,
            run_id=train_ids.get(run_name),
        )
        for run_name in SFT_TRAIN_RUN_NAMES
    ]

    eval_runs: list[EvalRunSpec] = []
    in_sft_eval_table = False
    for line in text.splitlines():
        if line.startswith("## Base and SFT evals"):
            in_sft_eval_table = True
            continue
        if in_sft_eval_table and line.startswith("## GRPO evals"):
            break
        if not in_sft_eval_table or not line.startswith("| sft |"):
            continue

        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        train_run_name = clean_markdown_cell(cells[1])
        prompt = prompt_from_train_run_name(train_run_name)
        checkpoint_match = re.search(r"hf_epoch_(\d{3})", cells[2])
        if checkpoint_match is None:
            raise ValueError(f"Cannot infer checkpoint epoch from inventory row: {line}")
        checkpoint_epoch = int(checkpoint_match.group(1))
        for eval_run_name in re.findall(r"`([^`]+)`", cells[3]):
            for suffix, (metric_label, wandb_metric) in EVAL_METRIC_BY_RUN_SUFFIX.items():
                if eval_run_name.endswith(suffix):
                    eval_runs.append(
                        EvalRunSpec(
                            prompt=prompt,
                            checkpoint_epoch=checkpoint_epoch,
                            metric_label=metric_label,
                            wandb_metric=wandb_metric,
                            run_name=eval_run_name,
                        )
                    )
                    break

    if not train_runs:
        raise ValueError(f"No SFT train runs found in {path}")
    if not eval_runs:
        raise ValueError(f"No SFT eval runs found in {path}")
    return train_runs, eval_runs


def project_runs_by_name(api: object, entity: str, project: str) -> dict[str, object]:
    runs = list(api.runs(f"{entity}/{project}", per_page=200))
    lookup: dict[str, object] = {}
    for run in runs:
        for value in (getattr(run, "id", None), getattr(run, "name", None), getattr(run, "display_name", None)):
            if value:
                lookup[str(value)] = run
    return lookup


def load_wandb_run(api: object, entity: str, project: str, spec: TrainRunSpec) -> object:
    if spec.run_id:
        return api.run(f"{entity}/{project}/{spec.run_id}")
    lookup = project_runs_by_name(api, entity, project)
    if spec.run_name not in lookup:
        raise KeyError(f"Could not find W&B train run {spec.run_name}")
    return lookup[spec.run_name]


def iter_metric_history(run: object, metric: str) -> Iterable[dict[str, object]]:
    for row in run.scan_history(keys=["_step", "epoch", metric], page_size=1000):
        value = row.get(metric)
        if value is not None:
            yield {"step": row.get("_step"), "epoch": row.get("epoch"), "metric": metric, "value": value}


def fetch_training_history(api: object, train_runs: list[TrainRunSpec], *, entity: str, project: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in train_runs:
        run = load_wandb_run(api, entity, project, spec)
        for metric in TRAIN_METRICS:
            for history_row in iter_metric_history(run, metric):
                rows.append(
                    {
                        **history_row,
                        "prompt": spec.prompt,
                        "prompt_label": PROMPT_LABELS[spec.prompt],
                        "run_name": spec.run_name,
                        "run_id": getattr(run, "id", spec.run_id),
                    }
                )
    if not rows:
        raise RuntimeError("No SFT training history rows were fetched from W&B.")
    df = pd.DataFrame(rows)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def metric_value_from_run(run: object, metric: str) -> float:
    summary = dict(run.summary)
    value = summary.get(metric)
    if value is not None and not isinstance(value, dict):
        return float(value)
    for row in run.scan_history(keys=[metric], page_size=100):
        value = row.get(metric)
        if value is not None:
            return float(value)
    raise KeyError(f"Metric {metric} not found in run {getattr(run, 'name', '<unknown>')}")


def fetch_epoch_evals(api: object, eval_runs: list[EvalRunSpec], *, entity: str, project: str) -> pd.DataFrame:
    lookup = project_runs_by_name(api, entity, project)
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for spec in eval_runs:
        run = lookup.get(spec.run_name)
        if run is None:
            missing.append(spec.run_name)
            continue
        rows.append(
            {
                "prompt": spec.prompt,
                "prompt_label": PROMPT_LABELS[spec.prompt],
                "checkpoint_epoch": spec.checkpoint_epoch,
                "metric_label": spec.metric_label,
                "accuracy": metric_value_from_run(run, spec.wandb_metric),
                "wandb_metric": spec.wandb_metric,
                "run_name": spec.run_name,
                "run_id": getattr(run, "id", None),
            }
        )
    if missing:
        raise KeyError("Could not find W&B eval runs: " + ", ".join(sorted(missing)))
    if not rows:
        raise RuntimeError("No SFT eval metrics were fetched from W&B.")
    return pd.DataFrame(rows)


def fetch_data(*, inventory: Path, entity: str, project: str, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    import wandb

    train_runs, eval_runs = parse_inventory(inventory)
    api = wandb.Api()
    training_df = fetch_training_history(api, train_runs, entity=entity, project=project)
    eval_df = fetch_epoch_evals(api, eval_runs, entity=entity, project=project)

    data_dir.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(data_dir / "sft_training_history.csv", index=False)
    eval_df.to_csv(data_dir / "sft_epoch_eval.csv", index=False)
    metadata = {
        "entity": entity,
        "project": project,
        "inventory": str(inventory),
        "train_runs": [spec.__dict__ for spec in train_runs],
        "eval_run_count": len(eval_runs),
    }
    (data_dir / "sft_wandb_fetch_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return training_df, eval_df


def load_cached_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_path = data_dir / "sft_training_history.csv"
    eval_path = data_dir / "sft_epoch_eval.csv"
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
    ax.legend()


def plot_loss(training_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    metric_specs = (
        ("sft/train/loss", "train", "-"),
        ("sft/eval/val_loss", "val", "--"),
    )
    for prompt in PROMPT_ORDER:
        for metric, metric_label, linestyle in metric_specs:
            part = training_df[(training_df["metric"] == metric) & (training_df["prompt"] == prompt)].dropna(subset=["step", "value"])
            part = part.sort_values("step")
            ax.plot(
                part["step"],
                part["value"],
                marker="o",
                markersize=3,
                linestyle=linestyle,
                label=f"{PROMPT_LABELS[prompt]} {metric_label}",
            )
    setup_axis(ax, "SFT loss", "Step", "Loss")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
    save_figure(fig, figures_dir, "sft_loss")


def plot_gsm8k(eval_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for prompt in PROMPT_ORDER:
        for metric_label in ("GSM8K simple", "GSM8K step-by-step"):
            part = eval_df[(eval_df["prompt"] == prompt) & (eval_df["metric_label"] == metric_label)].copy()
            part["accuracy_pct"] = 100.0 * pd.to_numeric(part["accuracy"], errors="coerce")
            part = part.dropna(subset=["accuracy_pct"]).sort_values("checkpoint_epoch")
            ax.plot(
                part["checkpoint_epoch"],
                part["accuracy_pct"],
                marker="o",
                label=f"{PROMPT_LABELS[prompt]}, {metric_label}",
            )
    setup_axis(ax, "SFT GSM8K accuracy", "Checkpoint epoch", "Accuracy (%)")
    ax.set_xticks([1, 2, 3])
    save_figure(fig, figures_dir, "sft_gsm8k_eval")


def plot_mmlu(eval_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for prompt in PROMPT_ORDER:
        part = eval_df[(eval_df["prompt"] == prompt) & (eval_df["metric_label"] == "MMLU")].copy()
        part["accuracy_pct"] = 100.0 * pd.to_numeric(part["accuracy"], errors="coerce")
        part = part.dropna(subset=["accuracy_pct"]).sort_values("checkpoint_epoch")
        ax.plot(part["checkpoint_epoch"], part["accuracy_pct"], marker="o", label=PROMPT_LABELS[prompt])
    setup_axis(ax, "SFT MMLU accuracy", "Checkpoint epoch", "Accuracy (%)")
    ax.set_xticks([1, 2, 3])
    save_figure(fig, figures_dir, "sft_mmlu_eval")


def plot_figures(training_df: pd.DataFrame, eval_df: pd.DataFrame, figures_dir: Path) -> None:
    plot_loss(training_df, figures_dir)
    plot_gsm8k(eval_df, figures_dir)
    plot_mmlu(eval_df, figures_dir)


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
