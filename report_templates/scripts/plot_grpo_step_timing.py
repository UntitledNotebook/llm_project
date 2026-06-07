"""Fetch GRPO backend timing metrics from W&B and render one bar chart.

Run from the repository root:

    WANDB_ENTITY=zcm25-cvhw .venv/bin/python report_templates/scripts/plot_grpo_step_timing.py

Use cached CSVs after the first successful fetch:

    .venv/bin/python report_templates/scripts/plot_grpo_step_timing.py --use-cache
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
import numpy as np
import pandas as pd


REPORT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = REPORT_DIR / "experiment_inventory.md"
DEFAULT_FIGURES_DIR = REPORT_DIR / "figures"
DEFAULT_DATA_DIR = DEFAULT_FIGURES_DIR / "data"
DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "zcm25-cvhw")
DEFAULT_PROJECT = "llm-course-project"

SELECTED_RUNS = (
    ("grpo_vllm_colocate_3epoch_grpo", "vLLM colocate"),
    ("grpo_vllm_server_pynccl_3epoch_grpo", "vLLM server, PyNccl"),
    ("grpo_vllm_server_tmp_3epoch_grpo", "vLLM server, tmp"),
    ("grpo_hf_3epoch_grpo", "HF generate"),
)
RUN_ORDER = tuple(run_name for run_name, _ in SELECTED_RUNS)
LABEL_BY_RUN = {run_name: label for run_name, label in SELECTED_RUNS}

TIMING_METRICS = (
    ("grpo/speed/step_time_sec", "Step time"),
    ("grpo/speed/sync_weights_time_sec", "Sync weights time"),
    ("grpo/speed/rollout_time_sec", "Rollout time"),
)
METRIC_ORDER = tuple(metric for metric, _ in TIMING_METRICS)
LABEL_BY_METRIC = {metric: label for metric, label in TIMING_METRICS}


@dataclass(frozen=True)
class TrainRunSpec:
    run_name: str
    run_label: str
    run_id: str | None


def clean_markdown_cell(cell: str) -> str:
    return cell.strip().strip("`").strip()


def parse_inventory(path: Path) -> list[TrainRunSpec]:
    text = path.read_text(encoding="utf-8")
    train_ids: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("| `grpo_"):
            continue
        cells = [clean_markdown_cell(cell) for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and cells[0] in RUN_ORDER and re.fullmatch(r"[a-z0-9]{8}", cells[3]):
            train_ids[cells[0]] = cells[3]

    train_runs = [
        TrainRunSpec(run_name=run_name, run_label=LABEL_BY_RUN[run_name], run_id=train_ids.get(run_name))
        for run_name in RUN_ORDER
    ]
    missing_ids = [spec.run_name for spec in train_runs if spec.run_id is None]
    if missing_ids:
        raise ValueError("Missing W&B train ids in inventory: " + ", ".join(missing_ids))
    return train_runs


def iter_metric_history(run: object, metric: str) -> Iterable[dict[str, object]]:
    for row in run.scan_history(keys=["_step", "epoch", metric], page_size=1000):
        value = row.get(metric)
        if value is not None:
            yield {"step": row.get("_step"), "epoch": row.get("epoch"), "metric": metric, "value": value}


def fetch_timing_history(api: object, train_runs: list[TrainRunSpec], *, entity: str, project: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in train_runs:
        run = api.run(f"{entity}/{project}/{spec.run_id}")
        for metric in METRIC_ORDER:
            for history_row in iter_metric_history(run, metric):
                rows.append(
                    {
                        **history_row,
                        "run_name": spec.run_name,
                        "run_label": spec.run_label,
                        "run_id": getattr(run, "id", spec.run_id),
                    }
                )
    if not rows:
        raise RuntimeError("No GRPO timing history rows were fetched from W&B.")

    df = pd.DataFrame(rows)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def summarize_timing(history_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_name in RUN_ORDER:
        for metric in METRIC_ORDER:
            part = history_df[(history_df["run_name"] == run_name) & (history_df["metric"] == metric)]
            if part.empty:
                if metric == "grpo/speed/sync_weights_time_sec" and run_name == "grpo_hf_3epoch_grpo":
                    value = 0.0
                else:
                    raise ValueError(f"Missing timing metric {metric} for run {run_name}")
            else:
                value = float(part["value"].mean())
            rows.append(
                {
                    "run_name": run_name,
                    "run_label": LABEL_BY_RUN[run_name],
                    "metric": metric,
                    "metric_label": LABEL_BY_METRIC[metric],
                    "mean_sec": value,
                }
            )
    return pd.DataFrame(rows)


def fetch_data(*, inventory: Path, entity: str, project: str, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    import wandb

    train_runs = parse_inventory(inventory)
    api = wandb.Api()
    history_df = fetch_timing_history(api, train_runs, entity=entity, project=project)
    summary_df = summarize_timing(history_df)

    data_dir.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(data_dir / "grpo_step_timing_history.csv", index=False)
    summary_df.to_csv(data_dir / "grpo_step_timing_summary.csv", index=False)
    metadata = {
        "entity": entity,
        "project": project,
        "inventory": str(inventory),
        "train_runs": [spec.__dict__ for spec in train_runs],
        "timing_metrics": list(METRIC_ORDER),
        "aggregation": "mean over logged history rows",
    }
    (data_dir / "grpo_step_timing_wandb_fetch_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return history_df, summary_df


def load_cached_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    history_path = data_dir / "grpo_step_timing_history.csv"
    summary_path = data_dir / "grpo_step_timing_summary.csv"
    if not history_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Cached data not found under {data_dir}; run without --use-cache first.")
    return pd.read_csv(history_path), pd.read_csv(summary_path)


def save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(figures_dir / f"{stem}{suffix}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def plot_step_timing(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(METRIC_ORDER))
    width = 0.18
    offsets = (np.arange(len(RUN_ORDER)) - (len(RUN_ORDER) - 1) / 2.0) * width

    colors = ("#4C78A8", "#F58518", "#54A24B", "#B279A2")
    max_value = float(summary_df["mean_sec"].max())
    label_pad = max(max_value * 0.012, 0.05)

    for offset, run_name, color in zip(offsets, RUN_ORDER, colors, strict=True):
        part = summary_df[summary_df["run_name"] == run_name].set_index("metric").loc[list(METRIC_ORDER)]
        values = part["mean_sec"].to_numpy(dtype=float)
        bars = ax.bar(x + offset, values, width=width, label=LABEL_BY_RUN[run_name], color=color)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + label_pad,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )

    ax.set_title("Average GRPO step timing by backend")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_BY_METRIC[metric] for metric in METRIC_ORDER])
    ax.set_ylim(0.0, max_value * 1.14 + label_pad)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, figures_dir, "grpo_step_timing")


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
        _, summary_df = load_cached_data(args.data_dir)
    else:
        _, summary_df = fetch_data(inventory=args.inventory, entity=args.entity, project=args.project, data_dir=args.data_dir)
    plot_step_timing(summary_df, args.figures_dir)
    print(f"Wrote figures to {args.figures_dir}")
    print(f"Wrote cached W&B data to {args.data_dir}")


if __name__ == "__main__":
    main()
