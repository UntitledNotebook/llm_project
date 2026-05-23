#!/usr/bin/env bash
set -euo pipefail
python -m llm_project.cli.plot_curves \
  --sft outputs/sft_qwen25_1p5b_numina_gsm8k/logs/sft_metrics.jsonl \
  --grpo outputs/grpo_qwen25_1p5b_gsm8k/logs/grpo_metrics.jsonl \
  --out_dir outputs/figures
