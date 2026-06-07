#!/usr/bin/env bash
set -euo pipefail
python -m llm_project.cli.eval_gsm8k \
  --config configs/templates/eval.yaml \
  --model Qwen/Qwen2.5-1.5B \
  --output outputs/eval/base_gsm8k.json \
  --stage base

python -m llm_project.cli.eval_mmlu \
  --config configs/templates/eval.yaml \
  --model Qwen/Qwen2.5-1.5B \
  --output outputs/eval/base_mmlu.json \
  --stage base
