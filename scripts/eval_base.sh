#!/usr/bin/env bash
set -euo pipefail
python -m llm_project.cli.eval_gsm8k \
  --config configs/eval.yaml \
  --model Qwen/Qwen2.5-1.5B-Base \
  --output outputs/eval/base_gsm8k.json

python -m llm_project.cli.eval_mmlu \
  --config configs/eval.yaml \
  --model Qwen/Qwen2.5-1.5B-Base \
  --output outputs/eval/base_mmlu.json
