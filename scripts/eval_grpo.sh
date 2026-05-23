#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH=${1:-outputs/grpo_qwen25_1p5b_gsm8k/hf}
python -m llm_project.cli.eval_gsm8k --config configs/eval.yaml --model "$MODEL_PATH" --output outputs/eval/grpo_gsm8k.json
python -m llm_project.cli.eval_mmlu --config configs/eval.yaml --model "$MODEL_PATH" --output outputs/eval/grpo_mmlu.json
