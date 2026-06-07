#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH=${1:-outputs/sft_qwen25_1p5b_numina_gsm8k/hf}
python -m llm_project.cli.eval_gsm8k --config configs/templates/eval.yaml --model "$MODEL_PATH" --output outputs/eval/sft_gsm8k.json --stage sft
python -m llm_project.cli.eval_mmlu --config configs/templates/eval.yaml --model "$MODEL_PATH" --output outputs/eval/sft_mmlu.json --stage sft
