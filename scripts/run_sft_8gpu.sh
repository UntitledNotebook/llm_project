#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

deepspeed --num_gpus 8 --module llm_project.cli.train_sft \
  --config configs/sft.yaml \
  --ds_config configs/deepspeed_zero2.json
