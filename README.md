# Choice A Part 1–2 code framework: SFT + GRPO

This repository is a code framework for the course project **Choice A: Post-training Implementation**, covering:

- **Part 1: SFT** on `Qwen/Qwen2.5-1.5B` using the `AI-MO/NuminaMath-CoT` subset with `source == "gsm8k"`.
- **Part 2: GRPO** on `Qwen/Qwen2.5-1.5B` using GSM8K.

Part 3 open exploration is intentionally omitted.

## Repository layout

```text
.
├── configs/
│   ├── sft.yaml                  # SFT model/data/training config
│   ├── grpo.yaml                 # GRPO model/data/reward/training config
│   ├── eval.yaml                 # GSM8K and MMLU evaluation config
│   ├── deepspeed_zero2.json      # recommended default for 8 × RTX 4090
│   └── deepspeed_zero3.json      # optional memory-saving config
├── scripts/
│   ├── doctor.py                 # environment sanity checks
│   ├── resolve_flash_attn_wheel.py
│   ├── run_sft_8gpu.sh
│   ├── run_grpo_8gpu.sh
│   ├── eval_base.sh
│   ├── eval_sft.sh
│   ├── eval_grpo.sh
│   └── plot_all.sh
├── src/llm_project/
│   ├── cli/                      # train/eval/plot entrypoints
│   ├── data/                     # dataset loading and prompt formatting
│   ├── evaluation/               # GSM8K and MMLU evaluators
│   ├── training/                 # losses, rewards, generation, checkpoint helpers
│   ├── config.py
│   ├── distributed.py
│   ├── math_utils.py
│   └── models.py
├── tests/
├── report_templates/
├── env_setup.md
└── reference.md
```

## 1. Environment setup

Follow `env_setup.md` first. The key steps are:

```bash
uv python install 3.10
uv venv .venv --python 3.10
source .venv/bin/activate

# Install PyTorch first, then project dependencies and flash-attn.
python scripts/doctor.py
```

The provided server has 8 × RTX 4090 GPUs, so the default scripts use `--num_gpus 8` and `configs/deepspeed_zero2.json`.

## 2. Where to modify configs

Most changes should be made in these files:

- `configs/sft.yaml`: model path, NuminaMath filtering, max sequence length, SFT learning rate, epoch count, output directory.
- `configs/grpo.yaml`: initial policy path, reference model path, rollout group size, max generation length, reward weights, KL coefficient, GRPO learning rate.
- `configs/eval.yaml`: evaluation model path defaults, GSM8K/MMLU sample limits, output paths.
- `configs/deepspeed_zero2.json`: micro-batch size, gradient accumulation, bf16, ZeRO stage.

For a quick smoke test, set these values before a full run:

```yaml
# configs/sft.yaml
dataset:
  max_train_samples: 64
  max_eval_samples: 32
train:
  num_train_epochs: 1
  eval_steps: 5
```

```yaml
# configs/grpo.yaml
dataset:
  max_train_samples: 32
  max_eval_samples: 32
rollout:
  group_size: 2
  max_new_tokens: 128
train:
  eval_steps: 10
```

## 3. Baseline evaluation before post-training

Run baseline GSM8K and MMLU evaluation on the base model:

```bash
source .venv/bin/activate
bash scripts/eval_base.sh
```

Outputs:

```text
outputs/eval/base_gsm8k.json
outputs/eval/base_mmlu.json
```

For a fast debug run:

```bash
python -m llm_project.cli.eval_gsm8k --model Qwen/Qwen2.5-1.5B --max_samples 16 --output outputs/eval/debug_gsm8k.json
python -m llm_project.cli.eval_mmlu --model Qwen/Qwen2.5-1.5B --subjects abstract_algebra --max_samples_per_subject 16 --output outputs/eval/debug_mmlu.json
```

## 4. Part 1: Run SFT

Launch SFT on all eight GPUs:

```bash
bash scripts/run_sft_8gpu.sh
```

Equivalent explicit command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --num_gpus 8 -m llm_project.cli.train_sft \
  --config configs/sft.yaml \
  --ds_config configs/deepspeed_zero2.json
```

Main outputs:

```text
outputs/sft_qwen25_1p5b_numina_gsm8k/
├── hf/                            # Hugging Face checkpoint for evaluation
├── logs/sft_metrics.jsonl         # train loss and validation loss
├── resolved_sft_config.yaml
└── deepspeed_config.json
```

Evaluate the SFT checkpoint:

```bash
bash scripts/eval_sft.sh outputs/sft_qwen25_1p5b_numina_gsm8k/hf
```

Plot the SFT curve:

```bash
python -m llm_project.cli.plot_curves \
  --sft outputs/sft_qwen25_1p5b_numina_gsm8k/logs/sft_metrics.jsonl \
  --out_dir outputs/figures
```

Report figure:

```text
outputs/figures/sft_train_validation_curve.png
```

## 5. Part 2: Run GRPO

By default, GRPO starts from `Qwen/Qwen2.5-1.5B`. To run RL after the SFT checkpoint, edit `configs/grpo.yaml`:

```yaml
model:
  init_from_sft_checkpoint: outputs/sft_qwen25_1p5b_numina_gsm8k/hf
  reference_model_name_or_path: Qwen/Qwen2.5-1.5B
```

Launch GRPO:

```bash
bash scripts/run_grpo_8gpu.sh
```

Equivalent explicit command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --num_gpus 8 -m llm_project.cli.train_grpo \
  --config configs/grpo.yaml \
  --ds_config configs/deepspeed_zero2.json
```

Main outputs:

```text
outputs/grpo_qwen25_1p5b_gsm8k/
├── hf/                            # Hugging Face checkpoint for evaluation
├── logs/grpo_metrics.jsonl        # reward, rollout accuracy, KL, optional test accuracy
├── eval/gsm8k_step*.json          # test-set accuracy snapshots
├── resolved_grpo_config.yaml
└── deepspeed_config.json
```

Evaluate the final GRPO checkpoint:

```bash
bash scripts/eval_grpo.sh outputs/grpo_qwen25_1p5b_gsm8k/hf
```

Plot reward and test accuracy:

```bash
python -m llm_project.cli.plot_curves \
  --grpo outputs/grpo_qwen25_1p5b_gsm8k/logs/grpo_metrics.jsonl \
  --out_dir outputs/figures
```

Report figure:

```text
outputs/figures/grpo_reward_accuracy_curve.png
```

## 6. Submit-ready result files to collect

For Part 1 SFT:

```text
outputs/figures/sft_train_validation_curve.png
outputs/eval/base_gsm8k.json
outputs/eval/base_mmlu.json
outputs/eval/sft_gsm8k.json
outputs/eval/sft_mmlu.json
```

For Part 2 GRPO:

```text
outputs/figures/grpo_reward_accuracy_curve.png
outputs/eval/grpo_gsm8k.json
outputs/eval/grpo_mmlu.json
```

Use `report_templates/results_tables.md` to copy results into the final report.

## 7. Implementation notes

SFT uses standard next-token prediction with labels masked over the prompt tokens. The data collator pads `input_ids`, `attention_mask`, and `labels`, using `-100` for label padding.

GRPO uses grouped rollouts per prompt. Rewards are normalized within each group to produce advantages. The loss uses a clipped policy-gradient term plus a reference-model KL penalty. GSM8K rewards are based on exact normalized numeric answer matching, with an optional small formatting reward.

MMLU evaluation is implemented as answer-choice log-likelihood scoring over `A/B/C/D`, not free-form generation. GSM8K evaluation uses generation and numeric answer extraction.

## 8. Practical memory settings for 8 × RTX 4090

The default config is conservative:

- `train_micro_batch_size_per_gpu = 1`
- `gradient_accumulation_steps = 8`
- bf16 enabled
- ZeRO-2
- gradient checkpointing enabled

For faster training, increase `train_micro_batch_size_per_gpu` and the matching `per_device_*_batch_size` in the YAML files after confirming memory headroom with `nvidia-smi`.
