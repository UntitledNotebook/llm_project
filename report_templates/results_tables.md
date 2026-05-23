# Report result table templates

## Part 1: SFT

| Model | GSM8K accuracy | MMLU accuracy | Notes |
|---|---:|---:|---|
| Qwen2.5-1.5B-Base before SFT | TODO | TODO | `scripts/eval_base.sh` |
| After NuminaMath-CoT source=gsm8k SFT | TODO | TODO | `scripts/eval_sft.sh` |

Figure: `outputs/figures/sft_train_validation_curve.png`

## Part 2: GRPO

| Model | GSM8K accuracy | MMLU accuracy | Notes |
|---|---:|---:|---|
| Qwen2.5-1.5B-Base before GRPO | TODO | TODO | Use the selected initial policy, often the base or SFT checkpoint |
| After GRPO on GSM8K | TODO | TODO | `scripts/eval_grpo.sh` |

Figure: `outputs/figures/grpo_reward_accuracy_curve.png`
