# Config Layout

- `templates/`: canonical hand-edited YAML templates.
- `generated/sft/`: SFT run configs generated from `templates/sft.yaml`.
- `generated/grpo/hf/`: HF-rollout GRPO configs generated from `templates/grpo.yaml`.
- `generated/grpo/vllm/`: vLLM-rollout GRPO configs generated from `templates/grpo.yaml`.
- `generated/eval/`: evaluation configs generated from `templates/eval.yaml`, grouped by stage and SFT prompt style.
