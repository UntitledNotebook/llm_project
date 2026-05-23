# References and acknowledgements

## Course requirement source

- `2026_llm_course_project.pdf`, Choice A: Post-training Implementation. This framework covers Part 1 SFT and Part 2 RL/GRPO only. Part 3 open exploration is intentionally not included.

## Code bases and documentation referenced

- Hugging Face Transformers documentation and examples for `AutoModelForCausalLM`, tokenizer loading, generation, and causal language-model loss.
- Hugging Face Datasets documentation for loading NuminaMath-CoT, GSM8K, and MMLU.
- Microsoft DeepSpeed documentation and examples for multi-GPU launch, ZeRO-2/ZeRO-3 configuration, and checkpointing.
- Dao-AILab FlashAttention release naming convention for choosing a pre-built wheel matching CUDA/PyTorch/Python/CXX11 ABI.
- PyTorch official previous-version installation table for selecting a PyTorch 2.7 wheel.
- vLLM package is included because the course environment lists `vllm==0.9.0`; rollout acceleration with vLLM is not implemented here because it is part of the optional open-exploration direction.

## Research papers

- GRPO / DeepSeekMath: Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, arXiv:2402.03300. Used for the group-relative reward normalization and clipped policy-gradient objective design.
- FlashAttention: Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, arXiv:2205.14135, and FlashAttention-2 follow-up materials.
- MMLU: Hendrycks et al., *Measuring Massive Multitask Language Understanding*, ICLR 2021.
- GSM8K: Cobbe et al., *Training Verifiers to Solve Math Word Problems*, arXiv:2110.14168.

## Models and datasets

- Base model: `Qwen/Qwen2.5-1.5B-Base`.
- SFT dataset: `AI-MO/NuminaMath-CoT`, filtered to `source == "gsm8k"`.
- RL dataset: `openai/gsm8k`, train split for GRPO rollouts and test split for accuracy curves.
- Evaluation datasets: `openai/gsm8k` test split and `cais/mmlu` test split.

## AI tool acknowledgement

- OpenAI ChatGPT / GPT-5.5 Pro was used to generate this framework. Review, test, and modify the implementation before submission, and acknowledge AI assistance in the final report as required by the course policy.
