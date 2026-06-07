# Experiment inventory

Generated on 2026-06-07 UTC.

Sources checked:

- Local eval outputs: `outputs/**/eval/*.json` (91 result files, grouped into 42 checkpoint rows below).
- Local train configs: `outputs/*/resolved_*_config.yaml`, with generated config fallbacks under `configs/generated/` when an output directory only has eval JSONs.
- Online W&B project: `zcm25-cvhw/llm-course-project`, queried with the W&B API on 2026-06-07 UTC.
- Training/eval logging code: `src/llm_project/cli/train_sft.py`, `src/llm_project/cli/train_grpo.py`, `src/llm_project/cli/eval_gsm8k.py`, and `src/llm_project/cli/eval_mmlu.py`.

## W&B project snapshot

The online project currently contains 105 runs with 105 unique run names: 14 train-run records and 91 eval-run records. After renaming the 256-token vLLM GRPO runs, all current online W&B run display names are unique.

| Online family | Unique eval run names |
|---|---:|
| `eval_base` | 3 |
| `eval_sft_simple` | 9 |
| `eval_sft_step_by_step` | 9 |
| `eval_grpo_hf_3epoch_grpo` | 6 |
| `eval_grpo_hf_3epoch_dr_grpo` | 4 |
| `eval_grpo_vllm_server_pynccl` | 6 |
| `eval_grpo_vllm_server_pynccl_256` | 6 |
| `eval_grpo_vllm_server_pynccl_dr_grpo` | 6 |
| `eval_grpo_vllm_server_tmp` | 6 |
| `eval_grpo_vllm_server_tmp_256` | 6 |
| `eval_grpo_vllm_server_tmp_dr_grpo` | 6 |
| `eval_grpo_vllm_colocate` | 6 |
| `eval_grpo_vllm_colocate_no_is` | 6 |
| `eval_grpo_vllm_colocate_256` | 6 |
| `eval_grpo_vllm_colocate_dr_grpo` | 6 |

Online train runs:

| Run name | Latest state | Latest created UTC | Latest id | Count |
|---|---|---|---|---:|
| `sft_qwen25_1p5b_numina_gsm8k_simple_3epoch` | finished | 2026-06-04T02:22:34Z | `fn4tbt56` | 1 |
| `sft_qwen25_1p5b_numina_gsm8k_step_by_step_3epoch` | finished | 2026-06-04T02:22:47Z | `80n03ogg` | 1 |
| `grpo_hf_3epoch_grpo` | finished | 2026-06-04T16:44:40Z | `gxjdrypi` | 1 |
| `grpo_hf_3epoch_dr_grpo` | crashed | 2026-06-05T04:38:03Z | `2iwypxqk` | 1 |
| `grpo_vllm_server_pynccl_3epoch_grpo_256` | finished | 2026-06-05T15:07:56Z | `0tc9jdzt` | 1 |
| `grpo_vllm_server_tmp_3epoch_grpo_256` | finished | 2026-06-05T17:56:14Z | `gaudn9i0` | 1 |
| `grpo_vllm_colocate_3epoch_grpo_256` | finished | 2026-06-05T21:21:49Z | `o2na6bht` | 1 |
| `grpo_vllm_server_pynccl_3epoch_dr_grpo` | finished | 2026-06-06T01:17:03Z | `o1elbj7u` | 1 |
| `grpo_vllm_server_tmp_3epoch_dr_grpo` | finished | 2026-06-06T05:41:03Z | `qso5u50v` | 1 |
| `grpo_vllm_colocate_3epoch_dr_grpo` | finished | 2026-06-06T10:44:45Z | `i04ll6u8` | 1 |
| `grpo_vllm_server_pynccl_3epoch_grpo` | finished | 2026-06-06T14:37:30Z | `fupdbes4` | 1 |
| `grpo_vllm_server_tmp_3epoch_grpo` | finished | 2026-06-06T18:58:44Z | `zcpzp8lp` | 1 |
| `grpo_vllm_colocate_3epoch_grpo` | finished | 2026-06-07T00:54:21Z | `hdt2gk51` | 1 |
| `grpo_vllm_colocate_3epoch_grpo_no_is` | finished | 2026-06-07T10:28:44Z | `kxisrn9f` | 1 |

Notes:

- W&B rename on 2026-06-07 UTC relabeled the older 256-token vLLM GRPO train/eval records with `_256` suffixes.
- W&B cleanup on 2026-06-07 UTC deleted two crashed memory-error `vllm_colocate` train records: `grpo_vllm_colocate_3epoch_grpo` (`628y992i`) and `grpo_vllm_colocate_3epoch_dr_grpo` (`x2xo8kbe`). Post-delete verification found zero crashed `vllm_colocate` runs online.
- W&B rename on 2026-06-07 UTC relabeled the finished colocate retry train records to their canonical names: `grpo_vllm_colocate_3epoch_grpo` (`hdt2gk51`) and `grpo_vllm_colocate_3epoch_dr_grpo` (`i04ll6u8`). Post-rename verification found zero duplicated W&B display names.
- `grpo_vllm_colocate_3epoch_grpo` eval JSONs are now under `outputs/grpo_vllm_colocate_3epoch_grpo/hf_epoch_00{1,2,3}/eval/`; the corresponding W&B eval run names are `eval_grpo_vllm_colocate_epoch00{1,2,3}_gsm8k_simple` and `eval_grpo_vllm_colocate_epoch00{1,2,3}_mmlu`.
- `grpo_vllm_colocate_3epoch_grpo_no_is` disables `rollout.vllm.importance_sampling`, so colocated vLLM rollouts do not request sampled-token logprobs and the GRPO rollout-loss importance weights stay at 1.0.
- W&B SDK re-verification on 2026-06-07 UTC found the finished no-IS train run `grpo_vllm_colocate_3epoch_grpo_no_is` (`kxisrn9f`) and all six finished eval runs: `4eo8v9sc`, `46px6002`, `zv08udvb`, `a46liyyp`, `gx2z0ob9`, and `njgj3g3o`.
- The local eval output directory for colocate Dr. GRPO remains `outputs/grpo_vllm_colocate_3epoch_dr_grpo`.

## Logged metrics

SFT train runs log:

- `epoch`
- `sft/train/loss`
- `sft/train/lr`
- `sft/eval/val_loss`
- `sft/eval/val_ppl`

GRPO train runs log:

- `epoch`
- `grpo/train/loss`
- `grpo/train/policy_loss`
- `grpo/train/kl`
- `grpo/train/mean_ratio`
- `grpo/train/reward`
- `grpo/train/rollout_accuracy`
- `grpo/train/avg_completion_tokens`
- `grpo/train/lr`
- `grpo/rollout/backend`
- `grpo/rollout/sync_method`
- `grpo/speed/rollout_time_sec`
- `grpo/speed/step_time_sec`
- `grpo/speed/sync_weights_time_sec`
- `grpo/speed/rollout_tokens_per_sec`

Eval runs log:

- GSM8K: `eval/{base|sft|grpo}/gsm8k_accuracy`, `eval/{stage}/gsm8k_correct`, `eval/{stage}/gsm8k_total`.
- MMLU: `eval/{base|sft|grpo}/mmlu_accuracy`, `eval/{stage}/mmlu_correct`, `eval/{stage}/mmlu_total`, and `eval/{stage}/mmlu/{subject}_accuracy`.

## Train configs represented in outputs

All train configs use `Qwen/Qwen2.5-1.5B` as the base policy. GRPO configs also use `Qwen/Qwen2.5-1.5B` as the reference model and `simple` GSM8K prompts. `Backend` is the GRPO rollout backend; all evals below use the vLLM eval path.

| Stage | Run/output family | Train prompt | Epochs | Backend | Loss | beta_kl | group | max prompt/new | ZeRO | Local source |
|---|---|---|---:|---|---|---:|---:|---|---:|---|
| sft | `sft_qwen25_1p5b_numina_gsm8k_simple_3epoch` | `simple` | 3 | - | - | - | - | -/- | 2 | `outputs/sft_simple_3epoch/resolved_sft_config.yaml` |
| sft | `sft_qwen25_1p5b_numina_gsm8k_step_by_step_3epoch` | `step_by_step` | 3 | - | - | - | - | -/- | 2 | `outputs/sft_step_by_step_3epoch/resolved_sft_config.yaml` |
| grpo | `grpo_hf_3epoch_grpo` | `simple` | 3 | `hf` | `grpo` | 0.0 | 4 | 768/512 | 2 | `outputs/grpo_hf_3epoch_grpo/resolved_grpo_config.yaml` |
| grpo | `grpo_hf_3epoch_dr_grpo` | `simple` | 3 | `hf` | `dr_grpo` | 0.0 | 4 | 768/512 | 2 | `outputs/grpo_hf_3epoch_dr_grpo/resolved_grpo_config.yaml` |
| grpo | `grpo_vllm_server_pynccl_3epoch_grpo` | `simple` | 3 | `vllm_server` | `grpo` | 0.02 | 4 | 512/512 | - | `configs/generated/grpo/vllm/grpo_vllm_server_pynccl_3epoch_grpo.yaml` |
| grpo | `grpo_vllm_server_pynccl_3epoch_grpo_256` | `simple` | 3 | `vllm_server` | `grpo` | 0.02 | 4 | 512/256 | 2 | `outputs/grpo_vllm_server_pynccl_3epoch_grpo_256/resolved_grpo_config.yaml` |
| grpo | `grpo_vllm_server_pynccl_3epoch_dr_grpo` | `simple` | 3 | `vllm_server` | `dr_grpo` | 0.02 | 4 | 512/512 | 2 | `outputs/grpo_vllm_server_pynccl_3epoch_dr_grpo/resolved_grpo_config.yaml` |
| grpo | `grpo_vllm_server_tmp_3epoch_grpo` | `simple` | 3 | `vllm_server` | `grpo` | 0.02 | 4 | 512/512 | - | `configs/generated/grpo/vllm/grpo_vllm_server_tmp_3epoch_grpo.yaml` |
| grpo | `grpo_vllm_server_tmp_3epoch_grpo_256` | `simple` | 3 | `vllm_server` | `grpo` | 0.02 | 4 | 512/256 | 2 | `outputs/grpo_vllm_server_tmp_3epoch_grpo_256/resolved_grpo_config.yaml` |
| grpo | `grpo_vllm_server_tmp_3epoch_dr_grpo` | `simple` | 3 | `vllm_server` | `dr_grpo` | 0.02 | 4 | 512/512 | 2 | `outputs/grpo_vllm_server_tmp_3epoch_dr_grpo/resolved_grpo_config.yaml` |
| grpo | `grpo_vllm_colocate_3epoch_grpo` | `simple` | 3 | `vllm_colocate` | `grpo` | 0.02 | 4 | 512/512 | - | `configs/generated/grpo/vllm/grpo_vllm_colocate_3epoch_grpo.yaml` |
| grpo | `grpo_vllm_colocate_3epoch_grpo_no_is` | `simple` | 3 | `vllm_colocate` | `grpo` | 0.02 | 4 | 512/512 | 2 | `outputs/grpo_vllm_colocate_3epoch_grpo_no_is/resolved_grpo_config.yaml` |
| grpo | `grpo_vllm_colocate_3epoch_grpo_256` | `simple` | 3 | `vllm_colocate` | `grpo` | 0.02 | 4 | 512/256 | 2 | `outputs/grpo_vllm_colocate_3epoch_grpo_256/resolved_grpo_config.yaml` |
| grpo | `grpo_vllm_colocate_3epoch_dr_grpo` | `simple` | 3 | `vllm_colocate` | `dr_grpo` | 0.02 | 4 | 512/512 | 2 | `outputs/grpo_vllm_colocate_3epoch_dr_grpo/resolved_grpo_config.yaml` |

## Base and SFT evals

GSM8K was tested with both `simple` and `step_by_step` eval prompts. MMLU uses the simple eval prompt in the generated configs. Scores are local JSON accuracies.

| Stage | Train run | Eval checkpoint/model | Eval run names | GSM8K simple | GSM8K step_by_step | MMLU |
|---|---|---|---|---:|---:|---:|
| base | `Qwen/Qwen2.5-1.5B` | `outputs/base` | `eval_base_gsm8k_simple`, `eval_base_gsm8k_step_by_step`, `eval_base_mmlu` | 52.31% | 67.78% | 33.29% |
| sft | `sft_qwen25_1p5b_numina_gsm8k_simple_3epoch` | `outputs/sft_simple_3epoch/hf_epoch_001` | `eval_sft_simple_epoch001_gsm8k_simple`, `eval_sft_simple_epoch001_gsm8k_step_by_step`, `eval_sft_simple_epoch001_mmlu` | 62.93% | 63.31% | 18.18% |
| sft | `sft_qwen25_1p5b_numina_gsm8k_simple_3epoch` | `outputs/sft_simple_3epoch/hf_epoch_002` | `eval_sft_simple_epoch002_gsm8k_simple`, `eval_sft_simple_epoch002_gsm8k_step_by_step`, `eval_sft_simple_epoch002_mmlu` | 66.11% | 66.72% | 21.26% |
| sft | `sft_qwen25_1p5b_numina_gsm8k_simple_3epoch` | `outputs/sft_simple_3epoch/hf_epoch_003` | `eval_sft_simple_epoch003_gsm8k_simple`, `eval_sft_simple_epoch003_gsm8k_step_by_step`, `eval_sft_simple_epoch003_mmlu` | 66.72% | 67.02% | 24.73% |
| sft | `sft_qwen25_1p5b_numina_gsm8k_step_by_step_3epoch` | `outputs/sft_step_by_step_3epoch/hf_epoch_001` | `eval_sft_step_by_step_epoch001_gsm8k_simple`, `eval_sft_step_by_step_epoch001_gsm8k_step_by_step`, `eval_sft_step_by_step_epoch001_mmlu` | 62.70% | 63.46% | 25.00% |
| sft | `sft_qwen25_1p5b_numina_gsm8k_step_by_step_3epoch` | `outputs/sft_step_by_step_3epoch/hf_epoch_002` | `eval_sft_step_by_step_epoch002_gsm8k_simple`, `eval_sft_step_by_step_epoch002_gsm8k_step_by_step`, `eval_sft_step_by_step_epoch002_mmlu` | 67.93% | 67.70% | 36.76% |
| sft | `sft_qwen25_1p5b_numina_gsm8k_step_by_step_3epoch` | `outputs/sft_step_by_step_3epoch/hf_epoch_003` | `eval_sft_step_by_step_epoch003_gsm8k_simple`, `eval_sft_step_by_step_epoch003_gsm8k_step_by_step`, `eval_sft_step_by_step_epoch003_mmlu` | 66.94% | 66.34% | 34.76% |

## GRPO evals

GRPO evals were run on `hf_epoch_001`, `hf_epoch_002`, and/or `hf_epoch_003` checkpoints depending on which checkpoints were produced. GSM8K evals here use the `simple` prompt only; MMLU uses the simple eval prompt. Scores are local JSON accuracies.

| Train/output family | Epoch | Train backend | Loss | beta_kl | Eval run names | GSM8K simple | MMLU |
|---|---:|---|---|---:|---|---:|---:|
| `grpo_hf_3epoch_grpo` | 001 | `hf` | `grpo` | 0.0 | `eval_grpo_hf_3epoch_grpo_epoch001_gsm8k_simple`, `eval_grpo_hf_3epoch_grpo_epoch001_mmlu` | 75.59% | 51.60% |
| `grpo_hf_3epoch_grpo` | 002 | `hf` | `grpo` | 0.0 | `eval_grpo_hf_3epoch_grpo_epoch002_gsm8k_simple`, `eval_grpo_hf_3epoch_grpo_epoch002_mmlu` | 77.03% | 50.13% |
| `grpo_hf_3epoch_grpo` | 003 | `hf` | `grpo` | 0.0 | `eval_grpo_hf_3epoch_grpo_epoch003_gsm8k_simple`, `eval_grpo_hf_3epoch_grpo_epoch003_mmlu` | 76.19% | 49.33% |
| `grpo_hf_3epoch_dr_grpo` | 001 | `hf` | `dr_grpo` | 0.0 | `eval_grpo_hf_3epoch_dr_grpo_epoch001_gsm8k_simple`, `eval_grpo_hf_3epoch_dr_grpo_epoch001_mmlu` | 76.65% | 52.41% |
| `grpo_hf_3epoch_dr_grpo` | 002 | `hf` | `dr_grpo` | 0.0 | `eval_grpo_hf_3epoch_dr_grpo_epoch002_gsm8k_simple`, `eval_grpo_hf_3epoch_dr_grpo_epoch002_mmlu` | 17.36% | 15.37% |
| `grpo_vllm_server_pynccl_3epoch_grpo` | 001 | `vllm_server` | `grpo` | 0.02 | `eval_grpo_vllm_server_pynccl_epoch001_gsm8k_simple`, `eval_grpo_vllm_server_pynccl_epoch001_mmlu` | 75.13% | 52.14% |
| `grpo_vllm_server_pynccl_3epoch_grpo` | 002 | `vllm_server` | `grpo` | 0.02 | `eval_grpo_vllm_server_pynccl_epoch002_gsm8k_simple`, `eval_grpo_vllm_server_pynccl_epoch002_mmlu` | 75.59% | 54.01% |
| `grpo_vllm_server_pynccl_3epoch_grpo` | 003 | `vllm_server` | `grpo` | 0.02 | `eval_grpo_vllm_server_pynccl_epoch003_gsm8k_simple`, `eval_grpo_vllm_server_pynccl_epoch003_mmlu` | 76.50% | 53.21% |
| `grpo_vllm_server_tmp_3epoch_grpo` | 001 | `vllm_server` | `grpo` | 0.02 | `eval_grpo_vllm_server_tmp_epoch001_gsm8k_simple`, `eval_grpo_vllm_server_tmp_epoch001_mmlu` | 75.44% | 52.14% |
| `grpo_vllm_server_tmp_3epoch_grpo` | 002 | `vllm_server` | `grpo` | 0.02 | `eval_grpo_vllm_server_tmp_epoch002_gsm8k_simple`, `eval_grpo_vllm_server_tmp_epoch002_mmlu` | 76.19% | 52.41% |
| `grpo_vllm_server_tmp_3epoch_grpo` | 003 | `vllm_server` | `grpo` | 0.02 | `eval_grpo_vllm_server_tmp_epoch003_gsm8k_simple`, `eval_grpo_vllm_server_tmp_epoch003_mmlu` | 77.03% | 54.41% |
| `grpo_vllm_colocate_3epoch_grpo` | 001 | `vllm_colocate` | `grpo` | 0.02 | `eval_grpo_vllm_colocate_epoch001_gsm8k_simple`, `eval_grpo_vllm_colocate_epoch001_mmlu` | 75.36% | 52.41% |
| `grpo_vllm_colocate_3epoch_grpo` | 002 | `vllm_colocate` | `grpo` | 0.02 | `eval_grpo_vllm_colocate_epoch002_gsm8k_simple`, `eval_grpo_vllm_colocate_epoch002_mmlu` | 75.97% | 54.28% |
| `grpo_vllm_colocate_3epoch_grpo` | 003 | `vllm_colocate` | `grpo` | 0.02 | `eval_grpo_vllm_colocate_epoch003_gsm8k_simple`, `eval_grpo_vllm_colocate_epoch003_mmlu` | 75.59% | 54.68% |
| `grpo_vllm_colocate_3epoch_grpo_no_is` | 001 | `vllm_colocate` | `grpo` | 0.02 | `eval_grpo_vllm_colocate_no_is_epoch001_gsm8k_simple`, `eval_grpo_vllm_colocate_no_is_epoch001_mmlu` | 75.59% | 53.21% |
| `grpo_vllm_colocate_3epoch_grpo_no_is` | 002 | `vllm_colocate` | `grpo` | 0.02 | `eval_grpo_vllm_colocate_no_is_epoch002_gsm8k_simple`, `eval_grpo_vllm_colocate_no_is_epoch002_mmlu` | 76.95% | 50.40% |
| `grpo_vllm_colocate_3epoch_grpo_no_is` | 003 | `vllm_colocate` | `grpo` | 0.02 | `eval_grpo_vllm_colocate_no_is_epoch003_gsm8k_simple`, `eval_grpo_vllm_colocate_no_is_epoch003_mmlu` | 73.92% | 47.59% |
| `grpo_vllm_server_pynccl_3epoch_dr_grpo` | 001 | `vllm_server` | `dr_grpo` | 0.02 | `eval_grpo_vllm_server_pynccl_dr_grpo_epoch001_gsm8k_simple`, `eval_grpo_vllm_server_pynccl_dr_grpo_epoch001_mmlu` | 75.66% | 52.14% |
| `grpo_vllm_server_pynccl_3epoch_dr_grpo` | 002 | `vllm_server` | `dr_grpo` | 0.02 | `eval_grpo_vllm_server_pynccl_dr_grpo_epoch002_gsm8k_simple`, `eval_grpo_vllm_server_pynccl_dr_grpo_epoch002_mmlu` | 75.97% | 52.81% |
| `grpo_vllm_server_pynccl_3epoch_dr_grpo` | 003 | `vllm_server` | `dr_grpo` | 0.02 | `eval_grpo_vllm_server_pynccl_dr_grpo_epoch003_gsm8k_simple`, `eval_grpo_vllm_server_pynccl_dr_grpo_epoch003_mmlu` | 75.97% | 54.81% |
| `grpo_vllm_server_tmp_3epoch_dr_grpo` | 001 | `vllm_server` | `dr_grpo` | 0.02 | `eval_grpo_vllm_server_tmp_dr_grpo_epoch001_gsm8k_simple`, `eval_grpo_vllm_server_tmp_dr_grpo_epoch001_mmlu` | 72.78% | 50.80% |
| `grpo_vllm_server_tmp_3epoch_dr_grpo` | 002 | `vllm_server` | `dr_grpo` | 0.02 | `eval_grpo_vllm_server_tmp_dr_grpo_epoch002_gsm8k_simple`, `eval_grpo_vllm_server_tmp_dr_grpo_epoch002_mmlu` | 75.44% | 52.94% |
| `grpo_vllm_server_tmp_3epoch_dr_grpo` | 003 | `vllm_server` | `dr_grpo` | 0.02 | `eval_grpo_vllm_server_tmp_dr_grpo_epoch003_gsm8k_simple`, `eval_grpo_vllm_server_tmp_dr_grpo_epoch003_mmlu` | 75.89% | 53.61% |
| `grpo_vllm_colocate_3epoch_dr_grpo` | 001 | `vllm_colocate` | `dr_grpo` | 0.02 | `eval_grpo_vllm_colocate_dr_grpo_epoch001_gsm8k_simple`, `eval_grpo_vllm_colocate_dr_grpo_epoch001_mmlu` | 73.09% | 51.87% |
| `grpo_vllm_colocate_3epoch_dr_grpo` | 002 | `vllm_colocate` | `dr_grpo` | 0.02 | `eval_grpo_vllm_colocate_dr_grpo_epoch002_gsm8k_simple`, `eval_grpo_vllm_colocate_dr_grpo_epoch002_mmlu` | 75.44% | 53.48% |
| `grpo_vllm_colocate_3epoch_dr_grpo` | 003 | `vllm_colocate` | `dr_grpo` | 0.02 | `eval_grpo_vllm_colocate_dr_grpo_epoch003_gsm8k_simple`, `eval_grpo_vllm_colocate_dr_grpo_epoch003_mmlu` | 76.19% | 52.54% |

`grpo_vllm_colocate_3epoch_grpo_no_is` uses the colocated vLLM backend with importance sampling disabled (`rollout.vllm.importance_sampling: false`), GRPO loss, `beta_kl=0.02`, group size 4, and 512/512 prompt/new-token limits. Local evals are complete for all three checkpoints: GSM8K simple is 75.59% (997/1319), 76.95% (1015/1319), and 73.92% (975/1319), while MMLU is 53.21% (398/748), 50.40% (377/748), and 47.59% (356/748). The best checkpoint is epoch 002 for GSM8K and epoch 001 for MMLU; compared with the importance-sampling colocate GRPO run, no-IS slightly improves epoch-002 GSM8K but loses MMLU after epoch 001 and degrades by epoch 003.

Renamed `_256` GRPO eval runs:

| Train/output family | Epoch | Train backend | Loss | beta_kl | rollout max new tokens | Eval run names | GSM8K simple | MMLU |
|---|---:|---|---|---:|---:|---|---:|---:|
| `grpo_vllm_server_pynccl_3epoch_grpo_256` | 001 | `vllm_server` | `grpo` | 0.02 | 256 | `eval_grpo_vllm_server_pynccl_epoch001_gsm8k_simple_256`, `eval_grpo_vllm_server_pynccl_epoch001_mmlu_256` | 71.19% | 44.52% |
| `grpo_vllm_server_pynccl_3epoch_grpo_256` | 002 | `vllm_server` | `grpo` | 0.02 | 256 | `eval_grpo_vllm_server_pynccl_epoch002_gsm8k_simple_256`, `eval_grpo_vllm_server_pynccl_epoch002_mmlu_256` | 74.07% | 43.58% |
| `grpo_vllm_server_pynccl_3epoch_grpo_256` | 003 | `vllm_server` | `grpo` | 0.02 | 256 | `eval_grpo_vllm_server_pynccl_epoch003_gsm8k_simple_256`, `eval_grpo_vllm_server_pynccl_epoch003_mmlu_256` | 74.30% | 42.25% |
| `grpo_vllm_server_tmp_3epoch_grpo_256` | 001 | `vllm_server` | `grpo` | 0.02 | 256 | `eval_grpo_vllm_server_tmp_epoch001_gsm8k_simple_256`, `eval_grpo_vllm_server_tmp_epoch001_mmlu_256` | 71.42% | 42.25% |
| `grpo_vllm_server_tmp_3epoch_grpo_256` | 002 | `vllm_server` | `grpo` | 0.02 | 256 | `eval_grpo_vllm_server_tmp_epoch002_gsm8k_simple_256`, `eval_grpo_vllm_server_tmp_epoch002_mmlu_256` | 73.24% | 41.98% |
| `grpo_vllm_server_tmp_3epoch_grpo_256` | 003 | `vllm_server` | `grpo` | 0.02 | 256 | `eval_grpo_vllm_server_tmp_epoch003_gsm8k_simple_256`, `eval_grpo_vllm_server_tmp_epoch003_mmlu_256` | 72.93% | 41.31% |
| `grpo_vllm_colocate_3epoch_grpo_256` | 001 | `vllm_colocate` | `grpo` | 0.02 | 256 | `eval_grpo_vllm_colocate_epoch001_gsm8k_simple_256`, `eval_grpo_vllm_colocate_epoch001_mmlu_256` | 70.81% | 44.52% |
| `grpo_vllm_colocate_3epoch_grpo_256` | 002 | `vllm_colocate` | `grpo` | 0.02 | 256 | `eval_grpo_vllm_colocate_epoch002_gsm8k_simple_256`, `eval_grpo_vllm_colocate_epoch002_mmlu_256` | 73.92% | 41.31% |
| `grpo_vllm_colocate_3epoch_grpo_256` | 003 | `vllm_colocate` | `grpo` | 0.02 | 256 | `eval_grpo_vllm_colocate_epoch003_gsm8k_simple_256`, `eval_grpo_vllm_colocate_epoch003_mmlu_256` | 74.37% | 39.71% |

