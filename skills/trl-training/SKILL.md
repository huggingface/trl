---
name: trl-training
description: Post-train LLMs with TRL (Transformers Reinforcement Learning) — SFT, DPO, GRPO, KTO, and reward-model training. Use when writing or debugging training code with the TRL Python API or the trl CLI.
license: Apache-2.0
metadata:
  author: huggingface
  documentation: https://huggingface.co/docs/trl
---

# TRL

Each method pairs a `*Trainer` class with a `*Config` dataclass. Configs extend `transformers.TrainingArguments`, so all of its arguments work in any trainer config.

| Trainer | Dataset type |
|---|---|
| `SFTTrainer` | language modeling or prompt-completion |
| `DPOTrainer` | preference (chosen/rejected pairs) |
| `GRPOTrainer` | prompt-only + reward function(s) |
| `DistillationTrainer` | prompt-only + a teacher model (on-policy distillation) |
| `KTOTrainer` | unpaired preference (per-sample bool label) |
| `RewardTrainer` | preference (chosen/rejected pairs); trains a scalar reward model, not a policy |

Many more trainers (PPO, OnlineDPO, ORPO, CPO, GKD, …) live in `trl.experimental` with unstable APIs: https://huggingface.co/docs/trl/experimental_overview

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",  # model ID or a PreTrainedModel instance
    args=SFTConfig(output_dir="Qwen2.5-0.5B-SFT"),
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```

Pass `model` as a string and route loading kwargs through `model_init_kwargs` (e.g. `{"dtype": "bfloat16", "attn_implementation": "kernels-community/flash-attn2"}`) instead of calling `from_pretrained` yourself. The tokenizer/processor is inferred from the model; pass `processing_class` only when it differs. For LoRA, pass `peft_config=LoraConfig(...)`.

## Dataset formats

Conversational: `{"messages": [{"role": ..., "content": ...}]}` (language modeling) or `{"prompt": [...], "completion": [...]}`. The chat template is applied automatically — never apply it yourself. Extra columns are allowed; GRPO forwards them to reward functions. Reference: https://huggingface.co/docs/trl/dataset_formats

## SFT: the fields that matter

```python
SFTConfig(
    max_length=1024,        # truncation length; None disables truncation
    packing=True,           # pack sequences into max_length blocks: fewer pad tokens, higher throughput
    padding_free=True,      # flatten batch, no padding; requires FlashAttention; implied by packing
    use_liger_kernel=True,  # fused Liger kernels, reduces peak memory
    assistant_only_loss=True,  # loss only on assistant turns (conversational datasets)
)
```

## GRPO: online RL

```python
def reward_len(completions, **kwargs):
    return [-abs(20 - len(c[0]["content"])) for c in completions]

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=reward_len,  # or a list; rewards are summed
    args=GRPOConfig(output_dir="Qwen2.5-0.5B-GRPO", max_completion_length=512),
    train_dataset=load_dataset("trl-lib/DeepMath-103K", split="train"),
)
```

Reward functions are called with keyword arguments `prompts`, `completions`, `completion_ids`, `trainer_state`, plus every extra dataset column — accept `**kwargs` for the ones you ignore. Return `list[float]`, one reward per completion. With conversational data, `completions` is a list of message lists, not strings.

The generation batch is `per_device_train_batch_size × num_processes × steps_per_generation` (or set `generation_batch_size` directly) and must be divisible by `num_generations` (default 8). Generation is the usual bottleneck — enable vLLM with `use_vllm=True`: `vllm_mode="colocate"` shares the training GPUs (size with `vllm_gpu_memory_utilization`); `vllm_mode="server"` uses a separate `trl vllm-serve --model <model_id>`.

`AsyncGRPOTrainer` (`trl.experimental.async_grpo`) implements the same algorithm with generation decoupled from training: a background worker streams completions from a vLLM server while the training loop consumes them, so the two overlap instead of alternating.

## CLI

Flags mirror the config fields: `trl sft --model_name_or_path Qwen/Qwen2.5-0.5B --dataset_name trl-lib/Capybara`. YAML via `--config`; distributed presets via `--accelerate_config zero3` (Python scripts: `accelerate launch train.py`).
