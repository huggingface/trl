---
name: trl
description: Post-train LLMs with TRL (Transformers Reinforcement Learning) — SFT, DPO, GRPO, RLOO, KTO, and reward-model training. Use when writing or debugging training code with the TRL Python API or the trl CLI.
license: Apache-2.0
metadata:
  version: "2.0.0"
  author: huggingface
  documentation: https://huggingface.co/docs/trl
---

# TRL

TRL post-trains transformer models, built on transformers, accelerate, peft, and datasets. The primary interface is the Python API: each method pairs a `*Trainer` class with a `*Config` dataclass. Configs extend `transformers.TrainingArguments`, so all of its arguments (`learning_rate`, `per_device_train_batch_size`, `gradient_checkpointing`, `bf16`, `report_to`, …) work in any trainer config.

## Picking a trainer

| Trainer | Dataset type | Use for |
|---|---|---|
| `SFTTrainer` | language modeling or prompt-completion | instruction tuning, continued pretraining |
| `DPOTrainer` | preference (chosen/rejected pairs) | offline preference alignment |
| `GRPOTrainer` | prompt-only + reward function(s) | online RL (reasoning, RLVR) |
| `RLOOTrainer` | prompt-only + reward function(s) | online RL, REINFORCE-style |
| `KTOTrainer` | unpaired preference (per-sample bool label) | alignment without preference pairs |
| `RewardTrainer` | preference (chosen/rejected pairs) | training a reward model |

Typical pipeline: SFT first, then preference or RL alignment on the SFT checkpoint.

## Minimal training run

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

Prefer passing `model` as a string and routing loading kwargs through `model_init_kwargs` instead of calling `from_pretrained` yourself:

```python
args = SFTConfig(
    model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "kernels-community/flash-attn2"},
)
```

The tokenizer/processor (`processing_class`) is inferred from the model; pass it explicitly only when it differs from the model's default.

## Dataset formats

Two formats, auto-detected. Conversational datasets get the chat template applied automatically — never apply it yourself before passing the dataset.

```python
# Standard
{"text": "The sky is blue."}                          # language modeling
{"prompt": "The sky is", "completion": " blue."}      # prompt-completion
{"prompt": "The sky is"}                              # prompt-only
{"chosen": "...", "rejected": "..."}                  # preference (implicit prompt)

# Conversational (same types, messages instead of strings)
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"prompt": [{"role": "user", "content": "..."}]}
```

Extra columns are allowed; GRPO/RLOO forward them to reward functions. Full reference: https://huggingface.co/docs/trl/dataset_formats

## SFT: the fields that matter

```python
SFTConfig(
    max_length=1024,        # truncation length; None disables truncation
    packing=True,           # pack sequences into max_length blocks: fewer pad tokens, higher throughput
    padding_free=True,      # flatten batch, no padding; requires FlashAttention; implied by packing
    use_liger_kernel=True,  # fused Liger kernels, reduces peak memory
    model_init_kwargs={"attn_implementation": "kernels-community/flash-attn2"},
)
```

With prompt-completion datasets, loss is computed only on completion tokens by default (`completion_only_loss=None` resolves per dataset type); language-modeling datasets train on the full sequence.

## GRPO / RLOO: online RL

```python
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

def reward_len(completions, **kwargs):
    return [-abs(20 - len(c)) for c in completions]

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=reward_len,  # or a list; rewards are summed
    args=GRPOConfig(output_dir="Qwen2.5-0.5B-GRPO", max_completion_length=512),
    train_dataset=load_dataset("trl-lib/tldr", split="train"),
)
trainer.train()
```

**Reward function contract**: called with keyword arguments `prompts`, `completions`, `completion_ids`, `trainer_state`, plus every extra dataset column (e.g. `ground_truth`) — accept `**kwargs` for the ones you ignore. Returns `list[float]` (one reward per completion; `None` entries opt a function out for those samples). With conversational data, `completions` is a list of message lists (`completions[0][0]["content"]`), not strings. Functions may be `async def`. Built-ins live in `trl.rewards` (`accuracy_reward`, `think_format_reward`, …). `reward_funcs` also accepts a reward-model ID string.

**Generation batching**: each optimization step generates `num_generations` (default 8) completions per prompt. The generation batch is `per_device_train_batch_size × num_processes × steps_per_generation`, or set `generation_batch_size` directly (mutually exclusive); it must be divisible by `num_generations`.

**vLLM for fast generation** (the default transformers generation is the usual bottleneck):

```python
GRPOConfig(use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.3)
```

- `"colocate"`: vLLM shares the training GPUs; raise/lower `vllm_gpu_memory_utilization` to trade generation throughput against training memory headroom.
- `"server"`: run `trl vllm-serve --model <model_id>` on separate GPUs and train with `vllm_mode="server"`.

## LoRA / QLoRA

Pass a peft config to any trainer:

```python
from peft import LoraConfig

trainer = SFTTrainer(model=..., args=..., train_dataset=..., peft_config=LoraConfig(r=32, lora_alpha=16))
```

For QLoRA, also pass `quantization_config=BitsAndBytesConfig(load_in_4bit=True)` to the trainer — SFT, DPO, GRPO, and RLOO all accept it alongside `peft_config`.

## Multi-GPU

Launch the same script with accelerate; the trainer handles the rest:

```bash
accelerate launch train.py                    # DDP
accelerate launch --config_file zero3.yaml train.py  # DeepSpeed/FSDP via accelerate config
```

## CLI

Every trainer is also exposed as a command (`trl sft`, `trl dpo`, `trl grpo`, `trl rloo`, `trl kto`, `trl reward`) whose flags mirror the config fields, plus `--model_name_or_path` and `--dataset_name`. Arguments can be stored in YAML and passed with `--config config.yaml`; `--accelerate_config zero3` (or `single_gpu`, `multi_gpu`, `fsdp1`, `fsdp2`, `zero1`, `zero2`, `zero3`) selects a predefined distributed setup.

```bash
trl sft --model_name_or_path Qwen/Qwen2.5-0.5B --dataset_name trl-lib/Capybara --output_dir Qwen2.5-0.5B-SFT
```

## Docs

- Trainer guides and API: https://huggingface.co/docs/trl
- Paper-to-implementation index: https://huggingface.co/docs/trl/paper_index
- Memory/throughput tuning: https://huggingface.co/docs/trl/reducing_memory_usage
