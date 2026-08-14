---
name: trl
description: Configure and run post-training with the TRL Python API, including SFT, preference optimization, online RL, custom rewards, dataset formats, and throughput-sensitive settings.
license: Apache-2.0
---

# TRL

Use TRL for supervised fine-tuning, preference optimization, reward modeling, and online reinforcement learning. Prefer the Python API when a workflow needs custom datasets, reward functions, callbacks, or model-loading behavior; the CLI is a thin entry point for standard recipes.

## Start by checking the installed API

TRL evolves quickly. Inspect the installed version and the signatures of the trainer and config you plan to use before copying an older recipe.

```python
import inspect
import trl
from trl import GRPOConfig, GRPOTrainer

print(trl.__version__)
print(inspect.signature(GRPOConfig))
print(inspect.signature(GRPOTrainer))
```

Use the documentation that matches the installed version: <https://huggingface.co/docs/trl>.

## Choose the trainer from the data and objective

| Objective | Trainer | Minimum dataset columns |
| --- | --- | --- |
| Language modeling or instruction tuning | `SFTTrainer` | `text`, `messages`, or `prompt` + `completion` |
| Pairwise preference optimization | `DPOTrainer` | `prompt`, `chosen`, `rejected` |
| Desirable/undesirable labels | `KTOTrainer` | `prompt`, `completion`, `label` |
| Pairwise reward model | `RewardTrainer` | `chosen`, `rejected` (optionally `prompt`) |
| Group-relative online RL | `GRPOTrainer` | `prompt`, plus columns used by reward functions |
| Leave-one-out online RL | `RLOOTrainer` | `prompt`, plus columns used by reward functions |

TRL accepts standard strings and conversational message lists. Do not flatten conversational data manually unless the selected trainer requires preformatted text; trainers apply the processing class's chat template when appropriate.

```python
standard_sft = {"text": "The sky is blue."}
conversational_sft = {
    "messages": [
        {"role": "user", "content": "What color is the sky?"},
        {"role": "assistant", "content": "Blue."},
    ]
}
preference = {
    "prompt": "The sky is",
    "chosen": " blue.",
    "rejected": " green.",
}
```

## SFT with the Python API

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

args = SFTConfig(
    output_dir="outputs/sft",
    max_length=2048,
    packing=True,
    packing_strategy="bfd",
    padding_free=True,
    model_init_kwargs={"attn_implementation": "flash_attention_2"},
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    report_to="none",
)
trainer = SFTTrainer(
    model=model_id,
    args=args,
    processing_class=tokenizer,
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```

`padding_free=True` requires FlashAttention 2 or 3. With `packing=True` and `packing_strategy="bfd"`, TRL enables padding-free operation automatically. Install and validate the chosen attention backend before a long run.

## GRPO reward functions

Custom reward functions receive keyword arguments. They must accept `completions` and may accept `prompts`, `completion_ids`, `trainer_state`, logging helpers, environment objects, and every dataset column other than `prompt`. Use `**kwargs` so the function remains compatible with the trainer contract. Return one float or `None` per completion; `None` lets a reward opt out for a sample.

```python
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

def exact_match_reward(completions, answer, **kwargs):
    return [float(completion.strip() == expected) for completion, expected in zip(completions, answer)]

dataset = Dataset.from_list(
    [
        {"prompt": "What is 2 + 2?", "answer": "4"},
        {"prompt": "What is 3 + 5?", "answer": "8"},
    ]
)
args = GRPOConfig(
    output_dir="outputs/grpo",
    num_generations=4,
    generation_batch_size=16,
    steps_per_generation=8,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_completion_length=128,
    report_to="none",
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=exact_match_reward,
    args=args,
    train_dataset=dataset,
)
trainer.train()
```

For conversational data, each completion is a list of message dictionaries rather than a string. Extract the assistant content before scoring it. Keep `remove_unused_columns=False` when rewards need extra dataset columns; this is the GRPO default.

## Settings that materially affect correctness or throughput

- `processing_class`: Pass the tokenizer or processor explicitly after setting its pad token or chat template. Online trainers require left padding; TRL configures this, but custom processors still need a valid pad token.
- `model_init_kwargs`: Applies only when `model` is a model identifier. Put `from_pretrained` options such as `attn_implementation`, `dtype`, or `device_map` here instead of assuming the trainer constructor forwards arbitrary keywords.
- `packing`: Groups short SFT examples into `max_length` blocks. It changes sample boundaries and should be validated with the intended loss mask.
- `padding_free`: Flattens a batch to remove padding overhead. It needs a compatible attention implementation and is implied by SFT BFD packing.
- `use_liger_kernel`: Enables Liger optimizations when the optional dependency and the selected trainer/model combination support them. Treat it as an explicit environment capability, not a portable default.
- `attn_implementation`: For Python trainer APIs, pass this through `model_init_kwargs`; `ModelConfig.attn_implementation` is used by TRL's script/CLI layer.
- `generation_batch_size`: Number of samples generated together for GRPO/RLOO. If omitted, it is derived from per-device batch size, process count, and `steps_per_generation`.
- `steps_per_generation`: Reuses one generated batch across this many optimizer steps. If omitted, it defaults to `gradient_accumulation_steps`.
- vLLM capacity: `vllm_gpu_memory_utilization`, `vllm_tensor_parallel_size`, and `vllm_max_model_length` configure colocated generation. The lower-level `VLLMGeneration.max_num_seqs` caps concurrent sequences; current trainers derive it from their batch configuration rather than exposing a `vllm_max_num_seqs` config field.

For GRPO, `num_generations` must divide the effective batch size. Check the relationship among world size, per-device batch size, gradient accumulation, `generation_batch_size`, and `steps_per_generation` before launching distributed jobs.

## Checkpoint handling

Resume training with `trainer.train(resume_from_checkpoint=...)`. Save the processing class with the model so inference uses the same tokens and chat template. PEFT checkpoints contain adapter weights; merge them only when the serving stack requires a standalone model, and verify the merged model with the intended inference backend.

## CLI footnote

The CLI remains useful for standard recipes and YAML configuration:

```bash
trl sft --config examples/configs/sft_config.yaml
```

Use Python when the workflow includes custom rewards, custom preprocessing, callbacks, or non-default model construction.
