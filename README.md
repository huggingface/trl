# TRL - Transformer Reinforcement Learning

<div style="text-align: center">
    <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl_banner_dark.png" alt="TRL Banner">
</div>

<hr> <br>

<h3 align="center">
    <p>A comprehensive library to post-train foundation models</p>
</h3>

<p align="center">
    <a href="https://github.com/huggingface/trl/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/huggingface/trl.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/trl/index"><img alt="Documentation" src="https://img.shields.io/website?label=documentation&url=https%3A%2F%2Fhuggingface.co%2Fdocs%2Ftrl%2Findex&down_color=red&down_message=offline&up_color=blue&up_message=online"></a>
    <a href="https://github.com/huggingface/trl/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/trl.svg"></a>
    <a href="https://huggingface.co/trl-lib"><img alt="Hugging Face Hub" src="https://img.shields.io/badge/🤗%20Hub-trl--lib-yellow"></a>
</p>

## 🎉 What's New

**OpenEnv Integration:** TRL now supports **[OpenEnv](https://huggingface.co/blog/openenv)**, the open-source framework from Meta for defining, deploying, and interacting with environments in reinforcement learning and agentic workflows.

Explore how to seamlessly integrate TRL with OpenEnv in our [dedicated documentation](https://huggingface.co/docs/trl/openenv).

## Overview

TRL is a cutting-edge library designed for post-training foundation models using advanced techniques like Supervised Fine-Tuning (SFT), Group Relative Policy Optimization (GRPO), and Direct Preference Optimization (DPO). Built on top of the [🤗 Transformers](https://github.com/huggingface/transformers) ecosystem, TRL supports a variety of model architectures and modalities, and can be scaled-up across various hardware setups.

## Highlights

- **Trainers**: Various fine-tuning methods are easily accessible via trainers like [`SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer), [`GRPOTrainer`](https://huggingface.co/docs/trl/grpo_trainer), [`DPOTrainer`](https://huggingface.co/docs/trl/dpo_trainer), [`RewardTrainer`](https://huggingface.co/docs/trl/reward_trainer) and more.

- **Efficient and scalable**:
  - Leverages [🤗 Accelerate](https://github.com/huggingface/accelerate) to scale from single GPU to multi-node clusters using methods like [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) and [DeepSpeed](https://github.com/deepspeedai/DeepSpeed).
  - Full integration with [🤗 PEFT](https://github.com/huggingface/peft) enables training on large models with modest hardware via quantization and LoRA/QLoRA.
  - Integrates [🦥 Unsloth](https://github.com/unslothai/unsloth) for accelerating training using optimized kernels.

- **Command Line Interface (CLI)**: A simple interface lets you fine-tune with models without needing to write code.

## Installation

### Python Package

Install the library using `pip`:

```bash
pip install trl
```

### From source

If you want to use the latest features before an official release, you can install TRL from source:

```bash
pip install git+https://github.com/huggingface/trl.git
```

### Repository

If you want to use the examples you can clone the repository with the following command:

```bash
git clone https://github.com/huggingface/trl.git
```

## Quick Start

For more flexibility and control over training, TRL provides dedicated trainer classes to post-train language models or PEFT adapters on a custom dataset. Each trainer in TRL is a light wrapper around the 🤗 Transformers trainer and natively supports distributed training methods like DDP, DeepSpeed ZeRO, and FSDP.

### `SFTTrainer`

Here is a basic example of how to use the [`SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer):

```python
from trl import SFTTrainer
from datasets import load_dataset

dataset = load_dataset("trl-lib/Capybara", split="train")

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
)
trainer.train()
```

### `GRPOTrainer`

[`GRPOTrainer`](https://huggingface.co/docs/trl/grpo_trainer) implements the [Group Relative Policy Optimization (GRPO) algorithm](https://huggingface.co/papers/2402.03300) that is more memory-efficient than PPO and was used to train [Deepseek AI's R1](https://huggingface.co/deepseek-ai/DeepSeek-R1).

```python
from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()
```

> [!NOTE]
> For reasoning models, use the `reasoning_accuracy_reward()` function for better results.

### `DPOTrainer`

[`DPOTrainer`](https://huggingface.co/docs/trl/dpo_trainer) implements the popular [Direct Preference Optimization (DPO) algorithm](https://huggingface.co/papers/2305.18290) that was used to post-train [Llama 3](https://huggingface.co/papers/2407.21783) and many other models. Here is a basic example of how to use the `DPOTrainer`:

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
training_args = DPOConfig(output_dir="Qwen2.5-0.5B-DPO")
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)
trainer.train()
```

### `RewardTrainer`

Here is a basic example of how to use the [`RewardTrainer`](https://huggingface.co/docs/trl/reward_trainer):

```python
from trl import RewardTrainer
from datasets import load_dataset

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = RewardTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=dataset,
)
trainer.train()
```

## Command Line Interface (CLI)

You can use the TRL Command Line Interface (CLI) to quickly get started with post-training methods like Supervised Fine-Tuning (SFT) or Direct Preference Optimization (DPO):

**SFT:**

```bash
trl sft --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset_name trl-lib/Capybara \
    --output_dir Qwen2.5-0.5B-SFT
```

**DPO:**

```bash
trl dpo --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name argilla/Capybara-Preferences \
    --output_dir Qwen2.5-0.5B-DPO 
```

Read more about CLI in the [relevant documentation section](https://huggingface.co/docs/trl/clis) or use `--help` for more details.

## Development

If you want to contribute to `trl` or customize it to your needs make sure to read the [contribution guide](https://github.com/huggingface/trl/blob/main/CONTRIBUTING.md) and make sure you make a dev install:

```bash
git clone https://github.com/huggingface/trl.git
cd trl/
pip install -e .[dev]
```

## Experimental

A minimal incubation area is available under `trl.experimental` for unstable / fast-evolving features. Anything there may change or be removed in any release without notice.

Example:

```python
from trl.experimental.new_trainer import NewTrainer
```

Read more in the [Experimental docs](https://huggingface.co/docs/trl/experimental_overview).

## Citation

```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```

## License

This repository's source code is available under the [Apache-2.0 License](LICENSE).
