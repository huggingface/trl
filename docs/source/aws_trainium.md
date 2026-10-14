# Training on AWS Trainium

[AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/) runs PyTorch natively through the `torch-neuronx` package, which registers the `neuron` device. TRL trainers run on it without code changes. Neuron compiles a graph for each input shape it sees, and each compilation takes about a minute, so most of this page is about keeping shapes static. For installation and the Transformers side, see the [Transformers AWS Trainium guide](https://huggingface.co/docs/transformers/main/en/aws_trainium).

## Installation

Install `torch-neuronx` by following the [AWS Neuron setup guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/pytorch/index.html), then install TRL in the same environment.

```bash
pip install trl
```

## Supervised fine-tuning

The [`SFTTrainer`] quick start runs as is:

```python
from datasets import load_dataset
from trl import SFTTrainer

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```

On Neuron, `pad_to_multiple_of` defaults to 128 (its `"auto"` value), so the sequences are padded to a multiple of 128. This bounds the number of shapes, and so of compilations, to `max_length / 128`. Without it, every new sequence length triggers a compilation. On one NeuronCore, 20 steps of the example above take 130 s with the default and 260 s with `pad_to_multiple_of=None`.

The same applies to every trainer that has a `pad_to_multiple_of` option: DPO, KTO, Reward, GRPO, RLOO and Distillation.

## Compiling the model

Set `torch_compile=True` to compile the model with the `neuron` backend. Dynamo then marks the sequence dimension as dynamic after the second shape, which the Neuron backend doesn't support yet ([#7660](https://github.com/huggingface/trl/issues/7660)). Turn automatic dynamic shapes off:

```python
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

torch._dynamo.config.automatic_dynamic_shapes = False

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    args=SFTConfig(torch_compile=True),
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```

The first step compiles for about 2 minutes. After that, Qwen3-0.6B trains at 1.15 s/step on one NeuronCore, against 1.71 s/step without compilation.

## Online methods

vLLM doesn't run on Neuron, so GRPO and RLOO generate with Transformers. With the default dynamic cache, every generated token has a new shape, and a step takes several minutes. Use the static cache, which keeps the decoding shapes fixed:

```python
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    args=GRPOConfig(cache_implementation="static", model_init_kwargs={"dtype": "bfloat16"}),
    train_dataset=load_dataset("trl-lib/DeepMath-103K", split="train"),
)
trainer.train()
```

The prompts are padded to `pad_to_multiple_of` before generation too, so the static cache only takes a few sizes. With both, a GRPO step on Qwen3-0.6B (16 completions of 128 tokens) takes 25 to 38 s, against 107 to 119 s with the static cache alone.

> [!NOTE]
> The static cache needs a bf16 model for now: with the default fp32 model, the keys and values end up in different dtypes ([#7664](https://github.com/huggingface/trl/issues/7664)).

## Data parallelism

A Trainium2 device exposes 4 logical NeuronCores. Train on all of them with [Accelerate](https://huggingface.co/docs/accelerate):

```bash
accelerate launch --num_processes 4 train.py
```

SFT scales to 3.6x on 4 NeuronCores. To pin a run to some NeuronCores with `NEURON_RT_VISIBLE_CORES`, use the pairs `0,1` or `2,3`, or all four: the Neuron runtime fails to set up the collectives on `1,2`.

## Memory

Gradient checkpointing, on by default in [`SFTTrainer`], is what lets Qwen3-0.6B fit on one NeuronCore (24 GB) with `per_device_train_batch_size=4` and `max_length=1024`. `activation_offloading=True` is an alternative, about 1.8x slower.
