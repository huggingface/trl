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

On Neuron, `pad_to_multiple_of` defaults to 128 (its `"auto"` value), so the sequences are padded to a multiple of 128. This bounds the number of shapes, and so of compilations, to `max_length / 128`. Without it, every new sequence length triggers a compilation. On one NeuronCore, with `per_device_train_batch_size=4`, 20 steps of the example above take 130 s with the default and 260 s with `pad_to_multiple_of=None`.

The same applies to every trainer that has a `pad_to_multiple_of` option: DPO, KTO, Reward, GRPO, RLOO and Distillation.

## Compiling the model

Set `torch_compile=True` to compile the model with the `neuron` backend. Dynamo's automatic dynamic shapes produce graphs the Neuron backend can't compile yet ([#7660](https://github.com/huggingface/trl/issues/7660)), so turn them off:

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

The first step compiles for about 2 minutes. After that, with `per_device_train_batch_size=4`, Qwen3-0.6B trains at 1.15 s/step on one NeuronCore, against 1.71 s/step without compilation.

## Online methods

vLLM doesn't run on Neuron, so GRPO and RLOO generate with Transformers. With the default dynamic cache, every generated token has a new shape, and a step takes several minutes. Use the static cache, which keeps the decoding shapes fixed:

```python
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer


# Reward completions close to 200 characters
def reward_len(completions, **kwargs):
    return [-abs(200 - len(completion)) / 100 for completion in completions]


trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=reward_len,
    args=GRPOConfig(
        per_device_train_batch_size=4,
        num_generations=4,
        max_completion_length=128,
        cache_implementation="static",
        model_init_kwargs={"dtype": "bfloat16"},
    ),
    train_dataset=load_dataset("trl-lib/tldr", split="train"),
)
trainer.train()
```

The prompts are padded to `pad_to_multiple_of` before generation too, so the static cache only takes a few sizes. Each new size still compiles for about 12 minutes on one NeuronCore; once a size is compiled, a step of this example takes about 25 s.

> [!NOTE]
> The static cache needs a bf16 model for now: with the default fp32 model, the keys and values end up in different dtypes ([#7664](https://github.com/huggingface/trl/issues/7664)).

## Data parallelism

On a `trn2.3xlarge` instance, which contains one Trainium2 chip, the default [Logical NeuronCore configuration](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html) (`LNC=2`) pairs the chip's 8 physical NeuronCores into 4 logical NeuronCores.

To train on all four, launch one process per NeuronCore with [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html):

```bash
torchrun --nproc_per_node 4 train.py
```

In this setup, SFT runs 3.6x faster on 4 NeuronCores than on 1.

To restrict a run to a subset of NeuronCores, set `NEURON_RT_VISIBLE_CORES` to `0,1`, `2,3`, or `0-3`. Other combinations such as `1,2` are not supported: the Neuron runtime fails to set up collectives across them.
