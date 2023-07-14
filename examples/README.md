# Examples

_The best place to learn about examples in TRL is our [docs page](https://huggingface.co/docs/trl/index)!_

## Introduction

The examples should work in any of the following settings (with the same script):
   - single CPU or single GPU
   - multi GPUS (using PyTorch distributed mode)
   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
   - fp16 (mixed-precision) or fp32 (normal precision)

To run it in each of these various modes, first initialize the accelerate
configuration with `accelerate config`

**NOTE for to train with a 8-bit model a more recent version of**
transformers is required, for example:

```bash
pip install --upgrade bitsandbytes datasets accelerate loralib
pip install git+https://github.com/huggingface/peft.git
```

## Installation

```bash
pip install trl
#optional: wandb
pip install wandb
```
Note: if you don't want to log with `wandb` remove `log_with="wandb"` in the scripts/notebooks. 
You can also replace it with your favourite experiment tracker that's [supported by `accelerate`](https://huggingface.co/docs/accelerate/usage_guides/tracking).

## Accelerate Config
For all the examples, you'll need to generate an `Accelerate` config with:

```shell
accelerate config # will prompt you to define the training configuration
```

Then, it is encouraged to launch jobs with `accelerate launch`!

## Categories
The examples are currently split over the following categories:

**1: [ppo_trainer](https://github.com/lvwerra/trl/tree/main/examples/scripts/sentiment_tuning.py)**: Learn about different ways of using PPOTrainer
**2: [sft_trainer](https://github.com/lvwerra/trl/tree/main/examples/scripts/sft_trainer.py)**: Learn about how to leverage `SFTTrainer` for supervised fine-tuning your pretrained language models easily.
**3: [reward_modeling](https://github.com/lvwerra/trl/tree/main/examples/scripts/reward_trainer.py)**: Learn about how to use `RewardTrainer` to easily train your own reward model to use it for your RLHF pipeline.
**4: [research_projects](https://github.com/lvwerra/trl/tree/main/examples/research_projects)**: Check out that folder to check the scripts used for some research projects that used TRL (LM de-toxification, Stack-Llama, etc.)
**5: [notebooks](https://github.com/lvwerra/trl/tree/main/examples/notebooks)**: Check out that folder to check some applications of TRL features directly on a Jupyter notebook. This includes running sentiment tuning and sentiment control on a notebook, but also how to use "Best of N sampling" strategy using TRL.