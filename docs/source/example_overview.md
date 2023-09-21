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


## Accelerate Config
For all the examples, you'll need to generate an `Accelerate` config with:

```shell
accelerate config # will prompt you to define the training configuration
```

Then, it is encouraged to launch jobs with `accelerate launch`!


# Maintained Examples


| File                                                                                           | Description                                                                                                              |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| [`examples/scripts/sft_trainer.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_trainer.py) | This script shows how to use the `SFTTrainer` to fine tune a model or adapters into a target dataset.                     |
| [`examples/scripts/reward_trainer.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/reward_trainer.py) | This script shows how to use the `RewardTrainer` to train a reward model on your own dataset.                            |
| [`examples/scripts/sentiment_tuning.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sentiment_tuning.py) | This script shows how to use the `PPOTrainer` to fine-tune a sentiment analysis model using IMDB dataset                 |
| [`examples/scripts/multi_adapter_rl.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/multi_adapter_rl.py) | This script shows how to use the `PPOTrainer` to train a single base model with multiple adapters. Requires you to run the example script with the reward model training beforehand. |
| [`examples/scripts/stable_diffusion_tuning_example.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/stable_diffusion_tuning_example.py) | This script shows to use DDPOTrainer to fine-tune a stable diffusion model using reinforcement learning.                 |

Here are also some easier-to-run colab notebooks that you can use to get started with TRL:


| File                                                                                           | Description                                                                                                              |
|----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| [`examples/notebooks/best_of_n.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/best_of_n.ipynb)                       | This notebook demonstrates how to use the "Best of N" sampling strategy using TRL when fine-tuning your model with PPO.  |
| [`examples/notebooks/gpt2-sentiment.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/gpt2-sentiment.ipynb)              | This notebook demonstrates how to reproduce the GPT2 imdb sentiment tuning example on a jupyter notebook.                |
| [`examples/notebooks/gpt2-control.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/gpt2-control.ipynb)                  | This notebook demonstrates how to reproduce the GPT2 sentiment control example on a jupyter notebook.                    |


We also have some other examples that are less maintained but can be used as a reference:
1. **[research_projects](https://github.com/huggingface/trl/tree/main/examples/research_projects)**: Check out this folder to find the scripts used for some research projects that used TRL (LM de-toxification, Stack-Llama, etc.)


## Distributed training

All of the scripts can be run on multiple GPUs by providing the path of an 🤗 Accelerate config file when calling `accelerate launch`. To launch one of them on $N$ GPUs, use:

```shell
accelerate launch --config_file=examples/accelerate_configs/multi_gpu.yaml --num_processes {NUM_GPUS} path_to_script.py --all_arguments_of_the_script
```

You can also adjust the parameters of the 🤗 Accelerate config file to suit your needs (e.g. training in mixed precision).

### Distributed training with DeepSpeed

Most of the scripts can be run on multiple GPUs together with DeepSpeed ZeRO-{1,2,3} for efficient sharding of the optimizer states, gradients, and model weights. To do so, run:

```shell
accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero{1,2,3}.yaml --num_processes {NUM_GPUS} path_to_script.py --all_arguments_of_the_script
```
