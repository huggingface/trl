# Examples


## Introduction

The examples should work in any of the following settings (with the same script):
   - single GPU
   - multi GPUS (using PyTorch distributed mode)
   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1, 2, & 3)
   - fp16 (mixed-precision), fp32 (normal precision), or bf16 (bfloat16 precision)

To run it in each of these various modes, first initialize the accelerate
configuration with `accelerate config`

**NOTE to train with a 4-bit or 8-bit model**, please run

```bash
pip install --upgrade trl[quantization]
```


## Accelerate Config
For all the examples, you'll need to generate a 🤗 Accelerate config file with:

```shell
accelerate config # will prompt you to define the training configuration
```

Then, it is encouraged to launch jobs with `accelerate launch`!


# Maintained Examples


| File                                                                                           | Description                                                                                                              |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| [`examples/scripts/sft.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py) | This script shows how to use the `SFTTrainer` to fine tune a model or adapters into a target dataset.                     |
| [`examples/scripts/vsft_llava.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/vsft_llava.py) | This script shows how to use the `SFTTrainer` to fine tune a Vision Language Model in a chat setting, the script has been tested on a llava1.5 model so users may see unexpected behaviour in other model architectures.                     |
| [`examples/scripts/reward_modeling.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py) | This script shows how to use the `RewardTrainer` to train a reward model on your own dataset.                            |
| [`examples/scripts/ppo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/ppo.py) | This script shows how to use the `PPOTrainer` to fine-tune a sentiment analysis model using IMDB dataset                 |
| [`examples/scripts/ppo_multi_adapter.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/ppo_multi_adapter.py) | This script shows how to use the `PPOTrainer` to train a single base model with multiple adapters. Requires you to run the example script with the reward model training beforehand. |
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

All of the scripts can be run on multiple GPUs by providing the path of an 🤗 Accelerate config file when calling `accelerate launch`. To launch one of them on one or multiple GPUs, run the following command (swapping `{NUM_GPUS}` with the number of GPUs in your machine and `--all_arguments_of_the_script` with your arguments.)

```shell
accelerate launch --config_file=examples/accelerate_configs/multi_gpu.yaml --num_processes {NUM_GPUS} path_to_script.py --all_arguments_of_the_script
```

You can also adjust the parameters of the 🤗 Accelerate config file to suit your needs (e.g. training in mixed precision).

### Distributed training with DeepSpeed

Most of the scripts can be run on multiple GPUs together with DeepSpeed ZeRO-{1,2,3} for efficient sharding of the optimizer states, gradients, and model weights. To do so, run following command (swapping `{NUM_GPUS}` with the number of GPUs in your machine, `--all_arguments_of_the_script` with your arguments, and `--deepspeed_config` with the path to the DeepSpeed config file such as `examples/deepspeed_configs/deepspeed_zero1.yaml`):

```shell
accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero{1,2,3}.yaml --num_processes {NUM_GPUS} path_to_script.py --all_arguments_of_the_script
```
