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

Scripts can be used as examples of how to use TRL trainers. They are located in the [`trl/scripts`](https://github.com/huggingface/trl/blob/main/trl/scripts) directory. Additionally, we provide examples in the [`examples/scripts`](https://github.com/huggingface/trl/blob/main/examples/scripts) directory. These examples are maintained and tested regularly.

| File                                                                                                                          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`examples/scripts/alignprop.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/alignprop.py)                 | This script shows how to use the [`AlignPropTrainer`] to fine-tune a diffusion model.                                                                                                                                                                                                                                                                                                                                                                             |
| [`examples/scripts/bco.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/bco.py)                             | This script shows how to use the [`KTOTrainer`] with the BCO loss to fine-tune a model to increase instruction-following, truthfulness, honesty and helpfulness using the [openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) dataset.                                                                                                                                                                                                 |
| [`examples/scripts/cpo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/cpo.py)                             | This script shows how to use the [`CPOTrainer`] to fine-tune a model to increase helpfulness and harmlessness using the [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset.                                                                                                                                                                                                                                                           |
| [`examples/scripts/ddpo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/ddpo.py)                           | This script shows how to use the [`DDPOTrainer`] to fine-tune a stable diffusion model using reinforcement learning.                                                                                                                                                                                                                                                                                                                                              |
| [`examples/scripts/dpo_vlm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo_vlm.py)                     | This script shows how to use the [`DPOTrainer`] to fine-tune a Vision Language Model to reduce hallucinations using the [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) dataset.                                                                                                                                                                                                                                               |
| [`examples/scripts/orpo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py)                           | This script shows how to use the [`ORPOTrainer`] to fine-tune a model to increase helpfulness and harmlessness using the [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset.                                                                                                                                                                                                                                                          |
| [`examples/scripts/ppo/ppo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo.py)                     | This script shows how to use the [`PPOTrainer`] to fine-tune a model to improve its ability to continue text with positive sentiment or physically descriptive language                                                                                                                                                                                                                                                                                           |
| [`examples/scripts/ppo/ppo_tldr.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo_tldr.py)           | This script shows how to use the [`PPOTrainer`] to fine-tune a model to improve its ability to generate TL;DR summaries.                                                                                                                                                                                                                                                                                                                                          |
| [`examples/scripts/reward_modeling.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py)     | This script shows how to use the [`RewardTrainer`] to train a reward model on your own dataset.                                                                                                                                                                                                                                                                                                                                                                   |
| [`examples/scripts/sft_vlm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py)                     | This script shows how to use the [`SFTTrainer`] to fine-tune a Vision Language Model in a chat setting. The script has only been tested with [LLaVA 1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf), [LLaVA 1.6](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf), and [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) models so users may see unexpected behaviour in other model architectures. |

Here are also some easier-to-run colab notebooks that you can use to get started with TRL:

| File                                                                                                                              | Description                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| [`examples/notebooks/best_of_n.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/best_of_n.ipynb)           | This notebook demonstrates how to use the "Best of N" sampling strategy using TRL when fine-tuning your model with PPO. |
| [`examples/notebooks/gpt2-sentiment.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/gpt2-sentiment.ipynb) | This notebook demonstrates how to reproduce the GPT2 imdb sentiment tuning example on a jupyter notebook.               |
| [`examples/notebooks/gpt2-control.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/gpt2-control.ipynb)     | This notebook demonstrates how to reproduce the GPT2 sentiment control example on a jupyter notebook.                   |


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
