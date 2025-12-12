# Examples

This directory contains a collection of examples that demonstrate how to use the TRL library for various applications. We provide both **scripts** for advanced use cases and **notebooks** for an easy start and interactive experimentation.

The notebooks are self-contained and can run on **free Colab**, while the scripts can run on **single GPU, multi-GPU, or DeepSpeed** setups.

**Getting Started**

Install TRL and additional dependencies as follows:

```bash
pip install --upgrade trl[quantization]
```

Check for additional optional dependencies [here](https://github.com/huggingface/trl/blob/main/pyproject.toml).

For scripts, you will also need an 🤗 Accelerate config (recommended for multi-gpu settings):

```bash
accelerate config # will prompt you to define the training configuration
```

This allows you to run scripts with `accelerate launch` in single or multi-GPU settings.

## Notebooks

These notebooks are easier to run and are designed for quick experimentation with TRL. The list of notebooks can be found in the [`trl/examples/notebooks/`](https://github.com/huggingface/trl/tree/main/examples/notebooks/) directory.


| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| [`grpo_agent.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/grpo_agent.ipynb) | GRPO for agent training | Not available due to OOM with Colab GPUs |
| [`grpo_rnj_1_instruct.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/grpo_rnj_1_instruct.ipynb) | GRPO rnj-1-instruct with QLoRA using TRL on Colab to add reasoning capabilities | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/grpo_rnj_1_instruct.ipynb) |
| [`sft_ministral3_vl.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/sft_ministral3_vl.ipynb) | Supervised Fine-Tuning (SFT) Ministral 3 with QLoRA using TRL on free Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/sft_ministral3_vl.ipynb) |
| [`grpo_ministral3_vl.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/grpo_ministral3_vl.ipynb) | GRPO Ministral 3 with QLoRA using TRL on free Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/grpo_ministral3_vl.ipynb) |
| [`openenv_wordle_grpo.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/openenv_wordle_grpo.ipynb) | GRPO to play Worldle on an OpenEnv environment | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb) |
| [`sft_trl_lora_qlora.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/sft_trl_lora_qlora.ipynb) | Supervised Fine-Tuning (SFT) using QLoRA on free Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/sft_trl_lora_qlora.ipynb) |
| [`sft_qwen_vl.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/sft_qwen_vl.ipynb) | Supervised Fine-Tuning (SFT) Qwen3-VL with QLoRA using TRL on free Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/sft_qwen_vl.ipynb) |
| [`grpo_qwen3_vl.ipynb`](https://github.com/huggingface/trl/tree/main/examples/notebooks/grpo_qwen3_vl.ipynb) | GRPO Qwen3-VL with QLoRA using TRL on free Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/grpo_qwen3_vl.ipynb) |

## Scripts

Scripts are maintained in the [`trl/scripts`](https://github.com/huggingface/trl/blob/main/trl/scripts) and [`examples/scripts`](https://github.com/huggingface/trl/blob/main/examples/scripts) directories. They show how to use different trainers such as `SFTTrainer`, `PPOTrainer`, `DPOTrainer`, `GRPOTrainer`, and more.

| File | Description |
| --- | --- |
| [`examples/scripts/bco.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/bco.py) | This script shows how to use the [`experimental.kto.KTOTrainer`] with the BCO loss to fine-tune a model to increase instruction-following, truthfulness, honesty, and helpfulness using the [openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) dataset. |
| [`examples/scripts/cpo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/cpo.py) | This script shows how to use the [`experimental.cpo.CPOTrainer`] to fine-tune a model to increase helpfulness and harmlessness using the [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset. |
| [`trl/scripts/dpo.py`](https://github.com/huggingface/trl/blob/main/trl/scripts/dpo.py) | This script shows how to use the [`DPOTrainer`] to fine-tune a model. |
| [`examples/scripts/dpo_vlm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo_vlm.py) | This script shows how to use the [`DPOTrainer`] to fine-tune a Vision Language Model to reduce hallucinations using the [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) dataset. |
| [`examples/scripts/evals/judge_tldr.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/evals/judge_tldr.py) | This script shows how to use [`experimental.judges.HfPairwiseJudge`] or [`experimental.judges.OpenAIPairwiseJudge`] to judge model generations. |
| [`examples/scripts/gkd.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/gkd.py) | This script shows how to use the [`experimental.gkd.GKDTrainer`] to fine-tune a model. |
| [`trl/scripts/grpo.py`](https://github.com/huggingface/trl/blob/main/trl/scripts/grpo.py) | This script shows how to use the [`GRPOTrainer`] to fine-tune a model. |
| [`trl/scripts/grpo_agent.py`](https://github.com/huggingface/trl/blob/main/trl/scripts/grpo_agent.py) | This script shows how to use the [`GRPOTrainer`] to fine-tune a model to enable agentic usage. |
| [`examples/scripts/grpo_vlm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/grpo_vlm.py) | This script shows how to use the [`GRPOTrainer`] to fine-tune a multimodal model for reasoning using the [lmms-lab/multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) dataset. |
| [`examples/scripts/gspo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/gspo.py) | This script shows how to use GSPO via the [`GRPOTrainer`] to fine-tune model for reasoning using the [AI-MO/NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) dataset. |
| [`examples/scripts/gspo_vlm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/gspo_vlm.py) | This script shows how to use GSPO via the [`GRPOTrainer`] to fine-tune a multimodal model for reasoning using the [lmms-lab/multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) dataset. |
| [`examples/scripts/kto.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/kto.py) | This script shows how to use the [`experimental.kto.KTOTrainer`] to fine-tune a model. |
| [`examples/scripts/mpo_vlm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/mpo_vlm.py) | This script shows how to use MPO via the [`DPOTrainer`] to align a model based on preferences using the [HuggingFaceH4/rlaif-v_formatted](https://huggingface.co/datasets/HuggingFaceH4/rlaif-v_formatted) dataset and a set of loss weights with weights. |
| [`examples/scripts/nash_md.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/nash_md.py) | This script shows how to use the [`experimental.nash_md.NashMDTrainer`] to fine-tune a model. |
| [`examples/scripts/online_dpo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/online_dpo.py) | This script shows how to use the [`experimental.online_dpo.OnlineDPOTrainer`] to fine-tune a model. |
| [`examples/scripts/online_dpo_vlm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/online_dpo_vlm.py) | This script shows how to use the [`experimental.online_dpo.OnlineDPOTrainer`] to fine-tune a a Vision Language Model. |
| [`examples/scripts/openenv/browsergym.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/browsergym.py) | Simple script to run GRPO training via the [`GRPOTrainer`] with OpenEnv's BrowserGym environment and vLLM |
| [`examples/scripts/openenv/catch.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/catch.py) | Simple script to run GRPO training via the [`GRPOTrainer`] with OpenEnv's Catch environment (OpenSpiel) and vLLM |
| [`examples/scripts/openenv/echo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/echo.py) | Simple script to run GRPO training via the [`GRPOTrainer`] with OpenEnv's Echo environment and vLLM. |
| [`examples/scripts/openenv/wordle.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/wordle.py) | Simple script to run GRPO training via the [`GRPOTrainer`] with OpenEnv's Wordle environment and vLLM. |
| [`examples/scripts/orpo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py) | This script shows how to use the [`experimental.orpo.ORPOTrainer`] to fine-tune a model to increase helpfulness and harmlessness using the [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset. |
| [`examples/scripts/ppo/ppo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo.py) | This script shows how to use the [`experimental.ppo.PPOTrainer`] to fine-tune a model to improve its ability to continue text with positive sentiment or physically descriptive language. |
| [`examples/scripts/ppo/ppo_tldr.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo_tldr.py) | This script shows how to use the [`experimental.ppo.PPOTrainer`] to fine-tune a model to improve its ability to generate TL;DR summaries. |
| [`examples/scripts/prm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/prm.py) | This script shows how to use the [`experimental.prm.PRMTrainer`] to fine-tune a Process-supervised Reward Model (PRM). |
| [`examples/scripts/reward_modeling.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py) | This script shows how to use the [`RewardTrainer`] to train an Outcome Reward Model (ORM) on your own dataset. |
| [`examples/scripts/rloo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/rloo.py) | This script shows how to use the [`RLOOTrainer`] to fine-tune a model to improve its ability to solve math questions. |
| [`examples/scripts/sft.py`](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py) | This script shows how to use the [`SFTTrainer`] to fine-tune a model. |
| [`examples/scripts/sft_gemma3.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_gemma3.py) | This script shows how to use the [`SFTTrainer`] to fine-tune a Gemma 3 model. |
| [`examples/scripts/sft_video_llm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_video_llm.py) | This script shows how to use the [`SFTTrainer`] to fine-tune a Video Language Model. |
| [`examples/scripts/sft_vlm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py) | This script shows how to use the [`SFTTrainer`] to fine-tune a Vision Language Model in a chat setting. The script has only been tested with [LLaVA 1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf), [LLaVA 1.6](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf), and [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) models, so users may see unexpected behaviour in other model architectures. |
| [`examples/scripts/sft_vlm_gemma3.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm_gemma3.py) | This script shows how to use the [`SFTTrainer`] to fine-tune a Gemma 3 model on vision to text tasks. |
| [`examples/scripts/sft_vlm_smol_vlm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm_smol_vlm.py) | This script shows how to use the [`SFTTrainer`] to fine-tune a SmolVLM model. |
| [`examples/scripts/xpo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/xpo.py) | This script shows how to use the [`experimental.xpo.XPOTrainer`] to fine-tune a model. |

## Distributed Training (for scripts)

You can run scripts on multiple GPUs with 🤗 Accelerate:

```shell
accelerate launch --config_file=examples/accelerate_configs/multi_gpu.yaml --num_processes {NUM_GPUS} path_to_script.py --all_arguments_of_the_script
```

For DeepSpeed ZeRO-{1,2,3}:

```shell
accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero{1,2,3}.yaml --num_processes {NUM_GPUS} path_to_script.py --all_arguments_of_the_script
```

Adjust `NUM_GPUS` and `--all_arguments_of_the_script` as needed.
