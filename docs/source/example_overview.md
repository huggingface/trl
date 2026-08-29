# Examples

The [`examples/`](https://github.com/huggingface/trl/tree/main/examples) directory contains a collection of self-contained examples that demonstrate how to use the TRL library for various applications. **Each example lives in its own folder** named after the method and the task it demonstrates (e.g. `grpo_wordle`, `sft_gpt_oss`), and holds everything the example needs: scripts, notebooks, prompts, chat templates, and evaluation code.

Basic single-trainer training scripts are not examples: they live in [`trl/scripts`](https://github.com/huggingface/trl/tree/main/trl/scripts) and are exposed through the [command line interface](clis) (`trl sft`, `trl dpo`, `trl grpo`, …). Each trainer's documentation page also contains a complete runnable snippet.

Shared resources sit at the root of `examples/`:

- [`examples/accelerate_configs`](https://github.com/huggingface/trl/tree/main/examples/accelerate_configs): 🤗 Accelerate configuration files for multi-GPU, DeepSpeed ZeRO, FSDP, and context-parallel setups, used by many examples.
- [`examples/datasets`](https://github.com/huggingface/trl/tree/main/examples/datasets): the scripts used to generate the `trl-lib` datasets used across the examples.

**Getting Started**

Install TRL and additional dependencies as follows:

```bash
pip install --upgrade trl[quantization]
```

Check for additional optional dependencies [here](https://github.com/huggingface/trl/blob/main/pyproject.toml). Notebook-based examples are self-contained and can run on **free Colab**; script-based examples run on single-GPU, multi-GPU, or DeepSpeed setups (see [Distributed Training](#distributed-training) below).

## Index

| Example | Description | Open in Colab |
| --- | --- | --- |
| [`async_distillation_math`](https://github.com/huggingface/trl/tree/main/examples/async_distillation_math) | Async on-policy distillation on GSM8K with [`experimental.async_distillation.AsyncDistillationTrainer`]: the teacher is served over HTTP with vLLM, including a multi-teacher (MOPD) math + code variant. | |
| [`async_grpo_math`](https://github.com/huggingface/trl/tree/main/examples/async_grpo_math) | Asynchronous GRPO on GSM8K with [`experimental.async_grpo.AsyncGRPOTrainer`], decoupling generation (vLLM server) from training. | |
| [`async_grpo_opencode`](https://github.com/huggingface/trl/tree/main/examples/async_grpo_opencode) | AsyncGRPO training of the real `opencode` coding agent on an [OpenEnv](openenv) environment (loop-owning: the external agent runs its own tool loop and TRL trains on its captured proxy trace), with a local subprocess sandbox or remote Hugging Face sandboxes. | |
| [`dpo_reduce_hallucinations`](https://github.com/huggingface/trl/tree/main/examples/dpo_reduce_hallucinations) | DPO fine-tuning of a Vision Language Model to reduce hallucinations using the [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) dataset. | |
| [`gold_chatbot_arena`](https://github.com/huggingface/trl/tree/main/examples/gold_chatbot_arena) | General Online Logit Distillation (GOLD) of a Qwen2 teacher into a Llama 3.2 student (cross-tokenizer) on chatbot_arena_completions with [`experimental.gold.GOLDTrainer`], with full-training and LoRA variants. | |
| [`gold_qwen3_vl`](https://github.com/huggingface/trl/tree/main/examples/gold_qwen3_vl) | General Online Logit Distillation (GOLD) of Qwen3-VL-8B into smaller VLM students with [`experimental.gold.GOLDTrainer`], covering same-family (JSD loss) and cross-family (ULD loss) distillation. | |
| [`grpo_2048`](https://github.com/huggingface/trl/tree/main/examples/grpo_2048) | GRPO with tool calling to teach a model to play the 2048 game. | |
| [`grpo_browsergym`](https://github.com/huggingface/trl/tree/main/examples/grpo_browsergym) | GRPO with the BrowserGym [OpenEnv](openenv) environment, with LLM and VLM variants. | |
| [`grpo_carla`](https://github.com/huggingface/trl/tree/main/examples/grpo_carla) | GRPO with the CARLA autonomous-driving [OpenEnv](openenv) environment, with LLM and VLM variants (multimodal camera-image tool responses). | |
| [`grpo_catch`](https://github.com/huggingface/trl/tree/main/examples/grpo_catch) | GRPO with the Catch (OpenSpiel) [OpenEnv](openenv) environment. | |
| [`grpo_continuous_batching`](https://github.com/huggingface/trl/tree/main/examples/grpo_continuous_batching) | GRPO with transformers' continuous batching engine for faster generation on large batches with variable completion lengths. | |
| [`grpo_echo`](https://github.com/huggingface/trl/tree/main/examples/grpo_echo) | Minimal GRPO training with the Echo [OpenEnv](openenv) environment. | |
| [`grpo_harbor`](https://github.com/huggingface/trl/tree/main/examples/grpo_harbor) | GRPO training against a Harbor task suite with a pluggable base agent (`bash` / `jupyter` / `terminal_notes` harnesses). See the [Harbor Integration](harbor) guide. | |
| [`grpo_ministral3_vl`](https://github.com/huggingface/trl/tree/main/examples/grpo_ministral3_vl) | GRPO Ministral 3 with QLoRA on free Colab. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/grpo_ministral3_vl/grpo_ministral3_vl.ipynb) |
| [`grpo_multi_env`](https://github.com/huggingface/trl/tree/main/examples/grpo_multi_env) | Multi-environment GRPO training: Wordle + Catch [OpenEnv](openenv) environments in the same training run. | |
| [`grpo_qlora`](https://github.com/huggingface/trl/tree/main/examples/grpo_qlora) | GRPO using QLoRA on free Colab. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/grpo_qlora/grpo_qlora.ipynb) |
| [`grpo_qwen3_vl`](https://github.com/huggingface/trl/tree/main/examples/grpo_qwen3_vl) | GRPO Qwen3-VL with QLoRA on free Colab. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/grpo_qwen3_vl/grpo_qwen3_vl.ipynb) |
| [`grpo_rnj_1_instruct`](https://github.com/huggingface/trl/tree/main/examples/grpo_rnj_1_instruct) | GRPO on rnj-1-instruct with QLoRA on Colab to add reasoning capabilities. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/grpo_rnj_1_instruct/grpo_rnj_1_instruct.ipynb) |
| [`grpo_seta`](https://github.com/huggingface/trl/tree/main/examples/grpo_seta) | GRPO training against the SETA ORS environment on the openreward.ai catalog. See the [OpenReward Integration](openreward) guide. | |
| [`grpo_sql_agent`](https://github.com/huggingface/trl/tree/main/examples/grpo_sql_agent) | GRPO to train an agent that answers questions by querying a SQL database (script and notebook; not runnable on free Colab due to OOM). | |
| [`grpo_sudoku`](https://github.com/huggingface/trl/tree/main/examples/grpo_sudoku) | GRPO to play Sudoku on an [OpenEnv](openenv) environment (script and notebook). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/grpo_sudoku/grpo_sudoku.ipynb) |
| [`grpo_visual_math`](https://github.com/huggingface/trl/tree/main/examples/grpo_visual_math) | GRPO fine-tuning of a multimodal model for reasoning using the [lmms-lab/multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) dataset. | |
| [`grpo_wordle`](https://github.com/huggingface/trl/tree/main/examples/grpo_wordle) | GRPO to play Wordle (TextArena) on an [OpenEnv](openenv) environment (script and notebook). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/grpo_wordle/grpo_wordle.ipynb) |
| [`gspo_math`](https://github.com/huggingface/trl/tree/main/examples/gspo_math) | GSPO via the [`GRPOTrainer`] for math reasoning on the [AI-MO/NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) dataset. | |
| [`gspo_visual_math`](https://github.com/huggingface/trl/tree/main/examples/gspo_visual_math) | GSPO via the [`GRPOTrainer`] to fine-tune a multimodal model for reasoning using the [lmms-lab/multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) dataset. | |
| [`mpo_visual_preferences`](https://github.com/huggingface/trl/tree/main/examples/mpo_visual_preferences) | MPO via the [`DPOTrainer`] to align a multimodal model based on preferences using the [HuggingFaceH4/rlaif-v_formatted](https://huggingface.co/datasets/HuggingFaceH4/rlaif-v_formatted) dataset and a set of loss weights. | |
| [`online_dpo_visual_math`](https://github.com/huggingface/trl/tree/main/examples/online_dpo_visual_math) | Online DPO fine-tuning of a Vision Language Model with [`experimental.online_dpo.OnlineDPOTrainer`]. | |
| [`ppo_sentiment`](https://github.com/huggingface/trl/tree/main/examples/ppo_sentiment) | PPO with [`experimental.ppo.PPOTrainer`] to continue text with positive sentiment or physically descriptive language. | |
| [`ppo_tldr`](https://github.com/huggingface/trl/tree/main/examples/ppo_tldr) | PPO with [`experimental.ppo.PPOTrainer`] to generate TL;DR summaries. | |
| [`rloo_math`](https://github.com/huggingface/trl/tree/main/examples/rloo_math) | RLOO with the [`RLOOTrainer`] for math reasoning on the [AI-MO/NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) dataset with vLLM. | |
| [`rloo_visual_math`](https://github.com/huggingface/trl/tree/main/examples/rloo_visual_math) | RLOO fine-tuning of a multimodal model for reasoning using the [lmms-lab/multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) dataset. | |
| [`sdft_privileged_context`](https://github.com/huggingface/trl/tree/main/examples/sdft_privileged_context) | Self-distillation fine-tuning with [`experimental.sdft.SDFTTrainer`], distilling privileged (teacher-only) context into the model. | |
| [`sdpo_math`](https://github.com/huggingface/trl/tree/main/examples/sdpo_math) | SDPO with [`experimental.sdpo.SDPOTrainer`] using verifiable math rewards and optional environment feedback on [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k). | |
| [`sft_diffusion_gemma`](https://github.com/huggingface/trl/tree/main/examples/sft_diffusion_gemma) | SFT of the DiffusionGemma block-diffusion language model on GSM8K by extending the [`SFTTrainer`] with a block-diffusion objective. | |
| [`sft_gemma3`](https://github.com/huggingface/trl/tree/main/examples/sft_gemma3) | SFT of Gemma 3 on the Codeforces COTS dataset. | |
| [`sft_gemma3_vision`](https://github.com/huggingface/trl/tree/main/examples/sft_gemma3_vision) | SFT of Gemma 3 on vision to text tasks. | |
| [`sft_gpt_oss`](https://github.com/huggingface/trl/tree/main/examples/sft_gpt_oss) | SFT of openai/gpt-oss-20b. | |
| [`sft_ministral3_vl`](https://github.com/huggingface/trl/tree/main/examples/sft_ministral3_vl) | SFT Ministral 3 with QLoRA on free Colab. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/sft_ministral3_vl/sft_ministral3_vl.ipynb) |
| [`sft_nemotron_3`](https://github.com/huggingface/trl/tree/main/examples/sft_nemotron_3) | SFT of NVIDIA Nemotron 3 models (script and LoRA notebook). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/sft_nemotron_3/sft_nemotron_3.ipynb) |
| [`sft_qlora`](https://github.com/huggingface/trl/tree/main/examples/sft_qlora) | SFT using QLoRA on free Colab. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/sft_qlora/sft_qlora.ipynb) |
| [`sft_qwen3_vl`](https://github.com/huggingface/trl/tree/main/examples/sft_qwen3_vl) | SFT Qwen3-VL with QLoRA on free Colab. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/sft_qwen3_vl/sft_qwen3_vl.ipynb) |
| [`sft_tool_calling`](https://github.com/huggingface/trl/tree/main/examples/sft_tool_calling) | Teaching tool calling to a model without native tool-calling support using SFT with QLoRA (script, chat template, and notebook). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/sft_tool_calling/sft_tool_calling.ipynb) |
| [`sft_visual_chat`](https://github.com/huggingface/trl/tree/main/examples/sft_visual_chat) | SFT of a Vision Language Model in a chat setting. Only tested with [LLaVA 1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf), [LLaVA 1.6](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf), and [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct); users may see unexpected behaviour in other model architectures. | |
| [`ssd_codegen`](https://github.com/huggingface/trl/tree/main/examples/ssd_codegen) | Simple Self-Distillation for code generation with [`experimental.ssd.SSDTrainer`], plus evaluation on LiveCodeBench. | |
| [`tpo_ultrafeedback`](https://github.com/huggingface/trl/tree/main/examples/tpo_ultrafeedback) | Triple Preference Optimization with [`experimental.tpo.TPOTrainer`] using the [tpo-alignment/triple-preference-ultrafeedback-40K](https://huggingface.co/datasets/tpo-alignment/triple-preference-ultrafeedback-40K) dataset. | |

## Distributed Training

You can run the example scripts on multiple GPUs with 🤗 Accelerate:

```shell
accelerate launch --config_file=examples/accelerate_configs/multi_gpu.yaml --num_processes {NUM_GPUS} path_to_script.py --all_arguments_of_the_script
```

For DeepSpeed ZeRO-{1,2,3}:

```shell
accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero{1,2,3}.yaml --num_processes {NUM_GPUS} path_to_script.py --all_arguments_of_the_script
```

Adjust `NUM_GPUS` and `--all_arguments_of_the_script` as needed.
