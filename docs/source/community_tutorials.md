# Community Tutorials

Community tutorials are made by active members of the Hugging Face community who want to share their knowledge and expertise with others. They are a great way to learn about the library and its features, and to get started with core classes and modalities.

# Language Models

| Task | Class | Description | Author | Tutorial | Colab |
| --- | --- | --- | --- | --- | --- |
| Reinforcement Learning | [`GRPOTrainer`] | Post training an LLM for reasoning with GRPO in TRL | [Sergio Paniego](https://huggingface.co/sergiopaniego) | [Link](https://huggingface.co/learn/cookbook/fine_tuning_llm_grpo_trl) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_llm_grpo_trl.ipynb) |
| Reinforcement Learning | [`GRPOTrainer`] | Mini-R1: Reproduce Deepseek R1 „aha moment“ a RL tutorial | [Philipp Schmid](https://huggingface.co/philschmid) | [Link](https://www.philschmid.de/mini-deepseek-r1) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb) |
| Instruction tuning | [`SFTTrainer`] | Fine-tuning Google Gemma LLMs using ChatML format with QLoRA | [Philipp Schmid](https://huggingface.co/philschmid) | [Link](https://www.philschmid.de/fine-tune-google-gemma) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/deep-learning-pytorch-huggingface/blob/main/training/gemma-lora-example.ipynb) |
| Structured Generation | [`SFTTrainer`] | Fine-tuning Llama-2-7B to generate Persian product catalogs in JSON using QLoRA and PEFT | [Mohammadreza Esmaeilian](https://huggingface.co/Mohammadreza) | [Link](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format.ipynb) |
| Preference Optimization | [`DPOTrainer`] | Align Mistral-7b using Direct Preference Optimization for human preference alignment | [Maxime Labonne](https://huggingface.co/mlabonne) | [Link](https://mlabonne.github.io/blog/posts/Fine_tune_Mistral_7b_with_DPO.html) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlabonne/llm-course/blob/main/Fine_tune_a_Mistral_7b_model_with_DPO.ipynb) |
| Preference Optimization | [`ORPOTrainer`] | Fine-tuning Llama 3 with ORPO combining instruction tuning and preference alignment | [Maxime Labonne](https://huggingface.co/mlabonne) | [Link](https://mlabonne.github.io/blog/posts/2024-04-19_Fine_tune_Llama_3_with_ORPO.html) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eHNWg9gnaXErdAa8_mcvjMupbSS6rDvi) |
| Instruction tuning | [`SFTTrainer`] | How to fine-tune open LLMs in 2025 with Hugging Face | [Philipp Schmid](https://huggingface.co/philschmid) | [Link](https://www.philschmid.de/fine-tune-llms-in-2025) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/deep-learning-pytorch-huggingface/blob/main/training/fine-tune-llms-in-2025.ipynb) |

<Youtube id="cnGyyM0vOes" />

# Vision Language Models

| Task | Class | Description | Author | Tutorial | Colab |
| --- | --- | --- | --- | --- | --- |
| Visual QA | [`SFTTrainer`] | Fine-tuning Qwen2-VL-7B for visual question answering on ChartQA dataset | [Sergio Paniego](https://huggingface.co/sergiopaniego) | [Link](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_vlm_trl.ipynb) |
| Visual QA | [`SFTTrainer`] | Fine-tuning SmolVLM with TRL on a consumer GPU | [Sergio Paniego](https://huggingface.co/sergiopaniego) | [Link](https://huggingface.co/learn/cookbook/fine_tuning_smol_vlm_sft_trl) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_smol_vlm_sft_trl.ipynb) |
| SEO Description | [`SFTTrainer`] | Fine-tuning Qwen2-VL-7B for generating SEO-friendly descriptions from images | [Philipp Schmid](https://huggingface.co/philschmid) | [Link](https://www.philschmid.de/fine-tune-multimodal-llms-with-trl) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/deep-learning-pytorch-huggingface/blob/main/training/fine-tune-multimodal-llms-with-trl.ipynb) |
| Visual QA | [`DPOTrainer`] | PaliGemma 🤝 Direct Preference Optimization | [Merve Noyan](https://huggingface.co/merve) | [Link](https://github.com/merveenoyan/smol-vision/blob/main/PaliGemma_DPO.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/merveenoyan/smol-vision/blob/main/PaliGemma_DPO.ipynb) |
| Visual QA | [`DPOTrainer`] | Fine-tuning SmolVLM using direct preference optimization (DPO) with TRL on a consumer GPU | [Sergio Paniego](https://huggingface.co/sergiopaniego) | [Link](https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_vlm_dpo_smolvlm_instruct.ipynb) |

## Contributing

If you have a tutorial that you would like to add to this list, please open a PR to add it. We will review it and merge it if it is relevant to the community.
