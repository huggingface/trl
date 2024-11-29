# Community Tutorials

Community tutorials are made by active members of the Hugging Face community that want to share their knowledge and expertise with others. They are a great way to learn about the library and its features, and to get started with core classes and modalities.

# Language Models

| Task                    | Class           | Description                                                                              | Author                                                         | Tutorial                                                                                                             |
| ----------------------- | --------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Instruction tuning      | [`SFTTrainer`]  | Fine-tuning Google Gemma LLMs using ChatML format with QLoRA                             | [Philipp Schmid](https://huggingface.co/philschmid)            | [Link](https://www.philschmid.de/fine-tune-google-gemma)                                                             |
| Structured Generation   | [`SFTTrainer`]  | Fine-tuning Llama-2-7B to generate Persian product catalogs in JSON using QLoRA and PEFT | [Mohammadreza Esmaeilian](https://huggingface.co/Mohammadreza) | [Link](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format) |
| Preference Optimization | [`DPOTrainer`]  | Align Mistral-7b using Direct Preference Optimization for human preference alignment     | [Maxime Labonne](https://huggingface.co/mlabonne)              | [Link](https://mlabonne.github.io/blog/posts/Fine_tune_Mistral_7b_with_DPO.html)                                     |
| Preference Optimization | [`ORPOTrainer`] | Fine-tuning Llama 3 with ORPO combining instruction tuning and preference alignment      | [Maxime Labonne](https://huggingface.co/mlabonne)              | [Link](https://mlabonne.github.io/blog/posts/2024-04-19_Fine_tune_Llama_3_with_ORPO.html)                            |

# Vision Language Models

| Task            | Class          | Description                                                                  | Author                                                 | Tutorial                                                             |
| --------------- | -------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------- |
| Visual QA       | [`SFTTrainer`] | Fine-tuning Qwen2-VL-7B for visual question answering on ChartQA dataset     | [Sergio Paniego](https://huggingface.co/sergiopaniego) | [Link](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl)    |
| SEO Description | [`SFTTrainer`] | Fine-tuning Qwen2-VL-7B for generating SEO-friendly descriptions from images | [Philipp Schmid](https://huggingface.co/philschmid)    | [Link](https://www.philschmid.de/fine-tune-multimodal-llms-with-trl) |

## Contributing

If you have a tutorial that you would like to add to this list, please open a PR to add it. We will review it and merge it if it is relevant to the community.