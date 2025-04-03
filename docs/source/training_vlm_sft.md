# Training a Multimodal Model Using SFT

<!-- Add diagram of training procedure -->

Large language models have evolved beyond the text-only approach and are now capable of processing multiple modalities within the same model. Specifically, many multimodal models can handle both image(s) and text, unlocking new possibilities for a wide range of applications. In this section, you'll learn how to train a multimodal model using Supervised Fine-Tuning (SFT).

## Standalone Script for Training

This section provides a detailed walkthrough of [this script](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm_gemma3.py), which demonstrates how to train a multimodal model (e.g., Gemma 3) using two different datasets. The first dataset consists of image-text pairs, while the second one includes samples with multiple images and corresponding text. Although this example uses two specific datasets, the approach can be extended to any other dataset.

### HuggingFaceH4/llava-instruct-mix-vsft Dataset (Image + Text)

This dataset is a reformatted version of [LLaVA Instruct Mix](theblackcat102/llava-instruct-mix), which includes conversations where a user provides both text and a single image during their interaction with the model. The model (referred to as "assistant") responds based on both the visual and textual information shared by the user.

<iframe
  src="https://huggingface.co/datasets/HuggingFaceH4/llava-instruct-mix-vsft/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

### FanqingM/MMIU-Benchmark Dataset (Multi-image + Text)

The **FanqingM/MMIU-Benchmark** dataset includes a context, a question, a series of images related to the question, and an answer. The context is part of the system prompt, the question and images represent the user's input, and the answer is the model's response.

<iframe
  src="https://huggingface.co/datasets/FanqingM/MMIU-Benchmark/embed/viewer/default/test"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

## Single Image+Text

<!-- Add diagram of single image -->
### Diagram

### Coding it

### Training the model

<!-- Add Wandb training results -->
### Results


## Multi-Images+Text or Interleaving

<!-- Add diagram of single image -->
### Diagram

### Coding it

### Training the model

<!-- Add Wandb training results -->
### Results


## Limitations



