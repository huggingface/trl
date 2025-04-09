# Fine-tuning a Multimodal Model Using SFT (Single or Multi-Image Dataset)

![VLM SFT training procedure](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/training_vlm_sft_training_procedure.png)  

## Overview  

This guide walks you through the process of fine-tuning a multimodal language model (e.g., **Gemma 3**) using **Supervised Fine-Tuning (SFT)**. We cover two cases:  

- **Single Image + Text**  
- **Multi-Image + Text**  

This guide serves as a **detailed walkthrough** and complements the existing [VLM SFT script](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm_gemma3.py). If you're already familiar with the concepts, you can use the script directly.  

We demonstrate the fine-tuning process using two datasets, but these principles extend to other **Vision-Language Models (VLMs)** and datasets.  

## Understanding the Datasets  

To address both **Single Image + Text** and **Multi-Image + Text** scenarios, we use two datasets that are well-suited for this task.  

### HuggingFaceH4/llava-instruct-mix-vsft Dataset (Image + Text)

This dataset is a reformatted version of [LLaVA Instruct Mix](https://huggingface.co/datasets/theblackcat102/llava-instruct-mix). It consists of conversations where a user provides both **text** and a **single image** as input.  

The model (referred to as the **"assistant"**) responds based on both the **visual and textual information** shared by the user. This dataset is particularly useful for training multimodal models to **understand and generate responses based on images and text**.  

<iframe
  src="https://huggingface.co/datasets/HuggingFaceH4/llava-instruct-mix-vsft/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

### FanqingM/MMIU-Benchmark Dataset (Multi-Image + Text)

The **FanqingM/MMIU-Benchmark** dataset consists of:  

- **Context:** Included in the system prompt.  
- **Question:** Provided as part of the user's input.  
- **Series of Images:** Multiple images related to the question.  
- **Answer:** The model's expected response.  

This dataset is designed for tasks where the model must reason over multiple images to generate an informed response based on both visual and textual inputs.

<iframe
  src="https://huggingface.co/datasets/FanqingM/MMIU-Benchmark/embed/viewer/default/test"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

## Developing a Fine-Tuning Script for Multimodal SFT

In this section, we build the script needed to fine-tune a multimodal model for both **Single Image + Text** and **Multi-Image + Text** use cases.  

### Setting Up the Environment

Before fine-tuning, we need to install the required dependencies. Let's start by setting up the environment:  

```bash
# Install the required libraries. Futher details: https://huggingface.co/docs/trl/installation 
pip install -U -q trl bitsandbytes peft hf_xet tensorboard
```

Once all dependencies are installed, we need to log in to the **Hugging Face Hub**. Since **Gemma 3** is a gated model, access permissions are required.  

If you haven’t requested access yet, visit the [Model Card](https://huggingface.co/google/gemma-3-4b-it) and request it.  

To log in, you’ll need to generate an [access token](https://huggingface.co/settings/tokens) from your Hugging Face account.  

```bash
huggingface-cli login
```

### **Loading the Data**

As mentioned earlier, we will cover two possible use cases. While the specific procedure may vary based on the dataset, the core principles remain consistent.  

This guide supports both use cases, so refer to the **Single Image + Text** or **Multi-Image + Text** sections depending on your specific scenario.

#### **Single Image + Text**

![Single Image + Text](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/training_vlm_sft_training_procedure_single_image.png)  

In this case, each sample in a batch consists of a **single image paired with text**. Since the dataset is already formatted for supervised fine-tuning (SFT), we can directly load it using `load_dataset`.

```python
from datasets import load_dataset

dataset_name = "HuggingFaceH4/llava-instruct-mix-vsft"

# Load Dataset
dataset = load_dataset(dataset_name)
```

#### **Multi-Image + Text (or Interleaving)**  

![Multi-Image + Text](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/training_vlm_sft_training_procedure_multi_image.png)  

Gemma 3 also supports **Multi-Image + Text** scenarios, where:  

- The model receives a **list of images** alongside a user message.  
- The model processes **interleaved images and text** within a conversation.  

For this dataset, some preprocessing is required before training.

```python
from datasets import load_dataset

dataset_name = "FanqingM/MMIU-Benchmark"

# Load Dataset
dataset = load_dataset(dataset_name)
```

After loading the dataset, we need to preprocess and format it into a conversational structure. Here’s an example of how the data might look:

```python
{"role": "system", "content": [{"type": "text", "text": "You are a judge in a photography competition, and now you are given the four images. Please examine the details and tell which one of them is most likely to be a real photograph.\nSelect from the following choices.\nA: the first image\nB: the second image\nC: the third image\nD: the fourth image"}]},
{"role": "user", "content": images_list + [{"type": "text", "text": "Which image is most likely to be a real photograph?"}]},
{"role": "assistant", "content": [{"type": "text", "text": "A: the first image\nB: the second image\nC: the third image\nD: the fourth image"}]},
```

Here, `images_list` is a list of images:

```python
images_list = [
  {"type": "image", "image": <class 'PIL.Image.Image'>},
  {"type": "image", "image": <class 'PIL.Image.Image'>},
  {"type": "image", "image": <class 'PIL.Image.Image'>},
  {"type": "image", "image": <class 'PIL.Image.Image'>},
  {"type": "image", "image": <class 'PIL.Image.Image'>},
]
```

This structure can be translated into code like this:

```python
import os
import zipfile
import io
from datasets import DatasetDict
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image

dataset_train_split = "test"

def format_data(samples: dict[str, any]) -> dict[str, list]:
    formatted_samples = {"messages": []}
    for cont in range(len(samples["question"])):
        images = []
        for img_path in samples["input_image_path"][cont]:
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append({"type": "image", "image": image})
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

        formatted_samples["messages"].append(
            [
                {"role": "system", "content": [{"type": "text", "text": samples["context"][cont]}]},
                {"role": "user", "content": images + [{"type": "text", "text": samples["question"][cont]}]},
                {"role": "assistant", "content": [{"type": "text", "text": samples["output"][cont]}]},
            ]
        )
    return formatted_samples

# For multi-image example
def prepare_dataset(dataset: DatasetDict, dataset_name: str, dataset_train_split: str) -> DatasetDict:
    all_files = list_repo_files(dataset_name, repo_type="dataset")
    zip_files = [f for f in all_files if f.endswith(".zip")]

    for zip_filename in zip_files:
        zip_path = hf_hub_download(repo_id=dataset_name, filename=zip_filename, repo_type="dataset")
        extract_folder = zip_filename.replace(".zip", "")
        os.makedirs(extract_folder, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

    dataset = dataset.map(format_data, batched=True, batch_size=4, num_proc=16)
    return dataset

dataset = prepare_dataset(dataset, dataset_name, dataset_train_split)
```

With this, your **Multi-Image + Text** dataset is now prepared for training.

### **Preparing for Training**  

We start by loading the model and processor. In this example, we use `google/gemma-3-4b-it`, but the same process applies to its other variants and similar models.  

To optimize memory usage, we configure `BitsAndBytes` to load the quantized version of the model.

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

model_id = "google/gemma-3-4b-it"

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

# Load model and tokenizer
model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    attn_implementation="eager", # Important (Ref: https://github.com/huggingface/transformers/blob/c15a7adb283fa984a40558c7fe7bed30ae975cdd/src/transformers/models/gemma3/modeling_gemma3.py#L934)
    quantization_config=bnb_config
)
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = "right"
```

Next, we set up [Quantized Low-Rank Adaptation (QLoRA)](https://huggingface.co/papers/2305.14314), an efficient fine-tuning technique for Large Language Models (LLMs) and Vision-Language Models (VLMs).  

```python
from peft import LoraConfig, get_peft_model

# Configure QLoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)
```

With QLoRA now set up, we need to define the training arguments for SFT. The [`SFTConfig`] class simplifies this process, providing an easy way to adjust parameters based on our specific needs.  

```python
from trl import SFTConfig

training_args = SFTConfig(
    output_dir="gemma-3-4b-it-trl-sft-llava-instruct-mix-vsft",     # Directory to save the model and push to the Hub. Use a specific repository id (e.g., gemma-3-4b-it-trl-sft-MMIU-Benchmark for multi-image datasets).
    num_train_epochs=1,                                             # Set the number of epochs to train the model.
    per_device_train_batch_size=8,                                  # Batch size for each device (e.g., GPU) during training. multi-image -> per_device_train_batch_size=1
    gradient_accumulation_steps=4,                                  # Number of steps before performing a backward/update pass to accumulate gradients. multi-image -> gradient_accumulation_steps=1
    gradient_checkpointing=True,                                    # Enable gradient checkpointing to reduce memory usage during training.
    optim="adamw_torch_fused",                                      # Use the fused AdamW optimizer for better performance.
    logging_steps=10,                                               # Frequency of logging training progress (log every 10 steps).
    save_strategy="epoch",                                          # Save checkpoints at the end of each epoch.
    learning_rate=2e-05,                                            # Learning rate for training.
    bf16=True,                                                      # Enable bfloat16 precision for training to save memory and speed up computations.
    push_to_hub=True,                                               # Automatically push the fine-tuned model to Hugging Face Hub after training.
    report_to="tensorboard",                                        # Automatically report metrics to tensorboard.
    gradient_checkpointing_kwargs={"use_reentrant": False},         # Set gradient checkpointing to non-reentrant to avoid issues.
    dataset_kwargs={"skip_prepare_dataset": True},                  # Skip dataset preparation to handle preprocessing manually.
    remove_unused_columns=False,                                    # Ensure unused columns are not removed in the collator (important for batch processing).
)
```

The `collate_fn` is responsible for processing and preparing individual examples to form a batch.  

Each example in the batch undergoes the following steps:  

1. The **chat template** is applied to the text.  
2. The **processor tokenizes** both `texts` and `images`, encoding them into tensors.  
3. The **labels** for training are set as the `input_ids` of the example.  
4. Certain **special tokens** are **masked (ignored)** during loss computation:  
   - `pad_token_id`  
   - `<image_token_id>`  
   - `<image_soft_token>` (corresponding to ID `262144`)  

This process is similar across different dataset types, with a minor variation in how images are handled:  

- **Single Image + Text** → A **list of images** is directly processed.  
- **Multi-Image + Text** → A **list of lists of images** is used, where each batch element contains multiple images.  

```python
from PIL import Image

# For multi-image cases
def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                if image is not None:
                    image = Image.open(io.BytesIO(image["bytes"]))
                    image_inputs.append(image.convert("RGB"))
    return image_inputs

def collate_fn(examples):
    texts = [processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False).strip() for example in examples]
    if "images" in examples[0]:  # single-image
        images = [
            [img.convert("RGB") for img in example["images"]]
            for example in examples
        ]
    else:  # multi-image
        images = [process_vision_info(example["messages"]) for example in examples]

    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=images, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    # Mask image tokens
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.special_tokens_map["boi_token"])
    ]
    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch  # Return the prepared batch
```

### **Training the Model**  

With all the components set up, we now configure the `SFTTrainer` using the previously defined settings and start the training process.

``` python
# Training
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=dataset["train"], # multi-image -> train_dataset=dataset["test"],
    processing_class=processor,
    peft_config=peft_config,
)

trainer.train()

# Save the final model
trainer.save_model()
```

We save the fine-tuned model to the Hub, making it easily accessible for future use. Additionally, TRL automatically logs the training results to **Weights & Biases (Wandb)** or **TensorBoard**, depending on the chosen configuration.  

<!-- Add Wandb training results -->
### Results

During and after trainig, we can inspect the results using **Weights & Biases (Wandb)** or **TensorBoard**. For example:

* [**gemma-3-4b-it-trl-sft-llava-instruct-mix-vsft (Single Image+Text)**](https://huggingface.co/sergiopaniego/gemma-3-4b-it-trl-sft-llava-instruct-mix-vsft)

* [**gemma-3-4b-it-trl-sft-MMIU-Benchmark (Multi-Images+Text or Interleaving)**](https://huggingface.co/sergiopaniego/gemma-3-4b-it-trl-sft-MMIU-Benchmark)

## Limitations  

Currently, fine-tuning Gemma has some [known limitations](https://github.com/huggingface/trl/issues/3121). We recommend following the procedure outlined in this guide to ensure the best results.  

## References  

For further reading and complementary resources, check out the following:  

- [Fine-Tuning Vision-Language Models with QLoRA](https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora)  
- [Fine-Tuning a Vision Language Model (Qwen2-VL-7B) with the Hugging Face Ecosystem (TRL)](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl)  

