# Unsloth Integration

Unsloth is an open‑source framework for fine‑tuning and reinforcement learning that trains LLMs (like Llama, OpenAI gpt-oss, Mistral, Gemma, DeepSeek, and more) up to 2× faster with up to 80% less VRAM. Unsloth allows [training](https://huggingface.co/docs/trl/en/unsloth_integration#Training), evaluation, running and [deployment](https://huggingface.co/docs/trl/en/unsloth_integration#Saving-the-model) with other inference engines like llama.cpp, Ollama and vLLM.

The library provides a streamlined, Hugging Face compatible workflow for training, evaluation, inference and deployment and is fully compatible with [`SFTTrainer`].

## Key Features

- Training support for all transformer compatible models: Text-to-speech (TTS), multimodal, BERT, RL and more
- Supports full fine-tuning, pretraining, LoRA, QLoRA, 8-bit training & more
- Works on Linux, Windows, Colab, Kaggle; NVIDIA GPUs, soon AMD & Intel setups
- Supports most features TRL supports, including RLHF (GSPO, GRPO, DPO etc.)
- Hand-written Triton kernels and a manual backprop engine ensure no accuracy degradation (0% approximation error)

## Installation

### pip install

Local Installation (Linux recommended):

```sh
pip install unsloth
```

You can also install `unsloth` according to the [official documentation](https://docs.unsloth.ai/get-started/installing-+-updating). Once installed, you can incorporate unsloth into your workflow in a very simple manner; instead of loading [`~transformers.AutoModelForCausalLM`], you just need to load a `FastLanguageModel` as follows:

```python
import torch
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

max_length = 2048 # Supports automatic RoPE Scaling, so choose any number

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b",
    max_seq_length=max_length,
    dtype="auto",  # For auto-detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit=True,  # Use 4bit quantization to reduce memory usage. Can be False
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Dropout = 0 is currently optimized
    bias="none",  # Bias = "none" is currently optimized
    use_gradient_checkpointing=True,
    random_state=3407,
)

training_args = SFTConfig(output_dir="./output", max_length=max_length)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

The saved model is fully compatible with Hugging Face's transformers library. Learn more about unsloth in their [official repository](https://github.com/unslothai/unsloth).

### Docker Install

```sh
docker run -d -e JUPYTER_PASSWORD="mypassword" \
  -p 8888:8888 -p 2222:22 \
  -v $(pwd)/work:/workspace/work \
  --gpus all \
  unsloth/unsloth
```

Access Jupyter Lab at ```http://localhost:8888``` and start fine-tuning!

## Training

These are some core settings you can toggle before training:

- ```max_seq_length = 2048``` – Controls context length. While Llama-3 supports 8192, we recommend 2048 for testing. Unsloth enables 4× longer context fine-tuning.
- ```dtype = "auto"``` – For auto-detection; use torch.float16 or torch.bfloat16 for newer GPUs.
- ```load_in_4bit = True``` – Enables 4-bit quantization, reducing memory use 4× for fine-tuning. Disabling it allows for LoRA 16-bit fine-tuning to be enabled.
- To enable full fine-tuning (FFT), set ```full_finetuning = True```. For 8-bit fine-tuning, set ```load_in_8bit = True```. Note: Only one training method can be set to True at a time.

For more information on configuring Unsloth's hyperparameters and features, read their [documentation guide here](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide).

## Saving the model

Unsloth allows you to directly save the finetuned model as a small file called a LoRA adapter. You can instead push to the Hugging Face hub as well if you want to upload your model! Remember to get a [Hugging Face token](https://huggingface.co/settings/tokens) and add your token!

### Saving to GGUF

To save to GGUF, Unsloth uses llama.cpp. To save locally:

```python
model.save_pretrained_gguf("directory", tokenizer, quantization_method = "q4_k_m")
model.save_pretrained_gguf("directory", tokenizer, quantization_method = "q8_0")
model.save_pretrained_gguf("directory", tokenizer, quantization_method = "f16")
```

To push to the hub:

```python
model.push_to_hub_gguf("hf_username/directory", tokenizer, quantization_method = "q4_k_m")
model.push_to_hub_gguf("hf_username/directory", tokenizer, quantization_method = "q8_0")
```

### Saving to vLLM

To save to 16-bit for vLLM, use:

```python
model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")
```
