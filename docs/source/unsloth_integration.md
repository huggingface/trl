# Unsloth Integration

<Tip warning={true}>

Section under construction. Feel free to contribute!

</Tip>

Unsloth is an open‑source framework for fine‑tuning and reinforcement learning that trains LLMs (like Llama, Mistral, Gemma, DeepSeek, and more) up to 2× faster with up to 70% less VRAM, while providing a streamlined, Hugging Face–compatible workflow for training, evaluation, and deployment.
Unsloth library that is fully compatible with [`SFTTrainer`]. Some benchmarks on 1 x A100 listed below:

| 1 A100 40GB     | Dataset   | 🤗   | 🤗 + Flash Attention 2 | 🦥 Unsloth | 🦥 VRAM saved |
| --------------- | --------- | --- | --------------------- | --------- | ------------ |
| Code Llama 34b  | Slim Orca | 1x  | 1.01x                 | **1.94x** | -22.7%       |
| Llama-2 7b      | Slim Orca | 1x  | 0.96x                 | **1.87x** | -39.3%       |
| Mistral 7b      | Slim Orca | 1x  | 1.17x                 | **1.88x** | -65.9%       |
| Tiny Llama 1.1b | Alpaca    | 1x  | 1.55x                 | **2.74x** | -57.8%       |

First, install `unsloth` according to the [official documentation](https://github.com/unslothai/unsloth). Once installed, you can incorporate unsloth into your workflow in a very simple manner; instead of loading [`~transformers.AutoModelForCausalLM`], you just need to load a `FastLanguageModel` as follows:

```python
import torch
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

max_length = 2048 # Supports automatic RoPE Scaling, so choose any number

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b",
    max_seq_length=max_length,
    dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit=True,  # Use 4bit quantization to reduce memory usage. Can be False
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
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
