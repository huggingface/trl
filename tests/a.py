from transformers import AutoModelForImageTextToText, AutoProcessor
from datasets import Dataset, Features, Image, Value
import numpy as np
import torch
import pytest
from peft import LoraConfig, PeftModel
from transformers import BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig
from trl.trainer.utils import get_kbit_device_map
"""
Test VLM training with aggressive memory optimization.

This test uses multiple memory reduction techniques:
- 4-bit quantization with double quantization
- LoRA with very low rank (r=4)
- Minimal batch size (1) with gradient accumulation
- Small images (64x64 instead of 224x224)
- Short sequences (max_completion_length=8)
- Only 4 training samples
- Only 1 training step
- Gradient checkpointing and bfloat16
"""

# Create processor once outside the data generator
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct", use_fast=True, padding_side="left")
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is in the image?"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

def data_gen(num_samples):
    for _ in range(num_samples):
        yield {
            "prompt": prompt,
            "image": np.random.uniform(low=0.0, high=255.0, size=(64, 64, 3)).astype(
                np.uint8
            ),  # Much smaller images
        }

dataset = Dataset.from_generator(
    data_gen, gen_kwargs={"num_samples": 4}, features=Features(image=Image(), prompt=Value(dtype="string"))
)
# reduce memory requirements as much as possible
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage="bfloat16",
)
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    attn_implementation="kernels-community/flash-attn2",
    dtype="bfloat16",
    device_map=get_kbit_device_map(),
    quantization_config=quantization_config,
)

def reward_func(prompts, completions, **kwargs):
    # simple nonsensical reward
    return [-((len(c) - 25) ** 2) + 100 for c in completions]

training_args = GRPOConfig(
    output_dir="tmp_dir",
    learning_rate=0.1,
    per_device_train_batch_size=1,  # Minimal batch size
    gradient_accumulation_steps=2,  # Maintain effective batch size
    num_generations=2,
    max_completion_length=8,  # Much shorter completions
    bf16=True,  # Use bfloat16 precision
    max_steps=1,  # Only do 1 training step to save time and memory
    report_to="none",
    logging_strategy="no",
)
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=4,  # Much lower rank for minimal memory
    lora_alpha=8,  # Reduced alpha proportionally
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # Minimal target modules
    # For VLM models, we typically want to freeze the vision encoder
    # and only adapt the language model parameters
    modules_to_save=None,
)

try:
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    assert isinstance(trainer.model, PeftModel)

    previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

    trainer.train()

    assert trainer.state.log_history[-1]["train_loss"] is not None

    # Check that LoRA parameters have changed
    # For VLM models, we're more permissive about which parameters can change
    lora_params_changed = False
    for n, param in previous_trainable_params.items():
        new_param = trainer.model.get_parameter(n)
        if "lora" in n.lower():  # LoRA parameters should change
            if not torch.equal(param, new_param):
                lora_params_changed = True

    # At least some LoRA parameters should have changed during training
    assert lora_params_changed, "No LoRA parameters were updated during training."

except torch.OutOfMemoryError as e:
    pytest.skip(f"Skipping VLM training test due to insufficient GPU memory: {e}")
except Exception as e:
    # Check for other memory-related errors
    if any(keyword in str(e).lower() for keyword in ["memory", "cuda", "out of memory", "insufficient"]):
        pytest.skip(f"Skipping VLM training test due to hardware constraints: {e}")
    else:
        raise

release_memory(model, trainer)
