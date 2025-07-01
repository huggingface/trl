# Iterative Trainer

[![](https://img.shields.io/badge/All_models-Iterative_SFT-blue)](https://huggingface.co/models?other=iterative-sft,trl)

Iterative fine-tuning is a training method that enables to perform custom actions (generation and filtering for example) between optimization steps. In TRL we provide an easy-to-use API to fine-tune your models in an iterative way in just a few lines of code.

## Quickstart

To get started quickly, you can either pass a model identifier or a pre-instantiated model to the trainer:

```python
from trl import IterativeSFTConfig, IterativeSFTTrainer

# Using a model identifier
trainer = IterativeSFTTrainer(
    "facebook/opt-350m",
    args=IterativeSFTConfig(
        max_length=512,
        output_dir="./output",
    ),
)

# Or using a pre-instantiated model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

trainer = IterativeSFTTrainer(
    model,
    args=IterativeSFTConfig(
        max_length=512,
        output_dir="./output",
    ),
    processing_class=tokenizer,
)
```

## Usage

The [`IterativeSFTTrainer`] supports two ways of providing input data to the `step` function:

### Using a list of tensors as input:

```python
inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
}

trainer.step(**inputs)
```

### Using a list of strings as input:

```python
inputs = {
    "texts": texts,
    "texts_labels": texts_labels,  # Optional, defaults to texts
}

trainer.step(**inputs)
```

For causal language models, labels will automatically be created from `input_ids` or from `texts`. When using sequence to sequence models you will have to provide your own labels or `text_labels`.

## Configuration

The [`IterativeSFTConfig`] class provides several parameters to customize the training:

```python
from trl import IterativeSFTConfig

config = IterativeSFTConfig(
    # Model initialization parameters
    model_init_kwargs={"torch_dtype": "bfloat16"},

    # Data preprocessing parameters
    max_length=512,
    truncation_mode="keep_end",

    # Training parameters
    output_dir="./output",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=1000,
    save_steps=100,
    optim="adamw_torch",
    report_to="wandb",
)
```

### Model Initialization

You can control how the model is initialized by passing keyword arguments to `model_init_kwargs`:

```python
config = IterativeSFTConfig(
    model_init_kwargs={
        "torch_dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": True,
    }
)
```

### Data Preprocessing

The trainer supports two truncation modes:

- `keep_end`: Truncates from the start of the sequence
- `keep_start`: Truncates from the end of the sequence

```python
config = IterativeSFTConfig(
    max_length=512,
    truncation_mode="keep_end",  # or "keep_start"
)
```

### Training Optimization

You can optimize CUDA cache usage for more memory-efficient training:

```python
config = IterativeSFTConfig(
    optimize_device_cache=True,
)
```

## IterativeSFTTrainer

[[autodoc]] IterativeSFTTrainer

## IterativeSFTConfig

[[autodoc]] IterativeSFTConfig
