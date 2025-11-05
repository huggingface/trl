# Merge Model Callback

This feature allows merging the policy model (the model being trained) with another model during or after training using [mergekit](https://github.com/arcee-ai/mergekit). This can be useful for techniques like model interpolation, task arithmetic, or other model merging strategies.

> **Note**: This is an experimental feature. The Mergekit integration has seen low usage and may be removed in future versions.

## Usage

```python
from trl.experimental.mergekit import MergeConfig, MergeModelCallback
from trl import DPOTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your dataset and model
pref_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")

# Configure the merge
config = MergeConfig()

# Create the callback
merge_callback = MergeModelCallback(
    merge_config=config,
    merge_at_every_checkpoint=False,  # Merge only at the end of training
    push_to_hub=False,  # Optionally push merged model to Hub
)

# Use with any trainer
trainer = DPOTrainer(
    model=model,
    train_dataset=pref_dataset,
    processing_class=tokenizer,
    callbacks=[merge_callback],
)

trainer.train()
```

## Installation

The mergekit integration requires the `mergekit` package:

```bash
pip install mergekit
```

## API Reference

[[autodoc]] experimental.mergekit.MergeModelCallback
