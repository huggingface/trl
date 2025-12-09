# BEMA for Reference Model

This feature implements the BEMA algorithm to update the reference model during DPO training.

## Usage

```python
from trl.experimental.bema_for_ref_model import BEMACallback, DPOTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


pref_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
ref_model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")

bema_callback = BEMACallback(update_ref_model=True)

model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
tokenizer.pad_token = tokenizer.eos_token

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=pref_dataset,
    processing_class=tokenizer,
    callbacks=[bema_callback],
)

trainer.train()
```

## DPOTrainer

[[autodoc]] experimental.bema_for_ref_model.DPOTrainer
    - train
    - save_model
    - push_to_hub

## BEMACallback

[[autodoc]] experimental.bema_for_ref_model.BEMACallback
