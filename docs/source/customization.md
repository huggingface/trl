# Training customization

TRL is designed with modularity in mind so that users to be able to efficiently customize the training loop for their needs. Below are some examples on how you can apply and test different techniques.  Note: Although these examples use the DPOTrainer, the customization applies to most (if not all) trainers.



## Use different optimizers and schedulers

By default, the `DPOTrainer` creates a `torch.optim.AdamW` optimizer. You can create and define a different optimizer and pass it to `DPOTrainer` as follows:

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import optim
from trl import DPOConfig, DPOTrainer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
training_args = DPOConfig(output_dir="Qwen2.5-0.5B-DPO")

optimizer = optim.SGD(model.parameters(), lr=training_args.learning_rate)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    optimizers=(optimizer, None),
)
trainer.train()
```

### Add a learning rate scheduler

You can also play with your training by adding learning rate schedulers.

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import optim
from trl import DPOConfig, DPOTrainer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
training_args = DPOConfig(output_dir="Qwen2.5-0.5B-DPO")

optimizer = optim.AdamW(model.parameters(), lr=training_args.learning_rate)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    optimizers=(optimizer, lr_scheduler),
)
trainer.train()
```

## Memory efficient fine-tuning by sharing layers

Another tool you can use for more memory efficient fine-tuning is to share layers between the reference model and the model you want to train.

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import create_reference_model, DPOConfig, DPOTrainer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ref_model = create_reference_model(model, num_shared_layers=6)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:1%]")
training_args = DPOConfig(output_dir="Qwen2.5-0.5B-DPO")

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

## Pass 8-bit reference models 
 
Since `trl` supports all keyword arguments when loading a model from `transformers` using `from_pretrained`, you can also leverage `load_in_8bit` from `transformers` for more memory efficient fine-tuning.

Read more about 8-bit model loading in `transformers` [here](https://huggingface.co/docs/transformers/en/peft#load-in-8bit-or-4bit).

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", quantization_config= quantization_config)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
training_args = DPOConfig(output_dir="Qwen2.5-0.5B-DPO")

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

## Use the accelerator cache optimizer

When training large models, you should better handle the accelerator cache by iteratively clearing it. To do so, simply pass `optimize_device_cache=True` to `DPOConfig`:

```python
training_args = DPOConfig(..., optimize_device_cache=True)
```
