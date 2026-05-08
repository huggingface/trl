# Training customization

TRL is designed with modularity in mind so that users are able to efficiently customize the training loop for their needs. Below are examples on how you can apply and test different techniques.

> [!NOTE]
> Although these examples use the [`DPOTrainer`], these customization methods apply to most (if not all) trainers in TRL.

## Use different optimizers and schedulers

By default, the [`DPOTrainer`] creates a `torch.optim.AdamW` optimizer. You can create and define a different optimizer and pass it to [`DPOTrainer`] as follows:

```python
from datasets import load_dataset
from torch import optim
from transformers import AutoModelForCausalLM
from trl import DPOTrainer

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
optimizer = optim.SGD(model.parameters(), lr=1e-6)

trainer = DPOTrainer(
    model=model,
    train_dataset=dataset,
    optimizers=(optimizer, None),
)
trainer.train()
```

### Add a learning rate scheduler

You can also add learning rate schedulers by passing both optimizer and scheduler:

```python
from torch import optim

optimizer = optim.AdamW(model.parameters(), lr=1e-6)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

trainer = DPOTrainer(..., optimizers=(optimizer, lr_scheduler))
```

## Pass 8-bit reference models

Since `trl` supports all keyword arguments when loading a model from `transformers` using `from_pretrained`, you can also leverage `load_in_8bit` from `transformers` for more memory efficient fine-tuning.

Read more about 8-bit model loading in `transformers` [Load in 8bit or 4bit](https://huggingface.co/docs/transformers/en/peft).

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", quantization_config=quantization_config)

trainer = DPOTrainer(..., ref_model=ref_model)
```

## Add custom callbacks

You can customize the training loop by adding callbacks for logging, monitoring, or early stopping. Callbacks allow you to execute custom code at specific points during training.

```python
from transformers import TrainerCallback


class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"Step {state.global_step}: {logs}")


trainer = DPOTrainer(..., callbacks=[CustomLoggingCallback()])
```

## Add custom evaluation metrics

You can define custom evaluation metrics to track during training. This is useful for monitoring model performance on specific tasks.

```python
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # Add your metric computation here
    return {"custom_metric": 0.0}


training_args = DPOConfig(..., eval_strategy="steps", eval_steps=100)

trainer = DPOTrainer(..., eval_dataset=eval_dataset, compute_metrics=compute_metrics)
```

## Use mixed precision training

Mixed precision training can significantly speed up training and reduce memory usage. You can enable it by setting `bf16=True` or `fp16=True` in the training config.

```python
# Use bfloat16 precision (recommended for modern GPUs)
training_args = DPOConfig(..., bf16=True)
```

Note: Use `bf16=True` for Ampere GPUs (A100, RTX 30xx) or newer, and `fp16=True` for older GPUs.

## Use gradient accumulation

When training with limited GPU memory, gradient accumulation allows you to simulate larger batch sizes by accumulating gradients over multiple steps before updating weights.

```python
# Simulate a batch size of 32 with per_device_train_batch_size=4 and gradient_accumulation_steps=8
training_args = DPOConfig(
    ...,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
)
```
