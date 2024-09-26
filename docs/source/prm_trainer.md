# Process-supervised Reward Models (PRMs)

TRL supports custom reward modeling for anyone to perform reward modeling on their dataset and model.

Check out a complete flexible example at [`examples/scripts/prm_trainer.py`](https://github.com/huggingface/trl/tree/main/examples/scripts/prm_modeling.py).

## Expected dataset format



## Using the `PRMTrainer`

After preparing your dataset, you can use the [`PRMTrainer`] in the same way as the `Trainer` class from 🤗 Transformers.
You should pass an `AutoModelForTokenClassification` model to the [`PRMTrainer`], along with a [`PRMConfig`] which configures the hyperparameters of the training.

### Leveraging 🤗 PEFT to train a reward model

Just pass a `peft_config` in the keyword arguments of [`PRMTrainer`], and the trainer should automatically take care of converting the model into a PEFT model!

```python
from peft import LoraConfig, TaskType
from transformers import AutoModelForTokenClassification, AutoTokenizer
from trl import PRMTrainer, PRMConfig

model = AutoModelForTokenClassification.from_pretrained("gpt2")
peft_config = LoraConfig(
    task_type=None,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

...

trainer = PRMTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()

```

## PRMTrainer

[[autodoc]] PRMTrainer

## PRMConfig

[[autodoc]] PRMConfig
