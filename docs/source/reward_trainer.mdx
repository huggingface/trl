# Reward Modeling

[![](https://img.shields.io/badge/All_models-Reward_Trainer-blue)](https://huggingface.co/models?other=reward-trainer,trl)

TRL supports custom reward modeling for anyone to perform reward modeling on their dataset and model.

Check out a complete flexible example at [`examples/scripts/reward_modeling.py`](https://github.com/huggingface/trl/tree/main/examples/scripts/reward_modeling.py).

## Expected dataset type

The [`RewardTrainer`] requires a [*implicit prompt* preference dataset](dataset_formats#preference). It means that the dataset should only contain the columns `"chosen"` and `"rejected"` (and not `"prompt"`).
The [`RewardTrainer`] supports both [conversational](dataset_formats#conversational) and [standard](dataset_formats#standard) dataset format. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

You can also use a pretokenized dataset, in which case the dataset should contain the following columns: `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`.

## Using the `RewardTrainer`

After preparing your dataset, you can use the [`RewardTrainer`] in the same way as the `Trainer` class from 🤗 Transformers.
You should pass an `AutoModelForSequenceClassification` model to the [`RewardTrainer`], along with a [`RewardConfig`] which configures the hyperparameters of the training.

### Leveraging 🤗 PEFT to train a reward model

Just pass a `peft_config` in the keyword arguments of [`RewardTrainer`], and the trainer should automatically take care of converting the model into a PEFT model!

```python
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

model = AutoModelForSequenceClassification.from_pretrained("gpt2")
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

...

trainer = RewardTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()

```

### Adding a margin to the loss

As in the [Llama 2 paper](https://huggingface.co/papers/2307.09288), you can add a margin to the loss by adding a `margin` column to the dataset. The reward collator will automatically pass it through and the loss will be computed accordingly.

```python
def add_margin(row):
    # Assume you have a score_chosen and score_rejected columns that you want to use to compute the margin
    return {'margin': row['score_chosen'] - row['score_rejected']}

dataset = dataset.map(add_margin)
```

### Centering rewards

In many scenarios, it's preferable to ensure that a reward model's output is mean zero. This is often done by first calculating the model's average score and then subtracting it.

[[Eisenstein et al., 2023]](https://huggingface.co/papers/2312.09244) proposed an auxiliary loss function designed to directly learn a centered reward model. This auxiliary loss minimizes the squared sum of the rewards, encouraging the model to naturally produce mean-zero outputs:

$$\Big( R(p, r_1) + R(p, r_2) \Big)^2 $$

This auxiliary loss is combined with the main loss function, weighted by the parameter `center_rewards_coefficient` in the `[RewardConfig]`. By default, this feature is deactivated (`center_rewards_coefficient = None`).

```python
training_args = RewardConfig(
    center_rewards_coefficient=0.01,
    ...
)
```

For reference results, please refer PR [#1932](https://github.com/huggingface/trl/pull/1932).

## RewardTrainer

[[autodoc]] RewardTrainer

## RewardConfig

[[autodoc]] RewardConfig
