# Reward Modeling

[![model badge](https://img.shields.io/badge/All_models-Reward_Trainer-blue)](https://huggingface.co/models?other=reward-trainer,trl)

## Overview

TRL supports the Outcome-supervised Reward Modeling (ORM) Trainer for training reward models.

This post-training method was contributed by [Younes Belkada](https://huggingface.co/ybelkada).

## Quick start

This example demonstrates how to train a reward model using the [`RewardTrainer`] from TRL. We train a [Qwen 3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model on the [UltraFeedback dataset](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized), large-scale, fine-grained, diverse preference dataset.

```python
from trl import RewardTrainer
from datasets import load_dataset

trainer = RewardTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
)
trainer.train()
```

<iframe src="https://trl-lib-trackio.hf.space/?project=trl-documentation&metrics=train*&sidebar=hidden&runs=reward_qwen3-0.6B_ultrafeedback2" style="width: 100%; min-width: 300px; max-width: 800px;" height="830" frameBorder="0"></iframe>

## Expected dataset type and format

[`RewardTrainer`] supports [preference](dataset_formats#preference) datasets type (both implicit and explicit prompt). The [`RewardTrainer`] is compatible with both [standard](dataset_formats#standard) and [conversational](dataset_formats#conversational) dataset formats. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

```python
# Standard preference (implicit prompt)
{"chosen": "The sky is blue.",
 "rejected": "The sky is green."}

# Conversational preference (implicit prompt)
{"chosen": [{"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "It is blue."}],
 "rejected": [{"role": "user", "content": "What color is the sky?"},
              {"role": "assistant", "content": "It is green."}]}

# Standard preference (explicit prompt)
{"prompt": "The sky is",
 "chosen": " blue.",
 "rejected": " green."}

# Conversational preference (explicit prompt)
{"prompt": [{"role": "user", "content": "What color is the sky?"}],
 "chosen": [{"role": "assistant", "content": "It is blue."}],
 "rejected": [{"role": "assistant", "content": "It is green."}]}
```

If your dataset is not in one of these formats, you can preprocess it to convert it into the expected format. Here is an example with the [lmarena-ai/arena-human-preference-55k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) dataset:

```python
from datasets import load_dataset
import json

dataset = load_dataset("lmarena-ai/arena-human-preference-55k")

# Filter out ties
dataset = dataset.filter(lambda example: example["winner_tie"] == 0)

# Create 'chosen' and 'rejected' fields based on the winner column
def response_a_b_to_chosen_rejected(example):
    if example["winner_model_a"] == 1:
        example["chosen"] = example["response_a"]
        example["rejected"] = example["response_b"]
    else:
        example["chosen"] = example["response_b"]
        example["rejected"] = example["response_a"]
    return example

dataset = dataset.map(response_a_b_to_chosen_rejected)

# Convert to conversational format
def make_conversation(example):
    prompt = json.loads(example["prompt"])[0]  # '["What color is the sky?"]' -> "What color is the sky?"
    chosen = json.loads(example["chosen"])[0]
    rejected = json.loads(example["rejected"])[0]
    return {
        "chosen": [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}],
        "rejected": [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}],
    }


dataset = dataset.map(make_conversation)

# Keep only necessary columns
dataset = dataset.select_columns(["chosen", "rejected"])

print(next(iter(dataset["train"])))
```

```json
{
    "chosen": [
        {"role": "user", "content": "Is it morally right to try to have a certain percentage of females on managerial positions?"},
        {"role": "assistant", "content": "The question of whether it is morally right to aim for a certain percentage of females..."},
    ],
    "rejected": [
        {"role": "user", "content": "Is it morally right to try to have a certain percentage of females on managerial positions?"},
        {"role": "assistant", "content": "As an AI, I don't have personal beliefs or opinions. However, ..."},
    ],
}
```

## Looking deeper into the training method

Reward Models (RMs) are typically trained using supervised learning on datasets containing pairs of preferred and non-preferred responses. The goal is to learn a function that assigns higher scores to preferred responses, enabling the model to rank outputs based on preferences.

This section breaks down how reward modeling works in practice, covering the key steps: **preprocessing** and **loss computation**.

### Preprocessing and tokenization

During training, each example is expected to contain a **chosen** and **rejected** field. For more details on the expected formats, see [Dataset formats - Preference](dataset_formats#preference).
The [`RewardTrainer`] tokenizes each input using the model's tokenizer. If prompts and completions (chosen and rejected) are provided separately (explicit prompt case), they are concatenated before tokenization.

### Computing the loss

Let  \\( x \\) be the input sequence (prompt) and  \\( y^+ \\) and  \\( y^- \\) be the chosen and rejected sequences respectively. Under the Bradley-Terry model ([Bradley & Terry, 1952](https://www.jstor.org/stable/2334029)), the probability that  \\( y^+ \\) is preferred over  \\( y^- \\) given a reward function  \\( r \\) is  \\( p(y^+ ≻ y^- |x) = \sigma(r(x, y^+)−r(x, y^-)) \\), where  \\( σ \\) is the sigmoid function.

The reward model  \\( r_\theta(x, y) \\) is trained to assign higher scores to preferred responses  \\( y^+ \\) over non-preferred ones  \\( y^- \\). The loss is then defined as the negative log-likelihood of the observed preferences:

$$
\mathcal{L}(\theta) = - \mathbb{E}_{(x,y^+,y^-) \sim \mathcal{D}} \left[ \log \sigma(r_\theta(x, y^+) - r_\theta(x, y^-)) \right].
$$

> [!TIP]
> The Bradley-Terry model is underdetermined, meaning that adding a constant to all rewards does not change the preference probabilities. To address this, [Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking](https://huggingface.co/papers/2312.09244) proposes adding an auxiliary loss term that encourages the rewards to be centered around zero. This is controlled by the `center_rewards_coefficient` parameter in the [`RewardConfig`]. The recommended value is `1e-2`.

## Logged metrics

While training and evaluating we record the following reward metrics:

* `global_step`: The total number of optimizer steps taken so far.
* `epoch`: The current epoch number, based on dataset iteration.
* `num_tokens`: The total number of tokens processed so far.
* `loss`: The average loss over the last logging interval.
* `accuracy`: The proportion of correct predictions (i.e., the model assigned a higher score to the chosen response than to the rejected one) averaged over the last logging interval.
* `min_reward`: The minimum reward score assigned by the model. This value is averaged over the logging interval.
* `mean_reward`: The average reward score assigned by the model over the last logging interval.
* `max_reward`: The maximum reward score assigned by the model. This value is averaged over the logging interval.
* `margin`: The average margin (difference between chosen and rejected rewards) over the last logging interval.
* `learning_rate`: The current learning rate, which may change dynamically if a scheduler is used.
* `grad_norm`: The L2 norm of the gradients, computed before gradient clipping.

## Customization

### Model initialization

You can directly pass the kwargs of the [`~transformers.AutoModelForSequenceClassification.from_pretrained()`] method to the [`RewardConfig`]. For example, if you want to load a model in a different precision, analogous to

```python
model = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
```

you can do so by passing the `model_init_kwargs={"dtype": torch.bfloat16}` argument to the [`RewardConfig`].

```python
from trl import RewardConfig

training_args = RewardConfig(
    model_init_kwargs={"dtype": torch.bfloat16},
)
```

Note that all keyword arguments of [`~transformers.AutoModelForSequenceClassification.from_pretrained()`] are supported, except for `num_labels`, which is automatically set to 1.

### Train adapters with PEFT

We support tight integration with 🤗 PEFT library, allowing any user to conveniently train adapters and share them on the Hub, rather than training the entire model.

```python
from datasets import load_dataset
from trl import RewardTrainer
from peft import LoraConfig

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = RewardTrainer(
    "Qwen/Qwen3-4B",
    train_dataset=dataset,
    peft_config=LoraConfig(modules_to_save=["score"])  # important to include the score head when base model is not a sequence classification model
)

trainer.train()
```

You can also continue training your [`~peft.PeftModel`]. For that, first load a `PeftModel` outside [`RewardTrainer`] and pass it directly to the trainer without the `peft_config` argument being passed.

```python
from datasets import load_dataset
from trl import RewardTrainer
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("trl-lib/Qwen3-4B-Reward-LoRA", is_trainable=True)
dataset = load_dataset("trl-lib/Capybara", split="train")

trainer = RewardTrainer(
    model=model,
    train_dataset=dataset,
)

trainer.train()
```

> [!TIP]
> When training adapters, you typically use a higher learning rate (≈1e‑3) since only new parameters are being learned.
>
> ```python
> RewardConfig(learning_rate=1e-3, ...)
> ```

## Tool Calling with Reward Modeling

The [`RewardTrainer`] fully supports fine-tuning models with _tool calling_ capabilities. In this case, each dataset example should include:

* The conversation messages, including any tool calls (`tool_calls`) and tool responses (`tool` role messages)
* The list of available tools in the `tools` column, typically provided as JSON `str` schemas

For details on the expected dataset structure, see the [Dataset Format — Tool Calling](dataset_formats#tool-calling) section.

## RewardTrainer

[[autodoc]] RewardTrainer
    - train
    - save_model
    - push_to_hub

## RewardConfig

[[autodoc]] RewardConfig

## DataCollatoForPreference

[[autodoc]] trainer.reward_trainer.DataCollatorForPreference
