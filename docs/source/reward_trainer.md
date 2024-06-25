# Reward Modeling

TRL supports custom reward modeling for anyone to perform reward modeling on their dataset and model.


## Expected dataset format

The [`RewardTrainer`] expects a very specific format for the dataset since the model will be trained on pairs of examples to predict which of the two is preferred. We usually preprocess the dataset to have `chosen` and `rejected` examples using [Hugging Face's chat template](https://huggingface.co/docs/transformers/main/en/chat_templating). Here are some examples:

* [trl-internal-testing/sentiment-trl-style](https://huggingface.co/datasets/trl-internal-testing/sentiment-trl-style)
* [trl-internal-testing/descriptiveness-trl-style](https://huggingface.co/datasets/trl-internal-testing/descriptiveness-trl-style)
* [trl-internal-testing/tldr-preference-trl-style](https://huggingface.co/datasets/trl-internal-testing/tldr-preference-trl-style)
* [trl-internal-testing/hh-rlhf-trl-style](https://huggingface.co/datasets/trl-internal-testing/hh-rlhf-trl-style)
* [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)


The dataset format roughly looks like the following:

<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/hh-rlhf-helpful-base-trl-style%20.png?download=true", width="80%">
</div>




The script to create these datasets can be found at [https://github.com/huggingface/trl/tree/main/examples/datasets](https://github.com/huggingface/trl/tree/main/examples/datasets)



## Get started

Here is a command to train a simple reward model on the sentiment dataset taken from [https://arxiv.org/abs/1909.08593](https://arxiv.org/abs/1909.08593) using the `pythia-1b-deduped` model. The training should take for a few minutes on a single GPU.

```bash
python examples/scripts/rm/rm.py \
    --dataset_name trl-internal-testing/sentiment-trl-style \
    --dataset_train_split train \
    --dataset_eval_split test \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --chat_template simple_concat \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --eval_strategy steps \
    --max_token_length 1024 \
    --max_prompt_token_lenth 1024 \
    --remove_unused_columns False \
    --num_train_epochs 1 \
    --eval_steps=100 \
    --output_dir models/minimal/rm_sentiment_1b \
    --push_to_hub
```


## How does it work?

You can use the [`RewardTrainer`] in the same way as the `Trainer` class from 🤗 Transformers.
You should pass an `AutoModelForSequenceClassification` model to the [`RewardTrainer`], along with a [`RewardConfig`] which configures the hyperparameters of the training.


### Tokenization

Ultimately, the `RewardTrainer` takes tokenized data as inputs: the post-processed dataset object should contain the following entries

-   `input_ids_chosen`
-   `attention_mask_chosen`
-   `input_ids_rejected`
-   `attention_mask_rejected`


To make data processing easier, we typically deal with chat-style dataset. For example, a chosen or rejected chat message would look like the following `chat` variable. We can set the `tokenizer.chat_template` to a template string that will be used to tokenize the chat, then call `tokenizer.apply_chat_template` to tokenize the chat, like demonstrated below:

```python
from transformers import AutoTokenizer
chat = [
    {"content": "How can I store food if I don't have a pantry?", "role": "user"},
    {
        "content": "You could store the food in a refrigerator, the top cupboards in your kitchen, the freezer, or even in a hole in the ground.",
        "role": "assistant",
    },
]
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.chat_template = "{% for message in messages %}{{'\n\n' if not loop.first else ''}}{{message['role']|capitalize + ': ' +message['content']}}{% endfor %}{{eos_token}}"
print(tokenizer.apply_chat_template(chat, tokenize=False))
# User: How can I store food if I don't have a pantry?
#
# Assistant: You could store the food in a refrigerator, the top cupboards in your kitchen, the freezer, or even in a hole in the ground.<|endoftext|>
```

To make sure the tokenization process is as transparent as possible. We provide a `DatasetProcessor` class that can be used to tokenize the dataset and visualize the tokenization process. Here is an example of how to use it:


```python
from transformers import AutoTokenizer
from datasets import load_dataset
from trl.dataset_processor import (
    CHAT_TEMPLATES,
    INPUT_IDS_CHOSEN_KEY,
    DatasetConfig,
    PreferenceDatasetProcessor,
    visualize_token,
)
dataset_config = DatasetConfig(
    dataset_name="trl-internal-testing/sentiment-trl-style",
    chat_template="simple_chat",
    max_token_length=1024,
    max_prompt_token_lenth=1024,
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.chat_template = CHAT_TEMPLATES["simple_chat"]
dataset = load_dataset(dataset_config.dataset_name)
dataset_processor = PreferenceDatasetProcessor(tokenizer=tokenizer, config=dataset_config)
dataset_processor.sanity_check_(dataset)
dataset = dataset_processor.tokenize(dataset)
dataset = dataset_processor.filter(dataset)
dataset_processor.get_token_length_visualization(dataset, save_path="tmp.png")
train_dataset = dataset[dataset_config.dataset_train_split]
eval_dataset = dataset[dataset_config.dataset_eval_split]
visualize_token(train_dataset[0][INPUT_IDS_CHOSEN_KEY], tokenizer)
```


The `visualize_token` will output the following colored tokens:

<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/Visualize token.png?download=true", width="80%">
</div>

The `dataset_processor.get_token_length_visualization` will output the visualization on the token length for the `chosen`, `rejected` and `prompt` in `tmp.png`.
<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/token_length_visualization.png?download=true", width="80%">
</div>



## Explanation of logged metrics

The logged metrics are as follows. Here is an example [tracked run at Weights and Biases](https://wandb.ai/huggingface/trl/runs/7m41xyvm):



* `eval/accuracy`: The accuracy of the model on the evaluation dataset.
* `eval/loss`: The loss of the model on a batch of the evaluation dataset.
* `eval/runtime`: The runtime of the evaluation.
* `eval/samples_per_second`: The number of samples processed per second during evaluation.
* `eval/steps_per_second`: The number of steps processed per second during evaluation.
* `total_flos`: The total number of floating operations done by the model since the beginning of training.
* `train/epoch`: The epoch number throughout training (e.g., 3.3 means 3 epochs and 30% of the 4th epoch).
* `train/global_step`: The step number throughout training.
* `train/grad_norm`: The norm of the gradients.
* `train/learning_rate`: The learning rate used during training.
* `train/loss`: The loss of the model on a batch of the training dataset.


## What is my model doing exactly?


To help you understand what your model is doing, we periodically log some the produced reward logits of the trained model. Here is an example of a completion. In an [example tracked run at Weights and Biases](https://wandb.ai/huggingface/trl/runs/7m41xyvm), it looks like the following, allowing you to see the model’s reward logits at different stages of training. You can tune the frequency of the logging with the `--eval_strategy steps` and `--eval_steps=100` options.


<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/rewardtrainer.gif?download=true", width="80%">
</div>



## Cookbook

* Debugging TIP: `eval/accuracy`: this is the ultimate evaluation of the reward model training. Ideally it should keep going up.
* Debugging TIP: `train/loss`: this is the loss of the reward model's objective, and it should keep going down.
* Debugging TIP: Always look into the dataset and the output reward logits which are logged in weights and biases to understand what the model is doing.
* Memory TIP: If you are running out of memory, you can try to reduce the `--per_device_train_batch_size` or increase the `--gradient_accumulation_steps` to reduce the memory footprint.
* Memory TIP: If you have multiple GPUs, you can also run training with DeepSpeed stage 3 to reduce the memory footprint `accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml`.
* Usage TIP: Make sure you understand the dataset by looking the tokenized inputs: do something like `print(train_dataset[0][INPUT_IDS_CHOSEN_KEY])` and `print(tokenizer.decode(train_dataset[0][INPUT_IDS_CHOSEN_KEY]))`. You should also see the token length distribution by running `dataset_processor.get_token_length_visualization`. Make sure nothing weird happens like the token length being too long or too short. You can customize by tweaking the `max_token_length=1024` and `max_prompt_token_lenth=1024` options.


## Leveraging 🤗 PEFT to train a reward model

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
    tokenizer=tokenizer,
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

## RewardConfig

[[autodoc]] RewardConfig

## RewardTrainer

[[autodoc]] RewardTrainer
