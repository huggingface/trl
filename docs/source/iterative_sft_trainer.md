# Iterative Trainer

[![](https://img.shields.io/badge/All_models-Iterative_SFT-blue)](https://huggingface.co/models?other=iterative-sft,trl)


Iterative fine-tuning is a training method that enables to perform custom actions (generation and filtering for example) between optimization steps. In TRL we provide an easy-to-use API to fine-tune your models in an iterative way in just a few lines of code.

## Usage

To get started quickly, instantiate an instance a model, and a tokenizer.

```python

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

trainer = IterativeSFTTrainer(
    model,
    tokenizer
)

```

You have the choice to either provide a list of strings or a list of tensors to the step function. 

#### Using a list of tensors as input:

```python

inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask
}

trainer.step(**inputs)

```

#### Using a list of strings as input:

```python

inputs = {
    "texts": texts
}

trainer.step(**inputs)

```

For causal language models, labels will automatically be created from input_ids or from texts. When using sequence to sequence models you will have to provide your own labels or text_labels.

## IterativeTrainer

[[autodoc]] IterativeSFTTrainer
