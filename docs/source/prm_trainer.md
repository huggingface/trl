# PRM Trainer

[![](https://img.shields.io/badge/All_models-PRM-blue)](https://huggingface.co/models?other=prm,trl)

<Tip warning={true}>

PRM Trainer is an experimental API which is subject to change at any time.

</Tip>

## Overview

Process-supervised Reward Models (PRM) were proposed in [Solving math word problems with process- and outcome-based feedback](https://huggingface.co/papers/2211.14275) by Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins.

The abstract from the paper is the following:

> Recent work has shown that asking language models to generate reasoning steps improves performance on many reasoning tasks. When moving beyond prompting, this raises the question of how we should supervise such models: outcome-based approaches which supervise the final result, or process-based approaches which supervise the reasoning process itself? Differences between these approaches might naturally be expected not just in final-answer errors but also in reasoning errors, which can be difficult to detect and are problematic in many real-world domains such as education. We run the first comprehensive comparison between process- and outcome-based approaches trained on a natural language task, GSM8K. We find that pure outcome-based supervision produces similar final-answer error rates with less label supervision. However, for correct reasoning steps we find it necessary to use processbased supervision or supervision from learned reward models that emulate process-based feedback. In total, we improve the previous best results from 16.8% → 12.7% final-answer error and 14.0% → 3.4% reasoning error among final-answer-correct solutions.

This post-training method was contributed by [Gaetan Lopez](https://github.com/gaetanlop), [Lewis Tunstall](https://huggingface.co/lewtun), [Quentin Gallouédec](https://huggingface.co/qgallouedec) and [Agustín Piqueres](https://huggingface.co/plaguss).


## Quick start

This example demonstrates how to train a model using the PRM method. We use the [Qwen 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B) as the base model. We use the stepwise supervision data from the [Math Shepherd dataset](https://huggingface.co/datasets/trl-lib/math_shepherd). You can view the data in the dataset here:

<iframe
  src="https://huggingface.co/datasets/trl-lib/math_shepherd/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

Below is the script to train the model:

```python
# train_prm.py
from datasets import load_dataset
from trl import PRMConfig, PRMTrainer
from transformers import AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained("Qwen/Qwen2-0.5B", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
train_dataset = load_dataset("trl-lib/math_shepherd", split="train[:10%]")

training_args = PRMConfig(output_dir="Qwen2-0.5B-Reward-Math-Sheperd", logging_steps=10)
trainer = PRMTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
```

Execute the script using the following command:

```bash
accelerate launch train_prm.py
```

Distributed across 8 GPUs, the training takes approximately 1 hour.

To see how the [trained model](https://huggingface.co/trl-lib/Qwen2-0.5B-Reward-Math-Sheperd) performs, you can use the following script.


```python
from datasets import load_dataset
from transformers import pipeline

pipe = pipeline("token-classification", model="trl-lib/Qwen2-0.5B-Reward-Math-Sheperd")
dataset = load_dataset("trl-lib/math_shepherd")
example = {
    "prompt": "Musa is the class teacher of a class of 45 students. He wants to split them into three groups by age. If a third of the class is under 11 years, and two-fifths are above 11 but under 13, how many students will be in the third group (13 years and above)?",
    "completions": [
        "Step 1: A third of the class is under 11 years because 11 - 1/3 = <<11-1/3=7>>7.",
        "Step 2: Two-fifths of the class are above 11 but under 13 because 2/5 * 11 = <<2/5*11=8>>8.",
        "Step 3: There are 45 students, so the third group will have 45 - 7 - 8 = <<45-7-8=20>>20 students. The answer is: 20",
    ],
    "labels": [True, False, False],
}


separator = "\n"  # It's important to use the same separator as the one used during training

for idx in range(1, len(example["completions"]) + 1):
    steps = example["completions"][0:idx]
    text = separator.join((example["prompt"], *steps)) + separator  # Add a separator between the prompt and each steps
    pred_entity = pipe(text)[-1]["entity"]
    pred = {"LABEL_0": False, "LABEL_1": True}[pred_entity]
    label = example["labels"][idx - 1]
    print(f"Step {idx}\tPredicted: {pred} \tLabel: {label}")
```

```text
Step 1  Predicted: True         Label: True
Step 2  Predicted: False        Label: False
Step 3  Predicted: False        Label: False
```

It's a win!

## Expected dataset type

PRM requires a [stepwise supervision](dataset_formats#stepwise-supervision).
The dataset should contain the following columns: `prompt`, `completions` and `labels`, where `completions` contains a list of reasoning steps and `labels` a list of booleans or floats indicating the correctness of each step.

The [`PRMTrainer`] only supports [standard](dataset_formats#standard) dataset format.

## Example script

We provide an example script to train a model using the PRM method. The script is available in [`examples/scripts/prm.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/prm.py)

To use the PRM script with the [Qwen2 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B) on the [Math Shepherd dataset](https://huggingface.co/datasets/trl-lib/math_shepherd), run the following command:

```bash
accelerate launch examples/scripts/prm.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/math_shepherd \
    --num_train_epochs 1 \
    --logging_steps 25 \
    --output_dir Qwen2-0.5B-Reward-Math-Sheperd
```

## PRMTrainer

[[autodoc]] PRMTrainer

## PRMConfig

[[autodoc]] PRMConfig
