# KTO Trainer

[![](https://img.shields.io/badge/All_models-KTO-blue)](https://huggingface.co/models?other=kto,trl)

## Overview

Kahneman-Tversky Optimization (KTO) was introduced in [KTO: Model Alignment as Prospect Theoretic Optimization](https://huggingface.co/papers/2402.01306) by [Kawin Ethayarajh](https://huggingface.co/kawine), [Winnie Xu](https://huggingface.co/xwinxu), [Niklas Muennighoff](https://huggingface.co/Muennighoff), Dan Jurafsky, [Douwe Kiela](https://huggingface.co/douwekiela).


The abstract from the paper is the following:

> Kahneman & Tversky's prospect theory tells us that humans perceive random variables in a biased but well-defined manner; for example, humans are famously loss-averse. We show that objectives for aligning LLMs with human feedback implicitly incorporate many of these biases -- the success of these objectives (e.g., DPO) over cross-entropy minimization can partly be ascribed to them being human-aware loss functions (HALOs). However, the utility functions these methods attribute to humans still differ from those in the prospect theory literature. Using a Kahneman-Tversky model of human utility, we propose a HALO that directly maximizes the utility of generations instead of maximizing the log-likelihood of preferences, as current methods do. We call this approach Kahneman-Tversky Optimization (KTO), and it matches or exceeds the performance of preference-based methods at scales from 1B to 30B. Crucially, KTO does not need preferences -- only a binary signal of whether an output is desirable or undesirable for a given input. This makes it far easier to use in the real world, where preference data is scarce and expensive.

The official code can be found in [ContextualAI/HALOs](https://github.com/ContextualAI/HALOs).

This post-training method was contributed by [Kashif Rasul](https://huggingface.co/kashif), [Younes Belkada](https://huggingface.co/ybelkada), [Lewis Tunstall](https://huggingface.co/lewtun) and Pablo Vicente.

## Quick start

This example demonstrates how to train a model using the KTO method. We use the [Qwen 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) as the base model. We use the preference data from the [KTO Mix 14k](https://huggingface.co/datasets/trl-lib/kto-mix-14k). You can view the data in the dataset here:

<iframe
  src="https://huggingface.co/datasets/trl-lib/kto-mix-14k/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

Below is the script to train the model:

```python
# train_kto.py
from datasets import load_dataset
from trl import KTOConfig, KTOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/kto-mix-14k", split="train")

training_args = KTOConfig(output_dir="Qwen2-0.5B-KTO", logging_steps=10)
trainer = KTOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
```

Execute the script using the following command:

```bash
accelerate launch train_kto.py
```

Distributed across 8 x H100 GPUs, the training takes approximately 30 minutes. You can verify the training progress by checking the reward graph. An increasing trend in the reward margin indicates that the model is improving and generating better responses over time.

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/kto-qwen2-reward-margin.png)

To see how the [trained model](https://huggingface.co/trl-lib/Qwen2-0.5B-KTO) performs, you can use the [TRL Chat CLI](clis#chat-interface).

<pre><code>$ trl chat --model_name_or_path trl-lib/Qwen2-0.5B-KTO
<strong><span style="color: red;">&lt;quentin_gallouedec&gt;:</span></strong>
What is the best programming language?

<strong><span style="color: blue;">&lt;trl-lib/Qwen2-0.5B-KTO&gt;:</span></strong>
The best programming language can vary depending on individual preferences, industry-specific requirements, technical skills, and familiarity with the specific use case or task. Here are some widely-used programming languages that have been noted as popular and widely used:                                                                                  

Here are some other factors to consider when choosing a programming language for a project:

 <strong><span style="color: green;">1</span> JavaScript</strong>: JavaScript is at the heart of the web and can be used for building web applications, APIs, and interactive front-end applications like frameworks like React and Angular. It's similar to C, C++, and F# in syntax structure and is accessible and easy to learn, making it a popular choice for beginners and professionals alike.                                                                   
 <strong><span style="color: green;">2</span> Java</strong>: Known for its object-oriented programming (OOP) and support for Java 8 and .NET, Java is used for developing enterprise-level software applications, high-performance games, as well as mobile apps, game development, and desktop applications.                                                                                                                                                            
 <strong><span style="color: green;">3</span> C++</strong>: Known for its flexibility and scalability, C++ offers comprehensive object-oriented programming and is a popular choice for high-performance computing and other technical fields. It's a powerful platform for building real-world applications and games at scale.                                                                                                                                         
 <strong><span style="color: green;">4</span> Python</strong>: Developed by Guido van Rossum in 1991, Python is a high-level, interpreted, and dynamically typed language known for its simplicity, readability, and versatility.   
</code></pre>

## Expected dataset format

KTO requires an [unpaired preference dataset](dataset_formats#unpaired-preference). Alternatively, you can provide a *paired* preference dataset (also known simply as a *preference dataset*). In this case, the trainer will automatically convert it to an unpaired format by separating the chosen and rejected responses, assigning `label = True` to the chosen completions and `label = False` to the rejected ones.

The [`KTOTrainer`] supports both [conversational](dataset_formats#conversational) and [standard](dataset_formats#standard) dataset format. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

In theory, the dataset should contain at least one chosen and one rejected completion. However, some users have successfully run KTO using *only* chosen or only rejected data. If using only rejected data, it is advisable to adopt a conservative learning rate.

## Example script

We provide an example script to train a model using the KTO method. The script is available in [`trl/scripts/kto.py`](https://github.com/huggingface/trl/blob/main/trl/scripts/kto.py)

To test the KTO script with the [Qwen2 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) on the [UltraFeedback dataset](https://huggingface.co/datasets/trl-lib/kto-mix-14k), run the following command:

```bash
accelerate launch trl/scripts/kto.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/kto-mix-14k \
    --num_train_epochs 1 \
    --logging_steps 25 \
    --output_dir Qwen2-0.5B-KTO
```

## Usage tips

### For Mixture of Experts Models: Enabling the auxiliary loss

MOEs are the most efficient if the load is about equally distributed between experts.  
To ensure that we train MOEs similarly during preference-tuning, it is beneficial to add the auxiliary loss from the load balancer to the final loss.

This option is enabled by setting `output_router_logits=True` in the model config (e.g. [`~transformers.MixtralConfig`]).  
To scale how much the auxiliary loss contributes to the total loss, use the hyperparameter `router_aux_loss_coef=...` (default: `0.001`) in the model config.


### Batch size recommendations

Use a per-step batch size that is at least 4, and an effective batch size between 16 and 128. Even if your effective batch size is large, if your per-step batch size is poor, then the KL estimate in KTO will be poor.

### Learning rate recommendations

Each choice of `beta` has a maximum learning rate it can tolerate before learning performance degrades. For the default setting of `beta = 0.1`, the learning rate should typically not exceed `1e-6` for most models. As `beta` decreases, the learning rate should also be reduced accordingly. In general, we strongly recommend keeping the learning rate between `5e-7` and `5e-6`. Even with small datasets, we advise against using a learning rate outside this range. Instead, opt for more epochs to achieve better results.

### Imbalanced data

The `desirable_weight` and `undesirable_weight` of the [`KTOConfig`] refer to the weights placed on the losses for desirable/positive and undesirable/negative examples.
By default, they are both 1. However, if you have more of one or the other, then you should upweight the less common type such that the ratio of (`desirable_weight`  \\(\times\\) number of positives) to (`undesirable_weight`  \\(\times\\) number of negatives) is in the range 1:1 to 4:3.

## Logged metrics

While training and evaluating we record the following reward metrics:

- `rewards/chosen`: the mean log probabilities of the policy model for the chosen responses scaled by beta
- `rewards/rejected`: the mean log probabilities of the policy model for the rejected responses scaled by beta
- `rewards/margins`: the mean difference between the chosen and corresponding rejected rewards
- `logps/chosen`: the mean log probabilities of the chosen completions
- `logps/rejected`: the mean log probabilities of the rejected completions
- `logits/chosen`: the mean logits of the chosen completions
- `logits/rejected`: the mean logits of the rejected completions
- `kl`: the KL divergence between the policy model and the reference model

## KTOTrainer

[[autodoc]] KTOTrainer

## KTOConfig

[[autodoc]] KTOConfig
