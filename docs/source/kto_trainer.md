# KTO Trainer

[![All_models-KTO-blue](https://img.shields.io/badge/All_models-KTO-blue)](https://huggingface.co/models?other=kto,trl)

## Overview

TRL supports the Kahneman-Tversky Optimization (KTO) Trainer for training language models, as described in the paper [KTO: Model Alignment as Prospect Theoretic Optimization](https://huggingface.co/papers/2402.01306) by [Kawin Ethayarajh](https://huggingface.co/kawine), [Winnie Xu](https://huggingface.co/xwinxu), [Niklas Muennighoff](https://huggingface.co/Muennighoff), Dan Jurafsky, [Douwe Kiela](https://huggingface.co/douwekiela).

The abstract from the paper is the following:

> Kahneman & Tversky's prospect theory tells us that humans perceive random variables in a biased but well-defined manner; for example, humans are famously loss-averse. We show that objectives for aligning LLMs with human feedback implicitly incorporate many of these biases -- the success of these objectives (e.g., DPO) over cross-entropy minimization can partly be ascribed to them being human-aware loss functions (HALOs). However, the utility functions these methods attribute to humans still differ from those in the prospect theory literature. Using a Kahneman-Tversky model of human utility, we propose a HALO that directly maximizes the utility of generations instead of maximizing the log-likelihood of preferences, as current methods do. We call this approach Kahneman-Tversky Optimization (KTO), and it matches or exceeds the performance of preference-based methods at scales from 1B to 30B. Crucially, KTO does not need preferences -- only a binary signal of whether an output is desirable or undesirable for a given input. This makes it far easier to use in the real world, where preference data is scarce and expensive.

The official code can be found in [ContextualAI/HALOs](https://github.com/ContextualAI/HALOs).

This post-training method was contributed by [Kashif Rasul](https://huggingface.co/kashif), [Younes Belkada](https://huggingface.co/ybelkada), [Lewis Tunstall](https://huggingface.co/lewtun), Pablo Vicente, and later refactored by [Albert Villanova del Moral](https://huggingface.co/albertvillanova).

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

training_args = KTOConfig(output_dir="Qwen2-0.5B-KTO")
trainer = KTOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
```

Execute the script using the following command:

```bash
accelerate launch train_kto.py
```

Distributed across 8 x H100 GPUs, the training takes approximately 30 minutes. You can verify the training progress by checking the reward graph. An increasing trend in the reward margin indicates that the model is improving and generating better responses over time.

![kto qwen2 reward margin](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/kto-qwen2-reward-margin.png)

To see how the [trained model](https://huggingface.co/trl-lib/Qwen2-0.5B-KTO) performs, you can use the [Transformers Chat CLI](https://huggingface.co/docs/transformers/quicktour#chat-with-text-generation-models).

<pre><code>$ transformers chat trl-lib/Qwen2-0.5B-KTO
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

## Expected dataset type and format

KTO requires an [unpaired preference](dataset_formats#unpaired-preference) dataset. Alternatively, you can provide a *paired* preference dataset (also known simply as a *preference dataset*). In this case, the trainer will automatically convert it to an unpaired format by separating the chosen and rejected responses, assigning `label = True` to the chosen completions and `label = False` to the rejected ones.

The [`KTOTrainer`] is compatible with both [standard](dataset_formats#standard) and [conversational](dataset_formats#conversational) dataset formats. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

```python
# Standard format
unpaired_preference_example = {"prompt": "The sky is", "completion": " blue.", "label": True}

# Conversational format
unpaired_preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                               "completion": [{"role": "assistant", "content": "It is blue."}],
                               "label": True}
```

In theory, the dataset should contain at least one chosen and one rejected completion. However, some users have successfully run KTO using *only* chosen or only rejected data. If using only rejected data, it is advisable to adopt a conservative learning rate.

## Looking deeper into the KTO method

Kahneman-Tversky Optimization (KTO) is a training method designed to align a language model using only a binary signal of whether an output is *desirable* or *undesirable* for a given input, rather than pairs of preferred/dispreferred completions. Drawing on the prospect theory of Kahneman and Tversky, it defines a *human-aware loss* (HALO) that directly maximizes the utility of generations, weighting desirable and undesirable examples asymmetrically to reflect human loss aversion.

This section breaks down how KTO works in practice, covering the key steps: **preprocessing** and **loss computation**.

### Preprocessing and tokenization

During training, each example is expected to contain a prompt, a `completion`, and a boolean `label` indicating whether the completion is desirable (`True`) or undesirable (`False`). For more details on the expected formats, see [Dataset formats](dataset_formats).
The [`KTOTrainer`] tokenizes each input using the model's tokenizer.

### Computing the loss

The KL divergence term used by the KTO loss is estimated by pairing each prompt with a mismatched completion drawn from elsewhere in the batch. For this reason, loss types that estimate the KL term (all except `apo_zero_unpaired`) require a per-device train batch size greater than 1 and a sequential sampling strategy, so that the mismatched pairs are stable across a batch.

The loss used in KTO (Eq. 7 of the [paper](https://huggingface.co/papers/2402.01306)) is defined as follows:

$$
\mathcal{L}_{\mathrm{KTO}}(\theta) = \mathbb{E}_{(x,y)}\!\left[ w(y)\Big(1 - v(x, y)\Big) \right]
$$

where the value  \\( v(x, y) \\) is

$$
v(x, y) = \begin{cases}
\sigma\!\left(\beta\big(\log\frac{\pi_{\theta}(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} - \mathrm{KL}\big)\right) & \text{if } y \text{ is desirable} \\
\sigma\!\left(\beta\big(\mathrm{KL} - \log\frac{\pi_{\theta}(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}\big)\right) & \text{if } y \text{ is undesirable}
\end{cases}
$$

Here  \\( x \\) is the prompt,  \\( y \\) is the completion,  \\( \pi_{\theta} \\) is the policy model being trained,  \\( \pi_{\mathrm{ref}} \\) is the reference model,  \\( \sigma \\) is the sigmoid function,  \\( \beta > 0 \\) controls the deviation from the reference model, and  \\( \mathrm{KL} \\) is the estimated KL divergence term. The weight  \\( w(y) \\) is `desirable_weight` for desirable completions and `undesirable_weight` for undesirable ones, used to counter an imbalance between the number of desirable and undesirable examples.

#### Loss Types

| `loss_type=` | Description |
| --- | --- |
| `"kto"` (default) | The KTO loss from the [KTO](https://huggingface.co/papers/2402.01306) paper. A human-aware loss (HALO) based on Kahneman-Tversky prospect theory that maximizes the utility of desirable completions and minimizes it for undesirable ones, using an estimated KL divergence term as a reference point. |
| `"apo_zero_unpaired"` | The unpaired variant of the APO-zero loss from the [APO](https://huggingface.co/papers/2408.06266) paper. It increases the likelihood of desirable completions and decreases the likelihood of undesirable ones without estimating the KL divergence term. Use this loss when you believe the desirable completions are better than the model's default output. |

## Logged metrics

While training and evaluating, we record the following metrics:

* `global_step`: The total number of optimizer steps taken so far.
* `epoch`: The current epoch number, based on dataset iteration.
* `num_tokens`: The total number of tokens processed so far.
* `loss`: The average KTO loss over the current logging interval.
* `entropy`: The average entropy of the model's predicted token distribution over non-masked tokens.
* `kl`: The average estimated KL divergence between the policy and reference model, used as the reference point in the KTO loss.
* `learning_rate`: The current learning rate, which may change dynamically if a scheduler is used.
* `grad_norm`: The L2 norm of the gradients, computed before gradient clipping.
* `logits/chosen`: The average logit values assigned by the model to the tokens in the chosen (desirable) completion.
* `logits/rejected`: The average logit values assigned by the model to the tokens in the rejected (undesirable) completion.
* `logps/chosen`: The average log-probability assigned by the model to the tokens in the chosen (desirable) completion.
* `logps/rejected`: The average log-probability assigned by the model to the tokens in the rejected (undesirable) completion.
* `rewards/chosen`: The average implicit reward computed for the chosen (desirable) completion, computed as  \\( \beta \log \frac{\pi_{\theta}(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} \\).
* `rewards/rejected`: The average implicit reward computed for the rejected (undesirable) completion, computed as  \\( \beta \log \frac{\pi_{\theta}(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} \\).
* `rewards/margins`: The average implicit reward margin between the chosen and rejected completions.

## Customization

### Compatibility and constraints

Some argument combinations are intentionally restricted in the current [`KTOTrainer`] implementation:

* With `use_liger_kernel=True`:
  * only `loss_type="kto"` is supported (not `"apo_zero_unpaired"`),
  * `compute_metrics` is not supported,
  * `precompute_ref_log_probs=True` is not supported,
  * PEFT models are not supported.
* `sync_ref_model=True` is not supported when training with PEFT models that do not keep a standalone `ref_model`.
* `sync_ref_model=True` cannot be combined with `precompute_ref_log_probs=True`.
* `precompute_ref_log_probs=True` is not supported with `IterableDataset` (train or eval) or with vision datasets.
* Loss types that estimate the KL divergence term (all except `"apo_zero_unpaired"`) require `train_sampling_strategy="sequential"` and a per-device train batch size greater than 1.

### Model initialization

You can directly pass the kwargs of the [`~transformers.AutoModelForCausalLM.from_pretrained()`] method to the [`KTOConfig`]. For example, if you want to load a model in a different precision, analogous to

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", dtype=torch.bfloat16)
```

you can do so by passing the `model_init_kwargs={"dtype": torch.bfloat16}` argument to the [`KTOConfig`].

```python
from trl import KTOConfig

training_args = KTOConfig(
    model_init_kwargs={"dtype": torch.bfloat16},
)
```

Note that all keyword arguments of [`~transformers.AutoModelForCausalLM.from_pretrained()`] are supported.

### Train adapters with PEFT

We support tight integration with 🤗 PEFT library, allowing any user to conveniently train adapters and share them on the Hub, rather than training the entire model.

```python
from datasets import load_dataset
from trl import KTOTrainer
from peft import LoraConfig

dataset = load_dataset("trl-lib/kto-mix-14k", split="train")

trainer = KTOTrainer(
    "Qwen/Qwen2-0.5B-Instruct",
    train_dataset=dataset,
    peft_config=LoraConfig(),
)

trainer.train()
```

You can also continue training your [`~peft.PeftModel`]. For that, first load a `PeftModel` outside [`KTOTrainer`] and pass it directly to the trainer without the `peft_config` argument being passed.

> [!TIP]
> When training adapters, you typically use a higher learning rate than full fine-tuning since only new parameters are being learned.

### Train with Liger Kernel

Liger Kernel is a collection of Triton kernels for LLM training that boosts multi-GPU throughput by 20%, cuts memory use by 60% (enabling up to 4× longer context), and works seamlessly with tools like FlashAttention, PyTorch FSDP, and DeepSpeed. For more information, see [Liger Kernel Integration](liger_kernel_integration).

## Tool Calling with KTO

The [`KTOTrainer`] fully supports fine-tuning models with _tool calling_ capabilities. In this case, each dataset example should include:

* The conversation messages (prompt and completion), including any tool calls (`tool_calls`) and tool responses (`tool` role messages)
* The list of available tools in the `tools` column, typically provided as JSON schemas

For details on the expected dataset structure, see the [Dataset Format — Tool Calling](dataset_formats#tool-calling) section.

## Training Vision Language Models

[`KTOTrainer`] fully supports training Vision-Language Models (VLMs). To train a VLM, provide a dataset with either an `image` column (single image per sample) or an `images` column (list of images per sample). For more information on the expected dataset structure, see the [Dataset Format — Vision Dataset](dataset_formats#vision-dataset) section.

```python
from trl import KTOConfig, KTOTrainer
from datasets import load_dataset

trainer = KTOTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    args=KTOConfig(max_length=None),
    train_dataset=load_dataset("trl-internal-testing/zen-image", "conversational_unpaired_preference", split="train"),
)
trainer.train()
```

> [!TIP]
> For VLMs, truncating may remove image tokens, leading to errors during training. To avoid this, set `max_length=None` in the [`KTOConfig`]. This allows the model to process the full sequence length without truncating image tokens.
>
> ```python
> KTOConfig(max_length=None, ...)
> ```
>
> Only use `max_length` when you've verified that truncation won't remove image tokens for the entire dataset.

## Example script

We provide an example script to train a model using the KTO method. The script is available in [`trl/scripts/kto.py`](https://github.com/huggingface/trl/blob/main/trl/scripts/kto.py)

To test the KTO script with the [Qwen2 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) on the [UltraFeedback dataset](https://huggingface.co/datasets/trl-lib/kto-mix-14k), run the following command:

```bash
accelerate launch trl/scripts/kto.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/kto-mix-14k \
    --num_train_epochs 1 \
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

## KTOTrainer

[[autodoc]] KTOTrainer
    - train
    - save_model
    - push_to_hub

## KTOConfig

[[autodoc]] KTOConfig
