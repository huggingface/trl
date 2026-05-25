# CPO Trainer

[![All_models-CPO-blue](https://img.shields.io/badge/All_models-CPO-blue)](https://huggingface.co/models?other=cpo,trl)

## Overview

TRL supports the Contrastive Preference Optimization (CPO) Trainer for training language models, as described in the paper [Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation](https://huggingface.co/papers/2401.08417) by [Haoran Xu](https://huggingface.co/haoranxu), [Amr Sharaf](https://huggingface.co/amrsharaf), [Yunmo Chen](https://huggingface.co/yunmochen), Weiting Tan, Lingfeng Shen, Benjamin Van Durme, [Kenton Murray](https://huggingface.co/Kenton), and [Young Jin Kim](https://huggingface.co/ykim362).

The abstract from the paper is the following:

> Moderate-sized large language models (LLMs) -- those with 7B or 13B parameters -- exhibit promising machine translation (MT) performance. However, even the top-performing 13B LLM-based translation models, like ALMA, does not match the performance of state-of-the-art conventional encoder-decoder translation models or larger-scale LLMs such as GPT-4. In this study, we bridge this performance gap. We first assess the shortcomings of supervised fine-tuning for LLMs in the MT task, emphasizing the quality issues present in the reference data, despite being human-generated. Then, in contrast to SFT which mimics reference translations, we introduce Contrastive Preference Optimization (CPO), a novel approach that trains models to avoid generating adequate but not perfect translations. Applying CPO to ALMA models with only 22K parallel sentences and 12M parameters yields significant improvements. The resulting model, called ALMA-R, can match or exceed the performance of the WMT competition winners and GPT-4 on WMT'21, WMT'22 and WMT'23 test datasets.

## Quick start

This example demonstrates how to train a language model using the [`experimental.cpo.CPOTrainer`] from TRL. We train a [Qwen 3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model on the [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback).

```python
from trl.experimental.cpo import CPOTrainer
from datasets import load_dataset

trainer = CPOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
)
trainer.train()
```

## Expected dataset type and format

CPO requires a [preference](dataset_formats#preference) dataset. The [`experimental.cpo.CPOTrainer`] is compatible with both [standard](dataset_formats#standard) and [conversational](dataset_formats#conversational) dataset formats. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

```python
# Standard format
## Explicit prompt (recommended)
preference_example = {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}
# Implicit prompt
preference_example = {"chosen": "The sky is blue.", "rejected": "The sky is green."}

# Conversational format
## Explicit prompt (recommended)
preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                      "chosen": [{"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "assistant", "content": "It is green."}]}
## Implicit prompt
preference_example = {"chosen": [{"role": "user", "content": "What color is the sky?"},
                                 {"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "user", "content": "What color is the sky?"},
                                   {"role": "assistant", "content": "It is green."}]}
```

If your dataset is not in one of these formats, you can preprocess it to convert it into the expected format. Here is an example with the [Vezora/Code-Preference-Pairs](https://huggingface.co/datasets/Vezora/Code-Preference-Pairs) dataset:

```python
from datasets import load_dataset

dataset = load_dataset("Vezora/Code-Preference-Pairs")


def preprocess_function(example):
    return {
        "prompt": [{"role": "user", "content": example["input"]}],
        "chosen": [{"role": "assistant", "content": example["accepted"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"]}],
    }


dataset = dataset.map(preprocess_function, remove_columns=["instruction", "input", "accepted", "ID"])
print(next(iter(dataset["train"])))
```

```json
{
    "prompt": [{"role": "user", "content": "Create a nested loop to print every combination of numbers [...]"}],
    "chosen": [{"role": "assistant", "content": "Here is an example of a nested loop in Python [...]"}],
    "rejected": [{"role": "assistant", "content": "Here is an example of a nested loop in Python [...]"}],
}
```

## Looking deeper into the CPO method

Contrastive Preference Optimization (CPO) is a training method designed to align a language model with preference data. Instead of supervised input–output pairs, the model is trained on pairs of completions to the same prompt, where one completion is preferred over the other. The objective directly optimizes the model to widen the margin between the log-likelihoods of preferred and dispreferred completions, **without requiring a reference model**. To recover the regularization signal that the reference model would otherwise provide, CPO adds an SFT-style negative-log-likelihood (NLL) term on the chosen completions, weighted by `cpo_alpha`.

This section breaks down how CPO works in practice, covering the key steps: **preprocessing** and **loss computation**.

### Preprocessing and tokenization

During training, each example is expected to contain a prompt along with a preferred (`chosen`) and a dispreferred (`rejected`) completion. For more details on the expected formats, see [Dataset formats](dataset_formats).
The [`experimental.cpo.CPOTrainer`] tokenizes each input using the model's tokenizer.

### Computing the loss

The loss used in CPO is defined as follows:
$$
\mathcal{L}_{\mathrm{CPO}}(\theta) = -\mathbb{E}_{(x,y^{+},y^{-})}\!\left[\log \sigma\!\left(\beta\Big(\log \pi_{\theta}(y^{+}\!\mid x) - \log \pi_{\theta}(y^{-}\!\mid x)\Big)\right)\right] + \alpha \, \mathcal{L}_{\mathrm{NLL}}(\theta; y^{+})
$$
  
where  \\( x \\)  is the prompt,  \\( y^+ \\) is the preferred completion and  \\( y^- \\)  is the dispreferred completion.  \\( \pi_{\theta} \\)  is the policy model being trained,  \\( \sigma \\)  is the sigmoid function,  \\( \beta > 0 \\)  is a hyperparameter that controls the strength of the preference signal, and  \\( \alpha \\)  (`cpo_alpha`) is the weight of the SFT NLL term on the chosen completion.

#### Loss Types

Several formulations of the objective have been proposed in the literature. Initially, the objective of CPO was defined as presented above.

| `loss_type=` | Description |
| --- | --- |
| `"sigmoid"` (default) | Given the preference data, we can fit a binary classifier according to the Bradley-Terry model and in fact the [DPO](https://huggingface.co/papers/2305.18290) authors propose the sigmoid loss on the normalized likelihood via the `logsigmoid` to fit a logistic regression. |
| `"hinge"` | The [RSO](https://huggingface.co/papers/2309.06657) authors propose to use a hinge loss on the normalized likelihood from the [SLiC](https://huggingface.co/papers/2305.10425) paper. In this case, the `beta` is the reciprocal of the margin. |
| `"ipo"` | The [IPO](https://huggingface.co/papers/2310.12036) authors argue the logit transform can overfit and propose the identity transform to optimize preferences directly; TRL exposes this as `loss_type="ipo"`. |
| `"simpo"` | [SimPO](https://huggingface.co/papers/2405.14734) uses length-normalized log-probabilities as the implicit reward and adds a target reward margin γ (`simpo_gamma`) in the Bradley-Terry sigmoid. Typically combined with `cpo_alpha=0` (no NLL term). |
| `"alphapo"` | Syntactic sugar for `loss_type=["simpo"]` with `cpo_alpha=0.0`. The actual [AlphaPO](https://huggingface.co/papers/2501.03884) reward reshaping `r = (1 - p^(-α)) / α` is controlled by the `alpha` config field and can be combined with any loss type. |

## Logged metrics

While training and evaluating we record the following reward metrics:

* `global_step`: The total number of optimizer steps taken so far.
* `epoch`: The current epoch number, based on dataset iteration.
* `num_tokens`: The total number of tokens processed so far.
* `loss`: The average cross-entropy loss computed over non-masked tokens in the current logging interval.
* `entropy`: The average entropy of the model's predicted token distribution over non-masked tokens.
* `mean_token_accuracy`: The proportion of non-masked tokens for which the model’s top-1 prediction matches the token from the chosen completion.
* `learning_rate`: The current learning rate, which may change dynamically if a scheduler is used.
* `grad_norm`: The L2 norm of the gradients, computed before gradient clipping.
* `logits/chosen`: The average logit values assigned by the model to the tokens in the chosen completion.
* `logits/rejected`: The average logit values assigned by the model to the tokens in the rejected completion.
* `logps/chosen`: The average log-probability assigned by the model to the tokens in the chosen completion.
* `logps/rejected`: The average log-probability assigned by the model to the tokens in the rejected completion.
* `rewards/chosen`: The average implicit reward computed for the chosen completion, computed as  \\( \beta \log \pi_{\theta}(y^{+}\!\mid x) \\).
* `rewards/rejected`: The average implicit reward computed for the rejected completion, computed as  \\( \beta \log \pi_{\theta}(y^{-}\!\mid x) \\).
* `rewards/margins`: The average implicit reward margin between the chosen and rejected completions.
* `rewards/accuracies`: The proportion of examples where the implicit reward for the chosen completion is higher than that for the rejected completion.

## Customization

### Compatibility and constraints

Some argument combinations are intentionally restricted in the current [`experimental.cpo.CPOTrainer`] implementation:

* With `use_liger_kernel=True`:
  * only a single `loss_type` is supported,
  * `compute_metrics` is not supported,
  * training with PEFT models is not supported.

### Multi-loss combinations

The CPO trainer supports combining multiple loss functions with different weights. To combine multiple losses, specify the loss types and corresponding weights as lists:

```python
training_args = CPOConfig(
    loss_type=["sigmoid", "hinge", "ipo"],  # loss types to combine
    loss_weights=[1.0, 0.5, 0.5],  # corresponding weights
)
```

### Model initialization

You can directly pass the kwargs of the [`~transformers.AutoModelForCausalLM.from_pretrained()`] method to the [`experimental.cpo.CPOConfig`]. For example, if you want to load a model in a different precision, analogous to

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
```

you can do so by passing the `model_init_kwargs={"dtype": torch.bfloat16}` argument to the [`experimental.cpo.CPOConfig`].

```python
from trl.experimental.cpo import CPOConfig

training_args = CPOConfig(
    model_init_kwargs={"dtype": torch.bfloat16},
)
```

Note that all keyword arguments of [`~transformers.AutoModelForCausalLM.from_pretrained()`] are supported.

### Train adapters with PEFT

We support tight integration with 🤗 PEFT library, allowing any user to conveniently train adapters and share them on the Hub, rather than training the entire model.

```python
from datasets import load_dataset
from trl.experimental.cpo import CPOTrainer
from peft import LoraConfig

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = CPOTrainer(
    "Qwen/Qwen3-0.6B",
    train_dataset=dataset,
    peft_config=LoraConfig(),
)

trainer.train()
```

You can also continue training your [`~peft.PeftModel`]. For that, first load a `PeftModel` outside [`experimental.cpo.CPOTrainer`] and pass it directly to the trainer without the `peft_config` argument being passed.

```python
from datasets import load_dataset
from trl.experimental.cpo import CPOTrainer
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("trl-lib/Qwen3-4B-LoRA", is_trainable=True)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = CPOTrainer(
    model=model,
    train_dataset=dataset,
)

trainer.train()
```

> [!TIP]
> When training adapters, you typically use a higher learning rate (≈1e‑5) than full fine-tuning since only new parameters are being learned.
>
> ```python
> CPOConfig(learning_rate=1e-5, ...)
> ```

### Train with Liger Kernel

Liger Kernel is a collection of Triton kernels for LLM training that boosts multi-GPU throughput by 20%, cuts memory use by 60% (enabling up to 4× longer context), and works seamlessly with tools like FlashAttention, PyTorch FSDP, and DeepSpeed. For more information, see [Liger Kernel Integration](liger_kernel_integration).

### Train with Unsloth

Unsloth is an open‑source framework for fine‑tuning and reinforcement learning that trains LLMs (like Llama, Mistral, Gemma, DeepSeek, and more) up to 2× faster with up to 70% less VRAM, while providing a streamlined, Hugging Face–compatible workflow for training, evaluation, and deployment. For more information, see [Unsloth Integration](unsloth_integration).

## Tool Calling with CPO

The [`experimental.cpo.CPOTrainer`] fully supports fine-tuning models with _tool calling_ capabilities. In this case, each dataset example should include:

* The conversation messages (prompt, chosen and rejected), including any tool calls (`tool_calls`) and tool responses (`tool` role messages)
* The list of available tools in the `tools` column, typically provided as JSON schemas

For details on the expected dataset structure, see the [Dataset Format — Tool Calling](dataset_formats#tool-calling) section.

## Training Vision Language Models

[`experimental.cpo.CPOTrainer`] fully supports training Vision-Language Models (VLMs). To train a VLM, provide a dataset with either an `image` column (single image per sample) or an `images` column (list of images per sample). For more information on the expected dataset structure, see the [Dataset Format — Vision Dataset](dataset_formats#vision-dataset) section.
An example of such a dataset is the [RLAIF-V Dataset](https://huggingface.co/datasets/HuggingFaceH4/rlaif-v_formatted) dataset.

```python
from trl.experimental.cpo import CPOConfig, CPOTrainer
from datasets import load_dataset

trainer = CPOTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    args=CPOConfig(max_length=None),
    train_dataset=load_dataset("HuggingFaceH4/rlaif-v_formatted", split="train"),
)
trainer.train()
```

> [!TIP]
> For VLMs, truncating may remove image tokens, leading to errors during training. To avoid this, set `max_length=None` in the [`experimental.cpo.CPOConfig`]. This allows the model to process the full sequence length without truncating image tokens.
>
> ```python
> CPOConfig(max_length=None, ...)
> ```
>
> Only use `max_length` when you've verified that truncation won't remove image tokens for the entire dataset.

## CPOTrainer

[[autodoc]] experimental.cpo.CPOTrainer
    - train
    - save_model
    - push_to_hub

## CPOConfig

[[autodoc]] experimental.cpo.CPOConfig
