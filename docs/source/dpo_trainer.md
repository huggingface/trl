# DPO Trainer

[![All_models-DPO-blue](https://img.shields.io/badge/All_models-DPO-blue)](https://huggingface.co/models?other=dpo,trl) [![smol_course-Chapter_2-yellow](https://img.shields.io/badge/smol_course-Chapter_2-yellow)](https://github.com/huggingface/smol-course/tree/main/2_preference_alignment)

## Overview

TRL supports the Direct Preference Optimization (DPO) Trainer for training language models, as described in the paper [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://huggingface.co/papers/2305.18290) by [Rafael Rafailov](https://huggingface.co/rmrafailov), Archit Sharma, Eric Mitchell, [Stefano Ermon](https://huggingface.co/ermonste), [Christopher D. Manning](https://huggingface.co/manning), [Chelsea Finn](https://huggingface.co/cbfinn).

The abstract from the paper is the following:

> While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.

This post-training method was contributed by [Kashif Rasul](https://huggingface.co/kashif) and later refactored by [Quentin Gallouédec](https://huggingface.co/qgallouedec).

## Quick start

This example demonstrates how to train a language model using the [`DPOTrainer`] from TRL. We train a [Qwen 3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model on the [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback).

```python
from trl import DPOTrainer
from datasets import load_dataset

trainer = DPOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
)
trainer.train()
```

<iframe src="https://trl-lib-trackio.hf.space/?project=trl-documentation&metrics=train*&sidebar=hidden&runs=dpo_qwen3-0.6B_ultrafeedback" style="width: 100%; min-width: 300px; max-width: 800px;" height="830" frameBorder="0"></iframe>

## Expected dataset type and format

DPO requires a [preference](dataset_formats#preference) dataset. The [`DPOTrainer`] is compatible with both [standard](dataset_formats#standard) and [conversational](dataset_formats#conversational) dataset formats. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

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

## Looking deeper into the DPO method

Direct Preference Optimization (DPO) is a training method designed to align a language model with preference data. Instead of supervised input–output pairs, the model is trained on pairs of completions to the same prompt, where one completion is preferred over the other. The objective directly optimizes the model to widen the margin between the log-likelihoods of preferred and dispreferred completions, relative to a reference model, without requiring an explicit reward model. In practice, this is typically achieved by suppressing the likelihood of dispreferred completions rather than by increasing the likelihood of preferred ones.

This section breaks down how DPO works in practice, covering the key steps: **preprocessing** and **loss computation**.

### Preprocessing and tokenization

During training, each example is expected to contain a prompt along with a preferred (`chosen`) and a dispreferred (`rejected`) completion. For more details on the expected formats, see [Dataset formats](dataset_formats).
The [`DPOTrainer`] tokenizes each input using the model's tokenizer.

### Computing the loss

![dpo_figure](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/dpo_figure.png)

The loss used in DPO is defined as follows:
$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}_{(x,y^{+},y^{-})}\!\left[\log \sigma\!\left(\beta\Big(\log\frac{\pi_{\theta}(y^{+}\!\mid x)}{\pi_{\mathrm{ref}}(y^{+}\!\mid x)}-\log \frac{\pi_{\theta}(y^{-}\!\mid x)}{\pi_{\mathrm{ref}}(y^{-}\!\mid x)}\Big)\right)\right]
$$
  
where  \\( x \\)  is the prompt,  \\( y^+ \\) is the preferred completion and  \\( y^- \\)  is the dispreferred completion.  \\( \pi_{\theta} \\)  is the policy model being trained,  \\( \pi_{\mathrm{ref}} \\)  is the reference model,  \\( \sigma \\)  is the sigmoid function, and  \\( \beta > 0 \\)  is a hyperparameter that controls the strength of the preference signal.

#### Loss Types

Several formulations of the objective have been proposed in the literature. Initially, the objective of DPO was defined as presented above.

| `loss_type=` | Description |
| --- | --- |
| `"sigmoid"` (default) | Given the preference data, we can fit a binary classifier according to the Bradley-Terry model and in fact the [DPO](https://huggingface.co/papers/2305.18290) authors propose the sigmoid loss on the normalized likelihood via the `logsigmoid` to fit a logistic regression. |
| `"hinge"` | The [RSO](https://huggingface.co/papers/2309.06657) authors propose to use a hinge loss on the normalized likelihood from the [SLiC](https://huggingface.co/papers/2305.10425) paper. In this case, the `beta` is the reciprocal of the margin. |
| `"ipo"` | The [IPO](https://huggingface.co/papers/2310.12036) authors argue the logit transform can overfit and propose the identity transform to optimize preferences directly; TRL exposes this as `loss_type="ipo"`. |
| `"exo_pair"` | The [EXO](https://huggingface.co/papers/2402.00856) authors propose reverse-KL preference optimization. `label_smoothing` must be strictly greater than `0.0`; a recommended value is `1e-3` (see Eq. 16 for the simplified pairwise variant). The full method uses `K>2` SFT completions and approaches PPO as `K` grows. |
| `"nca_pair"` | The [NCA](https://huggingface.co/papers/2402.05369) authors shows that NCA optimizes the absolute likelihood for each response rather than the relative likelihood. |
| `"robust"` | The [Robust DPO](https://huggingface.co/papers/2403.00409) authors propose an unbiased DPO loss under noisy preferences. Use `label_smoothing` in [`DPOConfig`] to model label-flip probability; valid values are in the range `[0.0, 0.5)`. |
| `"bco_pair"` | The [BCO](https://huggingface.co/papers/2404.04656) authors train a binary classifier whose logit serves as a reward so that the classifier maps {prompt, chosen completion} pairs to 1 and {prompt, rejected completion} pairs to 0. For unpaired data, we recommend the dedicated [`experimental.bco.BCOTrainer`]. |
| `"sppo_hard"` | The [SPPO](https://huggingface.co/papers/2405.00675) authors claim that SPPO is capable of solving the Nash equilibrium iteratively by pushing the chosen rewards to be as large as 1/2 and the rejected rewards to be as small as -1/2 and can alleviate data sparsity issues. The implementation approximates this algorithm by employing hard label probabilities, assigning 1 to the winner and 0 to the loser. |
| `"aot"`  or `loss_type="aot_unpaired"` | The [AOT](https://huggingface.co/papers/2406.05882) authors propose Distributional Preference Alignment via Optimal Transport. `loss_type="aot"` is for paired data; `loss_type="aot_unpaired"` is for unpaired data. Both enforce stochastic dominance via sorted quantiles; larger per-GPU batch sizes help. |
| `"apo_zero"` or `loss_type="apo_down"` | The [APO](https://huggingface.co/papers/2408.06266) method introduces an anchored objective. `apo_zero` boosts winners and downweights losers (useful when the model underperforms the winners). `apo_down` downweights both, with stronger pressure on losers (useful when the model already outperforms winners). |
| `"discopop"` | The [DiscoPOP](https://huggingface.co/papers/2406.08414) paper uses LLMs to discover more efficient offline preference optimization losses. In the paper the proposed DiscoPOP loss (which is a log-ratio modulated loss) outperformed other optimization losses on different tasks (IMDb positive text generation, Reddit TLDR summarization, and Alpaca Eval 2.0). |
| `"sft"` | SFT (Supervised Fine-Tuning) loss is the negative log likelihood loss, used to train the model to generate preferred responses. |

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
* `rewards/chosen`: The average implicit reward computed for the chosen completion, computed as  \\( \beta \log \frac{\pi_{\theta}(y^{+}\!\mid x)}{\pi_{\mathrm{ref}}(y^{+}\!\mid x)} \\).
* `rewards/rejected`: The average implicit reward computed for the rejected completion, computed as  \\( \beta \log \frac{\pi_{\theta}(y^{-}\!\mid x)}{\pi_{\mathrm{ref}}(y^{-}\!\mid x)} \\).
* `rewards/margins`: The average implicit reward margin between the chosen and rejected completions.
* `rewards/accuracies`: The proportion of examples where the implicit reward for the chosen completion is higher than that for the rejected completion.

## Customization

### Compatibility and constraints

Some argument combinations are intentionally restricted in the current [`DPOTrainer`] implementation:

* `use_weighting=True` is not supported with `loss_type="aot"` or `loss_type="aot_unpaired"`.
* With `use_liger_kernel=True`:
  * only a single `loss_type` is supported,
  * `compute_metrics` is not supported,
  * `precompute_ref_log_probs=True` is not supported.
* `sync_ref_model=True` is not supported when training with PEFT models that do not keep a standalone `ref_model`.
* `sync_ref_model=True` cannot be combined with `precompute_ref_log_probs=True`.
* `precompute_ref_log_probs=True` is not supported with `IterableDataset` (train or eval).

### Multi-loss combinations

The DPO trainer supports combining multiple loss functions with different weights, enabling more sophisticated optimization strategies. This is particularly useful for implementing algorithms like MPO (Mixed Preference Optimization). MPO is a training approach that combines multiple optimization objectives, as described in the paper [Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization](https://huggingface.co/papers/2411.10442).

To combine multiple losses, specify the loss types and corresponding weights as lists:

```python
# MPO: Combines DPO (sigmoid) for preference and BCO (bco_pair) for quality
training_args = DPOConfig(
    loss_type=["sigmoid", "bco_pair", "sft"],  # loss types to combine
    loss_weights=[0.8, 0.2, 1.0]  # corresponding weights, as used in the MPO paper
)
```

### Model initialization

You can directly pass the kwargs of the [`~transformers.AutoModelForCausalLM.from_pretrained()`] method to the [`DPOConfig`]. For example, if you want to load a model in a different precision, analogous to

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
```

you can do so by passing the `model_init_kwargs={"dtype": torch.bfloat16}` argument to the [`DPOConfig`].

```python
from trl import DPOConfig

training_args = DPOConfig(
    model_init_kwargs={"dtype": torch.bfloat16},
)
```

Note that all keyword arguments of [`~transformers.AutoModelForCausalLM.from_pretrained()`] are supported.

### Train adapters with PEFT

We support tight integration with 🤗 PEFT library, allowing any user to conveniently train adapters and share them on the Hub, rather than training the entire model.

```python
from datasets import load_dataset
from trl import DPOTrainer
from peft import LoraConfig

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = DPOTrainer(
    "Qwen/Qwen3-0.6B",
    train_dataset=dataset,
    peft_config=LoraConfig(),
)

trainer.train()
```

You can also continue training your [`~peft.PeftModel`]. For that, first load a `PeftModel` outside [`DPOTrainer`] and pass it directly to the trainer without the `peft_config` argument being passed.

```python
from datasets import load_dataset
from trl import DPOTrainer
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("trl-lib/Qwen3-4B-LoRA", is_trainable=True)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = DPOTrainer(
    model=model,
    train_dataset=dataset,
)

trainer.train()
```

> [!TIP]
> When training adapters, you typically use a higher learning rate (≈1e‑5) than full fine-tuning since only new parameters are being learned.
>
> ```python
> DPOConfig(learning_rate=1e-5, ...)
> ```

### Train with Liger Kernel

Liger Kernel is a collection of Triton kernels for LLM training that boosts multi-GPU throughput by 20%, cuts memory use by 60% (enabling up to 4× longer context), and works seamlessly with tools like FlashAttention, PyTorch FSDP, and DeepSpeed. For more information, see [Liger Kernel Integration](liger_kernel_integration).

### Rapid Experimentation for DPO

RapidFire AI is an open-source experimentation engine that sits on top of TRL and lets you launch multiple DPO configurations at once, even on a single GPU. Instead of trying configurations sequentially, RapidFire lets you **see all their learning curves earlier, stop underperforming runs, and clone promising ones with new settings in flight** without restarting. For more information, see [RapidFire AI Integration](rapidfire_integration).

### Train with Unsloth

Unsloth is an open‑source framework for fine‑tuning and reinforcement learning that trains LLMs (like Llama, Mistral, Gemma, DeepSeek, and more) up to 2× faster with up to 70% less VRAM, while providing a streamlined, Hugging Face–compatible workflow for training, evaluation, and deployment. For more information, see [Unsloth Integration](unsloth_integration).

## Tool Calling with DPO

The [`DPOTrainer`] fully supports fine-tuning models with _tool calling_ capabilities. In this case, each dataset example should include:

* The conversation messages (prompt, chosen and rejected), including any tool calls (`tool_calls`) and tool responses (`tool` role messages)
* The list of available tools in the `tools` column, typically provided as JSON `str` schemas

For details on the expected dataset structure, see the [Dataset Format — Tool Calling](dataset_formats#tool-calling) section.

## Training Vision Language Models

[`DPOTrainer`] fully supports training Vision-Language Models (VLMs). To train a VLM, provide a dataset with either an `image` column (single image per sample) or an `images` column (list of images per sample). For more information on the expected dataset structure, see the [Dataset Format — Vision Dataset](dataset_formats#vision-dataset) section.
An example of such a dataset is the [RLAIF-V Dataset](https://huggingface.co/datasets/HuggingFaceH4/rlaif-v_formatted) dataset.

```python
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    args=DPOConfig(max_length=None),
    train_dataset=load_dataset("HuggingFaceH4/rlaif-v_formatted", split="train"),
)
trainer.train()
```

> [!TIP]
> For VLMs, truncating may remove image tokens, leading to errors during training. To avoid this, set `max_length=None` in the [`DPOConfig`]. This allows the model to process the full sequence length without truncating image tokens.
>
> ```python
> DPOConfig(max_length=None, ...)
> ```
>
> Only use `max_length` when you've verified that truncation won't remove image tokens for the entire dataset.

## DPOTrainer

[[autodoc]] DPOTrainer
    - train
    - save_model
    - push_to_hub

## DPOConfig

[[autodoc]] DPOConfig

## DataCollatorForPreference

[[autodoc]] trainer.dpo_trainer.DataCollatorForPreference

## DataCollatorForVisionPreference

[[autodoc]] trainer.dpo_trainer.DataCollatorForVisionPreference
