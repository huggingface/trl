# TPO Trainer

[![All_models-TPO-blue](https://img.shields.io/badge/All_models-TPO-blue)](https://huggingface.co/models?other=tpo,trl)

## Overview

Triple Preference Optimization (TPO) was introduced in the paper [Triple Preference Optimization: Achieving Better Alignment using a Single Step Optimization](https://huggingface.co/papers/2405.16681) by Amir Saeidi, Shivanshu Verma, Aswin RRV, and Chitta Baral. TPO enhances the instruction-following and reasoning capabilities of large language models in a single training step, starting from a pre-trained or instruction-tuned model.

The abstract from the paper is the following:

> Reinforcement Learning with Human Feedback (RLHF) enhances the alignment of Large Language Models (LLMs). However, its limitations have led to the development of Direct Preference Optimization (DPO), an RL-free approach designed to overcome these shortcomings. While studies have shown that DPO improves instruction-following capabilities, it negatively impacts the reasoning ability of LLMs. Additionally, DPO is highly sensitive to judgment noise in preference datasets and the size of the training set. Although several modifications to DPO have been proposed, they still fail to fully resolve these issues. To address these limitations, we propose Triple Preference Optimization (TPO), a new preference learning method designed to enhance both reasoning and instruction-following abilities through one-step optimization. We compare TPO against DPO and its recent variants using state-of-the-art training setups, including both base and instructiontuned models such as Mistral and Llama 3. Our evaluation covers a comprehensive range of chat-based and reasoning benchmarks. The results demonstrate that TPO achieves significant improvements over existing methods without substantially increasing response length across different dataset sizes. Specifically, TPO outperforms DPO and SimPO by up to 7.0% and 7.3% points on Arena-Hard, 12.2% and 13.3% points on MixEval-Hard, 10.4% and 10.1% points on MMLU-Pro, and 19.0% and 19.2% points on GSM8K, respectively. Furthermore, TPO achieves these improvements while requiring less data than DPO.

This post-training method was contributed by [Kashif Rasul](https://huggingface.co/kashif).

## Quick start

This example demonstrates how to train a model using the TPO method. We use the [Qwen 3 0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B) as the base model. TPO requires a *triple-preference* dataset (`prompt`, `chosen`, `rejected`, `reference`) — see [Expected dataset type](#expected-dataset-type-and-format) below.

Below is the script to train the model:

```python
# train_tpo.py
from datasets import load_dataset
from trl.experimental.tpo import TPOConfig, TPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
train_dataset = load_dataset("tpo-alignment/triple-preference-ultrafeedback-40K", split="train")

training_args = TPOConfig(output_dir="Qwen3-0.6B-TPO")
trainer = TPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
```

Execute the script using the following command:

```bash
accelerate launch train_tpo.py
```

## Expected dataset type and format

TPO requires a *triple-preference* dataset: each example must contain a `prompt`, a `chosen` (preferred) completion, a `rejected` (dispreferred) completion **and** a `reference` (gold) completion. The [`experimental.tpo.TPOTrainer`] supports both [conversational](dataset_formats#conversational) and [standard](dataset_formats#standard) dataset formats. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

```python
# Standard format
triple_preference_example = {
    "prompt": "The sky is",
    "reference": " a beautiful shade of blue.",  # gold response (used for the NLL term)
    "chosen": " blue.",
    "rejected": " green.",
}

# Conversational format
triple_preference_example = {
    "prompt": [{"role": "user", "content": "What color is the sky?"}],
    "reference": [{"role": "assistant", "content": "It is a beautiful shade of blue."}],
    "chosen": [{"role": "assistant", "content": "It is blue."}],
    "rejected": [{"role": "assistant", "content": "It is green."}],
}
```

The reference response is typically the highest-quality completion available for the prompt; in the original TPO paper it is taken from the response with the highest score in [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback), with the second-highest used as the chosen completion and the lowest as the rejected completion.

## Example script

We provide an example script to train a model using the TPO method. The script is available at [`trl/experimental/tpo/tpo.py`](https://github.com/huggingface/trl/blob/main/trl/experimental/tpo/tpo.py).

To test the TPO script with the [Qwen 3 0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B) on a triple-preference dataset, run the following command:

```bash
accelerate launch trl/experimental/tpo/tpo.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name tpo-alignment/triple-preference-ultrafeedback-40K \
    --beta 0.01 \
    --tpo_alpha 1.0 \
    --learning_rate 5e-7 \
    --num_train_epochs 1 \
    --output_dir Qwen3-0.6B-TPO
```

## Looking deeper into the TPO method

Triple Preference Optimization (TPO) extends preference-based alignment from pairs to *triples* `(y_gold, y_chosen, y_rejected)`. The model is jointly optimized with two objectives in a single step:

1. A **contrastive loss** between the chosen and rejected completions, similar in spirit to DPO/SimPO but computed directly from the policy log-probabilities (no separate reference policy is required).
2. A **supervised negative log-likelihood (NLL) loss** on the gold (`reference`) completion, weighted by `tpo_alpha`. This term replaces the standalone SFT stage typically required before DPO.

The total TPO loss is:

$$
\mathcal{L}_{\mathrm{TPO}}(\theta) = \mathcal{L}_{\mathrm{contrast}}(\theta) + \alpha \cdot \mathcal{L}_{\mathrm{NLL}}(\theta; y_{\text{gold}})
$$

where  \\( \alpha \\) is `tpo_alpha` and  \\( \mathcal{L}_{\mathrm{contrast}} \\) is selected via `loss_type`.

### Loss types

| `loss_type=` | Description |
| --- | --- |
| `"sigmoid"` (default) | Sigmoid loss on the (sum) log-probability difference between the chosen and rejected completions, as in the original [TPO](https://huggingface.co/papers/2405.16681) paper. |
| `"hinge"` | Hinge loss on the normalized likelihood from the [SLiC](https://huggingface.co/papers/2305.10425) paper. In this case, `beta` is the reciprocal of the margin. |
| `"ipo"` | IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper, computed on length-normalized log-probabilities. |
| `"tpo-l"` | Length-normalized TPO variant: uses average per-token log-probabilities and adds a target reward margin `tpo_l_gamma` to the Bradley-Terry objective, in the spirit of [SimPO](https://huggingface.co/papers/2405.14734). |

Setting `tpo_alpha=0.0` disables the NLL term entirely (the reference response is then unused, and the corresponding cross-entropy is skipped to save compute).

## Logged metrics

While training and evaluating we record the following metrics:

* `loss`: The total TPO loss (contrastive + `tpo_alpha` × NLL) averaged over the current logging interval.
* `entropy`: The average entropy of the model's predicted token distribution over completion tokens.
* `mean_token_accuracy`: The proportion of completion tokens for which the model's top-1 prediction matches the chosen completion.
* `num_tokens`: The total number of tokens processed so far.
* `logits/chosen`: The average logit values assigned by the model to the tokens in the chosen completion.
* `logits/rejected`: The average logit values assigned by the model to the tokens in the rejected completion.
* `logps/chosen`: The average log-probability assigned by the model to the chosen completion.
* `logps/rejected`: The average log-probability assigned by the model to the rejected completion.
* `rewards/chosen`: The average implicit reward computed for the chosen completion, defined as  \\( \beta \log \pi_{\theta}(y^{+}\!\mid x) \\).
* `rewards/rejected`: The average implicit reward computed for the rejected completion, defined as  \\( \beta \log \pi_{\theta}(y^{-}\!\mid x) \\).
* `rewards/margins`: The average implicit reward margin between the chosen and rejected completions.
* `rewards/accuracies`: The proportion of examples where the implicit reward for the chosen completion is higher than that for the rejected completion.

## TPOTrainer

[[autodoc]] experimental.tpo.TPOTrainer
    - train
    - save_model
    - push_to_hub

## TPOConfig

[[autodoc]] experimental.tpo.TPOConfig
