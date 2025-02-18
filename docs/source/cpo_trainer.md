# CPO Trainer

[![](https://img.shields.io/badge/All_models-CPO-blue)](https://huggingface.co/models?other=cpo,trl)

## Overview

Contrastive Preference Optimization (CPO) as introduced in the paper [Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation](https://huggingface.co/papers/2401.08417) by [Haoran Xu](https://huggingface.co/haoranxu), [Amr Sharaf](https://huggingface.co/amrsharaf), [Yunmo Chen](https://huggingface.co/yunmochen), Weiting Tan, Lingfeng Shen, Benjamin Van Durme, [Kenton Murray](https://huggingface.co/Kenton), and [Young Jin Kim](https://huggingface.co/ykim362). At a high-level, CPO trains models to avoid generating adequate, but not perfect translations in Machine Translation (MT) tasks. However, CPO is a general approximation of the DPO loss and can be applied to other domains, such as chat.

CPO aims to mitigate two fundamental shortcomings of SFT. First, SFT’s methodology of minimizing the discrepancy between predicted outputs and gold-standard references inherently caps model performance at the quality level of the training data. Secondly, SFT lacks a mechanism to prevent the model from rejecting mistakes in translations. The CPO objective is derived from the DPO objective.

## Quick start

This example demonstrates how to train a model using the CPO method. We use the [Qwen 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) as the base model. We use the preference data from the [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback). You can view the data in the dataset here:

<iframe
  src="https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

Below is the script to train the model:

```python
# train_cpo.py
from datasets import load_dataset
from trl import CPOConfig, CPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = CPOConfig(output_dir="Qwen2-0.5B-CPO", logging_steps=10)
trainer = CPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
```

Execute the script using the following command:

```bash
accelerate launch train_cpo.py
```

## Expected dataset type

CPO requires a [preference dataset](dataset_formats#preference). The [`CPOTrainer`] supports both [conversational](dataset_formats#conversational) and [standard](dataset_formats#standard) dataset format. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

## Example script

We provide an example script to train a model using the CPO method. The script is available in [`examples/scripts/cpo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/cpo.py)

To test the CPO script with the [Qwen2 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) on the [UltraFeedback dataset](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized), run the following command:

```bash
accelerate launch examples/scripts/cpo.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --num_train_epochs 1 \
    --logging_steps 25 \
    --output_dir Qwen2-0.5B-CPO
```

## Logged metrics

While training and evaluating we record the following reward metrics:

* `rewards/chosen`: the mean log probabilities of the policy model for the chosen responses scaled by beta
* `rewards/rejected`: the mean log probabilities of the policy model for the rejected responses scaled by beta
* `rewards/accuracies`: mean of how often the chosen rewards are > than the corresponding rejected rewards
* `rewards/margins`: the mean difference between the chosen and corresponding rejected rewards
* `nll_loss`: the mean negative log likelihood loss of the policy model for the chosen responses

## CPO variants

### Simple Preference Optimization (SimPO)

The [SimPO](https://huggingface.co/papers/2405.14734) method is also implemented in the [`CPOTrainer`]. SimPO is an alternative loss that adds a reward margin, allows for length normalization, and does not use BC regularization. To use this loss, we can use SimPO easily by turning on `loss_type="simpo"` and `cpo_alpha=0.0` in the [`CPOConfig`].

### CPO-SimPO

We also offer the combined use of CPO and SimPO, which enables more stable training and improved performance. Learn more details at [CPO-SimPO GitHub](https://github.com/fe1ixxu/CPO_SIMPO). To use this method, simply enable SimPO by setting `loss_type="simpo"` and a non-zero `cpo_alpha` in the [`CPOConfig`].

## Loss functions

The CPO algorithm supports several loss functions. The loss function can be set using the `loss_type` parameter in the [`CPOConfig`]. The following loss functions are supported:

| `loss_type=`                           | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"sigmoid"` (default)                  | Given the preference data, we can fit a binary classifier according to the Bradley-Terry model and in fact the [DPO](https://huggingface.co/papers/2305.18290) authors propose the sigmoid loss on the normalized likelihood via the `logsigmoid` to fit a logistic regression.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `"hinge"`                              | The [RSO](https://huggingface.co/papers/2309.06657) authors propose to use a hinge loss on the normalized likelihood from the [SLiC](https://huggingface.co/papers/2305.10425) paper. In this case, the `beta` is the reciprocal of the margin.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `"ipo"`                                | The [IPO](https://huggingface.co/papers/2310.12036) authors provide a deeper theoretical understanding of the DPO algorithms and identify an issue with overfitting and propose an alternative loss. In this case, the `beta` is the reciprocal of the gap between the log-likelihood ratios of the chosen vs the rejected completion pair and thus the smaller the `beta` the larger this gaps is. As per the paper the loss is averaged over log-likelihoods of the completion (unlike DPO which is summed only).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

### For Mixture of Experts Models: Enabling the auxiliary loss

MOEs are the most efficient if the load is about equally distributed between experts.  
To ensure that we train MOEs similarly during preference-tuning, it is beneficial to add the auxiliary loss from the load balancer to the final loss.

This option is enabled by setting `output_router_logits=True` in the model config (e.g. [`~transformers.MixtralConfig`]).  
To scale how much the auxiliary loss contributes to the total loss, use the hyperparameter `router_aux_loss_coef=...` (default: `0.001`) in the model config.

## CPOTrainer

[[autodoc]] CPOTrainer

## CPOConfig

[[autodoc]] CPOConfig
