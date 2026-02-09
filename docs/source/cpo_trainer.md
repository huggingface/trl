# CPO Trainer

[![model badge](https://img.shields.io/badge/All_models-CPO-blue)](https://huggingface.co/models?other=cpo,trl)

## Overview

Contrastive Preference Optimization (CPO) as introduced in the paper [Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation](https://huggingface.co/papers/2401.08417) by [Haoran Xu](https://huggingface.co/haoranxu), [Amr Sharaf](https://huggingface.co/amrsharaf), [Yunmo Chen](https://huggingface.co/yunmochen), Weiting Tan, Lingfeng Shen, Benjamin Van Durme, [Kenton Murray](https://huggingface.co/Kenton), and [Young Jin Kim](https://huggingface.co/ykim362). At a high level, CPO trains models to avoid generating adequate, but not perfect, translations in Machine Translation (MT) tasks. However, CPO is a general approximation of the DPO loss and can be applied to other domains, such as chat.

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
from trl.experimental.cpo import CPOConfig, CPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = CPOConfig(output_dir="Qwen2-0.5B-CPO")
trainer = CPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
```

Execute the script using the following command:

```bash
accelerate launch train_cpo.py
```

## Expected dataset type

CPO requires a [preference dataset](dataset_formats#preference). The [`experimental.cpo.CPOTrainer`] supports both [conversational](dataset_formats#conversational) and [standard](dataset_formats#standard) dataset formats. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

## Example script

We provide an example script to train a model using the CPO method. The script is available in [`examples/scripts/cpo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/cpo.py)

To test the CPO script with the [Qwen2 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) on the [UltraFeedback dataset](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized), run the following command:

```bash
accelerate launch examples/scripts/cpo.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --num_train_epochs 1 \
    --output_dir Qwen2-0.5B-CPO
```

## Logged metrics

While training and evaluating, we record the following reward metrics:

* `rewards/chosen`: the mean log probabilities of the policy model for the chosen responses scaled by beta
* `rewards/rejected`: the mean log probabilities of the policy model for the rejected responses scaled by beta
* `rewards/accuracies`: mean of how often the chosen rewards are > than the corresponding rejected rewards
* `rewards/margins`: the mean difference between the chosen and corresponding rejected rewards
* `nll_loss`: the mean negative log likelihood loss of the policy model for the chosen responses

## CPO variants

### Simple Preference Optimization (SimPO)

[Simple Preference Optimization](https://huggingface.co/papers/2405.14734) (SimPO) by [Yu Meng](https://huggingface.co/yumeng5), [Mengzhou Xia](https://huggingface.co/mengzhouxia), and [Danqi Chen](https://huggingface.co/cdq10131) proposes a simpler and more effective preference optimization algorithm than DPO without using a reference model. The key designs in SimPO are (1) using length-normalized log likelihood as the implicit reward, and (2) incorporating a target reward margin in the Bradley-Terry ranking objective. The official code can be found at [princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO).

The abstract from the paper is the following:

> Direct Preference Optimization (DPO) is a widely used offline preference optimization algorithm that reparameterizes reward functions in reinforcement learning from human feedback (RLHF) to enhance simplicity and training stability. In this work, we propose SimPO, a simpler yet more effective approach. The effectiveness of SimPO is attributed to a key design: using the average log probability of a sequence as the implicit reward. This reward formulation better aligns with model generation and eliminates the need for a reference model, making it more compute and memory efficient. Additionally, we introduce a target reward margin to the Bradley-Terry objective to encourage a larger margin between the winning and losing responses, further enhancing the algorithm's performance. We compare SimPO to DPO and its latest variants across various state-of-the-art training setups, including both base and instruction-tuned models like Mistral and Llama3. We evaluated on extensive instruction-following benchmarks, including AlpacaEval 2, MT-Bench, and the recent challenging Arena-Hard benchmark. Our results demonstrate that SimPO consistently and significantly outperforms existing approaches without substantially increasing response length. Specifically, SimPO outperforms DPO by up to 6.4 points on AlpacaEval 2 and by up to 7.5 points on Arena-Hard. Our top-performing model, built on Llama3-8B-Instruct, achieves a remarkable 44.7 length-controlled win rate on AlpacaEval 2 -- surpassing Claude 3 Opus on the leaderboard, and a 33.8 win rate on Arena-Hard -- making it the strongest 8B open-source model.

The SimPO loss is integrated in the [`experimental.cpo.CPOTrainer`], as it's an alternative loss that adds a reward margin, allows for length normalization, and does not use BC regularization. To use this loss, just turn on `loss_type="simpo"` and `cpo_alpha=0.0` in the [`experimental.cpo.CPOConfig`] and set the `simpo_gamma` to a recommended value.

### CPO-SimPO

We also offer the combined use of CPO and SimPO, which enables more stable training and improved performance. Learn more details at [CPO-SimPO GitHub](https://github.com/fe1ixxu/CPO_SIMPO). To use this method, simply enable SimPO by setting `loss_type="simpo"` and a non-zero `cpo_alpha` in the [`experimental.cpo.CPOConfig`].

### AlphaPO

The [AlphaPO -- Reward shape matters for LLM alignment](https://huggingface.co/papers/2501.03884) (AlphaPO) method by Aman Gupta, Shao Tang, Qingquan Song, Sirou Zhu, [Jiwoo Hong](https://huggingface.co/JW17), Ankan Saha, Viral Gupta, Noah Lee, Eunki Kim, Jason Zhu, Natesh Pillai, and S. Sathiya Keerthi is also implemented in the [`experimental.cpo.CPOTrainer`]. AlphaPO is an alternative method that applies a transformation to the reward function shape in the context of SimPO loss. The abstract from the paper is the following:

> Reinforcement Learning with Human Feedback (RLHF) and its variants have made huge strides toward the effective alignment of large language models (LLMs) to follow instructions and reflect human values. More recently, Direct Alignment Algorithms (DAAs) have emerged in which the reward modeling stage of RLHF is skipped by characterizing the reward directly as a function of the policy being learned. Some popular examples of DAAs include Direct Preference Optimization (DPO) and Simple Preference Optimization (SimPO). These methods often suffer from likelihood displacement, a phenomenon by which the probabilities of preferred responses are often reduced undesirably. In this paper, we argue that, for DAAs the reward (function) shape matters. We introduce AlphaPO, a new DAA method that leverages an α-parameter to help change the shape of the reward function beyond the standard log reward. AlphaPO helps maintain fine-grained control over likelihood displacement and overoptimization. Compared to SimPO, one of the best performing DAAs, AlphaPO leads to about 7% to 10% relative improvement in alignment performance for the instruct versions of Mistral-7B and Llama3-8B while achieving 15% to 50% relative improvement over DPO on the same models. The analysis and results presented highlight the importance of the reward shape and how one can systematically change it to affect training dynamics, as well as improve alignment performance.

To use this loss as described in the paper, we can set the `loss_type="alphapo"` which automatically sets `loss_type="simpo"` and `cpo_alpha=0.0`, together with `alpha` and `simpo_gamma` to recommended values in the [`experimental.cpo.CPOConfig`]. Alternatively, you can manually set `loss_type="simpo"`, `cpo_alpha=0.0`, together with `alpha` and `simpo_gamma` to recommended values. Other variants of this method are also possible, such as setting `loss_type="ipo"` and `alpha` to any non-zero value.

## Loss functions

The CPO algorithm supports several loss functions. The loss function can be set using the `loss_type` parameter in the [`experimental.cpo.CPOConfig`]. The following loss functions are supported:

| `loss_type=` | Description |
| --- | --- |
| `"sigmoid"` (default) | Given the preference data, we can fit a binary classifier according to the Bradley-Terry model, and in fact, the [DPO](https://huggingface.co/papers/2305.18290) authors propose the sigmoid loss on the normalized likelihood via the `logsigmoid` to fit a logistic regression. |
| `"hinge"` | The [RSO](https://huggingface.co/papers/2309.06657) authors propose to use a hinge loss on the normalized likelihood from the [SLiC](https://huggingface.co/papers/2305.10425) paper. In this case, the `beta` is the reciprocal of the margin. |
| `"ipo"` | The [IPO](https://huggingface.co/papers/2310.12036) authors provide a deeper theoretical understanding of the DPO algorithms and identify an issue with overfitting and propose an alternative loss. In this case, the `beta` is the reciprocal of the gap between the log-likelihood ratios of the chosen vs the rejected completion pair, and thus the smaller the `beta`, the larger this gap is. As per the paper, the loss is averaged over log-likelihoods of the completion (unlike DPO, which is summed only). |
| `"simpo"` | The [SimPO](https://huggingface.co/papers/2405.14734) method is also implemented in the [`experimental.cpo.CPOTrainer`]. SimPO is an alternative loss that adds a reward margin, allows for length normalization, and does not use BC regularization. To use this loss, simply set `loss_type="simpo"` and `cpo_alpha=0.0` in the [`experimental.cpo.CPOConfig`] and `simpo_gamma` to a recommended value. |
| `"alphapo"` | The [AlphaPO](https://huggingface.co/papers/2501.03884) method is also implemented in the [`experimental.cpo.CPOTrainer`]. This is syntactic sugar that automatically sets `loss_type="simpo"` and `cpo_alpha=0.0`. AlphaPO applies a transformation to the reward function shape in the context of SimPO loss when the `alpha` parameter is non-zero. |

### For Mixture of Experts Models: Enabling the auxiliary loss

MOEs are the most efficient if the load is about equally distributed between experts.  
To ensure that we train MOEs similarly during preference-tuning, it is beneficial to add the auxiliary loss from the load balancer to the final loss.

This option is enabled by setting `output_router_logits=True` in the model config (e.g., [`~transformers.MixtralConfig`]).  
To scale how much the auxiliary loss contributes to the total loss, use the hyperparameter `router_aux_loss_coef=...` (default: `0.001`) in the model config.

## CPOTrainer

[[autodoc]] experimental.cpo.CPOTrainer
    - train
    - save_model
    - push_to_hub

## CPOConfig

[[autodoc]] experimental.cpo.CPOConfig
