# ORPO Trainer

[![](https://img.shields.io/badge/All_models-ORPO-blue)](https://huggingface.co/models?other=orpo,trl)

## Overview

Odds Ratio Preference Optimization (ORPO) wa introduced in [ORPO: Monolithic Preference Optimization without Reference Model](https://huggingface.co/papers/2403.07691) by [Jiwoo Hong](https://huggingface.co/JW17), [Noah Lee](https://huggingface.co/nlee-208), and [James Thorne](https://huggingface.co/j6mes).

The abstract from the paper is the following:

> While recent preference alignment algorithms for language models have demonstrated promising results, supervised fine-tuning (SFT) remains imperative for achieving successful convergence. In this paper, we study the crucial role of SFT within the context of preference alignment, emphasizing that a minor penalty for the disfavored generation style is sufficient for preference-aligned SFT. Building on this foundation, we introduce a straightforward and innovative reference model-free monolithic odds ratio preference optimization algorithm, ORPO, eliminating the necessity for an additional preference alignment phase. We demonstrate, both empirically and theoretically, that the odds ratio is a sensible choice for contrasting favored and disfavored styles during SFT across the diverse sizes from 125M to 7B. Specifically, fine-tuning Phi-2 (2.7B), Llama-2 (7B), and Mistral (7B) with ORPO on the UltraFeedback alone surpasses the performance of state-of-the-art language models with more than 7B and 13B parameters: achieving up to 12.20% on AlpacaEval_{2.0} (Figure 1), 66.19% on IFEval (instruction-level loose, Table 6), and 7.32 in MT-Bench (Figure 12). We release code and model checkpoints for Mistral-ORPO-alpha (7B) and Mistral-ORPO-beta (7B).

It studies the crucial role of SFT within the context of preference alignment. Using preference data the method posits that a minor penalty for the disfavored generation together with a strong adaption signal to the chosen response via a simple log odds ratio term appended to the NLL loss is sufficient for preference-aligned SFT.

Thus ORPO is a reference model-free preference optimization algorithm eliminating the necessity for an additional preference alignment phase thus saving compute and memory.

The official code can be found in [xfactlab/orpo](https://github.com/xfactlab/orpo).

This post-training method was contributed by [Kashif Rasul](https://huggingface.co/kashif), [Lewis Tunstall](https://huggingface.co/lewtun) and [Alvaro Bartolome](https://huggingface.co/alvarobartt).

## Quick start

This example demonstrates how to train a model using the ORPO method. We use the [Qwen 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) as the base model. We use the preference data from the [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback). You can view the data in the dataset here:

<iframe
  src="https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

Below is the script to train the model:

```python
# train_orpo.py
from datasets import load_dataset
from trl import ORPOConfig, ORPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = ORPOConfig(output_dir="Qwen2-0.5B-ORPO", logging_steps=10)
trainer = ORPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
```

Execute the script using the following command:

```bash
accelerate launch train_orpo.py
```

Distributed across 8 GPUs, the training takes approximately 30 minutes. You can verify the training progress by checking the reward graph. An increasing trend in the reward margin indicates that the model is improving and generating better responses over time.

![](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/orpo-qwen2-reward-margin.png)

To see how the [trained model](https://huggingface.co/trl-lib/Qwen2-0.5B-ORPO) performs, you can use the [TRL Chat CLI](clis#chat-interface).

<pre><code>$ trl chat --model_name_or_path trl-lib/Qwen2-0.5B-ORPO
<strong><span style="color: red;">&lt;quentin_gallouedec&gt;:</span></strong>
What is the best programming language?

<strong><span style="color: blue;">&lt;trl-lib/Qwen2-0.5B-ORPO&gt;:</span></strong>
The best programming language for specific applications can vary depending on the use case and knowledge level of the programmer. Here are some general factors that can be used as input to choose the best programming language:

 <strong><span style="color: green;">1</span></strong> Ease of use: Some programming languages are more user-friendly than others, such as Python, Java, or Ruby. Python is popular due to its simplicity and great scalability.
 <strong><span style="color: green;">2</span></strong> Versatility: The ability to work with a wide range of data structures and frameworks can define the language as versatile.
 <strong><span style="color: green;">3</span></strong> Ease of learning: Different programming languages have different learning curves, so users must be willing to take some time to master one.
 <strong><span style="color: green;">4</span></strong> Community support: The broader community of developers and enthusiasts in the selected programming language can provide great support and resources.
 <strong><span style="color: green;">5</span></strong> Reusability: Languages that emphasize code reuse and can be easily modifiable can be more suitable for software development.

The best programming language based on these factors is subjective and depends on what the programmer intends to accomplish.
</code></pre>


## Expected dataset format

ORPO requires a [preference dataset](dataset_formats#preference). The [`ORPOTrainer`] supports both [conversational](dataset_formats#conversational-dataset-format) and [standard](dataset_formats#standard-dataset-format) dataset format. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

Although the [`ORPOTrainer`] supports both explicit and implicit prompts, we recommend using explicit prompts. If provided with an implicit prompt dataset, the trainer will automatically extract the prompt from the `"chosen"` and `"rejected"` columns. For more information, refer to the [preference style](dataset_formats#preference) section.

## Example script

We provide an example script to train a model using the ORPO method. The script is available in [`examples/scripts/orpo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py)

To test the ORPO script with the [Qwen2 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) on the [UltraFeedback dataset](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized), run the following command:

```bash
accelerate launch examples/scripts/orpo.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --num_train_epochs 1 \
    --logging_steps 25 \
    --output_dir Qwen2-0.5B-DPO
```

## Usage tips

### For Mixture of Experts Models: Enabling the auxiliary loss

MOEs are the most efficient if the load is about equally distributed between experts.  
To ensure that we train MOEs similarly during preference-tuning, it is beneficial to add the auxiliary loss from the load balancer to the final loss.

This option is enabled by setting `output_router_logits=True` in the model config (e.g. [`~transformers.MixtralConfig`]).  
To scale how much the auxiliary loss contributes to the total loss, use the hyperparameter `router_aux_loss_coef=...` (default: `0.001`) in the model config.

## Logged metrics

While training and evaluating we record the following reward metrics:

- `rewards/chosen`: the mean log probabilities of the policy model for the chosen responses scaled by beta
- `rewards/rejected`: the mean log probabilities of the policy model for the rejected responses scaled by beta
- `rewards/accuracies`: mean of how often the chosen rewards are > than the corresponding rejected rewards
- `rewards/margins`: the mean difference between the chosen and corresponding rejected rewards
- `log_odds_chosen`: the mean log odds ratio of the chosen responses over the rejected responses
- `log_odds_ratio`: the mean of the `log(sigmoid(log_odds_chosen))`
- `nll_loss`: the mean negative log likelihood loss from the SFT part of the loss over chosen responses
 
## ORPOTrainer

[[autodoc]] ORPOTrainer

## ORPOConfig

[[autodoc]] ORPOConfig
