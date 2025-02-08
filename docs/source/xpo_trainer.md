# XPO Trainer

[![](https://img.shields.io/badge/All_models-XPO-blue)](https://huggingface.co/models?other=xpo,trl)

## Overview

Exploratory Preference Optimization (XPO) was proposed in the paper [Exploratory Preference Optimization: Harnessing Implicit Q*-Approximation for Sample-Efficient RLHF](https://huggingface.co/papers/2405.21046) by Tengyang Xie, Dylan J. Foster, Akshay Krishnamurthy, [Corby Rosset](https://huggingface.co/corbyrosset), [Ahmed Awadallah](https://huggingface.co/AhmedAwadallah), and Alexander Rakhlin. It is a simple online preference tuning method based on the DPO loss together with a reward model (RM). XPO augments the DPO objective with an exploration bonus allowing the method to explore outside the support of the initial model and human feedback data.

The abstract from the paper is the following:

> Reinforcement learning from human feedback (RLHF) has emerged as a central tool for language model alignment. We consider online exploration in RLHF, which exploits interactive access to human or AI feedback by deliberately encouraging the model to produce diverse, maximally informative responses. By allowing RLHF to confidently stray from the pre-trained model, online exploration offers the possibility of novel, potentially super-human capabilities, but its full potential as a paradigm for language model training has yet to be realized, owing to computational and statistical bottlenecks in directly adapting existing reinforcement learning techniques. We propose a new algorithm for online exploration in RLHF, Exploratory Preference Optimization (XPO), which is simple and practical -- a one-line change to (online) Direct Preference Optimization (DPO; Rafailov et al., 2023) -- yet enjoys the strongest known provable guarantees and promising empirical performance. XPO augments the DPO objective with a novel and principled exploration bonus, empowering the algorithm to explore outside the support of the initial model and human feedback data. In theory, we show that XPO is provably sample-efficient and converges to a near-optimal language model policy under natural exploration conditions, irrespective of whether the initial model has good coverage. Our analysis, which builds on the observation that DPO implicitly performs a form of Q*-approximation (or, Bellman error minimization), combines previously disparate techniques from language modeling and theoretical reinforcement learning in a serendipitous fashion through the perspective of KL-regularized Markov decision processes. Empirically, we find that XPO is more sample-efficient than non-exploratory DPO variants in a preliminary evaluation.

This post-training method was contributed by [Kashif Rasul](https://huggingface.co/kashif),  [Quentin Gallouédec](https://huggingface.co/qgallouedec) and [Lewis Tunstall](https://huggingface.co/lewtun).

## Quick start

This example demonstrates how to train a model using the XPO method. We use the [Qwen 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) as the base model and [`PairRMJudge`] as a judge. We use the prompts from the [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback). You can view the prompts in the dataset here:
<iframe
  src="https://huggingface.co/datasets/trl-lib/ultrafeedback-prompt/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

Below is the script to train the model:

```python
# train_xpo.py
from datasets import load_dataset
from trl import PairRMJudge, XPOConfig, XPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
judge = PairRMJudge()
train_dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

training_args = XPOConfig(output_dir="Qwen2-0.5B-XPO", logging_steps=10)
trainer = XPOTrainer(
    model=model, judge=judge, args=training_args, processing_class=tokenizer, train_dataset=train_dataset
)
trainer.train()
```

Execute the script using the following command:

```bash
accelerate launch train_xpo.py
```

Distributed across 8 GPUs, the training takes approximately 1 hour.

To see how the [trained model](https://huggingface.co/trl-lib/Qwen2-0.5B-XPO) performs, you can use the [TRL Chat CLI](clis#chat-interface).

<pre><code>$ trl chat --model_name_or_path trl-lib/Qwen2-0.5B-XPO
<strong><span style="color: red;">&lt;quentin_gallouedec&gt;:</span></strong>
What is the best programming language?

<strong><span style="color: blue;">&lt;trl-lib/Qwen2-0.5B-XPO&gt;:</span></strong>
The best programming language depends on individual preferences and familiarity with coding concepts. Some popular languages include Python, Java, C++, and JavaScript. 
</code></pre>

## Expected dataset type

XPO requires a [prompt-only dataset](dataset_formats#prompt-only). The [`XPOTrainer`] supports both [conversational](dataset_formats#conversational) and [standard](dataset_formats#standard) dataset format. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

## Usage tips

### Use a reward model

Instead of a judge, you can chose to use a reward model -- see [Reward Bench](https://huggingface.co/spaces/allenai/reward-bench) for a leaderboard of public models you can use. Below is a code example showing how to replace a judge with the [trl-lib/Qwen2-0.5B-Reward](https://huggingface.co/trl-lib/Qwen2-0.5B-Reward) model:

```diff
- from trl import PairRMJudge
+ from transformers import AutoModelForSequenceClassification

- judge = PairRMJudge()
+ reward_model = AutoModelForSequenceClassification.from_pretrained("trl-lib/Qwen2-0.5B-Reward", num_labels=1)

  trainer = XPOTrainer(
      ...
-     judge=judge,
+     reward_model=reward_model,
  )
```

<Tip warning={true}>

Make sure that the SFT model and reward model use the _same_ chat template and the same tokenizer. Otherwise, you may find the model completions are scored incorrectly during training.

</Tip>

### Encourage EOS token generation

When using a reward model, we may want the model to generate completions within a given length. During training, the model will generate completions up to the maximum length specified in the `max_new_tokens` argument of [`XPOConfig`]. If you want to penalize the model for not generating an EOS token before reaching the maximum length, you can use the `missing_eos_penalty` argument of [`XPOConfig`]:

```python
training_args = XPOConfig(..., max_new_tokens=128, missing_eos_penalty=1.0)
```

### Logging Completions

To better understand your model’s behavior during training, you can log sample completions periodically using the [`LogCompletionsCallback`].

```python
trainer = XPOTrainer(..., eval_dataset=eval_dataset)
completions_callback = LogCompletionsCallback(trainer, num_prompts=8)
trainer.add_callback(completions_callback)
```

This callback logs the model's generated completions directly to Weights & Biases.

![Logged Completions](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/wandb_completions.png)

## Example script

We provide an example script to train a model using the XPO method. The script is available in [`examples/scripts/xpo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/xpo.py)

To test the XPO script with the [Qwen2.5 0.5B model](https://huggingface.co/trl-lib/Qwen/Qwen2.5-0.5B-Instruct) on the [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback), run the following command:

```bash
python examples/scripts/xpo.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --judge pair_rm \
    --dataset_name trl-lib/ultrafeedback-prompt \
    --learning_rate 5.0e-7 \
    --logging_steps 25 \
    --output_dir Qwen2.5-0.5B-XPO-PairRM \
    --warmup_ratio 0.1 \
    --push_to_hub
```

## Logged metrics

The logged metrics are as follows:

* `loss/xpo`: The mean xpo part of the full loss.
* `loss/dpo`: The mean dpo part of the full loss.
* `objective/kl`: The mean KL divergence between the model and reference data.
* `objective/entropy`: The mean entropy of the model and reference data.
* `objective/model_scores`: The mean scores (according to the reward model) of the model completions.
* `objective/ref_scores`: The mean scores (according to the reward model) of the reference completions.
* `objective/scores_margin`: The mean score margin (according to the external reward model) between the chosen and rejected completions.
* `rewards/chosen`: The mean reward (according to XPO's DPO implicit reward model) of the chosen completions.
* `rewards/rejected`: The mean reward (according to XPO's DPO implicit reward model) of the rejected completions.
* `rewards/accuracies`: The accuracies of the XPO's implicit reward model.
* `rewards/margins`: The mean reward margin (according to online DPO's implicit reward model) between the chosen and rejected completions.
* `logps/chosen`: The mean log probabilities of the chosen completions.
* `logps/rejected`: The mean log probabilities of the rejected completions.
* `val/model_contain_eos_token`: The amount of times the model's output contains the eos token.
* `val/ref_contain_eos_token`: The amount of times the reference's output contains the eos token.
* `alpha`: The weight of the XPO loss term. Typically fixed, but can be made dynamic by passing a list to [`XPOConfig`].
* `beta`: The parameter that controls the weight of the loss term representing the deviation from the reference model. Typically fixed, but can be made dynamic by passing a list to [`XPOConfig`].


## XPOTrainer

[[autodoc]] XPOTrainer

## XPOConfig

[[autodoc]] XPOConfig
