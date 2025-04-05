# Online DPO Trainer

[![](https://img.shields.io/badge/All_models-Online_DPO-blue)](https://huggingface.co/models?other=online-dpo,trl)

## Overview 

Online DPO was proposed in [Direct Language Model Alignment from Online AI Feedback](https://huggingface.co/papers/2402.04792) by Shangmin Guo, Biao Zhang, Tianlin Liu, Tianqi Liu, Misha Khalman, Felipe Llinares, Alexandre Rame, Thomas Mesnard, Yao Zhao, Bilal Piot, Johan Ferret, and Mathieu Blondel. 

The abstract from the paper is the following:

> Direct alignment from preferences (DAP) methods, such as DPO, have recently emerged as efficient alternatives to reinforcement learning from human feedback (RLHF), that do not require a separate reward model. However, the preference datasets used in DAP methods are usually collected ahead of training and never updated, thus the feedback is purely offline. Moreover, responses in these datasets are often sampled from a language model distinct from the one being aligned, and since the model evolves over training, the alignment phase is inevitably off-policy. In this study, we posit that online feedback is key and improves DAP methods. Our method, online AI feedback (OAIF), uses an LLM as annotator: on each training iteration, we sample two responses from the current model and prompt the LLM annotator to choose which one is preferred, thus providing online feedback. Despite its simplicity, we demonstrate via human evaluation in several tasks that OAIF outperforms both offline DAP and RLHF methods. We further show that the feedback leveraged in OAIF is easily controllable, via instruction prompts to the LLM annotator.

This post-training method was contributed by [Michael Noukhovitch](https://huggingface.co/mnoukhov), [Shengyi Costa Huang](https://huggingface.co/vwxyzjn), [Quentin Gallouédec](https://huggingface.co/qgallouedec), and [Edward Beeching](https://huggingface.co/edbeeching).

## Quick start

This example demonstrates how to train a model using the online DPO method. We use the [Qwen 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) as the base model and [`PairRMJudge`] as a judge. We use the prompts from the [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback). You can view the prompts in the dataset here:

<iframe
  src="https://huggingface.co/datasets/trl-lib/ultrafeedback-prompt/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

Below is the script to train the model:

```python
# train_online_dpo.py
from datasets import load_dataset
from trl import OnlineDPOConfig, OnlineDPOTrainer, PairRMJudge
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
judge = PairRMJudge()
train_dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

training_args = OnlineDPOConfig(output_dir="Qwen2-0.5B-OnlineDPO", logging_steps=10)
trainer = OnlineDPOTrainer(
    model=model, judge=judge, args=training_args, processing_class=tokenizer, train_dataset=train_dataset
)
trainer.train()
```

Execute the script using the following command:

```bash
accelerate launch train_online_dpo.py
```

Distributed across 8 GPUs, the training takes approximately 1 hour. You can verify the training progress by checking the reward graph. An increasing trend in both the reward for rejected and chosen completions indicates that the model is improving and generating better responses over time.

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/online-dpo-qwen2.png)

To see how the [trained model](https://huggingface.co/trl-lib/Qwen2-0.5B-OnlineDPO) performs, you can use the [Transformers Chat CLI](https://huggingface.co/docs/transformers/quicktour#chat-with-text-generation-models).

<pre><code>$ transformers-cli chat --model_name_or_path trl-lib/Qwen2-0.5B-OnlineDPO
<strong><span style="color: red;">&lt;quentin_gallouedec&gt;:</span></strong>
What is the best programming language?

<strong><span style="color: blue;">&lt;trl-lib/Qwen2-0.5B-OnlineDPO&gt;:</span></strong>
The best programming language depends on your specific needs and priorities. Some people prefer imperative programming languages (like Haskell or Lisp), while others prefer functional programming languages (like Scala or Python). It's important to consider your work style, programming environment, and project requirements when choosing a programming language.
</code></pre>

## Expected dataset type

Online DPO only requires a [prompt-only dataset](dataset_formats#prompt-only) (unlike offline DPO, that expects [preference dataset](dataset_formats#preference)). The [`OnlineDPOTrainer`] supports both [conversational](dataset_formats#conversational) and [standard](dataset_formats#standard) dataset format. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

## Usage tips

### Use a reward model

Instead of a judge, you can chose to use a reward model -- see [Reward Bench](https://huggingface.co/spaces/allenai/reward-bench) for a leaderboard of public models you can use. Below is a code example showing how to replace a judge with the [trl-lib/Qwen2-0.5B-Reward](https://huggingface.co/trl-lib/Qwen2-0.5B-Reward) model:

```diff
- from trl import PairRMJudge
+ from transformers import AutoModelForSequenceClassification

- judge = PairRMJudge()
+ reward_model = AutoModelForSequenceClassification.from_pretrained("trl-lib/Qwen2-0.5B-Reward", num_labels=1)
+ reward_tokenizer = AutoTokenizer.from_pretrained("trl-lib/Qwen2-0.5B-Reward")

  trainer = OnlineDPOTrainer(
      ...
-     judge=judge,
+     reward_model=reward_model,
+     reward_processing_class=reward_tokenizer,
      ...
  )
```

### Encourage EOS token generation

When using a reward model, we may want the model to generate completions within a given length. During training, the model will generate completions up to the maximum length specified in the `max_new_tokens` argument of [`OnlineDPOConfig`]. If you want to penalize the model for not generating an EOS token before reaching the maximum length, you can use the `missing_eos_penalty` argument of [`OnlineDPOConfig`]:

```python
training_args = OnlineDPOConfig(..., max_new_tokens=128, missing_eos_penalty=1.0)
```

### Logging Completions

To better understand your model’s behavior during training, you can log sample completions periodically using the [`LogCompletionsCallback`].

```python
trainer = OnlineDPOTrainer(..., eval_dataset=eval_dataset)
completions_callback = LogCompletionsCallback(trainer, num_prompts=8)
trainer.add_callback(completions_callback)
```

This callback logs the model's generated completions directly to Weights & Biases.

![Logged Completions](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/wandb_completions.png)


## Example script

We provide an example script to train a model using the online DPO method. The script is available in [`examples/scripts/dpo_online.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo_online.py)

To test the online DPO script with the [Qwen2.5 0.5B model](https://huggingface.co/trl-lib/Qwen/Qwen2.5-0.5B-Instruct) on the [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback), run the following command:

```bash
python examples/scripts/dpo_online.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --judge pair_rm \
    --dataset_name trl-lib/ultrafeedback-prompt \
    --learning_rate 5.0e-7 \
    --logging_steps 25 \
    --output_dir Qwen2.5-0.5B-Online-DPO-PairRM \
    --warmup_ratio 0.1 \
    --push_to_hub
```

## Logged metrics

The logged metrics are as follows. Here is an example [tracked run at Weights and Biases](https://wandb.ai/huggingface/trl/runs/w4apmsi9)

* `objective/kl`: The mean Kullback-Leibler (KL) divergence between the current model and reference model.
* `objective/entropy`: The mean entropy of the model, indicating the randomness of the actions chosen by the model.
* `objective/non_score_reward`: The mean reward from non-score-related sources, basically `beta * kl.sum(1)`, where `beta` is the KL penalty coefficient and `kl` is the per-token KL divergence.
* `objective/rlhf_reward`: The mean RLHF reward, which is `scores - non_score_reward`. The `rlhf_reward` is the ultimate objective of online DPO training. If training works as intended, this metric should keep going up.
* `objective/scores`: The mean scores returned by the reward model.
* `objective/scores_margin`: The mean score margin (according to the external reward model) between the chosen and rejected completions.
* `rewards/chosen`: The mean reward (according to online DPO's implicit reward model)of the chosen completions.
* `rewards/rejected`: The mean reward (according to online DPO's implicit reward model) of the rejected completions.
* `rewards/accuracies`: The accuracies of the online DPO's implicit reward model.
* `rewards/margins`: The mean reward margin (according to online DPO's implicit reward model) between the chosen and rejected completions.
* `logps/chosen`: The mean log probabilities of the chosen completions.
* `logps/rejected`: The mean log probabilities of the rejected completions.
* `val/contain_eos_token`: The fraction of completions which contain an EOS token.
* `beta`: The parameter that controls the weight of the loss term representing the deviation from the reference model. Typically fixed, but can be made dynamic by passing a list to [`OnlineDPOConfig`].

## Benchmark experiments

To validate the online DPO implementation works, we ran experiments with the Pythia 1B, 2.8B, and 6.9B models on a single node of 8 x H100s. Here are the commands we used to run the experiments. We take the SFT / RM models directly from [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://huggingface.co/papers/2403.17031).


```
# 1B Online DPO experiment
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml \
    examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-1b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --output_dir pythia-1b-deduped-tldr-online-dpo \
    --beta 0.1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --max_new_tokens 53 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --logging_steps 20 \
    --save_steps 0.1 \
    --push_to_hub

# 2.8B Online DPO experiment
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-2.8b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-2.8b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --output_dir pythia-2.8b-deduped-tldr-online-dpo \
    --beta 0.1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --max_new_tokens 53 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --bf16 \
    --logging_steps 20 \
    --save_steps 0.1 \
    --push_to_hub

# 6.9B Online DPO experiment
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-6.9b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-6.9b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --output_dir pythia-6.9b-deduped-tldr-online-dpo \
    --beta 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --max_new_tokens 53 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 20 \
    --save_steps 0.1 \
    --push_to_hub
```

Checkpoints and experiment tracking are available at:

- [🤗 Model checkpoints](https://huggingface.co/collections/trl-lib/online-dpo-66acd3fa38a331a9cd457b07)
- [🐝 Tracked experiment](https://wandb.ai/huggingface/trl/reports/Online-DPO-experiments-for-TL-DR-summarisation--Vmlldzo5MTczMDU0)


To evaluate, we use [vLLM](https://github.com/vllm-project/vllm) to load the checkpoints and GPT-4o mini as a judge model to evaluate the generated TL;DR against the reference TL;DR.
For more information on how to use judges, see [Judges](judges).

```bash
$ python examples/scripts/evals/judge_tldr.py --model_name_or_path trl-lib/pythia-1b-deduped-tldr-sft --judge_model gpt-4o-mini --num_examples 1000
Model win rate: 33.00%
python examples/scripts/evals/judge_tldr.py --model_name_or_path trl-lib/pythia-6.9b-deduped-tldr-sft --judge_model gpt-4o-mini --num_examples 1000
Model win rate: 41.50%
python examples/scripts/evals/judge_tldr.py --model_name_or_path trl-lib/pythia-1b-deduped-tldr-online-dpo --judge_model gpt-4o-mini --num_examples 1000
Model win rate: 62.60%
python examples/scripts/evals/judge_tldr.py --model_name_or_path trl-lib/pythia-6.9b-deduped-tldr-online-dpo --judge_model gpt-4o-mini --num_examples 1000
Model win rate: 74.20%
```

We can then plot the RLHF scaling chart.

```python
import matplotlib.pyplot as plt

results = {
    "SFT": {1.0e9: 0.21, 2.8e9: 0.27, 6.9e9: 0.316},
    "online-dpo": {1.0e9: 0.542, 2.8e9: 0.746, 6.9e9: 0.796},
    "offline-dpo": {1.0e9: 0.422, 2.8e9: 0.517, 6.9e9: 0.701},
}


plt.plot(results["SFT"].keys(), results["SFT"].values(), label="SFT", marker="o")
plt.plot(results["online-dpo"].keys(), results["online-dpo"].values(), label="Online-dpo with RM judge", marker="o")
plt.plot(results["offline-dpo"].keys(), results["offline-dpo"].values(), label="Offline-dpo", marker="o")
plt.axhline(y=0.5, color="black", linestyle="-.", label="Human reference summary")
plt.xscale("log")
plt.xlabel("Model size")
plt.ylabel("Win rate against reference summaries\n(according to GPT-4-0613)")
plt.title("DPO scaling by model size")
plt.legend()
plt.xlim(5e8, 1.2e10)
plt.xticks([1e9, 3e9, 1e10], ["1B", "3B", "10B"])
plt.grid(True, which="both", ls="--", c="0.7")
plt.tight_layout()
plt.show()
```

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/online_dpo_scaling.png)

The online DPO checkpoint gets increasingly more win rate as we scale up the model sizes. This is a good sign that the online DPO implementation is working as intended.

## OnlineDPOTrainer

[[autodoc]] OnlineDPOTrainer

## OnlineDPOConfig

[[autodoc]] OnlineDPOConfig
