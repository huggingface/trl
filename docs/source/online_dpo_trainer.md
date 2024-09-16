# Online DPO Trainer

## Overview 

Online DPO was proposed in [Direct Language Model Alignment from Online AI Feedback](https://huggingface.co/papers/2402.04792) by Shangmin Guo, Biao Zhang, Tianlin Liu, Tianqi Liu, Misha Khalman, Felipe Llinares, Alexandre Rame, Thomas Mesnard, Yao Zhao, Bilal Piot, Johan Ferret, and Mathieu Blondel. 

The abstract from the paper is the following:

> Direct alignment from preferences (DAP) methods, such as DPO, have recently emerged as efficient alternatives to reinforcement learning from human feedback (RLHF), that do not require a separate reward model. However, the preference datasets used in DAP methods are usually collected ahead of training and never updated, thus the feedback is purely offline. Moreover, responses in these datasets are often sampled from a language model distinct from the one being aligned, and since the model evolves over training, the alignment phase is inevitably off-policy. In this study, we posit that online feedback is key and improves DAP methods. Our method, online AI feedback (OAIF), uses an LLM as annotator: on each training iteration, we sample two responses from the current model and prompt the LLM annotator to choose which one is preferred, thus providing online feedback. Despite its simplicity, we demonstrate via human evaluation in several tasks that OAIF outperforms both offline DAP and RLHF methods. We further show that the feedback leveraged in OAIF is easily controllable, via instruction prompts to the LLM annotator.

The current implementation uses reward models for scoring completions -- see [Reward Bench](https://huggingface.co/spaces/allenai/reward-bench) for a leaderboard of public models you can use.

This post-training method was contributed by [Michael Noukhovitch](https://huggingface.co/mnoukhov), [Shengyi Costa Huang](https://huggingface.co/vwxyzjn), [Quentin Gallouédec](https://huggingface.co/qgallouedec), and [Edward Beeching](https://huggingface.co/edbeeching).

## Usage tips

> [!WARNING]
> Make sure that the SFT model and reward model use the _same_ chat template. Otherwise, you may find the model completions are scored incorrectly during training.

## Expected dataset format

Online DPO only requires a [prompt-only dataset](dataset_format#preference) (unlike offline DPO, that expects [preference dataset](dataset_format#preference)). The [`OnlineDPOTrainer`] supports both [conversational](dataset_format#conversational-dataset-format) and [standard](dataset_format#standard-dataset-format) dataset format.

## Quick start

This example demonstrates how to train a model using the online DPO method. We use the [Qwen 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) as the base model and the [Qwen 0.5B reward model](https://huggingface.co/trl-lib/Qwen2-0.5B-Reward) as the reward model. We use the prompts from the [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback):

<iframe
  src="https://huggingface.co/datasets/trl-lib/ultrafeedback-prompt/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

The basic API is as follows:

```python
# train_online_dpo.py
from datasets import load_dataset
from trl import OnlineDPOConfig, OnlineDPOTrainer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
reward_model = AutoModelForSequenceClassification.from_pretrained("trl-lib/Qwen2-0.5B-Reward", num_labels=1)
train_dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

args = OnlineDPOConfig(output_dir="online-dpo-qwen2", logging_steps=10)
trainer = OnlineDPOTrainer(
    model=model,
    reward_model=reward_model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
)
trainer.train()
```

We run this script with the following command:

```bash
accelerate launch train_online_dpo.py
```

After approximately 1 hour of training, the model is trained and we can demonstrate the completions:

````python
>>> from transformers import pipeline
>>> generator = pipeline("text-generation", model="online-dpo-qwen2/checkpoint-500", device="cuda")
>>> question = "Can you tell me which shell command can be used to display the CPU usage of a specific process in Linux? And what is the syntax for this command?"
>>> output = generator([{"role": "user", "content": question}], max_new_tokens=200, return_full_text=False)[0]
>>> print(output["generated_text"])
Yes, you can use the `top` command in Linux to display the CPU usage of a specific process.
The syntax for the `top` command depends on your version of Linux. Here's an example of how to run `top` with the `-b` option:
```
top -b
```

This will display the CPU usage of all processes in the system at the top level. You can also specify additional options by adding them after the `-b` option using square brackets (`[]`). For example, to display only the running processes and their CPU usage, you would add the following options to the command:
```
top -b --running=1
```

Note that some versions of Linux may require you to set up a user account or enable logging before running the `top` command.
````

### Example script

To test the online DPO script with 1B parameter models, run:

```bash
python examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-1b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --output_dir pythia-1b-tldr-online-dpo \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 3 \
    --max_new_tokens 53 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --push_to_hub
```

Tips:

* `objective/rlhf_reward` is the ultimate objective of online DPO training. If training works as intended, this metric should keep going up.
* We recommend using the "EOS trick" via the `--missing_eos_penalty` argument, which subtracts from the rewards a fixed scalar penalty for completions that do not end with an EOS token. This can help the model learn to generate more coherent completions.



### Explanation of the logged metrics

The logged metrics are as follows. Here is an example [tracked run at Weights and Biases](https://wandb.ai/huggingface/trl/runs/dd2o3g35)

* `objective/kl`: The mean Kullback-Leibler (KL) divergence between the current model and reference model.
* `objective/entropy`: The mean entropy of the model, indicating the randomness of the actions chosen by the model.
* `objective/non_score_reward`: The mean reward from non-score-related sources, basically `beta * kl.sum(1)`, where `beta` is the KL penalty coefficient and `kl` is the per-token KL divergence.
* `objective/rlhf_reward`: The mean RLHF reward, which is `score - non_score_reward`.
* `objective/scores`: The mean scores returned by the reward model / environment.
* `objective/scores_margin`: The mean score margin (according to the external reward model) between the chosen and rejected completions.
* `rewards/accuracies`: The accuracies of the online DPO's implicit reward model.
* `rewards/chosen`: The mean reward (according to online DPO's implicit reward model)of the chosen completions.
* `rewards/rejected`: The mean reward (according to online DPO's implicit reward model) of the rejected completions.
* `rewards/margins`: The mean reward margin (according to online DPO's implicit reward model) between the chosen and rejected completions.
* `logps/chosen`: The mean log probabilities of the chosen completions.
* `logps/rejected`: The mean log probabilities of the rejected completions.
* `val/contain_eos_token`: The fraction of completions which contain an EOS token.


## What is my model doing exactly?

To help you understand what your model is doing, we periodically log some sample completions from the model via [`LogCompletionsCallback`]. You can find an example [tracked run at Weights and Biases](https://wandb.ai/huggingface/trl/runs/hlzevfro), which allows you to see the model's response at different stages of training. By default we generate during training, but you can customize the number of prompts to generate completions for in [`LogCompletionsCallback`]. 


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
    --push_to_hub \

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

![](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/online_dpo_scaling.png)

The online DPO checkpoint gets increasingly more win rate as we scale up the model sizes. This is a good sign that the online DPO implementation is working as intended.

## OnlineDPOTrainer

[[autodoc]] OnlineDPOTrainer


## OnlineDPOConfig

[[autodoc]] OnlineDPOConfig