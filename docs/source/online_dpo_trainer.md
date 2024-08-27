# Online DPO Trainer

TRL supports post-training LLMs with online DPO ([Guo et al., 2024](https://huggingface.co/papers/2402.04792)). The idea of online DPO is to generate completions per batch of prompts and have either a reward model or an LLM judge rank the responses as chosen or rejected. Then the model is updated with the ranked responses using the DPO loss.

While [Guo et al. (2024)](https://huggingface.co/papers/2402.04792) used an LLM judge to score model completions, the current implementation only supports reward models -- see [Reward Bench](https://huggingface.co/spaces/allenai/reward-bench) for a leaderboard of public models you can use.

## Get started

The basic API is as follows:

```python
from datasets import Dataset
from trl import OnlineDPOConfig, OnlineDPOTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
NUM_DUMMY_SAMPLES = 100
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
# The model to optimise
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
# The reference model to calculate the KL divergence against
ref_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
# The model to score completions with. In practice, you will need a reward model.
reward_model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct", num_labels=1)
train_dataset = Dataset.from_dict(
    {"prompt": ["Q: Hi how are you? A:"] * NUM_DUMMY_SAMPLES})
eval_dataset = Dataset.from_dict(
    {"prompt": ["Q: What do you like to eat A:"] * NUM_DUMMY_SAMPLES})
trainer = OnlineDPOTrainer(
    model=model,
    ref_model=ref_model,
    reward_model=reward_model,
    args=OnlineDPOConfig(output_dir="online-dpo-model"),
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

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
    --completion_length 53 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --push_to_hub
```

## Expected dataset format

Unlike standard DPO where one provides a dataset with chosen and rejected columns, for online DPO one just needs a dataset of prompts to generate the completions from. The [`OnlineDPOTrainer`] assumes that the dataset is preprocessed for model inference, so typically you will want to wrap your prompts in the messages format and then apply the chat template as follows:

```python
def prepare_dataset(row):
    """Apply chat template to messages"""
    row["prompt"] = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    return row

dataset = prepare_dataset(dataset)
```

## Explanation of the logged metrics

The logged metrics are as follows. Here is an example [tracked run at Weights and Biases](https://wandb.ai/huggingface/trl/runs/dd2o3g35)

* `objective/kl`: The mean Kullback-Leibler (KL) divergence between the current model and reference model.
* `objective/entropy`: The mean entropy of the model, indicating the randomness of the actions chosen by the model.
* `objective/non_score_reward`: The mean reward from non-score-related sources, basically `beta * kl.sum(1)`, where `beta` is the KL penalty coefficient and `kl` is the per-token KL divergence.
* `objective/rlhf_reward`: The mean RLHF reward, which is `score - non_score_reward`.
* `objective/scores`: The mean scores returned by the reward model / environment.
* `objective/scores_margin`: The mean score margin (according to the external reward model) between the chosen and rejected completions.
* `rewards/chosen`: The mean reward (according to online DPO's implicit reward model)of the chosen completions.
* `rewards/rejected`: The mean reward (according to online DPO's implicit reward model) of the rejected completions.
* `rewards/accuracies`: The accuracies of the online DPO's implicit reward model.
* `rewards/margins`: The mean reward margin (according to online DPO's implicit reward model) between the chosen and rejected completions.
* `logps/chosen`: The mean log probabilities of the chosen completions.
* `logps/rejected`: The mean log probabilities of the rejected completions.
* `val/contain_eos_token`: The fraction of completions which contain an EOS token.


## Cookbook

> [!IMPORTANT]
> Make sure the SFT model and reward model use the _same_ chat template. Otherwise you may find the model completions are scored incorrectly.


* Debugging TIP: `objective/rlhf_reward`: this is the ultimate objective of the RLHF training. If training works as intended, this metric should keep going up.
* Memory TIP: If you are running out of memory, you can try to reduce the `--per_device_train_batch_size` or increase the `--gradient_accumulation_steps` to reduce the memory footprint.
* Memory TIP: If you have multiple GPUs, you can also run training with DeepSpeed stage 3 to reduce the memory footprint `accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml`.
* Usage TIP: We recommend to use the "EOS trick" via `--missing_eos_penalty`, which replaces the score of completions that do not end with an EOS token with a static scalar penalty. This can help the model learn to generate more coherent completions.


## What is my model doing exactly?

To help you understand what your model is doing, we periodically log some sample completions from the model. Here is an example of a completion. In an example [tracked run at Weights and Biases](https://wandb.ai/huggingface/trl/runs/dd2o3g35), it looks like the following, allowing you to see the model's response at different stages of training. By default we generate during training, but you can customize the number of generations.


## Implementation details

Many online implementation details are borrowed from the [`PPOv2Trainer`], which is itself based on the [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://huggingface.co/papers/2403.17031). Here are some additional implementation details:

1. When we turn on the EOS trick (i.e., replacing the score of completions that do not end with an EOS token with a scalar penalty score like `-1`) via `--missing_eos_penalty`, it's possible that the chosen and rejected completions have the same score. In this case, we will naively select the completion with the lower index and the chosen completion.

## Benchmark experiments

To validate the online DPO implementation works, we ran experiments on the Pythia 1B, 2.8B, and 6.9B models on a single node of 8 x H100s. Here are the commands we used to run the experiments. We take the SFT / RM models directly from [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://huggingface.co/papers/2403.17031).


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
    --logging_steps 1 \
    --save_steps 0.1 \
    --push_to_hub
```

Checkpoints and experiment tracking are available at:

- [🤗 Model checkpoint](https://huggingface.co/collections/trl-lib/online-dpo-66acd3fa38a331a9cd457b07)
- [🐝 Tracked experiment](https://wandb.ai/huggingface/trl/runs/dd2o3g35)


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
