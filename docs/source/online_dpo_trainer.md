# Online DPO Trainer

TRL supports training LLMs with online DPO ([Guo et al., 2024](https://huggingface.co/papers/2402.04792)) with a reward model (RM). The idea of online DPO is to generate completions based on prompts and either have an RM or a LLM judge to rank the responses. Then the model is updated with the ranked responses using the DPO loss.

While [Guo et al. (2024)](https://huggingface.co/papers/2402.04792) used a LLM judge, in this implementation we just used a RM.


## Get started

The basic API looks as follows:

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
tok.add_special_tokens({"pad_token": "[PAD]"})
# The model to optimise
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
ref_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
# The model to score completions with. In practice, you will need a fine-tuned reward model. See Reward Bench for some good ones: https://huggingface.co/spaces/allenai/reward-bench
reward_model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct", num_labels=1)
train_dataset = Dataset.from_dict(
    {"input_ids": [tok.encode("Q: Hi how are you? A:")] * NUM_DUMMY_SAMPLES})
eval_dataset = Dataset.from_dict(
    {"input_ids": [tok.encode("Q: What do you like to eat A:")] * NUM_DUMMY_SAMPLES})
trainer = OnlineDPOTrainer(
    OnlineDPOConfig(
        output_dir="online-dpo-model",
    ),
    model=model,
    ref_model=ref_model,
    reward_model=reward_model,
    tokenizer=tok,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```


To just run the online DPO script to make sure the trainer can run, you can run the following command to train an online DPO model with a dummy reward model.

```bash
python examples/scripts/online_dpo.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-14m \
    --sft_model_path EleutherAI/pythia-14m \
    --reward_model_path EleutherAI/pythia-14m \
    --non_eos_penalty \
    --stop_token eos \
    --response_length 53 \
    --sanity_check
```


## Explanation of the logged metrics

The logged metrics are as follows. Here is an example [tracked run at Weights and Biases](https://wandb.ai/huggingface/trl/runs/dd2o3g35)

* `eps`: Tracks the number of episodes per second.
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
* `val/num_eos_tokens`: The number of end-of-sequence (EOS) tokens generated, which can indicate the number of complete responses.
* `lr`: lr: The current learning rate used by the optimizer.
* `episode`: episode: The current global step or episode count in the training process.


## Cookbook

* Debugging TIP: `objective/rlhf_reward`: this is the ultimate objective of the RLHF training. If training works as intended, this metric should keep going up.
* Memory TIP: If you are running out of memory, you can try to reduce the `--per_device_train_batch_size` or increase the `--gradient_accumulation_steps` to reduce the memory footprint.
* Memory TIP: If you have multiple GPUs, you can also run training with DeepSpeed stage 3 to reduce the memory footprint `accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml`.
* Usage TIP: We recommend to use the "EOS trick" via `--non_eos_penalty --stop_token eos`, which replaces the score of completions that do not end with an EOS token with a static scalar penalty `--penalty_reward_value`. This can help the model learn to generate more coherent completions.


## What is my model doing exactly?

To help you understand what your model is doing, we periodically log some sample completions from the model. Here is an example of a completion. In an example [tracked run at Weights and Biases](https://wandb.ai/huggingface/trl/runs/dd2o3g35), it looks like the following, allowing you to see the model's response at different stages of training. By default we generate `--num_sample_generations 10` during training, but you can customize the number of generations.

![](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/ppov2_completions.gif)


In the logs the sampled generations look like 

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ query                           ┃ model response                  ┃ score    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│  SUBREDDIT: r/AskReddit         │  I'm in love with a friend, and │ 3.921875 │
│                                 │ I don't know how to get rid of  │          │
│ TITLE: How do you get someone   │ those feelings. I'm             │          │
│ out of your head?               │ desperate.<|endoftext|>[PAD][P… │          │
│                                 │                                 │          │
│ POST: Hi,                       │                                 │          │
│ I'm 22, and I have been with my │                                 │          │
│ girlfriend for 5 years now. We  │                                 │          │
│ recently moved together. We've  │                                 │          │
│ always loved each other         │                                 │          │
│ intensely.                      │                                 │          │
│                                 │                                 │          │
│ Problem, I recently started to  │                                 │          │
│ have feelings for an other      │                                 │          │
│ person (a friend). This person  │                                 │          │
│ has had a boyfriend for now 3   │                                 │          │
│ years, and has absolutely no    │                                 │          │
│ ideas. Those feelings were so   │                                 │          │
│ strong, it was hard to hide     │                                 │          │
│ them. After 2 months of me      │                                 │          │
│ being distant and really sad,   │                                 │          │
│ my girlfriend forced me to say  │                                 │          │
│ what was bothering me. I'm not  │                                 │          │
│ a good liar, and now she knows. │                                 │          │
│                                 │                                 │          │
│ We decided to give us a week    │                                 │          │
│ alone, I went to my parents.    │                                 │          │
│                                 │                                 │          │
│ Now, I'm completely lost. I     │                                 │          │
│ keep on thinking about this     │                                 │          │
│ person, and I hate that. I      │                                 │          │
│ would like for those feelings   │                                 │          │
│ to go away, to leave me alone.  │                                 │          │
│ But I can't.                    │                                 │          │
│                                 │                                 │          │
│ What do I do? It's been 3       │                                 │          │
│ months now, and I'm just        │                                 │          │
│ desperate.                      │                                 │          │
│                                 │                                 │          │
│ TL;DR:                          │                                 │          │
├─────────────────────────────────┼─────────────────────────────────┼──────────┤
│  SUBREDDIT: r/pettyrevenge      │  My mom woke me up with a loud  │ 6.84375  │
│                                 │ TV. I blasted Gangnam Style on  │          │
│ TITLE: So, my mom woke me up    │ repeat, with the bass cranked   │          │
│ with a loud TV.                 │ up as high as it could          │          │
│                                 │ go.<|endoftext|>[PAD][PAD][PAD… │          │
│ POST: She was in her living     │                                 │          │
│ room, watching TV. This was at  │                                 │          │
│ about 8:30 in the morning, and  │                                 │          │
│ she was exercising. She turned  │                                 │          │
│ the TV up extra loud to hear it │                                 │          │
│ over her excercycle, and woke   │                                 │          │
│ me up. I went in there asking   │                                 │          │
│ for her to turn it down. She    │                                 │          │
│ said she didn't have to; I      │                                 │          │
│ explained that I always used    │                                 │          │
│ headphones so she didn't have   │                                 │          │
│ to deal with my noise and that  │                                 │          │
│ she should give me a little     │                                 │          │
│ more respect, given that I paid │                                 │          │
│ rent at the time.               │                                 │          │
│                                 │                                 │          │
│ She disagreed. I went back to   │                                 │          │
│ my room, rather pissed off at   │                                 │          │
│ the lack of equality. I had no  │                                 │          │
│ lock on my door; but I had a    │                                 │          │
│ dresser right next to it, so I  │                                 │          │
│ pulled one of the drawers out   │                                 │          │
│ enough so that it caused the    │                                 │          │
│ door to not be openable. Then,  │                                 │          │
│ I turned my speakers up really  │                                 │          │
│ loud and blasted Gangnam Style  │                                 │          │
│ on repeat, with the bass        │                                 │          │
│ cranked up as high as it could  │                                 │          │
│ go.                             │                                 │          │
│                                 │                                 │          │
│ If you hate Gangnam Style for   │                                 │          │
│ being overplayed, you will see  │                                 │          │
│ why I chose that particular     │                                 │          │
│ song. I personally don't mind   │                                 │          │
│ it. But here's the thing about  │                                 │          │
│ my bass; it vibrates the walls, │                                 │          │
│ making one hell of a lot of     │                                 │          │
│ noise. Needless to say, my mom  │                                 │          │
│ was not pleased and shut off    │                                 │          │
│ the internet. But it was oh so  │                                 │          │
│ worth it.                       │                                 │          │
│                                 │                                 │          │
│ TL;DR:                          │                                 │          │
└─────────────────────────────────┴─────────────────────────────────┴──────────┘
```

## Implementation details

Many online implementation details are borrowed from the PPOv2Trainer, which is itself based on the [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://huggingface.co/papers/2403.17031). Here are some additional implementation details:

1. When we turn on the EOS trick (i.e., replacing the score of completions that do not end with an EOS token with a scalar penalty score like `-1`) via `--non_eos_penalty --stop_token eos`, it's possible that the chosen and rejected completions have the same score. In this case, we will naively select the completion with the lower index and the chosen completion.

## Benchmark experiments

To validate the online DPO implementation works, we ran experiments on the 1B and 6.9B models. Here are the commands we used to run the experiments. We take the SFT / RM models directly from [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://huggingface.co/papers/2403.17031).


```
# 1B Online DPO experiment
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/online_dpo.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo_tldr \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --local_rollout_forward_batch_size 32 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 1000000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr  \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --save_strategy no \
    --non_eos_penalty \
    --stop_token eos \
    --beta 0.1 \
    --response_length 53 \
    --push_to_hub

# 6.9B Online DPO experiment
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/online_dpo.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo_tldr_6.9b \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --local_rollout_forward_batch_size 8 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 1000000 \
    --model_name_or_path EleutherAI/pythia-6.9b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-6.9b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-6.9b-deduped__reward__tldr \
    --save_strategy no \
    --non_eos_penalty \
    --stop_token eos \
    --beta 0.1 \
    --response_length 53 \
    --push_to_hub
```

Checkpoints and experiment tracking are available at:

- [🤗 Model checkpoint](https://huggingface.co/vwxyzjn/ppo_tldr)
- [🐝 Tracked experiment](https://wandb.ai/huggingface/trl/runs/dd2o3g35)


To evaluate, we use [vLLM](https://github.com/vllm-project/vllm) to load the checkpoints and GPT-4o mini as a judge model to evaluate the generated TL;DR against the reference TL;DR.
For more information on how to use judges, see [Judges](judges).

```bash
$ python examples/scripts/evals/judge_tldr.py --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr --judge_model gpt-4o-mini --num_examples 1000
Model win rate: 33.00%
python examples/scripts/evals/judge_tldr.py --model_name_or_path cleanrl/EleutherAI_pythia-6.9b-deduped__sft__tldr --judge_model gpt-4o-mini --num_examples 1000
Model win rate: 41.50%
python examples/scripts/evals/judge_tldr.py --model_name_or_path vwxyzjn/online_dpo_tldr --judge_model gpt-4o-mini --num_examples 1000
Model win rate: 62.60%
python examples/scripts/evals/judge_tldr.py --model_name_or_path vwxyzjn/online_dpo_tldr_6.9b --judge_model gpt-4o-mini --num_examples 1000
Model win rate: 74.20%
```

We can then plot the RLHF scaling chart.

```python
import matplotlib.pyplot as plt

data = {
    "SFT": [[1e9, 6.9e9], [0.33, 0.415]],
    "Online DPO": [[1e9, 6.9e9], [0.626, 0.742]],
}
for model, (x, y) in data.items():
    plt.scatter(x, y, label=model)

plt.axhline(y=0.5, color="black", linestyle="-.", label="Human reference summary")
plt.title("RLHF scaling by model size")
plt.xlabel("Model size")
plt.ylabel("Win rate against reference summaries\n(according to GPT-4o mini)")
plt.xscale("log")
plt.xlim(5e8, 1.2e10)
plt.xticks([1e9, 1e10], ["1B", "10B"])
plt.legend()
plt.grid(True, which="both", ls="--", c="0.7")
plt.tight_layout()
plt.savefig("plot.png")
```


![](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/online_dpo_scaling.png)

The online DPO checkpoint gets increasingly more win rate as we scale up the model sizes. This is a good sign that the online DPO implementation is working as intended.
