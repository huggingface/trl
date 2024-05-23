# PPOv2 Trainer

TRL supports training LLMs with [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347).

References:
- [Fine-Tuning Language Models from Human Preferences](https://github.com/openai/lm-human-preferences)
- [Learning to Summarize from Human Feedback](https://github.com/openai/summarize-from-feedback)
- [The N Implementation Details of RLHF with PPO](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)
- [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/pdf/2403.17031)

## Get started

To just run a PPO script to make sure the trainer can run, you can run the following command to train a PPO model with a dummy reward model.

```bash
python -i examples/scripts/ppo/ppo.py \
    --learning_rate 3e-6 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --non_eos_penalty \
```


## Explanation of the logged metrics

The logged metrics are as follows. Here is an example [tracked run at Weights and Biases](https://wandb.ai/huggingface/trl/runs/dd2o3g35)

* `eps`: Tracks the number of episodes per second.
* `objective/kl`: The mean Kullback-Leibler (KL) divergence between the current policy and reference policy.
* `objective/entropy`: The mean entropy of the policy, indicating the randomness of the actions chosen by the policy.
* `objective/non_score_reward`: The mean reward from non-score-related sources, basically `beta * kl.sum(1)`, where `beta` is the KL penalty coefficient and `kl` is the per-token KL divergence.
* `objective/rlhf_reward`: The mean RLHF reward, which is `score - non_score_reward`.
* `objective/scores`: The mean scores returned by the reward model / environment.
* `policy/approxkl_avg`: The average approximate KL divergence between consecutive PPO policies. Note that this is not the same as `objective/kl`.
* `policy/clipfrac_avg`: The average fraction of policy updates that are clipped, indicating how often the policy updates are constrained to prevent large changes.
* `loss/policy_avg`: The average policy loss, indicating how well the policy is performing.
* `loss/value_avg`: The average value loss, indicating the difference between the predicted value and the actual reward.
* `val/clipfrac_avg`: The average fraction of value function updates that are clipped, similar to policy/clipfrac_avg but for the value function.
* `policy/entropy_avg`: The average entropy of the policy during training, indicating how diverse the policy's actions are.
* `val/ratio`: The mean ratio of the current policy probability to the old policy probability, providing a measure of how much the policy has changed.
* `val/ratio_var`: The variance of the `val/ratio`, indicating the variability in policy changes.
* `val/num_eos_tokens`: The number of end-of-sequence (EOS) tokens generated, which can indicate the number of complete responses.
* `lr`: lr: The current learning rate used by the optimizer.
* `episode`: episode: The current global step or episode count in the training process.


## Cookbook

* Debugging TIP: `objective/rlhf_reward`: this is the ultimate objective of the RLHF training. If training works as intended, this metric should keep going up.
* Debugging TIP: `val/ratio`: this number should float around 1.0, and it gets clipped by `--cliprange 0.2` with PPO's surrogate loss. So if this `ratio` is too high like 2.0 or 1000.0 or too small like 0.1, it means the updates between consecutive policies are too drastic. You should try undertand why this is happening and try to fix it.
* Memory TIP: If you are running out of memory, you can try to reduce the `--per_device_train_batch_size` or increase the `--gradient_accumulation_steps` to reduce the memory footprint.
* Memory TIP: If you have multiple GPUs, you can also run training with DeepSpeed stage 3 to reduce the memory footprint `accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml`.
* Usage TIP: We recommend to use the "EOS trick" via `--non_eos_penalty --stop_token eos`, which replaces the score of completions that do not end with an EOS token with a static scalar penalty `--penalty_reward_value`. This can help the model learn to generate more coherent completions.


## What is my model doing exactly?

To help you understand what your model is doing, we periodically log some sample completions from the model. Here is an example of a completion. In an example [tracked run at Weights and Biases](https://wandb.ai/huggingface/trl/runs/dd2o3g35), it looks like the following, allowing you to see the model's response at different stages of training. By default we generate `--num_sample_generations 10` during training, but you can customize the number of generations.

![](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/ppov2_completions.gif?download=true)


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
├─────────────────────────────────┼─────────────────────────────────┼──────────┤
```

## Implementation details

This PPOv2 implementation is based on the [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/pdf/2403.17031).

## Benchmark experiments

To validate the PPO implementation works, we ran experiments on the 1B and 6.9B models. Here are the commands we used to run the experiments. We take the SFT / RM models directly from [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/pdf/2403.17031).


```
# 1B PPO experiment
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/ppo/ppo_tldr.py \
    --output_dir models/minimal/ppo_tldr \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes 1000000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --non_eos_penalty \
    --stop_token eos \

# 6.9B PPO experiment
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/ppo/ppo_tldr.py \
    --output_dir models/minimal/ppo_tldr_6.9b \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 100000 \
    --model_name_or_path EleutherAI/pythia-6.9b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-6.9b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-6.9b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 2 \
    --non_eos_penalty \
    --stop_token eos \
```

1B experiment can be found here:

- [🤗 Model checkpoint](https://huggingface.co/vwxyzjn/ppo_tldr)
- [🐝 Tracked experiment](https://wandb.ai/huggingface/trl/runs/dd2o3g35)


To evaluate, we use vLLM to load the checkpoints and GPT3.5 as a judge model to evaluate the generated TL;DR against the reference TL;DR.
```bash
python -i examples/scripts/evals/generate_tldr.py \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --output_path examples/scripts/minimal/evals/sft_tldr.csv \
    --n 1000
# preferred
# response1    656
# response0    344
# Name: count, dtype: int64
python -i examples/scripts/evals/generate_tldr.py \
    --model_name_or_path vwxyzjn/ppo_tldr \
    --output_path examples/scripts/minimal/evals/ppo_tldr.csv \
    --n 1000
# preferred
# response0    528
# response1    472
# Name: count, dtype: int64
```

The PPO checkpoint gets a 52.8% preferred rate vs the 34.4% preference rate of the SFT checkpoint. This is a good sign that the PPO training is working as intended.


Metrics:

![](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/benchmark/pr-1540/ppov2.png?download=true)


```bash
# pip install openrlbenchmark==0.2.1a5
# see https://github.com/openrlbenchmark/openrlbenchmark#get-started for documentation
# to use it, change `?we=huggingface&wpn=trl` to your own project and `?tag=pr-1540` to your own tag
python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=huggingface&wpn=trl&xaxis=train/episode&ceik=output_dir&cen=sft_model_path&metrics=train/objective/rlhf_reward&metrics=train/objective/scores&metrics=train/objective/kl&metrics=train/objective/non_score_reward&metrics=train/objective/entropy&metrics=train/policy/approxkl_avg&metrics=train/policy/clipfrac_avg&metrics=train/loss/policy_avg&metrics=train/loss/value_avg&metrics=train/val/clipfrac_avg&metrics=train/policy/entropy_avg&metrics=train/val/ratio&metrics=train/val/ratio_var&metrics=train/val/num_eos_tokens&metrics=train/lr&metrics=train/eps' \
        "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr?tag=pr-1540" \
    --env-ids models/minimal/ppo_tldr \
    --pc.ncols 4 \
    --pc.ncols-legend 1 \
    --pc.xlabel "Episode" \
    --output-filename benchmark/trl/pr-1540/ppov2 \
    --scan-history
```


6.9B experiment is still TBD (experiments got preempted due to resource constraints).
