# Zero-Sync GRPO Trainer

[![model badge](https://img.shields.io/badge/All_models-Zero_Sync_GRPO-blue)](https://huggingface.co/models?other=zero-sync-grpo,trl)

## Overview

[`experimental.zero_sync_grpo.ZeroSyncGRPOTrainer`] implements the same [GRPO](grpo_trainer) algorithm, but generation and training share **one copy of the weights**. A transformers [continuous batching](https://huggingface.co/docs/transformers/main/en/continuous_batching) engine generates from the parameter tensors the optimizer updates in place, so there is no inference server, no weight synchronization, and no second copy of the model in memory. Generation and training take turns: the engine decodes while the trainer waits for a step's samples, pauses for the step, and resumes with its rollouts in flight.

This trainer was contributed by [Quentin Gallouédec](https://huggingface.co/qgallouedec).

## Quick start

This example demonstrates how to train a model using the Zero-Sync GRPO method. We train a [Qwen3 0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B) with the prompts from the [DeepMath-103K dataset](https://huggingface.co/datasets/trl-lib/DeepMath-103K). You can view the data in the dataset here:

<iframe
  src="https://huggingface.co/datasets/trl-lib/DeepMath-103K/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

Below is the script to train the model. `accuracy_reward` needs math-verify, which you can install with `pip install trl[math_verify]`.

```python
# train_zero_sync_grpo.py
from datasets import load_dataset
from trl.experimental.zero_sync_grpo import ZeroSyncGRPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = ZeroSyncGRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()
```

Generation runs in the training process, on the same GPU. Execute the script using the following command:

```bash
python train_zero_sync_grpo.py
```

## Compatibility and constraints

Some argument combinations are intentionally restricted in the current [`experimental.zero_sync_grpo.ZeroSyncGRPOTrainer`] implementation:

- Generation always runs on the trainer's own continuous batching engine, in the training process. There is no vLLM.
- Prompts from the dataset must be in [conversational](dataset_formats#conversational) format. With an `environment_factory`, the dataset may have no `prompt` column, and `reset()` provides it.
- Reward functions must be Python callables. Reward models given as a model id or a [`~transformers.PreTrainedModel`] are not supported, see [Using a custom reward function](#using-a-custom-reward-function).
- The trainer needs `transformers>=5.19.0`.
- Every process holds the whole model, or its share of it with `tp_size`, see [Scaling](#scaling). FSDP and DeepSpeed are not supported, and neither are context and sequence parallelism (`cp_size > 1` or `sp_size > 1` in `parallelism_config`).
- `use_liger_kernel=True` is not supported.
- PEFT is not supported.
- Models with linear attention layers (Qwen3.5, Qwen3.6, Qwen3-Next, ...) need flash attention, see [Packed training](#packed-training).
- Images are not supported, see [Training Vision Language Models](#training-vision-language-models).

## Looking deeper into the Zero-Sync GRPO method

In [`GRPOTrainer`], generation and training are sequential: generate a batch, compute the loss, update the weights, repeat, and the whole batch waits for its slowest completion. In [`experimental.zero_sync_grpo.ZeroSyncGRPOTrainer`], they still take turns on the same weights, but generation never drains: the rollouts that are not finished when a step starts carry on after it. To understand how it works, it can be broken down into three main steps: **Generating completions**, **computing the advantage**, and **computing the loss**. There is no reference model, so there is no KL divergence term (GRPO's default, `beta=0.0`).

### Generating completions

The rollout loop keeps up to `max_concurrent_rollouts` requests in flight to the engine. For each prompt, it generates a group of  \\( G \\) completions (`num_generations`) and, once the whole group is done, scores it and computes its advantages. The resulting training samples go into a buffer. Once the buffer holds a step's samples, the trainer pauses the engine, trains on them, and resumes it. A training batch is formed from whichever scored samples are ready first, so a slow rollout never holds up a batch of fast ones: it lands in a later batch.

A rollout in flight when a step starts finishes with the updated weights, so its tokens can come from several versions of the policy. The engine returns, for each token, the log-probability computed by the weights that sampled it (see [Staleness](#staleness)).

The metrics and the rest of this page use the following terms for the stages a completion goes through:

- a **group** is one prompt and its  \\( G \\) rollouts. The advantage is computed within it.
- a **rollout** is one full conversation generated for a prompt, including every tool call. It is made of one or more **turns**, one engine request each.
- a **sample** is what the loss is computed on: the tokens of a rollout, with its advantage. A multi-turn rollout can produce more than one sample (see [Multi-turn rollouts](#multi-turn-rollouts)).
- a **row** is what one replica forwards in one micro-batch: several samples packed back to back (see [Packed training](#packed-training)). `gradient_accumulation_steps` micro-batches make one **step**, the optimizer step.

Token counts come in three kinds: **generated** tokens are what the model produced, **forwarded** tokens are every token of a sample (prompt and tool results included), and **trained** tokens are the ones the loss is taken over.

### Computing the advantage

For each of the  \\( G \\) completions of a group, the reward is the sum of the rewards from each reward function. The advantage is normalized within the group, as in GRPO:

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

There is no `scale_rewards` option: the advantage is always scaled by the group's standard deviation. A completion for which every reward function returned `None` is left out of the group's mean and standard deviation and gets an advantage of zero.

A group whose rewards are all equal is dropped: with the default advantage it is zero everywhere, so training on it would cost a forward and backward pass and move nothing. The fraction of such groups is logged as `frac_zero_std_groups`. The step is filled from the groups that are kept, so many dropped groups means more generation per step. A dropped group still counts toward the epoch, which stays one pass over the prompts.

### Computing the loss

The engine's log-probabilities never match the trainer's exactly: the engine decodes with different kernels than the training forward, and a rollout's tokens can come from a policy a few steps behind (see [Dealing with the Training-Inference Mismatch](grpo_trainer#dealing-with-the-training-inference-mismatch)). The trainer uses the sampler's log-probabilities as the old policy, so a single importance ratio covers both the mismatch and the staleness:

$$
r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,< t})}{\pi_{\text{sampler}}(o_{i,t} \mid q, o_{i,< t})}.
$$

Tokens whose ratio falls outside `[importance_sampling_clip_min, importance_sampling_clip_max]` are dropped from the loss. For more information, see [Masked Importance Sampling](paper_index#masked-importance-sampling). The trainer uses GRPO's clipped surrogate objective with this masked ratio:

$$
\mathcal{L}(\theta) = - \frac{1}{\sum_{i} |o_i|} \sum_{i} \sum_{t=1}^{|o_i|} \mathbb{1}\left[ C_{\min} \le r_{i,t}(\theta) \le C_{\max} \right] \min \left( r_{i,t}(\theta) \hat{A}_{i,t}, \, \text{clip}\left( r_{i,t}(\theta), 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}} \right) \hat{A}_{i,t} \right),
$$

where  \\( \epsilon_{\text{low}} \\) is `epsilon` and  \\( \epsilon_{\text{high}} \\) is `epsilon_high`.

The loss is normalized by the number of trained tokens in the micro-batch, counted across all replicas, and averaged over the `gradient_accumulation_steps` micro-batches of a step.

## Logged metrics

While training, we record the following metrics:

- `num_tokens`: The total number of tokens forwarded so far, prompts and completions.
- `completions/mean_length`: The average length of generated completions. When using tools, only non-tool tokens are counted.
- `completions/min_length`: The minimum length of generated completions. When using tools, only non-tool tokens are counted.
- `completions/max_length`: The maximum length of generated completions. When using tools, only non-tool tokens are counted.
- `completions/mean_terminated_length`: The average length of generated completions that terminate with EOS. When using tools, only non-tool tokens are counted.
- `completions/min_terminated_length`: The minimum length of generated completions that terminate with EOS. When using tools, only non-tool tokens are counted.
- `completions/max_terminated_length`: The maximum length of generated completions that terminate with EOS. When using tools, only non-tool tokens are counted.
- `completions/clipped_ratio`: The ratio of truncated (clipped) completions, whose last turn does not end with an EOS token.
- `completions/duplicate_frac`: The fraction of rollouts that repeat an earlier rollout of their group token for token. See [Saturation and duplicates](#saturation-and-duplicates).
- `tools/call_frequency`: The average number of tool calls per completion, over every rollout, including the groups that are then dropped. Logged only when the trainer has tools.
- `tools/failure_frequency`: The fraction of tool calls that failed (the tool was not found or raised an exception), over every tool call, including the groups that are then dropped. It is NaN in a logging window with no tool call, where GRPO logs `0.0`. Logged only when the trainer has tools.
- `rewards/{reward_func_name}/mean`: The average reward from a specific reward function, over every completion it scored, including the groups that are then dropped. When an environment owns the reward via `get_reward`, `{reward_func_name}` is the environment's class name.
- `rewards/{reward_func_name}/group_std`: The standard deviation of the reward from a specific reward function, computed within each group and averaged over the trained samples.
- `reward`: The overall average reward after summing rewards across functions, over every scored completion, including the groups that are then dropped. It is the policy's reward, so it is the one to track: `reward_trained` leaves out the groups whose rewards are all equal.
- `reward_group_std`: The standard deviation of summed rewards across functions, computed within each group and averaged over the trained samples. It is what the advantage is divided by, unlike GRPO's `reward_std`, which is taken over the whole batch.
- `reward_trained`: The average reward of the samples the trainer trained on. It leaves out the groups whose rewards are all equal.
- `reward_zero_std_groups`: The average reward of the groups whose rewards are all equal, which are dropped: the reward they are stuck at. See [Saturation and duplicates](#saturation-and-duplicates) to read it with `frac_zero_std_groups` and `reward`.
- `frac_zero_std_groups`: The fraction of groups with a reward std of zero (all answers are correct or incorrect), or with fewer than two scorable completions. These groups are dropped.
- `sampling/sampling_logp_difference/mean`: The average absolute difference between the log probabilities returned by the sampler (the engine) and the ones recomputed by the training model, over trained tokens.
- `sampling/sampling_logp_difference/max`: The largest such absolute difference.
- `sampling/importance_sampling_ratio/min`: The smallest importance sampling ratio  \\( r_{i,t}(\theta) \\) between the model and the sampler, over trained tokens, before the tokens outside `[importance_sampling_clip_min, importance_sampling_clip_max]` are dropped.
- `sampling/importance_sampling_ratio/mean`: The average such importance sampling ratio.
- `sampling/importance_sampling_ratio/max`: The largest such importance sampling ratio.
- `sampling/masked_mean`: The fraction of trained tokens dropped from the loss because their importance sampling ratio falls outside `[importance_sampling_clip_min, importance_sampling_clip_max]`.
- `entropy`: Average entropy of token predictions across trained tokens.
- `aux_loss`: The load-balancing auxiliary loss of a Mixture-of-Experts model, before it is scaled by `router_aux_loss_coef` and added to the loss. Logged only when the model is a MoE model and `router_aux_loss_coef` is nonzero.
- `clip_ratio/region_mean`: The ratio of token probabilities where the GRPO objective is clipped to stay within the trust region:  \\( \text{clip}\left( r_{i,t}(\theta), 1 - \epsilon_\mathrm{low}, 1 + \epsilon_\mathrm{high} \right) \\).
- `clip_ratio/low_mean`: The average ratio of token probabilities that were clipped on the lower bound of the trust region:  \\(r_{i,t}(\theta) < 1 - \epsilon_\mathrm{low}\\).
- `clip_ratio/low_min`: The smallest per-completion fraction of tokens clipped on the lower bound of the trust region:  \\(r_{i,t}(\theta) < 1 - \epsilon_\mathrm{low}\\).
- `clip_ratio/high_mean`: The average ratio of token probabilities that were clipped on the upper bound of the trust region:  \\(r_{i,t}(\theta) > 1 + \epsilon_\mathrm{high}\\).
- `clip_ratio/high_max`: The largest per-completion fraction of tokens clipped on the upper bound of the trust region:  \\(r_{i,t}(\theta) > 1 + \epsilon_\mathrm{high}\\).

The other metrics describe the rollout pipeline and the engine.

Metrics are logged once per **logging window**: everything recorded since the previous log, which covers the last `logging_steps` optimizer steps. Over the window:

- A mean measured per rollout, sample or call, like `reward`, `rollout/turns_mean` and `tools/latency_s`, is averaged over all the rollouts, samples or calls in the window. A rate like `rollout/generated_tokens_per_s` is the total tokens divided by the total seconds.
- Otherwise, a name whose last part contains the word `count` is summed: `tools/unknown_name_count` is the number of such calls since the last log, not since the start of training.
- A name whose last part contains the word `max` or `min`, like `sample/staleness_max` or `completions/max_length`, keeps the largest or smallest value.
- Any other metric is the mean of the values recorded in the window.

Units:

- `_s`: seconds.
- `tokens_per_s`: tokens per second.
- `_frac`: fraction between 0 and 1.
- `_per_step`: amount in one optimizer step, averaged over the steps in the window.

### Rollout metrics

- `rollout/duration_s`: The wall time of one rollout, from dispatch until the rollout loop collects it. It includes the training steps it spans.
- `rollout/generated_tokens_per_s`: The generation throughput, in generated tokens per second, since the previous report.
- `rollout/concurrency`: The number of rollouts in flight to the engine.
- `rollout/turns_mean`, `rollout/turns_max`: The average and maximum number of turns per rollout.
- `rollout/samples_per_rollout`: The number of training samples per rollout. Above `1.0`, some rollouts were forked; below `1.0`, some produced no trained token. See [Multi-turn rollouts](#multi-turn-rollouts).
- `rollout/fork_frac`: The fraction of turn transitions whose prompt does not start with the held tokens, so the turn started a new sample.
- `rollout/drift_tokens_mean`, `rollout/drift_tokens_max`: The number of held tokens a fork re-rendered, averaged over turn transitions (one that does not fork counts as `0`). The new sample shares the tokens before the divergence, so this is what each fork adds to the forward.
- `rollout/score_queue_size`: The number of completed groups waiting to be scored.
- `rollout/queue_size`: The number of scored samples waiting in the rollout buffer, measured where the trainer takes them. See [The rollout buffer](#the-rollout-buffer).
- `rollout/score_s`: The time to score a group.
- `rollout/score_wait_s`: The time a completed group waited before being scored.
- `rollout/score_block_s`: The time generation was blocked because the scoring queue was full.

### Tool metrics

- `tools/latency_s`, `tools/{tool_name}/latency_s`: The time spent executing a tool call, over all tools and per tool.
- `tools/{tool_name}/call_count`, `tools/{tool_name}/failure_count`: The number of calls and failed calls per tool.
- `tools/unknown_name_count`: The number of calls to a tool that does not exist. Unlike a tool that ran and raised, this is a policy error.
- `tools/parallel_calls_mean`: The average number of tool calls in an assistant message that calls tools.
- `tools/loop_exhausted_frac`: The fraction of rollouts cut off at `max_tool_calling_iterations` while still calling tools, over every rollout, including rollouts with no call and the groups that are then dropped. Logged only when the trainer has tools.

### Environment metrics

- `env/reset_s`: The time spent in the environment's `reset()` before a rollout starts. Logged only when `environment_factory` is provided.
- `env/reward_s`: The time spent in the environment's `get_reward()` after a rollout ends. Logged only when the environment defines `get_reward`.

### Sample metrics

- `sample/forwarded_tokens_mean`, `sample/forwarded_tokens_max`: The average and maximum number of forwarded tokens per sample.
- `sample/trained_tokens_mean`: The average number of trained tokens per sample.
- `sample/time_in_queue_s`: The time a sample waited in the rollout buffer before being trained on.
- `sample/staleness_mean`, `sample/staleness_max`: The number of optimizer steps between the start of the rollout that produced a sample and the step training on it. The `sampling/importance_sampling_ratio/*` and `sampling/sampling_logp_difference/*` metrics combine this staleness with the training-inference mismatch.

### Batch metrics

- `batch/forwarded_tokens_per_step`, `batch/trained_tokens_per_step`: The number of forwarded and trained tokens per step, over all replicas.
- `batch/samples_per_step`: The number of samples per step.
- `batch/microbatches_per_step`: The number of micro-batches per step.
- `batch/masked_token_frac`: The fraction of forwarded tokens that are not trained.
- `batch/row_imbalance`: The ratio of the most expensive replica's batch to the average one, where the cost is the sum of squared sample lengths. The slowest replica stalls the others at the gradient sum; `1.0` is perfect balance. Logged only with several replicas.
- `batch/from_other_replicas`: The fraction of a replica's batch that another replica generated. Logged only with several replicas, see [Scaling](#scaling).
- `batch/pool_exchange_s`: The time spent handing samples between replicas. Logged only with several replicas.

### Performance metrics

- `perf/step_s`: The wall time of a step, including the optimizer and the time waiting for samples.
- `perf/fwd_bwd_s`, `perf/fwd_s`: The time spent in the forward and backward passes of a step, and in its forward passes alone.
- `perf/optimizer_s`: The time spent in the optimizer step.
- `perf/rollout_wait_s`: The time training waited for the engine to finish a step's samples. See [The rollout buffer](#the-rollout-buffer).
- `perf/fwd_bwd_forwarded_tokens_per_s`, `perf/wall_clock_forwarded_tokens_per_s`: The number of forwarded tokens per second, over the forward and backward passes or over the whole step.
- `perf/wall_clock_trained_tokens_per_s`: The number of trained tokens per second, over the whole step.
- `perf/fwd_bwd_mfu`, `perf/wall_clock_mfu`: The model FLOPs utilization of the training devices, over the forward and backward passes or over the whole step. The whole step includes the time the same devices spend generating. Logged only when the peak compute of the training device at the training precision is known, and when the FLOPs estimate supports the model's architecture (such as Llama, Qwen2, Qwen3, Qwen3-MoE or Qwen3-VL).

### Engine metrics

- `generation/pause_s`: The time the pause took to be granted, plus releasing the KV cache with `kv_cache_during_step="release"`.
- `generation/lend_s`, `generation/give_back_s`: With `kv_cache_during_step="lend"`, the time to lend the KV cache to the step and to get it back.
- `generation/lent_copied_gib`, `generation/lent_beyond_gib`: With `kv_cache_during_step="lend"`, the kept KV cache the step overwrote, copied to the host and back, and the GPU memory it took beyond the cache.
- `memory/alloc_retries`: The allocator's retries. They stay at zero unless the KV cache leaves the training step too little room, see [Configuring the generation engine](#configuring-the-generation-engine).

## Customization

### Configuring the generation engine

The engine reserves its KV cache when generation first starts. `continuous_batching_config` takes the keyword arguments of [`~transformers.generation.ContinuousBatchingConfig`]. Size the cache from what generation actually needs, roughly `requests in flight x expected completion length x KV bytes per token`:

```python
from trl.experimental.zero_sync_grpo import ZeroSyncGRPOConfig

training_args = ZeroSyncGRPOConfig(
    ...,
    continuous_batching_config={"max_memory_percent": 0.25},
    max_concurrent_rollouts=32,
)
```

`kv_cache_during_step` decides what happens to the cache's GPU memory during the training step:

- `"release"` (default): the cache waits in host memory and comes back before generation resumes, so the rollouts in flight carry on where they were, and the forward and backward get its memory. This copies the cache to the host and back at every step.
- `"keep"`: the cache stays on the device, for small models or short completions, where memory is not the limit. An oversized cache then starves the training step, which shows up as an out-of-memory error in the backward pass, not as slow generation.
- `"lend"`: the cache's memory is lent to the step with [`sublet`](https://pypi.org/project/sublet/). The step takes the cache's free blocks first, then free GPU memory, then the blocks of rollouts in flight, which are copied to the host and back. Only what the step overwrites moves. Needs `pip install 'sublet>=0.2'`.

The optimizer updates fp32 copies of the parameters and writes them back after each step. The copies and the optimizer state live in host memory and come onto the GPU one parameter at a time during the step, so the GPU holds only the parameters and their gradients next to the KV cache and the activations.

### Concurrency

`max_concurrent_rollouts` (`32` by default) is the number of rollouts in flight at once. A rollout that finishes is replaced immediately by the next one, so the decode batch stays full instead of emptying and refilling once per step. A decode step costs about the same whatever number of sequences it carries, so more rollouts in flight means more tokens per second while the engine runs. Measured on one H100 with Qwen3-0.6B, GSM8K, 512-token completions, 8 generations per prompt, 64 samples per step:

| rollouts in flight | wait, share of the step | steps/s | completion tokens/s |
|---|---|---|---|
| 64 | 69% | 0.46 | 5,600 |
| 512 | 57% | 0.57 | 7,100 |

What it costs is staleness rather than time: a rollout spans every step that happens before it finishes, about `max_concurrent_rollouts / samples per step` of them (1 and 8 in the rows above), and the KV cache has to hold them all, or decode slows instead of the batch filling. Each rollout in flight also holds an environment, and its synchronous tool calls run on a thread pool of this size.

### The rollout buffer

The rollout loop puts scored samples in a buffer and the trainer takes a step's samples from it. Three metrics describe that buffer:

- `rollout/queue_size`: how many samples are left in it once the trainer has taken a step's samples.
- `sample/time_in_queue_s`: how long a single sample waited before being trained on, the time part of its staleness.
- `perf/rollout_wait_s`: how long training waited for the engine to finish a step's samples.

Generation only runs while the trainer waits, so a step costs the training step plus that wait, and the wait is the time to decode what was still missing when the previous step ended. On one GPU it is where most of the step goes (see [Concurrency](#concurrency)).

### Saturation and duplicates

The policy learns from the differences between the rollouts of a group: each advantage is taken against the group's mean reward. When every rollout of a group gets the same reward, they all get a zero advantage, so the group is dropped (see [Logged metrics](#logged-metrics)) and teaches the policy nothing. As a run saturates, more and more groups end up like this, either because every rollout solves the prompt or because none does. Four metrics tell these cases apart, and each one answers its own question, so you can read them straight off their plots:

| metric | question |
| --- | --- |
| `frac_zero_std_groups` | what share of groups carry no signal |
| `reward_zero_std_groups` | where those groups are stuck: their mean reward. With 0/1 rewards it is the share of them in which every rollout passed, so 0.9 means that almost all of them are solved, and 0.1 that almost all of them fail every rollout |
| `reward` | how well the policy does over every scored group, dropped ones included |
| `completions/duplicate_frac` | whether sampling still gives different rollouts for the same prompt |

Read the first three together, against the range of your reward:

| `frac_zero_std_groups` | `reward_zero_std_groups` | `reward` | diagnosis | what to do |
| --- | --- | --- | --- | --- |
| low | any (noisy over few groups) | any | **healthy**: most groups produce a gradient | nothing |
| climbing | near the top of the range | climbing | **saturating at the top**: the prompts have become too easy for the policy, which is how a successful run usually ends | stop the run, or move to harder prompts |
| high | near the bottom of the range | low and flat | **stuck at the bottom**: the prompts are too hard, or the reward function never gives credit | check `completions/clipped_ratio` first, since a completion cut off before its answer scores 0, then read a few completions to check that the reward function parses them |
| climbing | falling | falling | **regression**: groups that used to pass now fail together | check `completions/duplicate_frac` and `entropy` for a collapse |
| high | in the middle | any | **a mix**: with 0/1 rewards, some prompts are too easy and others too hard. With partial-credit rewards, groups can also be stuck on the same partial score | read the rollouts |

`completions/duplicate_frac` is the share of rollouts that repeat an earlier rollout of their group token for token, across all of their turns. Identical rollouts get identical rewards, so duplicates also push `frac_zero_std_groups` up.

| `completions/duplicate_frac` | diagnosis |
| --- | --- |
| near 0, with a small baseline on tasks whose completions are short enough to repeat by chance | **healthy**: the rollouts of a group are sampled independently |
| jumps towards `(num_generations - 1) / num_generations` | **sampling is no longer random**: every group comes back as copies of one rollout, for example with a temperature of 0 |
| climbs slowly, with `entropy` falling | **collapse**: the policy is converging onto one answer per prompt |

These metrics are recorded when the loop scores a group, not when its samples are trained on, so they run slightly ahead of training. A group in which no rollout could be scored, because every reward function returned `None`, is dropped and counted in `frac_zero_std_groups`, but it has no reward, so `reward_zero_std_groups` leaves it out.

### Packed training

The samples of a micro-batch are packed back to back in one row rather than padded to the longest, which on mixed lengths would waste up to a third of the forward on pad tokens. Position ids restart at each sample and the boundaries are passed as the flash varlen kwargs, so attention stays within a sample; each token keeps its own sample's advantage and behavior log-probability, so the loss is unchanged.

Unless `model_init_kwargs` sets one, the attention implementation is flash attention when it is available and `sdpa` otherwise, which builds the block-diagonal mask from the position ids. Models with linear attention layers require flash attention, whose kernels read the sample boundaries from the varlen kwargs.

Measured on Qwen3-4B with `tp_size=4` and batch 256 on GSM8K: 6.98 to 5.82 s/step against padding.

### Staleness

Because the weights change while a rollout is being generated, a long completion can span several optimizer steps: its tokens are sampled by different versions of the policy. The engine returns, for each token, the log-probability computed by the weights that actually sampled it, so the importance ratio in the loss is exact per token even though no single old policy exists.

What remains is that the KV cache of a rollout's prefix was computed with older weights, while the training forward recomputes it with current ones. Measured on Qwen3-0.6B at `learning_rate=1e-5`, the resulting log-probability gap peaks at about 0.16 nats on a rollout's first tokens (an importance ratio of 1.17, inside the default clip range) and falls to the kernel-numerics floor by token 50. It grows with the learning rate, the completion length, and `max_concurrent_rollouts`.

### Scaling

Every process holds a full copy of the weights and runs its own generation engine, so the default scaling is data parallel: `accelerate launch --num_processes N`. Generation issues no collectives there, so nothing ties one process's engine to another's: each pauses for its own training step and resumes after it.

Tensor parallelism splits one copy of the weights across processes instead, with `tp_size`:

```python
from trl.experimental.zero_sync_grpo import ZeroSyncGRPOConfig

training_args = ZeroSyncGRPOConfig(
    ...,
    tp_size=4,
)
```

Under tensor parallelism, generation and training never running at the same time is not a choice but a requirement. Both issue collectives, on separate NCCL communicators. NCCL requires that every rank issue the operations on its communicators in the same host-side order: [its user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html) states that "to remain deadlock free, users must ensure the order of host-side launches matches for all devices", and recommends "a deterministic order issued from a single host thread per-device". Two threads racing cannot provide that, because each rank's order then depends on its own timing. When the orders disagree the run deadlocks, and it does so in a way that is hard to read: a rank blocks inside an ordinary kernel launch, not inside a collective, because a CUDA call waits on the resident NCCL kernel and so prevents the other communicator's kernel from ever launching.

So the two take turns, at every `tp_size`. The engine keeps decoding in its own thread while the trainer collects and scores completions; once a step's samples are in, the trainer pauses the engine (`ContinuousBatchingManager.pause`), runs its forward, backward and optimizer step, and resumes it. It pauses once per optimizer step, whatever `gradient_accumulation_steps` is: the trainer collects every micro-batch of a step before it trains on the first. The pause is agreed on by every rank at the same engine step, and the engine finishes the step it had in flight on the device before it reports itself paused, so the trainer's collectives never share the device with the engine's, and the forward and backward have the device to themselves. Nothing is drained and no request is lost: the rollouts in flight keep their cache and carry on after the step. The ranks of a group generate together and train on the same batch, so the rollouts run on the group's first rank only: it submits the requests, reads the completions, runs the tools and the reward functions, computes the advantages and picks the batch, then sends it to the other ranks of its group, which join the pause and train on it. A tool or a reward function therefore runs once per completion, not once per rank.

Interleaving generation through the step instead, a decode step at every layer boundary, was measured and is not worth it: identical throughput on short completions (1.55 against 1.54 s/step) and about 7% better on long ones (2.28 against 2.44 s/step with 512-token completions and 8 rollouts per prompt), at the cost of a design that cannot ever hand the KV cache memory to the training step.

The two combine. `tp_size` only has to divide the world, and the ranks the model is not split across hold replicas of it: they draw their own prompts, and their gradients are summed. Pick the smallest `tp_size` the model fits in and spend the rest on replicas, because decode is bound by all-reduce latency, so a wider split slows generation. On 16 H100 with Qwen3-14B, `tp_size=4` with four replicas does 8220 trained tokens/s against 3280 for `tp_size=8` with two.

The replicas draw their training batches from one pool, exchanged between their first ranks. A sample is data, and nothing ties it to the replica that generated it, so waiting until the replicas together have enough and then handing them out to whoever carries the least, by sum of squared lengths, keeps their forwards aligned and their memory peaks alike. Otherwise a replica whose completions came back short trains first and then sits at the gradient sum waiting for one still decoding. `batch/row_imbalance` logs how even the result is, `1.0` being even, and `batch/from_other_replicas` how much of the batch came from elsewhere.

Each replica also keeps the optimizer state of only its share of the parameters, updates it, and passes the updated parameters back, since replicas hold the same weights and none of them needs a second copy of Adam's moments. The parameters stay whole everywhere, which is what generation reads.

Two consequences to know about. The vocabulary projection is replicated rather than split, which costs one copy of that matrix per process and removes a gather from every forward. And the engine decodes from the training parameters themselves, with the attention implementation the trainer chose, so a weight update is visible to the next decoding step without any copy.

Under tensor parallelism three more numbers say where a step's time goes: `generation/pause_s` is how long the pause took to be granted (one engine step, tens of milliseconds), `batch/pool_exchange_s` is the time spent handing samples between replicas, which grows when replicas reach the exchange at different moments, and `memory/alloc_retries` counts the allocator's retries.

On the training side, every micro-step pays a fixed host cost for dispatching the forward and backward, so `per_device_train_batch_size` should be the largest that fits, with gradient accumulation making up the rest. At the same samples per step, doubling it from 8 to 16 gave 5% on Qwen3-8B at `tp_size=2` and 22% on Qwen3-30B-A3B at `tp_size=4`, whose 48 sparse layers make the dispatch the longer part of a small micro-step.

### Using a custom reward function

The [`experimental.zero_sync_grpo.ZeroSyncGRPOTrainer`] supports using custom reward functions. To ensure compatibility, your reward function must satisfy the following requirements:

Reward functions can be either synchronous Python callables or asynchronous `async def` coroutines. When you provide multiple reward functions, they run concurrently: asynchronous ones are awaited together and synchronous ones run in threads, so their latency overlaps.

1. **Input arguments**:
   - The function must accept the following as keyword arguments:
     - `prompts` (contains the prompts),
     - `completions` (contains the generated completions),
     - `completion_ids` (contains the tokenized completions),
     - All column names (but `prompt`) that the dataset may have. For example, if the dataset contains a column named `ground_truth`, the function will be called with `ground_truth` as a keyword argument.

     The easiest way to comply with this requirement is to use `**kwargs` in the function signature.
   - The dataset is in [conversational format](dataset_formats#conversational), so `prompts` and `completions` are lists of message dictionaries.

2. **Return value**: The function must return a list of floats. Each float represents the reward corresponding to a single completion.

Reward functions run in the training process, on the first rank of each replica, while the engine is generating. A GPU-backed reward model shares the device with the policy and its KV cache.

#### Example 1: Reward longer completions

Below is an example of a reward function that rewards longer completions:

```python
def reward_func(completion_ids, **kwargs):
    """Reward function that assigns higher scores to longer completions (in terms of token count)."""
    return [float(len(ids)) for ids in completion_ids]
```

You can test it as follows:

```python
>>> prompts = [[{"role": "user", "content": "The sky is"}], [{"role": "user", "content": "The sun is"}]]  # not used in the reward function, but the trainer will pass it
>>> completions = [[{"role": "assistant", "content": " blue."}], [{"role": "assistant", "content": " in the sky."}]]  # not used in the reward function, but the trainer will pass it
>>> completion_ids = [[6303, 13], [304, 279, 12884, 13]]
>>> reward_func(prompts=prompts, completions=completions, completion_ids=completion_ids)
[2.0, 4.0]
```

#### Example 2: Reward completions with a specific format

Below is an example of a reward function that checks if the completion has a specific format. This example is inspired by the _format reward_ function used in the paper [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://huggingface.co/papers/2501.12948).
It is designed for a conversational format, where prompts and completions consist of structured messages.

```python
import re

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
```

You can test this function as follows:

```python
>>> prompts = [
...     [{"role": "user", "content": "What is the result of (1 + 2) * 4?"}],
...     [{"role": "user", "content": "What is the result of (3 + 1) * 2?"}],
... ]
>>> completions = [
...     [{"role": "assistant", "content": "<think>The sum of 1 and 2 is 3, which we multiply by 4 to get 12.</think><answer>(1 + 2) * 4 = 12</answer>"}],
...     [{"role": "assistant", "content": "The sum of 3 and 1 is 4, which we multiply by 2 to get 8. So (3 + 1) * 2 = 8."}],
... ]
>>> format_reward_func(prompts=prompts, completions=completions)
[1.0, 0.0]
```

#### Example 3: Reward completions based on a reference

Below is an example of a reward function that checks if the completion is correct. This example is inspired by the _accuracy reward_ function used in the paper [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://huggingface.co/papers/2501.12948). The dataset contains a column named `ground_truth`.

```python
import re

def reward_func(completions, ground_truth, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion[0]["content"]) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]
```

You can test this function as follows:

```python
>>> prompts = [
...     [{"role": "user", "content": "Solve the equation $2x + 3 = 7$."}],
...     [{"role": "user", "content": "Solve the equation $3x - 5 = 10$."}],
... ]
>>> completions = [
...     [{"role": "assistant", "content": r"The solution is \boxed{2}."}],
...     [{"role": "assistant", "content": r"The solution is \boxed{6}."}],
... ]
>>> ground_truth = ["2", "5"]
>>> reward_func(prompts=prompts, completions=completions, ground_truth=ground_truth)
[1.0, 0.0]
```

#### Example 4: Multi-task reward functions

Below is an example of using multiple reward functions in the [`experimental.zero_sync_grpo.ZeroSyncGRPOTrainer`]. In this example, we define two task-specific reward functions: `math_reward_func` and `coding_reward_func`. The `math_reward_func` rewards math problems based on their correctness, while the `coding_reward_func` rewards coding problems based on whether the solution works.

```python
from datasets import Dataset
from trl.experimental.zero_sync_grpo import ZeroSyncGRPOTrainer

# Define a dataset that contains both math and coding problems
dataset = Dataset.from_list(
    [
        {"prompt": [{"role": "user", "content": "What is 2+2?"}], "task": "math"},
        {"prompt": [{"role": "user", "content": "Write a function that returns the sum of two numbers."}], "task": "code"},
        {"prompt": [{"role": "user", "content": "What is 3*4?"}], "task": "math"},
        {"prompt": [{"role": "user", "content": "Write a function that returns the product of two numbers."}], "task": "code"},
    ]
)

# Math-specific reward function
def math_reward_func(prompts, completions, task, **kwargs):
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "math":
            # Calculate math-specific reward
            correct = check_math_solution(prompt, completion)
            reward = 1.0 if correct else -1.0
            rewards.append(reward)
        else:
            # Return None for non-math tasks
            rewards.append(None)
    return rewards

# Coding-specific reward function
def coding_reward_func(prompts, completions, task, **kwargs):
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "code":
            # Calculate coding-specific reward
            works = test_code_solution(prompt, completion)
            reward = 1.0 if works else -1.0
            rewards.append(reward)
        else:
            # Return None for non-coding tasks
            rewards.append(None)
    return rewards

# Use both task-specific reward functions
trainer = ZeroSyncGRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=[math_reward_func, coding_reward_func],
    train_dataset=dataset,
)

trainer.train()
```

In this example, the `math_reward_func` and `coding_reward_func` are designed to work with a mixed dataset that contains both math and coding problems. The `task` column in the dataset is used to determine which reward function to apply to each problem. If there is no relevant reward function for a sample in the dataset, the reward function will return `None`, and the [`experimental.zero_sync_grpo.ZeroSyncGRPOTrainer`] will continue with the valid functions and tasks. This allows the [`experimental.zero_sync_grpo.ZeroSyncGRPOTrainer`] to handle multiple reward functions with different applicability.

Note that the [`experimental.zero_sync_grpo.ZeroSyncGRPOTrainer`] will ignore the `None` rewards returned by the reward functions and only consider the rewards returned by the relevant functions. This ensures that the model is trained on the relevant tasks and ignores the tasks for which there is no relevant reward function.

#### Example 5: Asynchronous reward functions

Custom reward functions can also be defined as `async def` coroutines. This is useful if your reward depends on slow I/O (for example, calling a remote service). When you pass multiple async reward functions, [`experimental.zero_sync_grpo.ZeroSyncGRPOTrainer`] executes them concurrently so their latency overlaps.

Below is a minimal example of an async reward function that simulates an I/O-bound operation:

```python
import asyncio

async def async_reward_func(prompts, completions, **kwargs):
    # Simulate an I/O-bound call (e.g., HTTP request, database lookup)
    await asyncio.sleep(0.01)
    # Simple toy reward: 1.0 if the completion is non-empty, else 0.0
    return [1.0 if completion else 0.0 for completion in completions]
```

#### Passing the reward function to the trainer

To use your custom reward function, pass it to the [`experimental.zero_sync_grpo.ZeroSyncGRPOTrainer`] as follows:

```python
from trl.experimental.zero_sync_grpo import ZeroSyncGRPOTrainer

trainer = ZeroSyncGRPOTrainer(
    reward_funcs=reward_func,
    ...,
)
```

You can pass several reward functions as a list; this list may include both synchronous and asynchronous functions:

```python
from trl.experimental.zero_sync_grpo import ZeroSyncGRPOTrainer

trainer = ZeroSyncGRPOTrainer(
    reward_funcs=[reward_func, async_reward_func1, async_reward_func2],
    ...,
)
```

and the reward will be computed as the sum of the rewards from each function.

### Customizing the rollout loop

Generation, rewards, advantages and filtering run in [`experimental.zero_sync_grpo.ZeroSyncRolloutLoop`]. To change one of them, subclass the loop, override the method for that step, and set the subclass as `loop_cls` on a trainer subclass. The trainer's constructor doesn't change.

| Method | Called | Default |
|---|---|---|
| `filter_prompt(row_index, row, epoch)` | once per prompt, before generation | admits every prompt |
| `generate_one(prompt, tools, group_id)` | once per rollout | the multi-turn tool loop |
| `generate_one_turn(prompt_ids)` | once per sampling request | one engine request with the config's sampling settings |
| `compute_advantages(group)` | once per group, after the rewards are summed into `group.rewards` | reward standardized within the group |
| `setup()`, `close()` | when the loops start and stop | nothing |

For example, a curriculum that stops sampling prompts the policy already solves, with its threshold in a config subclass:

```python
from dataclasses import dataclass

import numpy as np

from trl.experimental.zero_sync_grpo import ZeroSyncGRPOConfig, ZeroSyncGRPOTrainer, ZeroSyncRolloutLoop


@dataclass
class CurriculumConfig(ZeroSyncGRPOConfig):
    pass_rate_cap: float = 0.9


class CurriculumRolloutLoop(ZeroSyncRolloutLoop):
    def filter_prompt(self, row_index, row, epoch):
        return self.pass_rates.get(row_index, 0.0) < self.config.pass_rate_cap

    async def setup(self):
        self.pass_rates = {}

    def compute_advantages(self, group):
        scored = group.rewards[~np.isnan(group.rewards)]
        if len(scored):
            self.pass_rates[group.row_index] = float(np.mean(scored > 0))
        return super().compute_advantages(group)


class CurriculumTrainer(ZeroSyncGRPOTrainer):
    loop_cls = CurriculumRolloutLoop
```

Rewards are summed over `reward_funcs`, environments that define `get_reward`, and the `reward` a custom `generate_one` sets on its [`experimental.zero_sync_grpo.Rollout`]: to score with a judge, pass an async reward function, or score inside `generate_one`. `compute_advantages` runs for every group, including one that is then dropped because all its scored rewards are equal; the drop looks at the rewards, not at the advantages. Create clients in `setup`, which runs on the loop's event loop, rather than overriding `__init__`. A prompt that `filter_prompt` rejects is skipped for that epoch. If it rejects every prompt of an epoch, the loop raises.

### Debugging

The generation engine runs in a background thread that is not a daemon, so a crash in the training step kills the main thread while the process stays alive with no progress and no traceback on screen. The trainer stops the engine on exit to avoid this, but if you build your own loop on top of continuous batching, a run that appears to hang is usually a training-side error that already happened: check the log rather than the GPU.

## Agent Training

Zero-Sync GRPO supports **agent training**: the model calls tools during generation and learns from the outcome.

- A **tool** is a plain Python function (sync or async) exposed to the model. Use `tools` for stateless calls (a calculator, a web search).
- An **environment** is the more general form: a stateful object built fresh per rollout, whose public methods are exposed as tools, plus a `reset` lifecycle hook and an optional `get_reward` that lets it own the reward. Use `environment_factory` when you need per-rollout state, a reset hook, or environment-owned reward.

They compose — you can pass standalone `tools` alongside an `environment_factory`.

### Tools

The `tools` argument expects a list of Python functions (sync or async) that define the tools available to the agent:

```python
from trl.experimental.zero_sync_grpo import ZeroSyncGRPOTrainer

trainer = ZeroSyncGRPOTrainer(
    tools=[tool1, tool2],
    ...,
)
```

Each tool must be a standard Python function with **type-hinted arguments and return types**, along with a **Google-style docstring** describing its purpose, arguments, and return value.
For more details, see the [Passing tools guide](https://huggingface.co/docs/transformers/en/chat_extras#passing-tools).

The model calls a tool by its name, so each tool is registered under its `__name__`. Synchronous tools run on a thread pool sized to `max_concurrent_rollouts`, so a slow tool never blocks the other rollouts in flight. Within one assistant turn, tool calls run in the order the model emitted them. Up to `max_concurrent_rollouts` synchronous tools can therefore run at once, so a tool that reads or writes shared state must be thread-safe.

> [!TIP]
> The tool call loop requires the chat template to be *prefix-preserving* (appending a tool message must not change how earlier messages are rendered). For known model families (e.g. Qwen3, DeepSeek-V3), TRL automatically swaps in a patched training template when tools are enabled. See [Chat Templates](chat_templates#training-templates) for the full list.

Use `max_tool_calling_iterations` in the [`experimental.zero_sync_grpo.ZeroSyncGRPOConfig`] to cap the number of tool-calling turns. By default there is no limit, and generation stops when the model produces a response turn with no tool calls.

Example:

```python
from trl.experimental.zero_sync_grpo import ZeroSyncGRPOTrainer

def multiply(a: int, b: int) -> int:
    """
    Multiplies two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The product of the two integers.
    """
    return a * b

async def async_add(a: int, b: int) -> int:
    """
    Asynchronously adds two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The sum of the two integers.
    """
    return a + b

trainer = ZeroSyncGRPOTrainer(
    tools=[multiply, async_add],
    ...,
)
```

### Environments

You can also provide tools through `environment_factory`. In this mode, [`experimental.zero_sync_grpo.ZeroSyncGRPOTrainer`] creates one environment instance per rollout and exposes the environment's public methods as tools.

When you use an `environment_factory`, the environment owns the data: there is no external `train_dataset`. On each rollout, `reset()` produces the task — self-sampling from a corpus the environment holds, or generating a state procedurally — and returns the prompt. `max_steps` sets the training length. The following is a minimal example of an environment that self-samples a target and exposes an `increment` method (exposed as a tool).

```python
import random

from trl.experimental.zero_sync_grpo import ZeroSyncGRPOConfig, ZeroSyncGRPOTrainer

class IncrementEnv:
    # Reserved methods
    def reset(self, **kwargs) -> str | None:  # required; called at the start of each rollout
        self.counter = 0
        self.target = random.randint(1, 6)  # self-sample the task
        return f"Increment the counter by {self.target}."  # returned string becomes the prompt

    def get_reward(self) -> float:  # optional: the environment scores itself from its own state
        return float(self.counter == self.target)

    # Public methods (exposed as tools)
    def increment(self, step: int) -> int:
        """
        Increment the internal counter.

        Args:
            step: Value to add to the counter.

        Returns:
            The updated counter value.
        """
        self.counter += step
        return self.counter

trainer = ZeroSyncGRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    args=ZeroSyncGRPOConfig(max_steps=1000, chat_template_kwargs={"enable_thinking": False}),
    environment_factory=IncrementEnv,
)
trainer.train()
```

An environment class has two reserved methods: `reset` and `get_reward`. Unlike any other public methods, these are not exposed to the model as tools.

- `reset` (required) is called when the rollout starts. It can return either `None` or a string. When it returns a string, that string is the user prompt.
- `get_reward` (optional, sync or async) takes no argument and returns a `float`: the environment scores the episode it just ran from its own internal state (did the game end in a win? was the word guessed?). It is called once per completed rollout and acts as a reward source.

Both run on the loop's tool thread pool, so a slow environment delays its own rollout but not the others in flight. Their durations are logged as `env/reset_s` and `env/reward_s`.

> [!NOTE]
> Because the environment self-samples on each `reset()`, the `G` members of a GRPO group may not share the same initial state, making the group baseline slightly noisier. For identical states within a group, have `reset()` derive the state deterministically from a shared key.

#### Providing an external dataset (optional)

Instead of letting the environment self-sample, you can provide an external `train_dataset`. For example when your tasks already live in a dataset, or to route between [multiple environments](#multiple-environments). Per rollout, the trainer samples one row, uses its `"prompt"` column as the prompt, and passes the row's columns to `reset()` as keyword arguments (so the environment reads the task from the dataset).

```python
from datasets import Dataset
from trl.experimental.zero_sync_grpo import ZeroSyncGRPOTrainer

# Each row carries the prompt and the task data (here, the counter target to reach).
dataset = Dataset.from_dict(
    {
        "prompt": [[{"role": "user", "content": f"Increment the counter to {i}."}] for i in range(1, 7)],
        "target": list(range(1, 7)),
    }
)

class IncrementEnv:
    def reset(self, target, **kwargs) -> None:  # the row's `target` column arrives as a keyword argument
        self.counter = 0
        self.target = target

    def get_reward(self) -> float:
        return float(self.counter == self.target)

    def increment(self, step: int) -> int:  # exposed as a tool (see the full docstring above)
        self.counter += step
        return self.counter

trainer = ZeroSyncGRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=dataset,  # the dataset owns the task; the "target" column reaches reset(target=...)
    environment_factory=IncrementEnv,
)
trainer.train()
```

### Rewards

Environment-owned and trainer-owned rewards are not exclusive. Pass `reward_funcs` for rewards that are natural to compute from the completion (e.g. a format check on the text) _and_ let the environment return its state-dependent reward through `get_reward`. All sources are summed with weight 1.

```python
def format_reward(completions, **kwargs):  # trainer-owned: scores the completion text
    ...

trainer = ZeroSyncGRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    args=ZeroSyncGRPOConfig(max_steps=1000),
    reward_funcs=format_reward,          # trainer-owned: scores the completion
    environment_factory=IncrementEnv,    # env-owned: scores its internal state via `get_reward`
)
```

### Multiple environments

To train on a mix of tasks in a single run (e.g. a coding task and a game), pass a dictionary mapping environment names to factories instead of a single callable. Each rollout selects its environment through an `environment` field in the dataset, and **only that environment's tools are exposed**. This avoids leaking irrelevant tools (the game's `move` tool to a coding example, and vice versa). Each environment still owns its data and returns the prompt from `reset()`, so the dataset only needs the `environment` column to route between them.

```python
import random

from datasets import Dataset
from trl.experimental.zero_sync_grpo import ZeroSyncGRPOConfig, ZeroSyncGRPOTrainer


class CodingEnv:
    def reset(self, **kwargs) -> str:
        n = random.randint(1, 20)
        return f"Compute the {n}th Fibonacci number."

    def run_code(self, code: str) -> str:  # exposed as a tool only for "coding" examples
        """Run the given Python code and return its stdout.

        Args:
            code: The Python source to execute.

        Returns:
            The captured standard output.
        """
        ...

class GameEnv:
    def reset(self, **kwargs) -> str:
        return "Reach the goal tile."

    def move(self, action: str) -> str:  # exposed as a tool only for "game" examples
        """Apply an action to the game and return the new observation.

        Args:
            action: One of "up", "down", "left", "right".

        Returns:
            The observation after the move.
        """
        ...

# The dataset only routes: each row picks the environment for that rollout
dataset = Dataset.from_dict({"environment": ["coding", "game"] * 500})

trainer = ZeroSyncGRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    args=ZeroSyncGRPOConfig(chat_template_kwargs={"enable_thinking": False}),
    train_dataset=dataset,
    reward_funcs=reward_func,
    environment_factory={"coding": CodingEnv, "game": GameEnv},
)
trainer.train()
```

Environment instances are reused across steps (reset between episodes), so an expensive `__init__` is paid once rather than every step. The `environment` field is consumed by the trainer and is **not** forwarded to `reset`.

### Multi-turn rollouts

Every turn renders the whole conversation through the chat template and generates from it. A chat template does not always render a growing conversation the same way (some drop the reasoning of earlier turns, for example), so a turn's prompt tokens can differ from the tokens held so far. At the end of the rollout, the loop reconciles the turns into training samples: a turn whose prompt starts with the held tokens is appended to the same sample, and any other turn starts a new one. Every generated token is trained in exactly one sample, so a fork costs forward tokens but no training signal.

Every sample of a rollout gets the rollout's advantage. `rollout/fork_frac` and `rollout/drift_tokens_*` show how the turns were reconciled.

## Training Vision Language Models

Vision-language models (Qwen3.5, Qwen3.6, Qwen3-VL, ...) can be trained on **text-only** datasets. Pass the model id as usual; nothing else changes.

**The vision tower is frozen.** A text-only dataset never produces image tokens, so the tower is never exercised by the forward pass. Everything outside the text model (vision tower, multimodal projector) has `requires_grad=False`: it gets no gradients and no optimizer state. The tower is still loaded, so it costs GPU memory for its weights.

> [!WARNING]
> **Images are not supported yet.** Prompts containing images are not passed to the engine or to the training forward pass.

## ZeroSyncGRPOTrainer

[[autodoc]] trl.experimental.zero_sync_grpo.ZeroSyncGRPOTrainer
    - train
    - save_model
    - push_to_hub

## ZeroSyncGRPOConfig

[[autodoc]] trl.experimental.zero_sync_grpo.ZeroSyncGRPOConfig

## ZeroSyncRolloutLoop

[[autodoc]] trl.experimental.zero_sync_grpo.ZeroSyncRolloutLoop
    - filter_prompt
    - generate_one
    - generate_one_turn
    - compute_advantages
    - setup
    - close

## Rollout

[[autodoc]] trl.experimental.zero_sync_grpo.Rollout

## TurnRecord

[[autodoc]] trl.experimental.zero_sync_grpo.zero_sync_grpo_trainer.TurnRecord

## RolloutGroup

[[autodoc]] trl.experimental.zero_sync_grpo.zero_sync_grpo_trainer.RolloutGroup
