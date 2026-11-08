# Zero-Sync GRPO

GRPO where generation and training share **one copy of the weights**. A transformers [continuous batching](https://huggingface.co/docs/transformers/main/en/continuous_batching) manager generates from the same parameter tensors the optimizer updates in place, so there is no inference server, no weight synchronization step, and no second copy of the model in memory.

Generation and training take turns on the same weights: prompts are submitted continuously, each completion is scored as it lands, a group's advantages are computed as soon as its last completion arrives, and once enough scored samples are ready the engine is paused, the trainer runs its step, and generation resumes where it left off. A training batch is formed from whichever scored samples are ready first. A slow rollout never holds up a batch of fast ones, it simply lands in a later batch.

## Quick start

```python
from datasets import load_dataset
from trl.experimental.zero_sync_grpo import ZeroSyncGRPOConfig, ZeroSyncGRPOTrainer

dataset = load_dataset("trl-lib/tldr-chat", split="train")


def reward_num_unique_chars(completions, **kwargs):
    return [len(set(completion[0]["content"])) for completion in completions]


trainer = ZeroSyncGRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=reward_num_unique_chars,
    args=ZeroSyncGRPOConfig(output_dir="Qwen3-0.6B-ZeroSyncGRPO"),
    train_dataset=dataset,
)
trainer.train()
```

The dataset must be [conversational](dataset_formats#conversational): every prompt is a list of messages.

## Memory split

The generation engine reserves its KV cache when the trainer starts, and frees it for each training step (`kv_cache_during_step="release"`, the default): the cache waits in host memory and comes back before generation resumes, so the rollouts in flight carry on where they were and the forward and backward can use that memory. The optimizer state waits in host memory too. Size the cache from what generation actually needs, roughly `requests in flight x expected completion length x KV bytes per token`:

```python
ZeroSyncGRPOConfig(
    continuous_batching_config={"max_memory_percent": 0.25},
    max_concurrent_rollouts=32,
)
```

Releasing the cache copies it to the host and back at every step. For small models or short completions, where memory is not the limit, `kv_cache_during_step="keep"` keeps it on the device instead; an oversized pool then starves the training step, which shows up as an out-of-memory error in the backward pass, not as slow generation.

## Packed training

The samples of a micro-batch are packed back to back in one row rather than padded to the longest, which on mixed lengths would waste up to a third of the forward on pad tokens. Position ids restart at each sample and the boundaries are passed as the flash varlen kwargs, so attention stays within a sample; each token keeps its own sample's advantage and behavior logprob, so the loss is unchanged.

Unless `model_init_kwargs` sets one, the attention implementation is flash attention when it is available and `sdpa` otherwise, which builds the block-diagonal mask from the position ids. Models with linear attention layers require flash attention, whose kernels read the sample boundaries from the varlen kwargs.

Measured on Qwen3-4B with `tp_size=4` and batch 256 on GSM8K: 6.98 to 5.82 s/step against padding.

## Multi-turn and tools

Pass tools to train an agent. Each turn re-renders the whole conversation through the chat template, so a template that rewrites history (dropping reasoning, summarizing earlier turns) can re-tokenize tokens the model already generated. When that happens, the conversation forks into a second training row rather than silently masking those tokens as context, and both rows carry the rollout's advantage:

```python
def get_weather(city: str) -> str:
    """Get the weather of a city.

    Args:
        city: The city to get the weather of.
    """
    return "sunny"


trainer = ZeroSyncGRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=reward_num_unique_chars,
    train_dataset=dataset,
    tools=[get_weather],
)
```

`rollout/fork_frac` logs how often this happens, and `tools/call_frequency` how many tool calls a rollout makes.

## Vision-language models

Vision-language models (Qwen3.5, Qwen3.6, Qwen3-VL, …) can be trained on **text-only** datasets. Pass the model id as usual; nothing else changes.

**The vision tower is frozen.** A text-only dataset never produces image tokens, so the tower is never exercised by the forward pass. Everything outside the text tower (vision tower, multimodal projector) has `requires_grad=False`: it gets no gradients and no optimizer state. The tower is still loaded, so it costs GPU memory for its weights.

> [!WARNING]
> **Images are not supported yet.** Prompts containing images are not passed to the engine or to the training forward pass.

## Staleness

Because the weights change while a rollout is being generated, a long completion can span several optimizer steps: its tokens are sampled by different versions of the policy. The engine returns, for each token, the logprob computed by the weights that actually sampled it, so the importance ratio in the clipped loss is exact per token even though no single old policy exists.

What remains is that the KV cache of a rollout's prefix was computed with older weights, while the training forward recomputes it with current ones. Measured on Qwen3-0.6B at `learning_rate=1e-5`, the resulting logprob gap peaks at about 0.16 nats on a rollout's first tokens (an importance ratio of 1.17, inside the default clip range) and falls to the kernel-numerics floor by token 50. It grows with the learning rate, the completion length, and `max_concurrent_rollouts`.

## Scaling

Every process holds a full copy of the weights and runs its own generation engine, so the default
scaling is data parallel: `accelerate launch --num_processes N`. Generation issues no collectives
there, so nothing ties one process's engine to another's: each pauses for its own training step and
resumes after it.

Tensor parallelism splits one copy of the weights across processes instead, with `tp_size`:

```python
ZeroSyncGRPOConfig(tp_size=4)
```

The two never run at the same time, and under tensor parallelism that is not a choice but a
requirement worth knowing about. Both generation and training issue collectives, on separate
NCCL communicators. NCCL requires that every rank issue the operations on its communicators in the
same host-side order: [its user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)
states that "to remain deadlock free, users must ensure the order of host-side launches matches for
all devices", and recommends "a deterministic order issued from a single host thread per-device".
Two threads racing cannot provide that, because each rank's order then depends on its own timing.
When the orders disagree the run deadlocks, and it does so in a way that is hard to read: a rank
blocks inside an ordinary kernel launch, not inside a collective, because a CUDA call waits on the
resident NCCL kernel and so prevents the other communicator's kernel from ever launching.

So the two take turns, at every `tp_size`. The engine keeps decoding in its own thread while the
trainer collects and scores completions; once a step's samples are in, the trainer pauses the engine
(`ContinuousBatchingManager.pause`), runs its forward, backward and optimizer step, and resumes it.
It pauses once per optimizer step, whatever `gradient_accumulation_steps` is: the `Trainer` collects
every micro-batch of a step before it trains on the first.
The pause is agreed on by every rank at the same engine step, and the engine finishes the step it had
in flight on the device before it reports itself paused, so the trainer's collectives never share the
device with the engine's, and the forward and backward have the device to themselves. Nothing is drained and no request is lost: the rollouts in flight keep their
cache and carry on after the step. The ranks of a group generate together and train on the same
batch, so the rollouts run on the group's first rank only: it submits the requests, reads the
completions, runs the tools and the reward functions, computes the advantages and picks the batch,
then sends it to the other ranks of its group, which join the pause and train on it. A tool or a
reward function therefore runs once per completion, not once per rank.

Interleaving generation through the step instead, a decode step at every layer boundary, was measured
and is not worth it: identical throughput on short completions (1.55 against 1.54 s/step) and about 7%
better on long ones (2.28 against 2.44 s/step with 512-token completions and 8 rollouts per prompt),
at the cost of a design that cannot ever hand the KV cache memory to the training step.

The two combine. `tp_size` only has to divide the world, and the ranks the model is not split across
hold replicas of it: they draw their own prompts, and their gradients are summed. Pick the smallest
`tp_size` the model fits in and spend the rest on replicas, because decode is bound by all-reduce
latency, so a wider split slows generation. On 16 H100 with Qwen3-14B, `tp_size=4` with four replicas
does 8220 trained tokens/s against 3280 for `tp_size=8` with two.

```python
ZeroSyncGRPOConfig(tp_size=4)  # on 16 processes: four replicas of a model split four ways
```

The replicas draw their training batches from one pool, exchanged between their first ranks. A
sample is data, and nothing ties it to the replica that generated it, so waiting until the replicas
together have enough and then handing them out to whoever carries the least, by sum of squared lengths, keeps their forwards aligned and their
memory peaks alike. Otherwise a replica whose completions came back short trains first and then sits
at the gradient sum waiting for one still decoding. `batch/row_imbalance` logs how even the result
is, 1.0 being even, and `batch/from_other_replicas` how much of the batch came from elsewhere.

Each replica also keeps the optimizer state of only its share of the parameters, updates it, and
passes the updated parameters back, since replicas hold the same weights and none of them needs a
second copy of Adam's moments. The parameters stay whole everywhere, which is what generation reads.

Two consequences to know about. The vocabulary projection is replicated rather than split, which
costs one copy of that matrix per process and removes a gather from every forward. And the engine
decodes from the training parameters themselves, with the attention implementation the trainer
chose, so a weight update is visible to the next decoding step without any copy.

Which knob to turn depends on whether generation or training takes the step's time, and
`perf/rollout_wait_s` says which: it logs how long each step waited for the engine after resuming it.
Generation and training take turns, so a step costs the training step plus that wait, and the wait
is the time to decode what was still missing when the previous step ended. Under tensor parallelism
three more numbers say where a step's time goes: `generation/pause_s` is how long the pause took to be
granted (one engine step, tens of milliseconds), `batch/pool_exchange_s` is the time spent handing
samples between replicas, which grows when replicas reach the exchange at different moments, and
`memory/alloc_retries` counts the allocator's retries, which stay at zero unless the KV pool leaves the
training step too little room (then lower `max_memory_percent` in `continuous_batching_config`).

On one GPU the wait is where most of the step goes, and the generation-side lever is
`max_concurrent_rollouts`: a decode step costs about the same whatever number of sequences it carries, so
more rollouts in flight means more tokens per second while the engine runs. Measured on one H100 with
Qwen3-0.6B, GSM8K, 512-token completions, 8 generations per prompt, 64 samples per step:

| rollouts in flight | wait, share of the step | steps/s | completion tokens/s |
|---|---|---|---|
| 64 | 69% | 0.46 | 5,600 |
| 512 | 57% | 0.57 | 7,100 |

What it costs is staleness rather than time: a rollout spans every optimizer step that happens before
it finishes, about `max_concurrent_rollouts / samples per step` of them (1 and 8 in the rows above), and
the KV pool has to hold them all, or decode slows instead of the batch filling. A bigger model or
longer completions move the balance further towards generation; more processes at `tp_size=1` do not
change it, since every replica generates for itself. Read the metric for your own setup rather than
copying these numbers.

On the training side, every micro-step pays a fixed host cost for dispatching the forward and
backward, so `per_device_train_batch_size` should be the largest that fits, with gradient accumulation
making up the rest. At the same samples per step, doubling it from 8 to 16 gave 5% on Qwen3-8B at
`tp_size=2` and 22% on Qwen3-30B-A3B at `tp_size=4`, whose 48 sparse layers make the dispatch the
longer part of a small micro-step.

## Debugging

The generation engine runs in a background thread that is not a daemon, so a crash in the
training step kills the main thread while the process stays alive with no progress and no
traceback on screen. The trainer stops the manager on exit to avoid this, but if you build your
own loop on top of continuous batching, a run that appears to hang is usually a training-side
error that already happened: check the log rather than the GPU.

## Customizing the rollout loop

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

## Design philosophy

This trainer is intentionally kept minimal and is not meant to grow into a general-purpose solution. If you need a feature that is not supported, we recommend cloning the repository and adapting the trainer to your needs directly. New features will only be considered when there is significant community demand.

## Logged metrics

The metrics follow [`experimental.async_grpo.AsyncGRPOTrainer`]'s: same names for the same quantities, split by the stage (or _entity_) they are about.

| namespace      | entity it counts                                                                       |
| -------------- | -------------------------------------------------------------------------------------- |
| `rollout/`     | represents one _full_ conversation: its turns, its forks, how long it took to generate |
| `completions/` | what the model generated for one prompt                                                |
| `tools/`       | tool calls metrics                                                                     |
| `sample/`      | represents _one_ training sample                                                       |
| `batch/`       | one micro-batch or one optimizer step (built from one or multiple samples)             |
| `perf/`        | measured seconds and FLOPs                                                             |
| `sampling/`    | how far the trained policy is from the engine that generated the tokens                |
| `generation/`  | the engine's pause, and what happens to its KV cache during the step                   |

Let's also define some terms to make token metrics unambiguous:

- **generated** tokens are what the model produced (`completions/*`).
- **forwarded** tokens are every token the _forward_ pass processes: the prompt, any tool results, and the generated tokens.
- **trained** tokens are the subset the loss is taken over where `completion_mask == 1`. They're the generated tokens, minus any that a turn's `output_mask` excludes.

### Completions

A **completion** is what the engine generated for one prompt, counted across _every turn_ of the rollout.

| metric                                             | meaning                                                                                               |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `completions/mean_length`                          | generated tokens per rollout, averaged over all of its turns                                          |
| `completions/min_length`, `completions/max_length` | shortest and longest rollout in the window                                                            |
| `completions/clipped_ratio`                        | fraction of rollouts whose last turn did not end on EOS, i.e. was cut off by `max_completion_length`. |

### Rollouts

A **rollout** is **one full** conversation: a prompt generated to completion, including every tool round-trip it took to get there. `num_generations` rollouts share a prompt and form a group.

| metric                                                             | meaning                                                                                                                                                                                                                                      |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rollout/duration_s`                                               | wall time for one rollout, from dispatch until the loop collects the finished rollout.                                                                                                                                                                              |
| `rollout/generated_tok_s`                                          | generation throughput over the last interval (windowed), so a stall shows up                                                                                                                                                                 |
| `rollout/concurrency`                                              | rollouts in flight to the engine.                                                                                                                                                                                                             |
| `rollout/turns_mean`, `rollout/turns_max`                          | mean and max turns per conversation                                                                                                                                                                                                          |
| `rollout/samples_per_rollout`                                      | training **samples** produced per conversation. 1.0 means no forking happens; (<1.0 means some conversations produced no trainable sample at all)                                                                                            |
| `rollout/fork_frac`                                                | fraction of turn boundaries where the re-tokenized prompt didn't extend the held tokens, so the turn started a new sample                                                                                                                    |
| `rollout/drift_tokens_mean`, `rollout/drift_tokens_max`            | how many held tokens a fork re-rendered, averaged over turn boundaries (a clean boundary counts as 0). It's what each fork adds to the trainer's forward, since the new sample shares the tokens before the divergence                       |
| `rollout/score_queue_size`                                         | completed groups waiting to be scored.                                                                                                                                                                                                       |
| `rollout/score_s`, `rollout/score_wait_s`, `rollout/score_block_s` | scoring: time to score a group, group wait time to be scored, and how long generation was blocked because the scoring queue was full                                                                                                         |
| `frac_reward_zero_std`, `reward_kept`                              | fraction of groups dropped because every scored reward was equal, or fewer than two rollouts were scorable (GRPO only counts the first case), and the mean reward of the groups that were trained on. `reward` is the policy's reward over every scored group, dropped ones included, so it is the one to track |

A group whose rollouts all received the same reward (or fewer than two of which were scorable) is dropped instead of enqueued: with the default advantages it has zero advantage everywhere, so it would cost a forward and backward and move nothing. The batch is unaffected, it still fills its `samples_per_step` from the kept groups, so a step takes more generation when many groups are dropped.

### Tools

Logged only when the model executes tools. `tools/<name>/*` repeats per tool name, the same way `rewards/<func>/*` repeats per reward function, because one failing tool is invisible in an average over all of them.

| metric                                                  | meaning                                                                                                    |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `tools/call_frequency`, `tools/failure_frequency`       | calls per rollout, and the fraction that failed                                                            |
| `tools/latency_s`, `tools/<name>/latency_s`             | time spent executing the tool                                                                              |
| `tools/<name>/call_total`, `tools/<name>/failure_total` | per-tool call and failure counts                                                                           |
| `tools/unknown_name_total`                              | the model asked for a tool that does not exist: tracks a policy error, unlike a tool that ran and raised |
| `tools/parallel_calls_mean`                             | tool calls requested in a single assistant message                                                         |
| `tools/loop_exhausted_frac`                             | conversations cut off at `max_tool_calling_iterations` while still asking for tools                       |

### Environments

Logged only with an `environment_factory`. Both calls run on the loop's tool thread pool, so a slow environment delays its own rollout but not the others in flight.

| metric         | meaning                                       |
| -------------- | --------------------------------------------- |
| `env/reset_s`  | time in `reset()` before the rollout starts   |
| `env/reward_s` | time in `get_reward()` after the rollout ends |

### Samples

Each sample is one `RolloutSample`: a reconciled sequence plus the group advantage, the reward and its per-sample metrics, assembled in `_score_group` once every generation in a group has finished and been scored.

| metric                                                        | meaning                                                                                                                                             |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sample/forwarded_tokens_mean`, `sample/forwarded_tokens_max` | tokens in one **sample**: prompt + tool context + generated                                                                                        |
| `sample/trained_tokens_mean`                                  | of those, how many the loss is taken over                                                                                                           |
| `sample/time_in_queue_s`                                      | how long this sample waited in the rollout buffer before being trained on                                                                           |
| `sample/staleness_mean`, `sample/staleness_max`               | how many policy versions behind the data is. `sampling/importance_sampling_ratio/*` and `sampling/sampling_logp_difference/*` show the _effect_ of off-policyness on the loss; this shows the cause                 |

### Rewards and off-policyness

| metric | meaning |
| --- | --- |
| `reward_group_std` | std of the reward within each sample's group, averaged over the samples that were trained on. This is what the default `compute_advantages` divides by. GRPO's `reward_std` is the std over the whole batch, a different quantity: a batch of easy prompts all at 1 and hard prompts all at 0 has a batch std of 0.5 but a group std of 0 |
| `rewards/<func>/mean`, `rewards/<func>/group_std` | each reward function's mean, and its std within each sample's group, averaged over the samples that were trained on. GRPO's `rewards/<func>/mean` is over every generated completion in the batch, and its `rewards/<func>/std` is the std over the batch |
| `sampling/importance_sampling_ratio/mean`, `/min`, `/max` | `π_θ / π_sampler` per completion token, where `π_sampler` is the engine's policy when it generated it. It includes staleness, not only the train/inference mismatch |
| `sampling/sampling_logp_difference/mean`, `/max` | `\|log π_θ − log π_sampler\|` per completion token |
| `sampling/masked_mean` | completion tokens dropped from the loss because their importance ratio falls outside [`importance_sampling_clip_min`, `importance_sampling_clip_max`] |
| `num_tokens` | tokens forwarded since the start of training. |

### Batches

One micro-batch is one packed row per replica, and `gradient_accumulation_steps` of those make one optimizer step. Every `_per_step` metric below is a sum over the whole step and across the replicas.

| metric                                                             | meaning                                                                                                                  |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `batch/forwarded_tokens_per_step`, `batch/trained_tokens_per_step` | tokens in one optimizer step, forwarded and trained respectively                                                         |
| `batch/samples_per_step`                                           | training samples per optimizer step                                                                                      |
| `batch/microbatches_per_step`                                      | counted, not read off the config                                                                                         |
| `batch/masked_token_frac`                                          | forwarded tokens with `completion_mask == 0` — the share of the forward that earns no gradient                           |
| `batch/row_imbalance`                                              | `max cost / mean cost` across the replicas' batches, where the cost is the sum of squared sample lengths. 1.0 is perfect |
| `batch/from_other_replicas`, `batch/pool_exchange_s`               | the share of a replica's batch that another replica generated, and the time the exchange took. See [Scaling](#scaling)   |

### Performance

Throughput measures how many tokens the trainer processes per second. **Model FLOPs Utilization (MFU)** estimates the percentage of the training devices' theoretical compute capacity used for model forward and backward passes. The capacity used for this estimate depends on the hardware and training precision.

Throughput and MFU are reported on two time bases:

- **`_fwd_bwd`** measures performance during the forward and backward passes.
- **`_wall_clock`** measures performance over the complete optimizer step, including optimizer updates and time spent waiting for rollouts.

The difference helps distinguish compute performance from the cost of waiting and coordination. Wall-clock MFU counts the time the engine generates as time the training devices are not training, even though the same devices do the generation.

MFU is reported only when peak compute capacity is known for the local training device at the selected precision. Otherwise, MFU is omitted; training continues and throughput and timing metrics remain available.

| metric                                                            | meaning                                                                                                                                               |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `perf/step_s`                                                     | wall time between optimizer steps: compute, optimizer and rollout waits included                                                           |
| `perf/fwd_bwd_s`                                                  | forward + backward, summed over the step's micro-batches. This is the denominator of every `_fwd_bwd` metric below                                    |
| `perf/fwd_s`                                                      | the forward part of it. `fwd_s / fwd_bwd_s` near 1/3 is the usual split; higher means the backward is cheap or recompute is being paid on the forward |
| `perf/optimizer_s`                                                | `optimizer.step()`                                                                                                                                    |
| `perf/rollout_wait_s`                                             | how long the trainer waited for the engine to finish a step's samples. See [Scaling](#scaling)                                             |
| `perf/forwarded_tok_s_fwd_bwd`, `perf/forwarded_tok_s_wall_clock` | forwarded tokens per second on each basis                                                                                                             |
| `perf/trained_tok_s_wall_clock`                                   | the same, counting only tokens the loss saw                                                                                                           |
| `perf/mfu_fwd_bwd`, `perf/mfu_wall_clock`                         | model FLOPs utilisation on each basis                                                                                                                 |

### Engine

| metric                                                       | meaning                                                                                                                                            |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generation/pause_s`                                         | how long the pause took to be granted, plus releasing the KV cache with `kv_cache_during_step="release"`                                         |
| `generation/lend_s`, `generation/give_back_s`                | with `kv_cache_during_step="lend"`, the time to lend the KV cache to the step and to get it back                                                 |
| `generation/lent_copied_gib`, `generation/lent_beyond_gib`   | with `kv_cache_during_step="lend"`, the kept KV cache the step overwrote, copied to the host and back, and the GPU memory it took beyond the cache |
| `memory/alloc_retries`                                       | the allocator's retries: the KV pool and the training step fighting over memory when it grows                                                      |

## ZeroSyncGRPOConfig

[[autodoc]] trl.experimental.zero_sync_grpo.ZeroSyncGRPOConfig

## ZeroSyncGRPOTrainer

[[autodoc]] trl.experimental.zero_sync_grpo.ZeroSyncGRPOTrainer

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
