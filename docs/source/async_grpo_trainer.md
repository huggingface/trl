# Asynchronous GRPO Trainer

[![model badge](https://img.shields.io/badge/All_models-Asynchronous_GRPO-blue)](https://huggingface.co/models?other=async-grpo,trl)

## Overview

TRL supports the Asynchronous GRPO Trainer for training language models with the GRPO method, as described in the paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300) by [Zhihong Shao](https://huggingface.co/syhia), [Peiyi Wang](https://huggingface.co/peiyiwang89), [Qihao Zhu](https://huggingface.co/zqh11), Runxin Xu, [Junxiao Song](https://huggingface.co/haha-point), Mingchuan Zhang, Y. K. Li, Y. Wu, [Daya Guo](https://huggingface.co/guoday).

The abstract from the paper is the following:

> Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, approaching the performance level of Gemini-Ultra and GPT-4. Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9% on MATH. The mathematical reasoning capability of DeepSeekMath is attributed to two key factors: First, we harness the significant potential of publicly available web data through a meticulously engineered data selection pipeline. Second, we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO.

[`experimental.async_grpo.AsyncGRPOTrainer`] trains with the same algorithm as [`GRPOTrainer`], but decouples generation from training: a background worker streams completions from a vLLM server while the training loop consumes them, so generation and gradient updates overlap instead of alternating.

This trainer was contributed by [Amine Dirhoussi](https://huggingface.co/aminediroHF) and [Quentin Gallouédec](https://huggingface.co/qgallouedec).

## Quick start

This example demonstrates how to train a model using the Asynchronous GRPO method. We train a [Qwen3.5 2B model](https://huggingface.co/Qwen/Qwen3.5-2B) with the prompts from the [DeepMath-103K dataset](https://huggingface.co/datasets/trl-lib/DeepMath-103K). You can view the data in the dataset here:

<iframe
  src="https://huggingface.co/datasets/trl-lib/DeepMath-103K/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

Below is the script to train the model. The trainer needs vLLM and kernels, and `accuracy_reward` needs math-verify, which you can install with `pip install trl[vllm,kernels,math_verify]`.

```python
# train_async_grpo.py
from datasets import load_dataset
from trl.experimental.async_grpo import AsyncGRPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = AsyncGRPOTrainer(
    model="Qwen/Qwen3.5-2B",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()
```

Generation runs on a vLLM server, on different GPUs than the trainer. Start it on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3.5-2B \
    --weight-transfer-config '{"backend": "nccl"}' \
    --logprobs-mode processed_logprobs \
    --api-server-count 8 \
    --max-model-len 4096
```

Then execute the script on GPU 1 using the following command:

```bash
CUDA_VISIBLE_DEVICES=1 accelerate launch train_async_grpo.py
```

## Looking deeper into the Asynchronous GRPO method

In [`GRPOTrainer`], generation and training are sequential: generate a batch, compute the loss, update the weights, repeat. Even in [colocate mode](grpo_trainer#option-1-colocate-mode), one phase must finish before the other begins. In [`experimental.async_grpo.AsyncGRPOTrainer`], a rollout worker generates and scores completions in a background process while the training loop computes the loss and updates the model. To understand how it works, it can be broken down into three main steps: **Generating completions**, **computing the advantage**, and **computing the loss**. There is no reference model, so there is no KL divergence term (GRPO's default, `beta=0.0`).

### Generating completions

The rollout worker keeps up to `max_concurrent_rollouts` requests in flight to the vLLM server. For each prompt, it generates a group of  \\( G \\) completions (`num_generations`) and, once the whole group is done, scores it and computes its advantages. The resulting training samples go into a queue, which the training loop pulls from. Generation never waits for a training step, and a training step only waits when the queue is empty.

Every `weight_sync_steps` training steps, the trainer sends its updated weights to the vLLM server over NCCL, so the next completions come from the latest policy. A completion can therefore come from a policy a few steps older than the one being trained: a sample more than `max_staleness` weight updates behind when the trainer pulls it is dropped.

The metrics and the rest of this page use the following terms for the stages a completion goes through:

- a **group** is one prompt and its  \\( G \\) rollouts. The advantage is computed within it.
- a **rollout** is one full conversation generated for a prompt, including every tool call. It is made of one or more **turns**, one vLLM request each.
- a **sample** is what the loss is computed on: the tokens of a rollout, with its advantage. A multi-turn rollout can produce more than one sample (see [Multi-turn rollouts](#multi-turn-rollouts)).
- a **row** is what one process forwards in one micro-batch: several samples packed together (see [Tree packing](#tree-packing)). A **micro-batch** holds one row per process, and `gradient_accumulation_steps` micro-batches make one **step**, the optimizer step.

Token counts come in three kinds: **generated** tokens are what the model produced, **forwarded** tokens are every token of a sample (prompt and tool results included), and **trained** tokens are the ones the loss is taken over.

### Computing the advantage

For each of the  \\( G \\) completions of a group, the reward is the sum of the rewards from each reward function. The advantage is normalized within the group, as in GRPO:

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

There is no `scale_rewards` option: the advantage is always scaled by the group's standard deviation. A completion for which every reward function returned `None` is left out of the group's mean and standard deviation and gets an advantage of zero.

A group whose rewards are all equal has zero advantage everywhere, so the worker drops it instead of spending a forward and backward pass on it. The fraction of such groups is logged as `frac_reward_zero_std`. The step is filled from the groups that are kept, so many dropped groups means more generation per step. A dropped group still counts toward the epoch, which stays one pass over the prompts.

### Computing the loss

The vLLM server's log-probabilities never match the trainer's exactly: different kernels, a different precision, and a policy a few steps behind (see [Dealing with the Training-Inference Mismatch](grpo_trainer#dealing-with-the-training-inference-mismatch)). The trainer uses the sampler's log-probabilities as the old policy, so a single importance ratio covers both the mismatch and the staleness:

$$
r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,< t})}{\pi_{\text{sampler}}(o_{i,t} \mid q, o_{i,< t})}.
$$

Tokens whose ratio falls outside `[vllm_importance_sampling_clip_min, vllm_importance_sampling_clip_max]` are dropped from the loss. For more information, see [Masked Importance Sampling](paper_index#masked-importance-sampling).

By default (`score_centering=True`), the trainer uses [Score Centering](paper_index#score-centering-stabilizes-off-policy-reinforcement-learning). Importance weights alone do not remove the mismatch: the score  \\( \nabla_\theta \log \pi_\theta \\) of a sampled token does not average to zero under the sampler  \\( \mu \\), so every update also pulls the policy toward the sampler. Score centering subtracts that expectation at every position:

$$
\nabla_\theta \mathcal{L} = - \mathbb{E}_{o \sim \mu} \left[ \hat{A} \left( f(r_t) \nabla_\theta \log \pi_\theta(o_t) - \mathbb{E}_{v \sim \mu} \left[ f(r_v) \nabla_\theta \log \pi_\theta(v) \right] \right) \right], \quad r_v = \frac{\pi_\theta(v)}{\mu(v)},
$$

where  \\( f(r) = r \\) inside  \\( [C_{\min}, C_{\max}] \\) and  \\( 0 \\) outside. The inner expectation is taken over the sampler's `num_sampler_logprobs` most likely tokens at each position, which the rollout worker requests from the server.

With `score_centering=False`, the trainer uses GRPO's clipped surrogate objective instead, with the same masked ratio:

$$
\mathcal{L}(\theta) = - \frac{1}{\sum_{i} |o_i|} \sum_{i} \sum_{t=1}^{|o_i|} \mathbb{1}\left[ C_{\min} \le r_{i,t}(\theta) \le C_{\max} \right] \min \left( r_{i,t}(\theta) \hat{A}_{i,t}, \, \text{clip}\left( r_{i,t}(\theta), 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}} \right) \hat{A}_{i,t} \right),
$$

where  \\( \epsilon_{\text{low}} \\) is `epsilon` and  \\( \epsilon_{\text{high}} \\) is `epsilon_high`.

In both cases, the loss is normalized by the number of trained tokens in the micro-batch, counted across all processes, and averaged over the `gradient_accumulation_steps` micro-batches of a step.

## Logged metrics

While training, we record the following metrics:

- `completions/mean_length`: The average length of generated completions. When using tools, only non-tool tokens are counted.
- `completions/min_length`: The minimum length of generated completions. When using tools, only non-tool tokens are counted.
- `completions/max_length`: The maximum length of generated completions. When using tools, only non-tool tokens are counted.
- `completions/clipped_ratio`: The ratio of truncated (clipped) completions, whose last turn does not end with an EOS token.
- `tools/call_frequency`: The average number of tool calls per completion. Logged only when the model calls tools.
- `tools/failure_frequency`: The fraction of tool calls that failed (the tool was not found or raised an exception). It is `0.0` for a completion with no tool call. Logged only when the model calls tools.
- `rewards/{reward_func_name}`: The average reward from a specific reward function, over the trained samples. When an environment owns the reward via `get_reward`, `{reward_func_name}` is the environment's class name.
- `reward`: The overall average reward after summing rewards across functions, over every scored completion, including the groups that are then dropped.
- `reward_std`: The standard deviation of summed rewards across functions, computed within each group and averaged over the trained samples.
- `reward_kept`: The average reward of the trained samples, which excludes the dropped groups.
- `frac_reward_zero_std`: The fraction of groups with a reward std of zero (all answers are correct or incorrect), or with fewer than two scorable completions. These groups are dropped.
- `sampling/masked_mean`: The fraction of trained tokens dropped from the loss because their importance sampling ratio falls outside `[vllm_importance_sampling_clip_min, vllm_importance_sampling_clip_max]`.
- `sampling/sampler_mass`: The average probability mass the sampler's `num_sampler_logprobs` most likely tokens cover. Logged only when `score_centering=True`.
- `ratio`: The average importance sampling ratio  \\( r_{i,t}(\theta) \\) between the model and the sampler, over trained tokens.
- `entropy`: Average entropy of token predictions across trained tokens.
- `aux_loss`: The load-balancing auxiliary loss of a Mixture-of-Experts model, before it is scaled by `router_aux_loss_coef` and added to the loss. Logged only when the model is a MoE model and `router_aux_loss_coef` is nonzero.
- `kl`: The average approximate KL divergence between the model and the sampler, over trained tokens, using the approximator of [Schulman et al. (2020)](http://joschu.net/blog/kl-approx.html). Unlike in GRPO, it is not measured against a reference model.
- `clip_ratio/region_mean`: The ratio of token probabilities where the GRPO objective is clipped to stay within the trust region:  \\( \text{clip}\left( r_{i,t}(\theta), 1 - \epsilon_\mathrm{low}, 1 + \epsilon_\mathrm{high} \right) \\). With `score_centering=True` the loss does not clip, so the `clip_ratio/*` metrics count the tokens outside the trust region.
- `clip_ratio/low_mean`: The average ratio of token probabilities that were clipped on the lower bound of the trust region:  \\(r_{i,t}(\theta) < 1 - \epsilon_\mathrm{low}\\).
- `clip_ratio/low_min`: The smallest per-completion fraction of tokens clipped on the lower bound of the trust region:  \\(r_{i,t}(\theta) < 1 - \epsilon_\mathrm{low}\\).
- `clip_ratio/high_mean`: The average ratio of token probabilities that were clipped on the upper bound of the trust region:  \\(r_{i,t}(\theta) > 1 + \epsilon_\mathrm{high}\\).
- `clip_ratio/high_max`: The largest per-completion fraction of tokens clipped on the upper bound of the trust region:  \\(r_{i,t}(\theta) > 1 + \epsilon_\mathrm{high}\\).

The other metrics describe the asynchronous pipeline. Metrics ending in `_total` count events during the logging window.

- `rollout/duration_s`: The wall time of one rollout, from dispatch to its last turn.
- `rollout/generated_tok_s`: The generation throughput, in generated tokens per second, since the previous report.
- `rollout/concurrency`: The number of rollouts in flight to the vLLM server.
- `rollout/concurrency_cap`: The current cap on rollouts in flight, which follows the server's load. See [Concurrency](#concurrency).
- `rollout/turns_mean`, `rollout/turns_max`: The average and maximum number of turns per rollout.
- `rollout/samples_per_rollout`: The number of training samples per rollout. Above `1.0`, some rollouts were forked; below `1.0`, some produced no trained token. See [Multi-turn rollouts](#multi-turn-rollouts).
- `rollout/fork_frac`, `rollout/realign_frac`: The fraction of turn transitions whose re-tokenization drift forked a new sample or was realigned as context.
- `rollout/drift_tokens_mean`, `rollout/drift_tokens_max`: The number of held tokens a turn's re-tokenization invalidated, to compare against `fork_threshold_tokens`.
- `rollout/score_queue_size`: The number of completed groups waiting to be scored.
- `rollout/score_s`: The time to score a group.
- `rollout/score_wait_s`: The time a completed group waited before being scored.
- `rollout/score_block_s`: The time generation was blocked because the scoring queue was full.
- `rollout/backpressure_s`: The time generation was blocked because the rollout queue was full. See [The rollout queue](#the-rollout-queue).
- `rollout/vllm_retry_total`: The number of retried vLLM requests. A degraded server otherwise looks like unexplained slowness.
- `rollout/kv_cache_usage`: The vLLM KV cache usage (maximum over engines) at the last poll.
- `rollout/vllm_running_requests`: The number of requests the server's engines are decoding (vLLM's `num_requests_running`, summed over engines). The gap to `rollout/concurrency` is what sits in the API server process, building responses.
- `rollout/vllm_queued_requests`: The number of requests the server has accepted but not yet scheduled (vLLM's `num_requests_waiting`, summed over engines).
- `rollout/vllm_preemptions_total`: The number of requests the server preempted (vLLM's `num_preemptions_total`, summed over engines). A weight sync resets the server's caches, which preempts every running request, so this is not a pressure signal.
- `tools/latency_s`, `tools/{tool_name}/latency_s`: The time spent executing a tool call, over all tools and per tool.
- `tools/{tool_name}/call_total`, `tools/{tool_name}/failure_total`: The number of calls and failed calls per tool.
- `tools/unknown_name_total`: The number of calls to a tool that does not exist. Unlike a tool that ran and raised, this is a policy error.
- `tools/parallel_calls_mean`: The average number of tool calls in an assistant message that calls tools.
- `tools/loop_exhausted_frac`: The fraction of rollouts cut off at `max_tool_calling_iterations` while still calling tools. Logged only when tools are provided.
- `env/reset_s`: The time spent in the environment's `reset()` before a rollout starts. Logged only when `environment_factory` is provided.
- `env/reward_s`: The time spent in the environment's `get_reward()` after a rollout ends. Logged only when the environment defines `get_reward`.
- `sample/forwarded_tokens_mean`, `sample/forwarded_tokens_max`: The average and maximum number of forwarded tokens per sample.
- `sample/trained_tokens_mean`: The average number of trained tokens per sample.
- `sample/rollout_queue_size`: The number of samples waiting in the rollout queue.
- `sample/time_in_queue_s`: The time a sample waited in the rollout queue before being trained on.
- `sample/staleness_mean`, `sample/staleness_max`: The number of weight updates between the policy that generated a sample and the one training on it.
- `sample/dropped_stale_total`: The number of samples dropped for exceeding `max_staleness`.
- `batch/forwarded_tokens_per_step`, `batch/trained_tokens_per_step`: The number of forwarded and trained tokens per step, over all processes.
- `batch/samples_per_step`, `batch/groups_per_step`: The number of samples and of distinct groups per step. They differ by the number of samples per group.
- `batch/microbatches_per_step`: The number of micro-batches per step.
- `batch/masked_token_frac`: The fraction of forwarded tokens that are not trained.
- `batch/samples_per_row`, `batch/row_tokens_mean`, `batch/row_tokens_max`: The number of samples per row, and the average and maximum number of forwarded tokens per row.
- `batch/row_fill_frac`: The fraction of `token_budget` the rows use.
- `batch/row_imbalance`: The ratio of the most expensive row's attention cost to the average one. The process with the most expensive row stalls the others at the gradient all-reduce; `1.0` is perfect balance.
- `batch/packing_ratio`: The number of tokens before packing per forwarded token: the factor by which packing shrinks the forward, `1.0` when nothing is shared.
- `batch/pad_frac`: The fraction of padding added to make the rows the same length. It only costs transfer, it is removed before the forward.
- `batch/dropped_oversize_total`: The number of samples dropped for exceeding `token_budget`.
- `perf/step_s`: The wall time of a step, including the optimizer, the weight sync and the time waiting for samples.
- `perf/fwd_bwd_s`, `perf/fwd_s`: The time spent in the forward and backward passes of a step, and in its forward passes alone.
- `perf/optimizer_s`: The time spent in the optimizer step.
- `perf/rollout_wait_s`: The time training waited because the rollout queue was empty. See [The rollout queue](#the-rollout-queue).
- `perf/weight_sync_s`: The time of a weight sync, split into `perf/weight_sync_pause_s` (waiting for vLLM to pause), `perf/weight_sync_barrier_s` (waiting for the other processes) and `perf/weight_sync_transfer_s` (sending the weights).
- `perf/forwarded_tok_s_fwd_bwd`, `perf/forwarded_tok_s_wall_clock`: The number of forwarded tokens per second, over the forward and backward passes or over the whole step.
- `perf/trained_tok_s_wall_clock`: The number of trained tokens per second, over the whole step.
- `perf/mfu_fwd_bwd`, `perf/mfu_wall_clock`: The model FLOPs utilization of the training devices, over the forward and backward passes or over the whole step. They exclude the vLLM server. Logged only when the peak compute of the training device at the training precision is known.

## Customization

### Compatibility and constraints

Some argument combinations are intentionally restricted in the current [`experimental.async_grpo.AsyncGRPOTrainer`] implementation:

- Generation always runs on a separate vLLM server, started with `vllm serve`. There is no colocate mode.
- The dataset must be in [conversational](dataset_formats#conversational) format.
- Reward functions must be Python callables. Reward models given as a model id or a [`~transformers.PreTrainedModel`] are not supported, see [Using a custom reward function](#using-a-custom-reward-function).
- The trainer needs `transformers>=5.2.0`.
- For distributed training, only FSDP2 is supported (DeepSpeed ZeRO is not). Context and sequence parallelism (`cp_size > 1` or `sp_size > 1` in `parallelism_config`) are not supported.
- `use_liger_kernel=True` is not supported.
- PEFT adapters targeting `lm_head` are not supported.
- Models with attention sinks are not supported. The only non-attention layers supported are the gated DeltaNet layers of hybrid models (Qwen3.5, Qwen3.6, Qwen3-Next, ...), which need `transformers>=5.15.0`.
- Images are not supported, see [Training Vision Language Models](#training-vision-language-models).

Training loads its attention kernel from the Hub through the [`kernels`](https://github.com/huggingface/kernels) library: [`kernels-community/aiter-flash-attn`](https://huggingface.co/kernels-community/aiter-flash-attn) on AMD GPUs (ROCm), [`kernels-community/flash-attn4`](https://huggingface.co/kernels-community/flash-attn4) on NVIDIA Blackwell GPUs and [`kernels-community/flash-attn3`](https://huggingface.co/kernels-community/flash-attn3) on other NVIDIA GPUs. Models that cap their attention logits (`attn_logit_softcapping`), such as Gemma 2, are not supported on AMD GPUs or on NVIDIA Blackwell GPUs.

### Configuring the vLLM server

The trainer generates with a vLLM server, which it reaches at `vllm_server_base_url` (`http://localhost:8000` by default):

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve <model_name> \
    --weight-transfer-config '{"backend": "nccl"}' \
    --logprobs-mode processed_logprobs \
    --api-server-count 8 \
    --max-model-len 4096
```

- `VLLM_SERVER_DEV_MODE=1` and `--weight-transfer-config` expose the endpoints the trainer sends its weights through.
- `--logprobs-mode processed_logprobs` makes the server return the log-probabilities of the distribution it sampled from, after temperature and top-k/top-p, which is what the importance ratio compares against.
- `--api-server-count 8` keeps the top log-probabilities that score centering requests cheap to serve (see the warning below).
- `--max-model-len` is the maximum total sequence length (prompt and completion) you expect. A lower value reduces GPU memory usage on the server, freeing more memory for the KV cache and increasing throughput. A good starting point is the prompt length plus `max_completion_length` from your config. It is also the default `token_budget`.

> [!WARNING]
> Make sure that the server is using different GPUs than the trainer, otherwise you may run into NCCL errors. You can specify the GPUs to use with the `CUDA_VISIBLE_DEVICES` environment variable.

> [!WARNING]
> Serializing the top log-probabilities is CPU work in vLLM's API server, and with its default single API server `num_sampler_logprobs=8` halves generation throughput. Start the server with `--api-server-count 8` on vLLM 0.32 or later: measured on one H100 with 1.5B to 7B models, the top-8 log-probabilities then cost under 3% of generation throughput. On vLLM 0.31 the same flag still leaves a 20 to 40% loss, fixed upstream by [vllm-project/vllm#58181](https://github.com/vllm-project/vllm/pull/58181).

### Concurrency

The number of concurrent requests sent to the vLLM server is capped by `max_concurrent_rollouts`. By default it is set automatically to `max(max_staleness, 1) × per_device_train_batch_size × gradient_accumulation_steps × num_processes`: the maximum number of samples the trainer can consume before they become stale. Generating more than this is wasteful, since the excess samples are dropped.

`max_concurrent_rollouts` is a ceiling, not the number actually in flight. The worker polls the server's `/metrics` endpoint every few seconds and keeps a live cap underneath it (`rollout/concurrency_cap`): past a certain concurrency, more requests make vLLM *slower*, because the KV cache fills, sequences are preempted and recomputed, and cached prefixes are evicted and re-prefilled. The cap starts at `KV cache tokens / max_model_len`, read from the server at startup: as many requests as fit at full length, so the first step cannot thrash. While usage stays under 0.6 with nothing queued, the cap grows by 25% each time a cap's worth of rollouts has completed, up to the ceiling. When KV cache usage passes 0.8, the cap is cut by 30%. No running request is cancelled: admissions stop until the pool has drained below the new cap, and no further cut is taken until then. If the server does not expose `/metrics`, the cap stays at the ceiling.

### The rollout queue

The worker pushes scored samples into a queue and the trainer pulls from it. Four metrics describe that queue:

- `sample/rollout_queue_size`: how many samples are waiting.
- `sample/time_in_queue_s`: how long a single sample waited before being trained on, the time part of its staleness.
- `perf/rollout_wait_s`: how long training waited because the queue was empty.
- `rollout/backpressure_s`: how long generation waited because the queue was full.

The last two are never both large. Read together with the queue size, they tell you which side is the bottleneck:

- queue near empty, `perf/rollout_wait_s` high: **generation-bound**. The trainer is starving; look at `rollout/generated_tok_s`, `rollout/concurrency` and `rollout/score_queue_size`.
- queue near full, `rollout/backpressure_s` high: **trainer-bound**. Generation is throttled and its output ages in the queue, so `sample/staleness_mean` climbs.
- both near zero: balanced.

### Tree packing

The samples of a micro-batch are packed into one row per process, without padding. A row is a **prefix trie**, not a concatenation: a token shared by several samples is forwarded once, and attention still shows each sample exactly its own history. The rollouts of a group all repeat its prompt, and every turn of a multi-turn rollout repeats the conversation so far, so the forward shrinks by the packing ratio (`batch/packing_ratio`). It is a layout change only: every trained token keeps its own target, advantage and old log-probability, and its log-probability matches forwarding the sample on its own.

On two rollouts of one prompt:

```
1. Rollouts             r0:  p1 p2 p3 a1 a2
                        r1:  p1 p2 p3 b1 b2                      10 raw tokens

2. Trie (one per row)   S0 [p1 p2] ─┬─ S1 [p3 a1 a2]          nodes are keyed on (token, next token),
                                   └─ S2 [p3 b1 b2]          so the fork happens one token early and
                                                              every node predicts exactly one token;
                                                              chains without a fork become segments

3. Packed row           idx           0  1 | 2  3  4 | 5  6  7
   (DFS order)          input_ids    p1 p2 | p3 a1 a2 | p3 b1 b2  8 tokens: MLPs, norms and projections
                        position_ids  0  1 |  2  3  4 |  2  3  4  run once per unique token
                        segment      S0    | S1       | S2

4. Attention            seg   queries      K/V gathered            kv_index
   (one FA3 call)       S0    [p1 p2]      [p1 p2]                 0 1
                        S1    [p3 a1 a2]   [p1 p2 | p3 a1 a2]      0 1 2 3 4
                        S2    [p3 b1 b2]   [p1 p2 | p3 b1 b2]      0 1 5 6 7

5. Loss                 p2 (idx 1) predicts p3 for both rollouts; every trained token of every rollout
                        points at the position that predicts it
```

A few consequences:

- **Any common prefix is shared**: the rollouts of a prompt, and across prompts a common system prompt or tool schema. A sample goes to the row that already holds its group, where its prompt already is. When that row is full, the sample goes to another row if a process still has an empty one, or if that does not raise the most loaded row, which sets the micro-batch's time; otherwise the micro-batch closes.
- **Nothing shared is plain packing**: every sample is then its own segment, which is ordinary padding-free packing.
- **Only K/V is duplicated**: a shared prefix is gathered once per descendant segment, which is about what unpacked training would hold, and small with grouped-query attention.
- **Gated DeltaNet hybrids are supported** (Qwen3.5, Qwen3.6, Qwen3-Next, ...): their convolution reads each token's own predecessors, and their chunked delta rule starts each segment from its parent's final state.

`token_budget` bounds the tokens of a row, counting a shared token once. By default, it is the server's `max_model_len`, so no sample can exceed it. With nothing shared, tree packing runs at about the speed of plain padding-free packing, and the speed-up grows roughly with the packing ratio (about 2.8 to 3x at a ratio of 3.4 on Qwen3-0.6B and 1.7B, on H100). The ratio is shared prefix times samples per row, so `batch/samples_per_row` is the leading indicator: near 1, nothing can be shared, and the lever is `token_budget`. `batch/row_fill_frac` matters when samples are long: a 10k-token sample tiles a 32k budget badly (three fit, four never do, so the planner often gets two and the row runs about 77% full), while 1k-token samples tile it almost perfectly.

Judge tree packing on throughput, `batch/trained_tokens_per_step / perf/fwd_bwd_s`. A packed micro-batch holds more samples, so `perf/fwd_bwd_s` rises and `perf/mfu_*`, which divides by forwarded tokens, falls: neither is a regression. More samples per row also means more rollouts drawn per step, so a generation-bound run becomes more so.

The batch metrics compose: one step holds `gradient_accumulation_steps × num_processes` rows, so `batch/samples_per_step` is about that many times `batch/samples_per_row`, and likewise for tokens. The two sides agree only up to the variation between the micro-batches of a step, since the per-step metrics are sums and the per-row ones are means.

> [!TIP]
> Anything that makes rows longer or fewer removes the sharing. Check `fork_threshold_tokens` first: a turn only starts a new sample when its re-tokenization drifts by that many tokens, and agentic drift is typically tens of tokens against a default of `1024`, so nothing forks and a conversation collapses into one row per process. `rollout/fork_frac` says which regime you are in.

### Top-k and top-p sampling

Top-k and top-p renormalize the sampling distribution over a restricted candidate set rather than the whole vocabulary. The server's log-probabilities reflect that (`--logprobs-mode processed_logprobs`), so the trainer has to normalize over the same set, otherwise every importance ratio is biased: a token's probability under the whole vocabulary is lower than under the candidate set it was drawn from ([DeepSeek-V3.2](https://huggingface.co/papers/2512.02556), section 3.1). With `top_p < 1` or `top_k > 0`, the worker asks the server for the candidate set of each sampled token and the trainer renormalizes its log-probabilities over it. This needs two more server flags, and `top_k` has to be set, since vLLM bounds the size of the candidate set with it:

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve <model_name> \
    --weight-transfer-config '{"backend": "nccl"}' \
    --logprobs-mode processed_logprobs \
    --api-server-count 8 \
    --max-model-len 4096 \
    --return-sampling-mask \
    --enable-scale-out
```

```python
from trl.experimental.async_grpo import AsyncGRPOConfig

training_args = AsyncGRPOConfig(
    ...,
    top_p=0.95,
    top_k=512,
)
```

`--return-sampling-mask` makes vLLM record the candidate set (vLLM 0.28 or later, on its V2 model runner: `VLLM_USE_V2_MODEL_RUNNER=1` if the server refuses the flag), and `--enable-scale-out` exposes the token-in-token-out endpoint that returns it. The worker checks both on the server when it starts. An older vLLM cannot return the set, so the trainer keeps normalizing over the whole vocabulary and warns about the biased ratio.

### Train adapters with PEFT

We support tight integration with the 🤗 PEFT library, letting you train adapters and share them on the Hub rather than training the whole model.

```python
from datasets import load_dataset
from peft import LoraConfig
from trl.experimental.async_grpo import AsyncGRPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = AsyncGRPOTrainer(
    model="Qwen/Qwen3.5-2B",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
    peft_config=LoraConfig(r=32, lora_alpha=64, target_modules="all-linear"),
)
trainer.train()
```

Every weight sync has to get the updated policy to the server. By default the trainer merges the adapter into the base model and sends the full merged weights, which is around 15 GB for a 7B model on every sync. Start the vLLM server with `--enable-lora` and it sends only the adapter instead, about 160 MB at `r=32`:

```bash
VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 vllm serve <model_name> \
    --weight-transfer-config '{"backend": "nccl"}' \
    --logprobs-mode processed_logprobs \
    --api-server-count 8 \
    --max-model-len 4096 \
    --enable-lora \
    --max-lora-rank 32 \
    --max-loras 6
```

`VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` exposes the endpoint the trainer posts each new adapter to. Keep `--weight-transfer-config` even with `--enable-lora`: the trainer only chooses a sync mode when it starts, by which point the server is already up, and merged sync is the fallback.

`--max-lora-rank` must be one of `1, 8, 16, 32, 64, 128, 256, 320, 512`. It sets the highest rank the server can serve rather than the rank it will serve, so an `r=4` adapter works fine under `8`. `--max-loras` must be at least `max_staleness + 2`: the trainer keeps `max_staleness + 1` adapter versions registered so a rollout that started under an older policy can finish under it instead of switching policies mid-generation, and each sync loads the next version before it unloads the oldest. The trainer checks both values when it starts and tells you what to restart the server with.

vLLM loads adapters from disk rather than over the network, so the trainer writes each version under `<output_dir>/.vllm_lora/` and gives the server the path. If the server runs on a different machine, that directory has to be on a filesystem both of them can see.

> [!WARNING]
> `<output_dir>/.vllm_lora/` is a serving cache, not where your trained adapter is kept. It only holds the most recent versions, and each one is deleted shortly after it falls out of the staleness window. Checkpoints are what preserve the adapter you trained, so do not turn them off: with `save_strategy="no"` a multi-hour run finishes and leaves nothing behind. `output_dir` also has to be readable by the server and has to outlive the job, so node-local scratch will not work, even though it would keep syncs off a slow shared filesystem.

> [!TIP]
> Adapter sync is not always faster. vLLM generates roughly 1.33 to 1.39x slower per token when it serves a LoRA adapter than when it serves merged weights, and generation is usually what dominates wall-clock time in RL. What you get in return is a much shorter pause at each sync and about 1% of the sync bandwidth. If generation throughput is your bottleneck, merged sync is often still the better choice.

### Using a custom reward function

The [`experimental.async_grpo.AsyncGRPOTrainer`] supports using custom reward functions. To ensure compatibility, your reward function must satisfy the following requirements:

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

> [!WARNING]
> Reward functions run in the rollout worker, a separate process, so everything passed to it is **pickled**. Each reward function, tool, and `environment_factory` (and anything they close over) must be picklable: use a module-level function, [`functools.partial`](https://docs.python.org/3/library/functools.html#functools.partial), or a **callable class instance**. Lambdas and closures raise a `TypeError` at `trainer.train()`. This is a difference from [`GRPOTrainer`], where reward functions are called in-process and closures work.
>
> The rollout process also runs with `CUDA_VISIBLE_DEVICES=""`, so it cannot use the GPU. A GPU-backed reward model (e.g. an `AutoModelForSequenceClassification` scorer) loads without error but silently falls back to the CPU. Keep reward functions lightweight (verifiers like `accuracy_reward`, format or length checks). If you need a GPU reward model, serve it behind its own inference engine (vLLM, TGI, ...) on separate GPUs and have a picklable reward function call it over HTTP.

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
...     [{"role": "assistant", "content": "What is the result of (1 + 2) * 4?"}],
...     [{"role": "assistant", "content": "What is the result of (3 + 1) * 2?"}],
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

Below is an example of using multiple reward functions in the [`experimental.async_grpo.AsyncGRPOTrainer`]. In this example, we define two task-specific reward functions: `math_reward_func` and `coding_reward_func`. The `math_reward_func` rewards math problems based on their correctness, while the `coding_reward_func` rewards coding problems based on whether the solution works.

```python
from datasets import Dataset
from trl.experimental.async_grpo import AsyncGRPOTrainer

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
        if t == "coding":
            # Calculate coding-specific reward
            works = test_code_solution(prompt, completion)
            reward = 1.0 if works else -1.0
            rewards.append(reward)
        else:
            # Return None for non-coding tasks
            rewards.append(None)
    return rewards

# Use both task-specific reward functions
trainer = AsyncGRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=[math_reward_func, coding_reward_func],
    train_dataset=dataset,
)

trainer.train()
```

In this example, the `math_reward_func` and `coding_reward_func` are designed to work with a mixed dataset that contains both math and coding problems. The `task` column in the dataset is used to determine which reward function to apply to each problem. If there is no relevant reward function for a sample in the dataset, the reward function will return `None`, and the [`experimental.async_grpo.AsyncGRPOTrainer`] will continue with the valid functions and tasks. This allows the [`experimental.async_grpo.AsyncGRPOTrainer`] to handle multiple reward functions with different applicability.

Note that the [`experimental.async_grpo.AsyncGRPOTrainer`] will ignore the `None` rewards returned by the reward functions and only consider the rewards returned by the relevant functions. This ensures that the model is trained on the relevant tasks and ignores the tasks for which there is no relevant reward function.

#### Example 5: Asynchronous reward functions

Custom reward functions can also be defined as `async def` coroutines. This is useful if your reward depends on slow I/O (for example, calling a remote service). When you pass multiple async reward functions, [`experimental.async_grpo.AsyncGRPOTrainer`] executes them concurrently so their latency overlaps.

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

To use your custom reward function, pass it to the [`experimental.async_grpo.AsyncGRPOTrainer`] as follows:

```python
from trl.experimental.async_grpo import AsyncGRPOTrainer

trainer = AsyncGRPOTrainer(
    reward_funcs=reward_func,
    ...,
)
```

You can pass several reward functions as a list; this list may include both synchronous and asynchronous functions:

```python
from trl.experimental.async_grpo import AsyncGRPOTrainer

trainer = AsyncGRPOTrainer(
    reward_funcs=[reward_func, async_reward_func1, async_reward_func2],
    ...,
)
```

and the reward will be computed as the sum of the rewards from each function.

### Resuming from a checkpoint

The base [`~transformers.Trainer`] resumes by skipping and replaying the batches it already trained on, which does not apply to a live rollout queue, so `ignore_data_skip` is always `True`. Instead, the index of the first prompt not yet trained on is saved to `rollout_state.json` alongside each checkpoint and restored on resume, so the worker fast-forwards to that prompt without replaying samples. It is the *trained* position, not the generator's: the worker runs ahead of training by the rollout queue depth, and those buffered samples are lost when the run ends, so resuming from the generator's position would skip prompts that were generated but never trained on. Streaming datasets ([`~datasets.IterableDataset`]) cannot be repositioned, so their worker restarts from the first prompt on resume.

Under FSDP with `bf16=True` (the default) or `fp16=True`, `model.safetensors` is written in that dtype even though the master weights are fp32, so a pushed checkpoint is the size of the served model, not twice it. The fp32 master weights and optimizer state still go to the resume files (`pytorch_model_fsdp.bin`, `optimizer.bin`), which are what a resume reads, so resuming is unaffected. Without FSDP, the checkpoint is also the resume file, so it keeps the master dtype.

## Agent Training

Asynchronous GRPO supports **agent training**: the model calls tools during generation and learns from the outcome.

- A **tool** is a plain Python function (sync or async) exposed to the model. Use `tools` for stateless calls (a calculator, a web search).
- An **environment** is the more general form: a stateful object built fresh per rollout, whose public methods are exposed as tools, plus a `reset` lifecycle hook and an optional `get_reward` that lets it own the reward. Use `environment_factory` when you need per-rollout state, a reset hook, or environment-owned reward.

They compose — you can pass standalone `tools` alongside an `environment_factory`.

### Tools

The `tools` argument expects a list of Python functions (sync or async) that define the tools available to the agent:

```python
from trl.experimental.async_grpo import AsyncGRPOTrainer

trainer = AsyncGRPOTrainer(
    tools=[tool1, tool2],
    ...,
)
```

Each tool must be a standard Python function with **type-hinted arguments and return types**, along with a **Google-style docstring** describing its purpose, arguments, and return value.
For more details, see the [Passing tools guide](https://huggingface.co/docs/transformers/en/chat_extras#passing-tools).

The model calls a tool by its name, so each tool is registered under its `__name__` and must be a **module-level function**, which also keeps it picklable for the rollout worker. Synchronous tools run on a thread pool sized to `max_concurrent_rollouts`, so a slow tool never blocks the other rollouts in flight. Within one assistant turn, tool calls run in the order the model emitted them. Up to `max_concurrent_rollouts` synchronous tools can therefore run at once, so a tool that reads or writes shared state must be thread-safe.

> [!TIP]
> The tool call loop requires the chat template to be *prefix-preserving* (appending a tool message must not change how earlier messages are rendered). For known model families (e.g. Qwen3, DeepSeek-V3), TRL automatically swaps in a patched training template when tools are enabled. See [Chat Templates](chat_templates#training-templates) for the full list.

Use `max_tool_calling_iterations` in the [`experimental.async_grpo.AsyncGRPOConfig`] to cap the number of tool-calling turns. By default there is no limit, and generation stops when the model produces a response turn with no tool calls.

Example:

```python
from trl.experimental.async_grpo import AsyncGRPOTrainer

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

trainer = AsyncGRPOTrainer(
    tools=[multiply, async_add],
    ...,
)
```

### Environments

You can also provide tools through `environment_factory`. In this mode, [`experimental.async_grpo.AsyncGRPOTrainer`] creates one environment instance per rollout and exposes the environment's public methods as tools.

When you use an `environment_factory`, the environment owns the data: there is no external `train_dataset`. On each rollout, `reset()` produces the task — self-sampling from a corpus the environment holds, or generating a state procedurally — and returns the prompt. `max_steps` sets the training length. The following is a minimal example of an environment that self-samples a target and exposes an `increment` method (exposed as a tool).

```python
import random

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer

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

trainer = AsyncGRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    args=AsyncGRPOConfig(max_steps=1000, chat_template_kwargs={"enable_thinking": False}),
    environment_factory=IncrementEnv,
)
trainer.train()
```

An environment class has two reserved methods: `reset` and `get_reward`. Unlike any other public methods, these are not exposed to the model as tools.

- `reset` (required) is called when the rollout starts. It can return either `None` or a string. When it returns a string, that string is the user prompt.
- `get_reward` (optional, sync or async) takes no argument and returns a `float`: the environment scores the episode it just ran from its own internal state (did the game end in a win? was the word guessed?). It is called once per completed rollout and acts as a reward source.

Both run on the worker's tool thread pool, so a slow environment delays its own rollout but not the others in flight. Their durations are logged as `env/reset_s` and `env/reward_s`.

> [!NOTE]
> Because the environment self-samples on each `reset()`, the `G` members of a GRPO group may not share the same initial state, making the group baseline slightly noisier. For identical states within a group, have `reset()` derive the state deterministically from a shared key.

#### Providing an external dataset (optional)

Instead of letting the environment self-sample, you can provide an external `train_dataset`. For example when your tasks already live in a dataset, or to route between [multiple environments](#multiple-environments). Per rollout, the trainer samples one row, uses its `"prompt"` column as the prompt, and passes the row's columns to `reset()` as keyword arguments (so the environment reads the task from the dataset).

```python
from datasets import Dataset
from trl.experimental.async_grpo import AsyncGRPOTrainer

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

trainer = AsyncGRPOTrainer(
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

trainer = AsyncGRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    args=AsyncGRPOConfig(max_steps=1000),
    reward_funcs=format_reward,          # trainer-owned: scores the completion
    environment_factory=IncrementEnv,    # env-owned: scores its internal state via `get_reward`
)
```

### Multiple environments

To train on a mix of tasks in a single run (e.g. a coding task and a game), pass a dictionary mapping environment names to factories instead of a single callable. Each rollout selects its environment through an `environment` field in the dataset, and **only that environment's tools are exposed**. This avoids leaking irrelevant tools (the game's `move` tool to a coding example, and vice versa). Each environment still owns its data and returns the prompt from `reset()`, so the dataset only needs the `environment` column to route between them.

```python
import random

from datasets import Dataset
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer


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

trainer = AsyncGRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    args=AsyncGRPOConfig(chat_template_kwargs={"enable_thinking": False}),
    train_dataset=dataset,
    reward_funcs=reward_func,
    environment_factory={"coding": CodingEnv, "game": GameEnv},
)
trainer.train()
```

Environment instances are reused across steps (reset between episodes), so an expensive `__init__` is paid once rather than every step. The `environment` field is consumed by the trainer and is **not** forwarded to `reset`.

### Multi-turn rollouts

Every turn renders the whole conversation through the chat template and generates from it. A chat template does not always render a growing conversation the same way (some drop the reasoning of earlier turns, for example), so a turn's prompt tokens can differ from the tokens held so far. At the end of the rollout, the worker reconciles the turns into training samples:

- when a turn's prompt starts with the held tokens, the turn is appended to the same sample.
- when it differs inside the last generated answer by fewer than `fork_threshold_tokens` tokens, the worker realigns: the drifted tokens are replaced by the new prompt's and become context, so they lose their training signal.
- otherwise, the turn starts a new sample, and the tokens of the previous one keep their training signal.

Every sample of a rollout gets the rollout's advantage. `rollout/fork_frac`, `rollout/realign_frac` and `rollout/drift_tokens_*` show how the turns were reconciled.

### Environment Integrations

To train agents that run inside their own harness, such as coding agents in Harbor sandboxes, see [Training on harnesses](openenv#training-on-harnesses-training-real-coding-agents-harbor) in the OpenEnv guide.

## Training Vision Language Models

Vision-language models (Qwen3.5, Qwen3.6, Qwen3-VL, ...) can be trained on **text-only** datasets. Pass the model id as usual; nothing else changes.

**The vision tower is frozen.** A text-only dataset never produces image tokens, so the tower is never exercised by the forward pass. Everything outside the text model (vision tower, multimodal projector) has `requires_grad=False`: it gets no gradients and no optimizer state, and weight sync skips it, so the server keeps the values it loaded from the checkpoint. The tower is still loaded, so it costs GPU memory for its weights.

> [!WARNING]
> **Images are not supported yet.** Prompts containing images are not passed to the vLLM server or to the training forward pass. Multimodal training also needs the padding-free packing path to build 3D M-RoPE positions from the image grid, which is not implemented here.

## AsyncGRPOTrainer

[[autodoc]] trl.experimental.async_grpo.AsyncGRPOTrainer
    - train
    - save_model
    - push_to_hub

## AsyncGRPOConfig

[[autodoc]] trl.experimental.async_grpo.AsyncGRPOConfig

## RolloutWorkerProtocol

[[autodoc]] trl.experimental.async_grpo.async_grpo_trainer.RolloutWorkerProtocol
