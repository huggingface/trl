# Zero-Sync GRPO

GRPO where generation and training share **one copy of the weights**. A transformers [continuous batching](https://huggingface.co/docs/transformers/main/en/continuous_batching) manager generates from the same parameter tensors the optimizer updates in place, so there is no inference server, no weight synchronization step, and no second copy of the model in memory.

Generation and training take turns on the same weights: prompts are submitted continuously, each completion is scored as it lands, a group's advantages are computed as soon as its last completion arrives, and once enough scored samples are ready the engine is paused, the trainer runs its step, and generation resumes where it left off. A training batch is formed from whichever scored samples are ready first. A slow rollout never holds up a batch of fast ones, it simply lands in a later batch.

## Usage

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

The generation engine reserves its KV cache when the trainer starts, and whatever it takes is unavailable to training. Size it from what generation actually needs, roughly `requests in flight x expected completion length x KV bytes per token`, and leave the rest to gradients, optimizer states and activations:

```python
ZeroSyncGRPOConfig(
    continuous_batching_config={"max_memory_percent": 0.25},
    rollouts_in_flight=32,
)
```

An oversized pool is not free: it starves the training step, which shows up as an out-of-memory error in the backward pass, not as slow generation.

## Packed training

Training rows are packed back to back rather than padded to the longest sample in the batch, which on mixed lengths would waste up to a third of the forward on pad tokens. Position ids restart at each sample and no attention mask is passed, so the model builds the block-diagonal mask itself; each token keeps its own sample's advantage and behavior logprob, so the loss is unchanged.

Unless `model_init_kwargs` sets one, the attention implementation defaults to `flex_attention`, so the block-diagonal mask skips the cross-sample blocks instead of computing and masking them. Models with linear attention layers get flash attention instead, whose kernels read the sample boundaries from the varlen kwargs.

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

`completions/forks` logs how often this happens, and `tools/call_count` how many tool calls a rollout makes.

## Staleness

Because the weights change while a rollout is being generated, a long completion can span several optimizer steps: its tokens are sampled by different versions of the policy. The engine returns, for each token, the logprob computed by the weights that actually sampled it, so the importance ratio in the clipped loss is exact per token even though no single old policy exists.

What remains is that the KV cache of a rollout's prefix was computed with older weights, while the training forward recomputes it with current ones. Measured on Qwen3-0.6B at `learning_rate=1e-5`, the resulting logprob gap peaks at about 0.16 nats on a rollout's first tokens (an importance ratio of 1.17, inside the default clip range) and falls to the kernel-numerics floor by token 50. It grows with the learning rate, the completion length, and `rollouts_in_flight`.

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
The pause is agreed on by every rank at the same engine step, and the engine finishes the step it had
in flight on the device before it reports itself paused, so the trainer's collectives never share the
device with the engine's, and the forward and backward have the device to themselves. Nothing is drained and no request is lost: the rollouts in flight keep their
cache and carry on after the step. Every rank receives every completion, so every rank reaches the
pause on the same sample count without any signal between them.

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

The replicas draw their training batches from one pool. A sample is data, and nothing ties it to the
replica that generated it, so waiting until the replicas together have enough and then handing them
out to whoever carries the least, by sum of squared lengths, keeps their forwards aligned and their
memory peaks alike. Otherwise a replica whose completions came back short trains first and then sits
at the gradient sum waiting for one still decoding. `batch/row_imbalance` logs how even the result
is, 1.0 being even, and `batch/from_other_replicas` how much of the batch came from elsewhere.

Each replica also keeps the optimizer state of only its share of the parameters, updates it, and
passes the updated parameters back, since replicas hold the same weights and none of them needs a
second copy of Adam's moments. The parameters stay whole everywhere, which is what generation reads.

Two consequences to know about. The vocabulary projection is replicated rather than split, which
costs one copy of that matrix per process and removes a gather from every forward. And the engine
decodes through a second view of the model, sharing every parameter, because continuous batching
switches a model to a paged attention implementation that cannot serve the training forward; that
view costs no extra memory.

Which knob to turn depends on whether generation or training takes the step's time, and
`generation_wait_s` says which: it logs how long each step waited for the engine after resuming it.
Generation and training take turns, so a step costs the training step plus that wait, and the wait
is the time to decode what was still missing when the previous step ended. Under tensor parallelism
three more numbers say where a step's time goes: `generation/pause_s` is how long the pause took to be
granted (one engine step, tens of milliseconds), `batch/pool_exchange_s` is the time spent handing
samples between replicas, which grows when replicas reach the exchange at different moments, and
`memory/alloc_retries` counts the allocator's retries, which stay at zero unless the KV pool leaves the
training step too little room (then lower `max_memory_percent` in `continuous_batching_config`).

On one GPU the wait is where most of the step goes, and the generation-side lever is
`rollouts_in_flight`: a decode step costs about the same whatever number of sequences it carries, so
more rollouts in flight means more tokens per second while the engine runs. Measured on one H100 with
Qwen3-0.6B, GSM8K, 512-token completions, 8 generations per prompt, 64 samples per step:

| rollouts in flight | wait, share of the step | steps/s | completion tokens/s |
|---|---|---|---|
| 64 | 69% | 0.46 | 5,600 |
| 512 | 57% | 0.57 | 7,100 |

What it costs is staleness rather than time: a rollout spans every optimizer step that happens before
it finishes, about `rollouts_in_flight / samples per step` of them (1 and 8 in the rows above), and
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

## ZeroSyncGRPOTrainer

[[autodoc]] experimental.zero_sync_grpo.ZeroSyncGRPOTrainer
    - train
    - save_model
    - push_to_hub

## ZeroSyncGRPOConfig

[[autodoc]] experimental.zero_sync_grpo.ZeroSyncGRPOConfig
