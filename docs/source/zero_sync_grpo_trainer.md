# Zero-Sync GRPO

GRPO where generation and training share **one copy of the weights**. A transformers [continuous batching](https://huggingface.co/docs/transformers/main/en/continuous_batching) manager generates from the same parameter tensors the optimizer updates in place, so there is no inference server, no weight synchronization step, and no second copy of the model in memory.

Generation never stops: prompts are submitted continuously, each completion is scored as it lands, a group's advantages are computed as soon as its last completion arrives, and a training batch is formed from whichever scored samples are ready first. A slow rollout never holds up a batch of fast ones, it simply lands in a later batch.

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
    generation_ahead=4,
)
```

An oversized pool is not free: it starves the training step, which shows up as an out-of-memory error in the backward pass, not as slow generation.

Under `tp_size > 1` the pool does not have to be held for the whole step. Generation is quiescent while the forward and backward run, so the cache can be freed and handed to them:

```python
ZeroSyncGRPOConfig(
    tp_size=2,
    release_kv_cache_during_step=True,
    continuous_batching_config={"use_async_batching": False},
)
```

Rollouts in flight are copied to host memory and back each step, so this buys room for longer completions or a larger batch at the cost of that copy. `generation/kv_released_gib` logs how much is handed over. It requires `use_async_batching=False`, since offloading a batch that is still in flight corrupts it.

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

What remains is that the KV cache of a rollout's prefix was computed with older weights, while the training forward recomputes it with current ones. Measured on Qwen3-0.6B at `learning_rate=1e-5`, the resulting logprob gap peaks at about 0.16 nats on a rollout's first tokens (an importance ratio of 1.17, inside the default clip range) and falls to the kernel-numerics floor by token 50. It grows with the learning rate, the completion length, and `generation_ahead`.

## Scaling

Every process holds a full copy of the weights and runs its own generation engine, so the default
scaling is data parallel: `accelerate launch --num_processes N`. Generation issues no collectives
there, so the engine free-runs in its background thread and never waits for the trainer.

Tensor parallelism splits one copy of the weights across processes instead, with `tp_size`:

```python
ZeroSyncGRPOConfig(tp_size=4)
```

Generation still runs throughout the training step, but it gets there differently, and the reason is
worth knowing. Under tensor parallelism both generation and training issue collectives, on separate
NCCL communicators. NCCL requires that every rank issue the operations on its communicators in the
same host-side order: [its user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)
states that "to remain deadlock free, users must ensure the order of host-side launches matches for
all devices", and recommends "a deterministic order issued from a single host thread per-device".
Two threads racing cannot provide that, because each rank's order then depends on its own timing.
When the orders disagree the run deadlocks, and it does so in a way that is hard to read: a rank
blocks inside an ordinary kernel launch, not inside a collective, because a CUDA call waits on the
resident NCCL kernel and so prevents the other communicator's kernel from ever launching.

So under `tp_size > 1` the trainer drives the engine itself, between its own steps, rather than
letting it run in a background thread. That is the single deterministic host thread NCCL asks for:
every rank runs the same program, so every rank issues the same collectives in the same order by
construction. `generation/decode_steps` reports how far generation got each step. Nothing is drained
and no request is lost.

Generation is therefore quiescent while the forward and backward run. Interleaving it through the
step instead, a decode step at every layer boundary, was measured and is not worth it: identical
throughput on short completions (1.55 against 1.54 s/step) and about 7% better on long ones (2.28
against 2.44 s/step with 512-token completions and 8 rollouts per prompt), at the cost of a design
that cannot ever hand the KV cache memory to the training step.

Two consequences to know about. The vocabulary projection is replicated rather than split, which
costs one copy of that matrix per process and removes a gather from every forward. And the engine
decodes through a second view of the model, sharing every parameter, because continuous batching
switches a model to a paged attention implementation that cannot serve the training forward; that
view costs no extra memory.

Which knob to turn depends on whether generation or training is the bottleneck, and
`generation_wait_s` tells you which regime you are in: it logs how long each step waited for the
engine.

When it is near zero, generation is fully hidden and the throughput lever is the batch. Measured on
one H100 with Qwen3-0.6B, GSM8K, 512-token completions, 8 generations per prompt, where the wait
stayed around a microsecond throughout:

| samples per step | steps/s | completion tokens/s |
|---|---|---|
| 16 | 0.95 | 2,800 |
| 32 | 0.71 | 4,400 |
| 64 | 0.53 | 6,000 |

Do not assume that regime. A bigger model, longer completions or more generations per prompt make
decoding dominate, and then the wait grows and the levers reverse: a larger batch only makes the
trainer wait longer, while `generation_ahead` and the size of the KV pool are what help. Read the
metric for your own setup rather than copying these numbers.

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
