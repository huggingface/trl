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

## ZeroSyncGRPOTrainer

[[autodoc]] experimental.zero_sync_grpo.ZeroSyncGRPOTrainer
    - train
    - save_model
    - push_to_hub

## ZeroSyncGRPOConfig

[[autodoc]] experimental.zero_sync_grpo.ZeroSyncGRPOConfig
