# OpenEnv Integration for Training LLMs with Environments

## Overview

[OpenEnv](https://github.com/meta-pytorch/OpenEnv) is an open-source framework from Meta's PyTorch team for defining, deploying, and interacting with environments in reinforcement learning (RL) and agentic workflows. It offers [Gymnasium-style APIs](https://gymnasium.farama.org) (e.g., `reset()` and `step()`) to interface with environments in a standard manner, and supports running these environments as backend servers (for example via HTTP or containerised execution). You can find a collection of ready-to-use OpenEnv environments on the [Hugging Face Hub](https://huggingface.co/collections/openenv/environment-hub).

In this guide, we’ll focus on **how to integrate OpenEnv with TRL**, but feel free to explore the links above to dive deeper into OpenEnv itself.

## Installation

To use OpenEnv with TRL, install the framework:

```bash
pip install openenv-core
```

## Using `rollout_func` with OpenEnv environments

TRL's [`GRPOTrainer`] supports _custom rollout logic_ through the `rollout_func` argument. This lets you override the trainer's default text-generation loop and directly interact with OpenEnv environments — for instance, to compute environment-driven rewards instead of relying solely on model-based signals.

### Rollout Function Signature

A rollout function must have the following signature:

```python
def rollout_func(
    prompts: list[str],
    args: GRPOConfig,
    processing_class
) -> dict[str, list]:
    """
    Custom rollout function for generation and reward computation.

    Args:
        prompts: List of prompts to generate from
        args: GRPOConfig containing sampling parameters (temperature, top_p, etc.)
        processing_class: Tokenizer/processor for encoding/decoding

    Returns:
        Dictionary containing:
        - prompt_ids: List of token IDs for each prompt
        - completion_ids: List of token IDs for each completion
        - logprobs: List of log probabilities for each token
        - Any additional fields are forwarded to reward functions as kwargs
    """
    pass
```

> [!NOTE]
> Any extra fields in the returned dictionary (beyond the required three) are automatically forwarded to your reward functions. This makes it easy to propagate signals such as environment rewards or auxiliary metrics from the rollout step.

### Integration pattern

The typical pattern when combining OpenEnv with TRL looks like this:

1. Start or connect to an OpenEnv environment (e.g., an HTTP endpoint or Dockerized env).
2. Generate completions from your model — for example, via a vLLM inference server (`use_vllm=True`, `vllm_mode="server"`).
3. Step through the environment using each completion to compute rewards or metrics.
4. Add environment results (e.g., `env_reward`) to the rollout result dict.
5. Access those rewards inside your reward function via `**kwargs`.

By using OpenEnv in this loop, you can:

* Train with realistic or interactive feedback (not just static reward functions).
* Plug in custom simulators, web APIs, or evaluators as environments.
* Pass structured reward signals back into RL training seamlessly.

## A simple example

The [echo.py](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/echo.py) script demonstrates a minimal, end-to-end integration between TRL and OpenEnv. In this example, the Echo environment rewards completions based on their text length, encouraging the model to generate longer outputs. This pattern can be extended to any custom environment that provides structured feedback or task-based rewards:

```python
from envs.echo_env import EchoEnv, EchoAction
from trl import GRPOConfig, GRPOTrainer

# Create HTTP client for Echo Environment
client = EchoEnv.from_docker_image("echo-env:latest")

def rollout_func(prompts, args, processing_class):
    # 1. Generate completions via vLLM inference server (running on port 8000)
    payload = {
        "prompts": prompts,
        "n": args.num_generations,
        "temperature": args.temperature,
        "max_tokens": args.max_completion_length,
    }
    response = requests.post("http://0.0.0.0:8000/generate/", json=payload)
    result = response.json()

    completions_text = processing_class.batch_decode(
        result["completion_ids"],
        skip_special_tokens=True
    )

    # 2. Step through the environment to get rewards
    client.reset()
    env_rewards = []
    for msg in completions_text:
        env_result = client.step(EchoAction(message=msg))
        env_rewards.append(env_result.reward)

    # 3. Add environment rewards as extra field
    result["env_reward"] = env_rewards
    return result

def reward_from_env(completions, **kwargs):
    """Extract environment rewards passed via rollout_func kwargs."""
    env_rewards = kwargs.get("env_reward", [])
    return [float(reward) for reward in env_rewards] if env_rewards else [0.0] * len(completions)

dataset = Dataset.from_dict({"prompt": ["You are an AI that interacts with an *Echo* environment. Word to echo:"] * 64})

# Setup trainer with custom rollout
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=reward_from_env,
    train_dataset=dataset,
    rollout_func=rollout_func,  # Use custom rollout
    args=GRPOConfig(
        vllm_mode="server",
        use_vllm=True,
        num_train_epochs=1,
        num_generations=8,
        max_completion_length=2048,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
    ),
)
trainer.train()
```

That's it! Now that you’ve seen the full example, let’s unpack how the main pieces fit together.

1. **Environment Client:** `EchoEnv` implements an HTTP interface to interact with the environment server.  
2. **Custom rollout:** The `rollout_func` generates completions and steps through the environment to collect rewards.  
3. **Extra fields:** The rollout adds `env_reward` to the result dictionary, which is automatically passed to reward functions.  
4. **Reward function:** Extracts `env_reward` from `kwargs` to apply environment-computed rewards during training.

> [!WARNING]
> The `rollout_func` is currently only supported when using vLLM in server mode (`use_vllm=True`, `vllm_mode="server"`).

### Running the Example

The example requires two GPUs:

```bash
# Terminal 1: Start vLLM inference server
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000

# Terminal 2: Run GRPO training with OpenEnv
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/echo.py
```

Below is the reward curve from training:

<iframe src="https://trl-lib-trackio.hf.space?project=openenv&metrics=train/rewards/reward_from_env/mean&runs=qgallouedec-1761202871&sidebar=hidden&navbar=hidden" style="width:600px; height:500px; border:0;"></iframe>

To learn more about how to create custom environments, see the [OpenEnv documentation](https://github.com/meta-pytorch/OpenEnv/blob/main/src/envs/README.md).
