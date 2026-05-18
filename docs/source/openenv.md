# OpenEnv Integration for Training LLMs with Environments

[OpenEnv](https://github.com/meta-pytorch/OpenEnv) is an open-source framework for defining, deploying, and interacting with environments in reinforcement learning (RL) and agentic workflows. It provides standardized APIs for environment interaction and supports running environments as backend servers (via WebSocket or containerised execution). You can find a collection of ready-to-use OpenEnv environments on the [Hugging Face Hub](https://huggingface.co/collections/openenv/openenv-environment-hub).

This guide covers **how to integrate OpenEnv with TRL**. For more on OpenEnv itself, see the [OpenEnv docs](https://meta-pytorch.org/OpenEnv/).

> [!NOTE]
> You can explore ready-to-use example [scripts](example_overview#openenv-scripts) and [notebooks](example_overview#openenv-notebooks) in the Examples Overview.

## When to use environments

[`GRPOTrainer`] can be used to train agents. For agentic tasks, it supports two modes: **tools**, where the model can call external functions but each call is stateless and independent, and **environments**, which maintain state across turns, enabling genuine multi-turn interaction where the agent's actions shape future observations. Use environments when continuity matters — for example, navigating a game, browsing a web page, or any task where what the agent sees next depends on what it did before.

## Installation

OpenEnv environments are hosted as Hugging Face Spaces, which are also pip-installable Git repositories:

```bash
# Echo environment
pip install "openenv-echo-env @ git+https://huggingface.co/spaces/openenv/echo_env"

# Wordle (TextArena) environment
pip install "openenv-textarena @ git+https://huggingface.co/spaces/openenv/wordle"

# Catch (OpenSpiel) environment
pip install "openenv-openspiel-env @ git+https://huggingface.co/spaces/openenv/openspiel_env"
```

This installs the **environment client** (e.g., `EchoEnv`) that communicates with the remote environment server via WebSocket, along with the action/observation models and all required dependencies (including `openenv-core`).

> [!TIP]
> You can find the install command for any environment on its HF Space page. Click the **⋮ (three dots)** menu and select **"Use this Space"** to see the install instructions.

> [!TIP]
> You can also install the core package from PyPI with `pip install "openenv-core[core]>=0.2.1"`, but note that environment-specific dependencies may need to be installed separately.

For development, you can clone the OpenEnv repo and install locally:

```bash
git clone https://github.com/meta-pytorch/OpenEnv.git
cd OpenEnv/envs/echo_env
pip install -e .
```

> [!NOTE]
> Each environment script in TRL includes inline dependency metadata (PEP 723) so you can also run them directly with [uv](https://docs.astral.sh/uv/):
>
> ```bash
> uv run examples/scripts/openenv/echo.py
> ```
>
> This automatically installs the required environment package in an isolated virtual environment.

## Quick start

The fastest way to understand the integration is a complete example. The [echo.py](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/echo.py) script trains a model with the [Echo environment](https://meta-pytorch.org/OpenEnv/environments/echo.html), which rewards completions based on their text length:

```python
from datasets import Dataset
from echo_env import EchoEnv
from echo_env.models import EchoAction

from trl import GRPOConfig, GRPOTrainer

ENV_URL = "https://openenv-echo-env.hf.space"

class EchoToolEnv:
    def __init__(self):
        self.env = EchoEnv(base_url=ENV_URL)
        self.reward = 0.0

    def reset(self, **kwargs) -> str | None:
        self.reward = 0.0
        return None

    def echo(self, message: str) -> str:
        """
        Echo the message back from the environment.

        Args:
            message: The message to echo

        Returns:
            The echoed message.
        """
        observation = self.env.step(EchoAction(message=message))
        self.reward = observation.observation.reward
        return observation.observation.echoed_message

def reward_func(environments, **kwargs):
    return [env.reward for env in environments]

dataset = Dataset.from_dict(
    {"prompt": [[{"role": "user", "content": "Try to echo 'Hello World!' in the environment."}]] * 64}
)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=dataset,
    reward_funcs=reward_func,
    args=GRPOConfig(
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
    ),
    environment_factory=EchoToolEnv,
)
trainer.train()
```

That's it. Here's what happens under the hood:

1. **`environment_factory=EchoToolEnv`**: The trainer creates one `EchoToolEnv` instance per generation (pass the class, not an instance).
2. **`reset()`** is called at the start of each episode to initialize state. Returns an observation string (or `None`).
3. **Tool discovery**: The trainer discovers all public methods on the environment instance (here, `echo()`) and exposes them as function-calling tools. Each method must have a proper docstring with typed arguments, which the trainer uses to build the tool schema.
4. **Multi-turn loop**: The trainer generates a completion, parses tool calls, executes `echo()`, appends the result, and generates again, until the model stops calling tools or `max_completion_length` is reached.
5. **Reward function**: Reads `env.reward` from each environment instance after the episode (before the environment is reset).

```bash
# Run the example
python examples/scripts/openenv/echo.py

# Customize model and environment URL
python examples/scripts/openenv/echo.py --model Qwen/Qwen3-0.6B --env-host https://openenv-echo-env.hf.space
```

Below is the reward curve from training:

<iframe src="https://trl-lib-trackio.hf.space?project=openenv&metrics=train/rewards/reward_from_env/mean&runs=qgallouedec-1761202871&sidebar=hidden&navbar=hidden" style="width:100%; max-width:800px; height:500px; border:0;"></iframe>

> [!NOTE]
> You can explore more ready-to-use example [scripts](example_overview#openenv-scripts) and [notebooks](example_overview#openenv-notebooks) in the Examples Overview.

## How `environment_factory` works

TRL's [`GRPOTrainer`] supports interactive environment training through the `environment_factory` argument. When provided, the trainer automatically handles the multi-turn tool-calling loop: it generates completions, parses tool calls, executes them against the environment, and feeds the results back to the model. All without custom rollout code.

### Environment class requirements

Your environment class must follow these rules:

- **`__init__(self)`** *(optional)*: If provided, must take no arguments. Use it to initialize state or clients. If you need external configuration (e.g., a URL), capture it from the enclosing scope or module-level variables.
- **`reset(self, **kwargs)`**: Called at the start of each episode. Receives all dataset columns as keyword arguments. Return a string observation (or `None` for no initial observation).
- **Tool methods**: Any public method (not starting with `_`) other than `reset` is automatically exposed as a tool. Each tool method must have a docstring with `Args:` descriptions, since the trainer uses these to generate the tool schema for the model.

### Tips for environment classes

- **State for reward**: You can store any state you want on the environment instance (e.g., `self.reward`, `self.done`, etc.) and access it in your reward function via the `environments` parameter. Refer to the [Quick Start guide](#quick-start) for an example of this pattern.
- **Error handling**: If a tool method raises an exception (e.g., `ValueError("Game over.")`), the trainer catches it and feeds the error message back to the model as a tool response. This is the recommended way to signal that an action is invalid or that the episode has ended.

```python
ENV_URL = "https://my-env.hf.space"

class MyEnv:
    def __init__(self):
        self.client = MyClient(base_url=ENV_URL)  # captured from enclosing scope
        self.reward = 0.0

    def reset(self, **kwargs) -> str | None:
        self.reward = 0.0
        return "Initial observation for the model"

    def my_tool(self, arg1: str, arg2: int) -> str:
        """
        Description of what this tool does.

        Args:
            arg1: Description of arg1
            arg2: Description of arg2

        Returns:
            The result message.
        """
        self.reward = 1.0
        return "Tool result"
```

> [!IMPORTANT]
> Tools must be **individual methods** with descriptive names and typed arguments (e.g., `guess(word: str)`, `move(direction: str)`). We do not recommend using generic methods like `step(action)`, since the model needs meaningful tool names and argument descriptions to learn tool calling.

### Reward functions

Reward functions receive the `environments` parameter (a list of environment instances), so you can access any state stored during the episode:

```python
def reward_func(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]
```

For more information on reward functions, see the [GRPO - Custom Reward Functions](grpo_trainer#custom-reward-functions).

### Tips for reward functions

A few things we've found helpful when working with OpenEnv environments and GRPO:

- **Simple rewards work well.** In our experiments with Wordle and Sudoku, binary rewards (1.0 for success, 0.0 otherwise) gave cleaner training signals than shaped rewards with partial credit. GRPO compares completions within a group, so the relative ranking matters more than the absolute values.
- **Check the final state, not the path.** When possible, let the environment judge the outcome (e.g., "did the model solve the puzzle?") rather than checking if it followed a specific sequence of actions. This gives the model freedom to discover its own strategies.
- **Test your reward before training.** Run a few episodes manually (see the [Wordle example notebook](https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb)) to confirm the environment returns sensible rewards. If a capable model can't score higher than a random baseline, the reward signal may need adjustment.

### `max_completion_length` in multi-turn episodes

The `max_completion_length` parameter limits the **total number of tokens across the entire multi-turn conversation** (all model generations + tool results combined), not just a single generation. For environments with many turns (e.g., Sudoku with dozens of moves), you may need to increase it:

```python
args = GRPOConfig(
    max_completion_length=4096,  # default is usually 256-1024, increase for long episodes
    # ...
)
```

If episodes are being cut short (model stops mid-game), this is likely the cause.

## Advanced example: Wordle

Let's train a model to play [Wordle](https://www.nytimes.com/games/wordle/index.html) using the [`TextArena`](https://meta-pytorch.org/OpenEnv/environments/textarena.html) environment. This demonstrates multi-turn interaction, cumulative feedback handling, and episode termination via exceptions.

> [!NOTE]
> You can explore the notebook version of this example in [the OpenEnv Wordle GRPO example](https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb).

### The TextArena Environment

[TextArena](https://huggingface.co/papers/2504.11442) is an open-source collection of competitive text-based games designed to evaluate reasoning skills in LLMs using textual games like Wordle, Snake, Tic-Tac-Toe, and more.

![image of TextArena](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/text_arena_evals.png)

### Why Wordle?

Wordle is a good benchmark for environment-based RL because it requires reasoning about feedback, is purely text-based, and models from 1B parameters can improve at it. Each guess is only 8 tokens, making it lightweight to experiment with.

> [!NOTE] How does Wordle work?
> Wordle is a word guessing game where the player has to guess a 5-letter word in 6 attempts. After each guess, the environment provides letter-by-letter feedback:
>
> ```
> G U E S S
> X G Y X X
> ```
> X = not in the word, G = correct position (green), Y = wrong position (yellow). Here, "U" is correct and in place, "E" is in the word but misplaced.

### Environment class

The `WordleEnv` class wraps the TextArena client and exposes `guess()` as the tool:

```python
from textarena_env import TextArenaAction, TextArenaEnv

class WordleEnv:
    def __init__(self):
        self.client = TextArenaEnv(base_url="https://openenv-wordle.hf.space")

    def reset(self, **kwargs) -> str | None:
        result = self.client.reset()
        self._last_full_feedback = result.observation.messages[0].content
        self.reward = 0.0
        self.done = False
        return self._last_full_feedback

    def guess(self, guess: str) -> str:
        """
        Make a guess in the Wordle environment.

        Args:
            guess: The guessed word, formatted as '[abcde]'

        Returns:
            The feedback message from the environment.
        """
        if self.done:
            raise ValueError("Game over.")
        result = self.client.step(TextArenaAction(message=guess))
        _full_feedback = result.observation.messages[0].content
        feedback = _full_feedback[len(self._last_full_feedback):]
        self._last_full_feedback = _full_feedback
        if "You attempted an invalid move" in feedback:
            self.reward = 0.0
        else:
            self.reward = result.reward
        self.done = result.done
        return feedback
```

Key design choices:

- **`reset()`** returns the initial game message as the first observation the model sees.
- **`guess()`** is the only tool. The model calls it each turn with a 5-letter word.
- **Cumulative feedback slicing**: TextArena returns the full game history each turn. We slice out only the new part to avoid repeating context.
- **Exception on done**: If the model tries to guess after the game ends, `guess()` raises a `ValueError`. The trainer catches this and feeds `"Game over."` back to the model as a tool response. The model learns to stop calling tools after this signal.

### Reward function and training

```python
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

def reward_func(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]

prompt = """You are an expert Wordle solver with deep knowledge of English vocabulary...
Use the tool `guess` to make a guess."""

dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": prompt}]] * 1000})

trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    reward_funcs=reward_func,
    train_dataset=dataset,
    args=GRPOConfig(
        use_vllm=True,
        vllm_mode="colocate",
        chat_template_kwargs={"enable_thinking": False},
        max_completion_length=1024,
        num_generations=4,
        gradient_accumulation_steps=64,
    ),
    environment_factory=WordleEnv,
)
trainer.train()
```

The environment returns `1.0` if the model wins and `0.0` otherwise.

### Running the example

<hfoptions id="wordle_vllm_mode">

<hfoption id="colocate">

**Colocate mode (1 GPU, recommended)**

```bash
python examples/scripts/openenv/wordle.py --vllm-mode colocate
```

This runs vLLM in the same process as training, requiring only a single GPU.

</hfoption>

<hfoption id="server">

**Server mode (2+ GPUs, scalable)**

```bash
# Terminal 1: Start vLLM inference server
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000

# Terminal 2: Run GRPO training with OpenEnv
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/wordle.py --vllm-mode server --vllm-server-url http://localhost:8000
```

</hfoption>

</hfoptions>

### Results

The model improves its performance by reducing repetitions and increasing correct guesses. However, Qwen3-1.7B with `enable_thinking=False` is not able to consistently win the game.

<iframe src="https://burtenshaw-wordle-grpo.hf.space?project=group-Qwen-Qwen3-17B&metrics=reward&runs=run-2025-10-26_09-39-49,run-2025-10-26_08-04-49&sidebar=hidden&navbar=hidden" style="width:100%; max-width:800px; height:500px; border:0;"></iframe>

> [!NOTE]
> With `enable_thinking=False` (the default in these examples), small models like Qwen3-1.7B can learn to improve their guesses but should not be expected to consistently solve the game. For significantly better results, use larger models or enable thinking mode (`enable_thinking=True`), which allows the model to reason before making a guess at the cost of longer completions.

We experimented with larger models like [`gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) and found that it was able to consistently win the game, though this requires significantly more compute.

## Multi-environment training

You can train a single model across multiple environments simultaneously. This is useful when you want a model to learn different skills in parallel. For example, playing Wordle (language reasoning) and Catch (spatial reasoning) in the same training run.

The key idea is to create a **meta-environment class** that wraps multiple environments and routes each sample to the correct one using a dataset column.

### How it works

1. Add an `"env"` column (or similar) to your dataset that identifies which environment each sample belongs to.
2. In `reset(**kwargs)`, read `kwargs["env"]` to select the active environment for that episode.
3. Expose tools from all environments; the trainer discovers all public methods.
4. Use separate reward functions per environment, returning `None` for samples that don't belong to that environment. TRL handles `None` values with `nansum`/`nanmean`.

### Example: Wordle + Catch

The [multi_env.py](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/multi_env.py) script trains on Wordle and Catch simultaneously:

```python
class MultiEnv:
    def __init__(self):
        self._wordle_client = None
        self._catch_client = None
        self.active = None
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str | None:
        self.active = kwargs.get("env", "wordle")
        self.reward = 0.0
        self.done = False

        if self.active == "wordle":
            if self._wordle_client is not None:
                try:
                    self._wordle_client.close()
                except Exception:
                    pass
            self._wordle_client = TextArenaEnv(base_url=WORDLE_URL)
            result = self._wordle_client.reset()
            self._last_full_feedback = result.observation.messages[0].content
            self.reward = 0.0
            return self._last_full_feedback
        elif self.active == "catch":
            if self._catch_client is not None:
                try:
                    self._catch_client.close()
                except Exception:
                    pass
            self._catch_client = OpenSpielEnv(base_url=CATCH_URL)
            result = self._catch_client.reset()
            self.done = result.observation.done
            return _format_catch_obs(result.observation.info_state)

    # Wordle tool
    def guess(self, guess: str) -> str:
        """Make a guess in the Wordle environment. ..."""
        ...

    # Catch tools
    def move(self, direction: str) -> str:
        """Move the paddle left or right. ..."""
        ...

    def stay(self) -> str:
        """Do nothing and let the ball fall one step. ..."""
        ...
```

Key patterns:

- **Lazy client initialization**: Create clients in `reset()`, not `__init__()`, to avoid unnecessary WebSocket connections.
- **Close before reopen**: Close the previous client before creating a new one to avoid server capacity errors.
- **`kwargs` routing**: The `"env"` column from the dataset is passed to `reset()` as a keyword argument.
- **All tools are exposed simultaneously**: The model sees `guess`, `move`, and `stay` as available tools regardless of the active environment. If it calls the wrong tool (e.g., `move` during Wordle), the method raises a `ValueError` that the trainer catches gracefully. In practice, models learn to use the correct tools based on the system prompt.

### Per-environment reward functions

Each reward function returns `None` for samples from other environments:

```python
def wordle_reward(environments, **kwargs) -> list[float | None]:
    return [env.reward if env.active == "wordle" else None for env in environments]

def catch_reward(environments, **kwargs) -> list[float | None]:
    rewards = []
    for env in environments:
        if env.active != "catch":
            rewards.append(None)
        elif env.done:
            rewards.append(max(env.reward, 0.0))
        else:
            rewards.append(0.0)
    return rewards
```

TRL converts `None` to `nan` internally and uses `nansum`/`nanmean` for aggregation, so each sample is only scored by its relevant reward function.

### Dataset with environment routing

```python
n = 500
dataset = Dataset.from_dict({
    "prompt": (
        [[{"role": "user", "content": wordle_prompt}]] * n
        + [[{"role": "user", "content": catch_prompt}]] * n
    ),
    "env": ["wordle"] * n + ["catch"] * n,
})
```

### Running the multi-environment example

```bash
python examples/scripts/openenv/multi_env.py \
    --wordle-url https://openenv-wordle.hf.space \
    --catch-url https://openenv-openspiel-env.hf.space \
    --vllm-mode colocate \
    --gradient-accumulation-steps 4 \
    --num-generations 8
```

> [!TIP]
> When training across multiple environments, monitor the per-reward-function metrics (`train/reward_func_0`, `train/reward_func_1`, etc.) rather than the combined `train/reward`. The combined metric alternates between environments and can appear noisy.

## Running the environments

When using `environment_factory`, the trainer connects to the environment server automatically. You just need the server to be running. There are three ways to run an OpenEnv environment server:

<hfoptions id="env_mode">

<hfoption id="space">

**Connect to a remote Hugging Face Space** *(simplest)*

Most example scripts default to a hosted Space (no setup needed):

```python
env = EchoEnv(base_url="https://openenv-echo-env.hf.space")
```

> [!WARNING]
> For training, **duplicate the Space to your own account** to avoid concurrency issues. The trainer opens N simultaneous WebSocket connections (one per generation), and shared Spaces may not support this. See [Server concurrency](#server-concurrency) for details.

</hfoption>

<hfoption id="docker">

**Docker container** *(recommended for production)*

```bash
docker run -d -p 8001:8000 --platform linux/amd64 registry.hf.space/openenv-echo-env:latest
```

Then connect:

```python
env = EchoEnv(base_url="http://0.0.0.0:8001")
```

We map port 8001 to 8000 to leave port 8000 available for a vLLM server.

You can also start the container programmatically:

```python
env = EchoEnv.from_docker_image("registry.hf.space/openenv-echo-env:latest")
```

> [!NOTE]
> You can find the Docker image for any Space on the Hub: open the Space page → **⋮ (three dots)** → **"Run locally."**
>
> ![open_env_launch_docker](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/open_env_launch_docker.png)

</hfoption>

<hfoption id="local">

**Local Python process** *(for development)*

```bash
hf download openenv/echo_env --repo-type=space --local-dir=echo_env
python -m uvicorn echo_env.src.envs.echo_env.server.app:app --host 0.0.0.0 --port 8001
```

Then connect:

```python
env = EchoEnv(base_url="http://0.0.0.0:8001")
```

For more details, see the [OpenEnv catalog](https://meta-pytorch.org/OpenEnv/environments.html).

</hfoption>

</hfoptions>

## Environments catalog

The best way to explore the current catalog of maintained environments is by visiting the official OpenEnv [catalog](https://huggingface.co/collections/openenv/environment-hub).

To create your own environment, check out the guide on [Building Your Own Environment with OpenEnv](https://meta-pytorch.org/OpenEnv/auto_getting_started/plot_03_building_environments.html). Environments are tightly integrated with the Hub, so you can push new environments for the community to reuse.

## Server concurrency

When using `environment_factory`, the trainer creates N environment instances (one per generation), each opening a WebSocket connection to the server. By default, OpenEnv servers allow only 1 concurrent session, which will cause failures during training.

To support parallel training, configure the server for concurrency:

1. In your environment file, declare concurrent session support:
```python
SUPPORTS_CONCURRENT_SESSIONS: bool = True
```

2. In your server app, set the concurrency limit:
```python
app = create_app(
    create_my_environment,
    MyAction,
    MyObservation,
    max_concurrent_envs=64,  # match or exceed generation_batch_size
)
```

> [!TIP]
> `max_concurrent_envs` should be ≥ `generation_batch_size` (which defaults to `per_device_train_batch_size × gradient_accumulation_steps`). For example, with `gradient_accumulation_steps=64` and batch size 1, you need at least 64 concurrent sessions.

## `environment_factory` vs `rollout_func`

[`GRPOTrainer`] supports two approaches for environment-based training:

- **`environment_factory`** (recommended): You define an environment class with tool methods, and the trainer handles generation, tool-call parsing, and the multi-turn loop automatically. This is the approach used throughout this guide.
- **`rollout_func`**: You write the entire generation and environment interaction loop yourself. This gives full control over how completions are produced, how tools are executed, and how rewards are computed.

Use `rollout_func` when `environment_factory` doesn't fit your use case. For example, **external agent servers** like [NeMo-Gym](nemo_gym), where an external server owns the generation loop and manages its own agent-environment interaction protocol.

### Migrating from `rollout_func` to `environment_factory`

If you have existing `rollout_func` code and want to migrate, here's the mapping:

| `rollout_func` pattern | `environment_factory` equivalent |
|------------------------|----------------------------------|
| Manual generation loop | Handled automatically by the trainer |
| `generate_rollout_completions()` | Not needed, trainer generates internally |
| `env.step(Action(...))` in rollout | Wrap in a tool method on the environment class |
| Reward via `kwargs["env_reward"]` | Reward via `environments` parameter |
| `env_mask` construction | Automatic, trainer builds `tool_mask` |
| Token concatenation | Automatic, trainer manages token sequences |

**Before** (`rollout_func`):

```python
def rollout_func(prompts, trainer):
    outputs = generate_rollout_completions(trainer, prompts)
    env_rewards = []
    for out in outputs:
        text = tokenizer.decode(out["completion_ids"], skip_special_tokens=True)
        result = client.step(EchoAction(message=text))
        env_rewards.append(result.reward)
    return {
        "prompt_ids": [out["prompt_ids"] for out in outputs],
        "completion_ids": [out["completion_ids"] for out in outputs],
        "logprobs": [out["logprobs"] for out in outputs],
        "env_reward": env_rewards,
    }

trainer = GRPOTrainer(..., rollout_func=rollout_func)
```

**After** (`environment_factory`):

```python
class EchoToolEnv:
    def __init__(self):
        self.env = EchoEnv(base_url=url)
        self.reward = 0.0

    def reset(self, **kwargs) -> str | None:
        self.reward = 0.0
        return None

    def echo(self, message: str) -> str:
        """Echo the message back.

        Args:
            message: The message to echo

        Returns:
            The echoed message.
        """
        result = self.env.step(EchoAction(message=message))
        self.reward = result.observation.reward
        return result.observation.echoed_message

def reward_func(environments, **kwargs):
    return [env.reward for env in environments]

trainer = GRPOTrainer(..., environment_factory=EchoToolEnv, reward_funcs=reward_func)
```
