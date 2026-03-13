# OpenEnv Integration for Training LLMs with Environments

[OpenEnv](https://github.com/meta-pytorch/OpenEnv) is an open-source framework for defining, deploying, and interacting with environments in reinforcement learning (RL) and agentic workflows. It offers [Gymnasium-style APIs](https://gymnasium.farama.org) (e.g., `reset()` and `step()`) to interface with environments in a standard manner, and supports running these environments as backend servers (for example, via HTTP or containerised execution). You can find a collection of ready-to-use OpenEnv environments on the [Hugging Face Hub](https://huggingface.co/collections/openenv/openenv-environment-hub).

In this guide, we'll focus on **how to integrate OpenEnv with TRL**, but feel free to explore the links above to dive deeper into OpenEnv itself.

> [!NOTE]
> You can explore ready-to-use example [scripts](example_overview#scripts) and [notebooks](example_overview#notebooks) in the Examples Overview.

> [!NOTE]
> Explore the [OpenEnv docs](https://meta-pytorch.org/OpenEnv/) for more details.

## Installation

To use OpenEnv with TRL, install the environment package. OpenEnv environments can be hosted as Hugging Face Spaces, which are also pip-installable Git repositories. When you run:

```bash
pip install "openenv-echo-env @ git+https://huggingface.co/spaces/openenv/echo_env"
```

this clones the Space's Git repository (which contains a `pyproject.toml` or `setup.py`) and installs it as a Python package. This gives you the **environment client** (e.g., `EchoEnv`) that communicates with the remote environment server via WebSocket, along with the action/observation models and all required dependencies (including `openenv-core`).

You have two options:

**Option A - Install from HF Space (recommended):**

```bash
# Echo environment
pip install "openenv-echo-env @ git+https://huggingface.co/spaces/openenv/echo_env"

# Wordle (TextArena) environment
pip install "openenv-textarena @ git+https://huggingface.co/spaces/openenv/wordle"

# Catch (OpenSpiel) environment
pip install "openenv-openspiel-env @ git+https://huggingface.co/spaces/openenv/openspiel_env"
```

> [!TIP]
> You can find the install command for any environment on its HF Space page. Click the **⋮ (three dots)** menu and select **"Use this Space"** to see the install instructions.

> [!TIP]
> You can also install the core package from PyPI with `pip install "openenv-core[core]>=0.2.1"`, but note that environment-specific dependencies may need to be installed separately.

**Option B - Clone OpenEnv repo (for development):**

```bash
git clone https://github.com/meta-pytorch/OpenEnv.git
cd OpenEnv/envs/echo_env
pip install -e .
```

> [!NOTE]
> Each environment script in TRL includes inline dependency metadata (PEP 723) so you can also run them directly with [uv](https://docs.astral.sh/uv/):
> ```bash
> uv run examples/scripts/openenv/echo.py
> ```
> This automatically installs the required environment package in an isolated environment.

## Using `environment_factory` with OpenEnv environments

TRL's [`GRPOTrainer`] supports interactive environment training through the `environment_factory` argument. When provided, the trainer automatically handles the multi-turn tool-calling loop: it generates completions, parses tool calls, executes them against the environment, and feeds the results back to the model, all without custom rollout code.

### How it works

1. You define an **environment class** with a `reset()` method and one or more **tool methods**.
2. You pass the class (not an instance) as `environment_factory` to [`GRPOTrainer`].
3. The trainer creates N instances of your environment (one per generation), discovers tool methods via introspection, and exposes them to the model as function-calling tools.
4. During training, the trainer runs a multi-turn loop: generate → parse tool calls → execute tools → append results → generate again, until the model stops calling tools or `max_completion_length` is reached.

### Environment class requirements

Your environment class must follow these rules:

- **`__init__(self)`**: Takes no arguments. Initialize any state here.
- **`reset(self, **kwargs)`**: Called at the start of each episode. Receives all dataset columns as keyword arguments. Return a string observation (or `None` for no initial observation).
- **Tool methods**: Any public method (not starting with `_`) other than `reset` is automatically exposed as a tool. Each tool method must have a docstring with argument descriptions, since the trainer uses these to generate the tool schema for the model.
- **Reward state**: Store reward information as instance attributes (e.g., `self.reward`) and access them in your reward function via the `environments` parameter.

```python
class MyEnv:
    def __init__(self):
        self.reward = 0.0

    def reset(self, **kwargs) -> str | None:
        self.reward = 0.0
        # kwargs contains all dataset columns for this sample
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
        # Interact with your environment
        self.reward = 1.0
        return "Tool result"
```

> [!IMPORTANT]
> Tools must be **individual methods** with descriptive names and typed arguments (e.g., `guess(word: str)`, `move(direction: str)`). Do NOT use generic methods like `step(action)`, since the model needs meaningful tool names and argument descriptions to learn tool calling.

### Reward functions

Reward functions receive the `environments` parameter, a list of environment instances, so you can access any state stored during the episode:

```python
def reward_func(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]
```

### vLLM Modes

TRL supports two vLLM execution modes for generation:

- **`colocate` mode** (default): vLLM runs in the same process as training. Requires 1 GPU.
- **`server` mode**: vLLM runs as a separate server process. Requires at least 2 GPUs (one for vLLM server, one for training), but is highly scalable.

Configure the mode via `GRPOConfig`:

```python
# Colocate mode (1 GPU)
args = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",
    # ... other args
)

# Server mode (2+ GPUs, scalable)
args = GRPOConfig(
    use_vllm=True,
    vllm_mode="server",
    vllm_server_base_url="http://localhost:8000",
    # ... other args
)

# Example: Start vLLM server with multiple GPUs for tensor parallelism
# CUDA_VISIBLE_DEVICES=0,1,2,3 trl vllm-serve --model Qwen/Qwen3-1.7B --tensor-parallel-size 4
```

## Running the Environments

You can run OpenEnv environments in three different ways:

- We can load the environment from the Hugging Face Hub and execute it as a Docker container.
- We can connect to a hosted environment running on the Hugging Face Hub.
- We can launch the environment directly using Uvicorn in Python.

<hfoptions id="env_mode">

<hfoption id="docker">

**Load from Hugging Face Hub** *(recommended)*

We can use the [`from_docker_image`](https://meta-pytorch.org/OpenEnv/core.html) method to load the environment from a Docker image. This method will automatically start a Docker container for the environment on your local machine.

```python
env = EchoEnv.from_docker_image("registry.hf.space/openenv-echo-env:latest")
```

If you want to launch the environment manually, you can use the following command to pull and run the Docker container:

```bash
docker run -d -p 8001:8000 --platform linux/amd64  registry.hf.space/openenv-echo-env:latest
```

And then you can connect to the environment using the following code:

```python
env = EchoEnv(base_url="http://0.0.0.0:8001")
```

Here, we map the ports from 8001 to 8000 to make space for a vLLM server, but you will need to manage the ports for your local machine.

> [!NOTE]
> You can find the Docker container for any space on the hub.
>
> * Open the space page on the hub.
> * Click the **⋮ (three dots)** menu.
> * Select **"Run locally."**
> * Copy and execute the provided command in your terminal.
>
> ![open_env_launch_docker](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/open_env_launch_docker.png)

> [!NOTE]
> You can also use the **Docker option** with `from_docker_image` by providing the image name.
> For more details, refer to the official [OpenEnv documentation](https://meta-pytorch.org/OpenEnv/core.html).

</hfoption>
<hfoption id="space">

**Connect to a remote Hugging Face Space**

You can connect to a hosted environment running on the Hugging Face Hub by passing the URL of the space to the `base_url` parameter of the environment class.

```python
env = EchoEnv(base_url="https://openenv-echo-env.hf.space")
```

> [!NOTE]
> You can find the connection URL of any space on the hub.
>
> * Open the space page on the hub.
> * Click the **⋮ (three dots)** menu.
> * Select **"Embed this Space."**
> * Copy the connection URL.

> [!WARNING]
> **Currently**, it is recommended to **duplicate the Space to your own account** to avoid potential concurrency issues.

</hfoption>

<hfoption id="local">

**Local Python process**

You can start the server manually as a local Python process. For more details about the available environments, refer to the [OpenEnv catalog](https://meta-pytorch.org/OpenEnv/environments.html).

```bash
hf download openenv/echo_env --repo-type=space --local-dir=echo_env
python -m uvicorn echo_env.src.envs.echo_env.server.app:app --host 0.0.0.0 --port 8001
```

And then you can connect to the environment using the following code:

```python
env = EchoEnv(base_url="http://0.0.0.0:8001")
```

</hfoption>

</hfoptions>

## Environments Catalog

Environment development is active and evolving.
The best way to explore the **current catalog of maintained environments** is by visiting the official OpenEnv [catalog](https://huggingface.co/collections/openenv/environment-hub).

Custom environments are also supported. To learn how to create your own, check out the guide on [Building Your Own Environment with OpenEnv](https://meta-pytorch.org/OpenEnv/auto_getting_started/plot_03_building_environments.html).

Environments are tightly integrated with the Hub, allowing you to **push new environments directly** so the community can easily pull, reuse, and adapt them for their own use cases.

## A simple example

> [!NOTE]
> You can explore more ready-to-use example scripts in the [`examples/scripts/openenv/`](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/) directory.

The [echo.py](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/echo.py) script demonstrates a minimal, end-to-end integration between TRL and OpenEnv. In this example, the [Echo environment](https://meta-pytorch.org/OpenEnv/environments/echo.html) rewards completions based on their text length, encouraging the model to generate longer outputs:

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

    def reset(self, **kwargs) -> None | str:
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

def reward_func(completions, environments, **kwargs):
    return [environment.reward for environment in environments]

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

That's it! Let's unpack how the main pieces fit together:

1. **Environment class:** `EchoToolEnv` wraps the OpenEnv client and exposes `echo()` as a tool method with a descriptive docstring.
2. **`reset()`:** Called at the start of each episode to reset state. Returns `None` (no initial observation needed for Echo).
3. **Tool discovery:** The trainer automatically discovers `echo()` via introspection and exposes it to the model as a function-calling tool.
4. **Reward function:** Reads `environment.reward` from each environment instance, no need to pass rewards through `kwargs`.
5. **`environment_factory`:** Pass the class itself (not an instance). The trainer creates one instance per generation.

### Running the Example

You can run the example in either colocate mode (1 GPU) or server mode (2 GPUs):

<hfoptions id="vllm_mode">

<hfoption id="colocate">

**Colocate mode (1 GPU, recommended)**

```bash
python examples/scripts/openenv/echo.py --vllm-mode colocate
```

This runs vLLM in the same process as training, requiring only a single GPU.

</hfoption>

<hfoption id="server">

**Server mode (2+ GPUs, scalable)**

```bash
# Terminal 1: Start vLLM inference server
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000

# Terminal 2: Run GRPO training with OpenEnv
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/echo.py --vllm-mode server --vllm-server-url http://localhost:8000
```

This runs vLLM as a separate server process, useful when you want to:
- Share the inference server across multiple training jobs
- Use multiple GPUs for the vLLM server (via `--tensor-parallel-size`)
- Scale up training to many GPUs while sharing a single inference endpoint

</hfoption>

</hfoptions>

Below is the reward curve from training:

<iframe src="https://trl-lib-trackio.hf.space?project=openenv&metrics=train/rewards/reward_from_env/mean&runs=qgallouedec-1761202871&sidebar=hidden&navbar=hidden" style="width:600px; height:500px; border:0;"></iframe>

## Advanced Example

Let's level this up by training a model to play [Wordle](https://www.nytimes.com/games/wordle/index.html) using the [`TextArena`](https://meta-pytorch.org/OpenEnv/environments/textarena.html) environment.

> [!NOTE]
> You can explore the notebook version of this example [here](https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb).

### The TextArena Environment

[TextArena](https://huggingface.co/papers/2504.11442) is an open-source collection of competitive text-based games designed to evaluate reasoning skills in LLMs using textual games like Wordle, Snake, Tic-Tac-Toe, and more. Research has shown that such games improve model performance on reasoning tasks.

![image of TextArena](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/text_arena_evals.png)

We will use the `TextArena` environment to train a model to play Wordle.

### Wordle

Wordle is a useful game to train a model on because it requires the model to reason about the word and the feedback provided by the environment. Also, it is a purely language based game that requires no external tools or knowledge. Furthermore, we found that models from 1 billion parameters and up are able to improve on wordle and only require 8 tokens to generate a guess, which makes the game a good benchmark to experiment with Reinforcement Learning environments without significant compute requirements.

> [!NOTE] How does Wordle work?
> Wordle is a word guessing game where the player has to guess a 5-letter word. The player can make 6 guesses, and for each guess, the environment will provide feedback on the correctness of the guess. The player wins if they guess the word in 6 guesses or fewer. It challenges the model to generate words that are likely to be correct, and to learn from the feedback provided by the environment.
>
> For example, if the wordle environment returns the following feedback:
>
> ```
> G U E S S
> X G Y X X
> ```
> The model has guessed the word "GUESS" and the environment has provided feedback as the letters X, G, and Y. Referring to colors in the original game as blank, green, and yellow. From this feedback, the model should learn that the word "GUESS" is incorrect. The letter "E" is in the word, but in the wrong position. The letter "U" is correct and in the correct position.

### Environment Class

The `WordleEnv` class wraps the TextArena OpenEnv client and exposes `guess()` as the tool the model uses to play:

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
            self.reward = 0.0
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
- **`guess()`** is the only tool, and the model calls it each turn with a 5-letter word.
- **Cumulative feedback slicing**: The TextArena environment returns the full game history each turn. We slice out only the new feedback to avoid repeating context.
- **`self.done`**: Tracks game completion. If the model tries to guess after the game ends, it raises an error (the trainer catches this gracefully).

### Reward Function

The reward function simply reads the environment's reward after the episode:

```python
def reward(environments, **kwargs) -> list[float]:
    return [environment.reward for environment in environments]
```

The environment returns `1.0` if the model wins and `0.0` otherwise.

### Training the Model

```python
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

prompt = """You are an expert Wordle solver with deep knowledge of English vocabulary...
Use the tool `guess` to make a guess."""

dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": prompt}]] * 1000})

trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    reward_funcs=reward,
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

### Running the Advanced Example

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

The resulting model improves its performance on the game, both by reducing the number of repetitions and by increasing the number of correct guesses. However, the Qwen3-1.7B model we trained is not able to consistently win the game. The following reward curve shows the coverage of the model's guesses and the coverage of correct Y and G letters.

<iframe src="https://burtenshaw-wordle-grpo.hf.space?project=group-Qwen-Qwen3-17B&metrics=reward&runs=run-2025-10-26_09-39-49,run-2025-10-26_08-04-49&sidebar=hidden&navbar=hidden" style="width:1600px; height:500px; border:0;"></iframe>

We experimented with larger models like `gpt-oss-20b` and found that the model was able to consistently win the game. However, this requires a lot of compute to train the model. Why not try this out yourself?

## Multi-Environment Training

You can train a single model across multiple environments simultaneously. This is useful when you want a model to learn different skills in parallel, for example, playing Wordle (language reasoning) and Catch (spatial reasoning) in the same training run.

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
        self._done = False

    def reset(self, **kwargs) -> str | None:
        self.active = kwargs.get("env", "wordle")
        self.reward = 0.0
        self._done = False

        if self.active == "wordle":
            if self._wordle_client is not None:
                try:
                    self._wordle_client.close()
                except Exception:
                    pass
            self._wordle_client = TextArenaEnv(base_url=WORDLE_URL)
            result = self._wordle_client.reset()
            self._last_full_feedback = result.observation.messages[0].content
            self.reward = -1.0
            return self._last_full_feedback
        elif self.active == "catch":
            if self._catch_client is not None:
                try:
                    self._catch_client.close()
                except Exception:
                    pass
            self._catch_client = OpenSpielEnv(base_url=CATCH_URL)
            result = self._catch_client.reset()
            self._done = result.observation.done
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
        elif env._done:
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

## Server Concurrency

When using `environment_factory`, the trainer creates N environment instances (one per generation), each opening a WebSocket connection to the server. By default, OpenEnv servers allow only 1 concurrent session.

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

## Migrating from `rollout_func`

If you have existing code using `rollout_func`, here's how to migrate to `environment_factory`:

| `rollout_func` pattern | `environment_factory` equivalent |
|------------------------|----------------------------------|
| Manual generation loop | Handled automatically by the trainer |
| `generate_rollout_completions()` | Not needed, trainer generates internally |
| `env.step(Action(...))` in rollout | Wrap in a tool method on the environment class |
| Reward via `kwargs["env_reward"]` | Reward via `environments` parameter |
| `env_mask` construction | Automatic, trainer builds `tool_mask` |
| Token concatenation | Automatic, trainer manages token sequences |

### Before (rollout_func)

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

### After (environment_factory)

```python
class EchoToolEnv:
    def __init__(self):
        self.env = EchoEnv(base_url=url)
        self.reward = 0.0

    def reset(self, **kwargs) -> None | str:
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

> [!NOTE]
> `rollout_func` remains available for advanced use cases that are incompatible with `environment_factory`, such as external agent servers (NeMo-Gym) or environments requiring multimodal tool responses (screenshots, images).
