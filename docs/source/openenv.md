# OpenEnv Integration for Training LLMs with Environments

## Overview

[OpenEnv](https://github.com/meta-pytorch/OpenEnv) is an open-source framework from Meta's PyTorch team for defining, deploying, and interacting with environments in reinforcement learning (RL) and agentic workflows. It offers [Gymnasium-style APIs](https://gymnasium.farama.org) (e.g., `reset()` and `step()`) to interface with environments in a standard manner, and supports running these environments as backend servers (for example, via HTTP or containerised execution). You can find a collection of ready-to-use OpenEnv environments on the [Hugging Face Hub](https://huggingface.co/collections/openenv/environment-hub).

In this guide, we’ll focus on **how to integrate OpenEnv with TRL**, but feel free to explore the links above to dive deeper into OpenEnv itself.

## Installation

To use OpenEnv with TRL, install the framework:

```bash
pip install git+https://github.com/meta-pytorch/OpenEnv.git
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

## Running the Environments

You can run OpenEnv environments in three different ways: 

- We can load the environment from the Hugging Face Hub and execute it as a Docker container.
- We can launch the environment directly using Uvicorn in Python, which you need on Google Colab.
- We can connect to a hosted environment running on the Hugging Face Hub.

<hfoptions id="env_mode">

<hfoption id="docker">

**Load from Hugging Face Hub** *(recommended)*

We can use the `from_hub` method to load the environment from the hub. This method will automatically start a Docker container for the environment on your local machine. `openenv/echo-env` is the repo_id of the space on the hub.

```python
env = EchoEnv.from_hub("openenv/echo-env")
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
> * Select **“Run locally.”**
> * Copy and execute the provided command in your terminal.
>
> ![open_env_launch_docker](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/open_env_launch_docker.png)

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
> * Select **“Embed this Space.”**
> * Copy the connection URL.

</hfoption>

<hfoption id="local">

**Local Python process**

You can start the server manually as a local Python process. For more details about the available environments, refer to the [OpenEnv repository](https://github.com/meta-pytorch/OpenEnv/tree/main/src/envs).
   
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

## A simple example

The [echo.py](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/echo.py) script demonstrates a minimal, end-to-end integration between TRL and OpenEnv. In this example, the Echo environment rewards completions based on their text length, encouraging the model to generate longer outputs. This pattern can be extended to any custom environment that provides structured feedback or task-based rewards:

```python
from envs.echo_env import EchoEnv, EchoAction
from trl import GRPOConfig, GRPOTrainer

# Create HTTP client for Echo Environment
client = EchoEnv.from_hub("openenv/echo-env")

"""
Alternatively, you can start the environment manually with Docker and connect to it:

# Step 1: Start the Echo environment
docker run -d -p 8001:8001 registry.hf.space/openenv-echo-env:latest

# Step 2: Connect the client to the running container
client = EchoEnv(base_url="http://0.0.0.0:8001")
"""

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

Alternatively, you can manually start the Echo environment in a Docker container before running the training:

```bash
# Launch the Echo environment
docker run -d -p 8001:8001 registry.hf.space/openenv-echo-env:latest
```

Then, initialize the client using:

`client = EchoEnv(base_url="http://0.0.0.0:8001")` 

instead of:

`client = EchoEnv.from_docker_image("echo-env:latest")`.

Below is the reward curve from training:

<iframe src="https://trl-lib-trackio.hf.space?project=openenv&metrics=train/rewards/reward_from_env/mean&runs=qgallouedec-1761202871&sidebar=hidden&navbar=hidden" style="width:600px; height:500px; border:0;"></iframe>

To learn more about how to create custom environments, see the [OpenEnv documentation](https://github.com/meta-pytorch/OpenEnv/blob/main/src/envs/README.md).

## Advanced Example

Let's level this up a bit by training a model to interact with a more complex environment. We'll use the game word guessing game [wordle](https://www.nytimes.com/games/wordle/index.html) from the `textarena` environment. 

### The TextArena Environment

[TextArena](https://huggingface.co/papers/2504.11442) is an open-source collection of competitive text-based games designed to evaluate reasoning skills in LLMs using textual games like Wordle, Snake, Tic-Tac-Toe, and more. Research has shown that such games improve model performance on reasoning tasks.

![image of textarena](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/text_arena_evals.png)

We will use the `textarena` environment to train a model to play Wordle. The environment is a simple text based response environment that allows the model to interact with the game by making guesses and receive feedback on them.

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
 
In the TextArena environment, a reward is only given when the model wins the game. The reward is 1.0 if the model wins, and 0.0 otherwise. This is not a very efficient reward signal for the model, so we have added a number of custom reward functions to the script to help the model learn to play the game. The extensible nature of `reward_funcs` and `rollout_func` allows you to add any custom reward function you want to the script.  

### Rollout Function

The rollout function runs one full Wordle episode, prompting the model for a guess each turn and capturing both environment rewards and auxiliary signals such as letter coverage and repetition penalties.

```python
def rollout_once(
    env: TextArenaEnv,
    tokenizer: AutoTokenizer,
    args: GRPOConfig,
    dataset_prompt: str,
    cli_args: argparse.Namespace,
    system_prompt: str,
) -> dict[str, list]:
    result = env.reset()
    observation = result.observation

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    raw_rewards: list[float] = []
    green_scores: list[float] = []
    yellow_scores: list[float] = []
    repetition_scores: list[float] = []
    correct_scores: list[float] = []
    guess_counts: dict[str, int] = {}

    for _turn in range(cli_args.max_turns):
        # when the game is over the environment will return a done=True
        if result.done:
            break

        # set up the prompt for the model
        base_prompt = observation.prompt or dataset_prompt
        user_prompt = make_user_prompt(base_prompt, observation.messages)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        # generate the completion from the model using vLLM
        vllm_result = request_vllm_completion(
            prompt_text,
            args,
            endpoint=cli_args.vllm_endpoint,
            timeout=cli_args.request_timeout,
            fallback=cli_args,
        )
        prompt_ids.extend(vllm_result["prompt_ids"])
        completion_ids.extend(vllm_result["completion_ids"])
        logprobs.extend(vllm_result["logprobs"])
        completion_text = vllm_result.get("text") or tokenizer.decode(
            vllm_result["completion_ids"], skip_special_tokens=True
        )
        # extract the guess from the completion
        guess = extract_guess(completion_text)

        # step the environment with the guess
        result = env.step(TextArenaAction(message=guess))
        raw_rewards.append(float(result.reward or 0.0))
        observation = result.observation
        correct_score = float(result.reward or 0.0)
        feedback = extract_wordle_feedback(observation)

        # Update guess counts
        previous_occurrences = guess_counts[guess]
        repetition_score = scale_repetition_score(previous_occurrences, len(guess_counts))
        guess_counts[guess] += 1

        # calculate custom reward signals from the feedback
        if not feedback:
            green_score = 0.0
            yellow_score = 0.0
        else:
            green_count, yellow_count = extract_feedback_counts(feedback)
            green_score = green_count / 5.0
            yellow_score = yellow_count / 5.0

        repetition_scores.append(repetition_score)
        green_scores.append(green_score)
        yellow_scores.append(yellow_score)
        correct_scores.append(correct_score)

    correct_reward_value = correct_scores[-1] if correct_scores else (raw_rewards[-1] if raw_rewards else 0.0)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "raw_rewards": raw_rewards,
        "correct_reward": correct_reward_value,
        "green_reward": green_scores[-1] if green_scores else 0.0,
        "yellow_reward": yellow_scores[-1] if yellow_scores else 0.0,
        "repetition_reward": repetition_scores[-1] if repetition_scores else 0.0,
    }
```

The environment has a reward signal based on the completion of the game. We found that most models struggle to ever win the game, so we have added a number of custom reward functions to the script to help the model learn to play the game more iteratively. At first, the model will learn to cover new letters and avoid repeating guesses. As it improves, it will learn to win the game.

### Reward Functions

We log four reward streams that encourage the model to solve the puzzle, cover new letters, and avoid repeating guesses:

- `reward_correct`: final win/loss signal from the environment.
- `reward_greens`: density of green letters in the last feedback.
- `reward_yellows`: density of yellow letters in the last feedback.
- `reward_repetition`: penalty for guessing the same token multiple times.

```python
def reward_correct(completions: List[str], **kwargs: Optional[Dict]) -> List[float]:
    rewards = kwargs.get("correct_reward") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)


def reward_greens(completions: List[str], **kwargs: Optional[Dict]) -> List[float]:
    rewards = kwargs.get("green_reward") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)


def reward_yellows(completions: List[str], **kwargs: Optional[Dict]) -> List[float]:
    rewards = kwargs.get("yellow_reward") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)


def reward_repetition(completions: List[str], **kwargs: Optional[Dict]) -> List[float]:
    rewards = kwargs.get("repetition_reward") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)
```

### Training the Model

The training script wires the custom rollout and rewards into `GRPOTrainer`. The CLI exposes the configuration used during development as defaults, so you can override endpoints or hyperparameters at launch time.

```python
parser = argparse.ArgumentParser()
# ... add CLI arguments with sensible defaults ...
cli_args = parser.parse_args()

trainer = GRPOTrainer(
    model=cli_args.model_id,
    processing_class=tokenizer,
    reward_funcs=[
        reward_correct,
        reward_greens,
        reward_yellows,
        reward_repetition,
    ],
    train_dataset=dataset,
    args=grpo_config,
    rollout_func=lambda prompts, args, processing_class: rollout_func(
        env=env,
        tokenizer=tokenizer,
        prompts=prompts,
        args=args,
        cli_args=cli_args,
        system_prompt=system_prompt,
    ),
)
trainer.train()
```

### Running the Advanced Example

The example requires two GPUs:

```bash
# Terminal 1: Start vLLM inference server
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000

# Terminal 2: Run GRPO training with OpenEnv
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/wordle.py
```

Again, you can manually start the TextArena environment in a Docker container before running the training.
In this case, initialize the client with
`client = TextArenaEnv(base_url="http://0.0.0.0:8001")`
instead of
`client = TextArenaEnv.from_docker_image("registry.hf.space/burtenshaw-textarena:latest")`:

```bash
# Launch the TextArena environment
docker run -d -p 8001:8001 registry.hf.space/burtenshaw-textarena:latest
```

### Results

The resulting model improves its performance on the game, both by reducing the number of repetitions and by increasing the number of correct guesses. However, the Qwen3-1.7B model we trained is not able to consistently win the game. The following reward curve shows the coverage of the model's guesses and the coverage of correct Y and G letters.

<iframe src="https://burtenshaw-wordle-grpo.hf.space?project=group-Qwen-Qwen3-17B&metrics=reward&runs=run-2025-10-26_09-39-49,run-2025-10-26_08-04-49&sidebar=hidden&navbar=hidden" style="width:1600px; height:500px; border:0;"></iframe>

We experimented with larger models like `gpt-oss-20b` and found that the model was able to consistently win the game. However, this requires a lot of compute to train the model. Why not try this out yourself?
