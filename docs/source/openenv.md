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

## Advanced Example

Let's level this up a bit by training a model to interact with a more complex environment. We'll use the game [wordle](https://www.nytimes.com/games/wordle/index.html) from the `textarena` environment. 

> [!NOTE]
> Wordle is a word guessing game where the player has to guess a 5-letter word. They will receive feedback on each guess, and the goal is to guess the word in 6 guesses or less.

### The TextArena Environment

[TextAren](https://huggingface.co/papers/2504.11442) is an open-source collection of competitive text-based games designed to evaluate reasoning skills in LLMs using textual games like Wordle, Snake, Tic-Tac-Toe, and more. Research has shown that such games improve model performance on reasoning tasks.

![image of textarena](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/text_arena_evals.png)

We will use the `textarena` environment to train a model to play Wordle. The environment is a simple text based response environment that allows the model to interact with the game by making guesses and receive feedback on them.

### Wordle

Wordle is a useful game to train a model on because it requires the model to reason about the word and the feedback provided by the environment. Also, it is a purely language based game that requires no external tools or knowledge. Furthermore, we found that models from 1 billion parameters and up are able to improve on wordle and only require 8 tokens to generate a guess, which makes the game a good benchmark to experiment with Reinforcement Learning environments without significant compute requirements.

> [!NOTE] How does Wordle work?
> Wordle is a word guessing game where the player has to guess a 5-letter word. The player can make 6 guesses, and for each guess, the environment will provide feedback on the correctness of the guess. The player wins if they guess the word in 6 guesses or less. It challenges the model to generate words that are likely to be correct, and to learn from the feedback provided by the environment. 
> 
> For example, if the wordle environment returns the following feedback:
>
> ```
> G U E S S
> X G Y X X
> ```
> The model has guessed the word "GUESS" and the environment has provided feedback as the letters X, G, and Y. Referring to colors in the original game blank, green, and yellow. From this feedback, the model should learn that the word is "GUESS" is incorrect. The letter "E" is in the word, but in the wrong position. The letter "U" is correct and in the correct position.
 
In the TextArena environment, reward is only given when the model wins the game. The reward is 1.0 if the model wins, and 0.0 otherwise. This is not a ver efficient reward signal for the model, so we have added a number of custom reward functions to the script to help the model learn to play the game. The extensible nature of `reward_funcs` and `rollout_func` allows you to add any custom reward function you want to the script.

### Rollout Function

The rollout function is an implementation of the actions that the agent will take within the environment. In this case, the agent will make guesses and receive feedback on them. The rollout function will generate the guesses and step through the environment to get the feedback. Multi-step environments will typically collect feedback from the environment after each action and aggregate those rewards into a single signal. 

This rollout function will iterate over the guesses of a word in the environment and at each step:

- construct a prompt for the model to generate a guess. 
- generate a guess with vLLM.
- extract the guess from the completion.
- act in the environment with the `env.step` method.
- collect the feedback from the environment to be returned to the model.
- aggregate and collect rewards from the environment and feedback.

After completing the game, either by winning or exhausting all 6 turns, the rollout function will return the rewards of the game to the trainer. Below is the rollout function for the Wordle environment:

<!-- TODO: @burtenshaw Add rollout function code here -->
```python
def rollout_once(
    env: TextArenaEnv,
    tokenizer: AutoTokenizer,
    args: GRPOConfig,
    dataset_prompt: str,
) -> Dict[str, List]:
    result = env.reset()
    observation = result.observation

    prompt_ids: List[int] = []
    completion_ids: List[int] = []
    logprobs: List[float] = []
    raw_rewards: List[float] = []
    coverage_rewards: List[float] = []
    repetition_rewards: List[float] = []

    for _turn in range(MAX_TURNS):
        if result.done:
            break
        base_prompt = observation.prompt or dataset_prompt
        user_prompt = make_user_prompt(base_prompt, observation.messages)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        vllm_result = request_vllm_completion(prompt_text, args)

        prompt_ids.extend(vllm_result["prompt_ids"])
        completion_ids.extend(vllm_result["completion_ids"])
        logprobs.extend(vllm_result["logprobs"])

        completion_text = vllm_result.get("text") or tokenizer.decode(
            vllm_result["completion_ids"], skip_special_tokens=True
        )
        guess = extract_guess(completion_text)

        result = env.step(TextArenaAction(message=guess))
        raw_rewards.append(float(result.reward or 0.0))

        observation = result.observation
        feedback = extract_wordle_feedback(observation)
        if not feedback:
            repetition_reward = 0.0
            coverage_reward = 0.0
        else:
            repetition_reward = repeated_guess_penalty(guess, feedback)
            coverage_reward = compute_feedback_score(feedback)
        repetition_rewards.append(repetition_reward)
        coverage_rewards.append(coverage_reward)

    solved = bool(result.done and raw_rewards and raw_rewards[-1] > 0.0)
    correctness_reward = 1.0 if solved else 0.0
    coverage_reward = coverage_rewards[-1] if coverage_rewards else 0.0
    repetition_reward = sum(repetition_rewards) / max(1, len(repetition_rewards))

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "raw_rewards": raw_rewards,
        "correct_reward": correctness_reward,
        "coverage_reward": coverage_reward,
        "repetition_reward": repetition_reward,
    }
```

### Reward Functions

For the Wordle environment, we use three different reward functions to help the model learn to play the game:

- `correctness_reward`: This reward score directly from the environment. The function gives a reward of 1.0 if the model wins the game, and 0.0 otherwise.
- `repeated_guess_penalty`: This reward function penalizes the model for making the same guess twice.
- `compute_feedback_score`: This reward function computes the score of the feedback from the environment. For example, if the model gets a `G` in the correct position, it should be rewarded. If the model gets a `Y` in the wrong position, it should be penalized.

Below is the reward function for the Wordle environment:

<!-- TODO: @burtenshaw Add reward function code here -->
```python
```

### Training the Model

To train the model, we use the `GRPOTrainer` with the `rollout_func` and `reward_funcs` arguments as the previous examples. Below is the training script for the Wordle environment:

<!-- TODO: @burtenshaw Add training script code here -->
```python
```

### Running the Example

The example requires two GPUs:

```bash
# Terminal 1: Start vLLM inference server
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000

# Terminal 2: Run GRPO training with OpenEnv
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/wordle.py
```

### Results

<!-- TODO: @burtenshaw Add results here -->
```
```