# Training Agents with Environments in TRL

Environments in TRL provide a powerful way to customize the rollout process for reinforcement learning (RL) with language models. This enables advanced agent training workflows, such as allowing your model to interact with external tools or APIs during RL training.

> **Note:**  
> - Environments currently require vLLM as the backend.
> - The only supported RL training method at this time is GRPO (Group Relative Policy Optimization).

## Installation

To use Environments and agent features, install TRL with the `[agents]` extra:

```bash
pip install trl[agents]
```

## What Are Environments?

An **Environment** defines how rollouts (model generations) are performed during RL training. By customizing the rollout, you can enable your model to interact with external systems, execute code, or follow any custom logic during generation.

TRL provides built-in environments, such as `CodeAgentEnvironment`, and you can also implement your own by subclassing `Environment`.

---

## Using the Built-in CodeAgentEnvironment

The `CodeAgentEnvironment` allows your agent to write and execute code during training, using a code interpreter like [E2B sandboxes](https://e2b.dev). This is useful for tasks where the model needs to solve problems by running code.

You can also use the `run_agent()` method to test and observe your agent's behavior outside of training.

### Example: Running the CodeAgent Environment

First, start a vLLM server with your model:

```bash
trl vllm-serve --model "Qwen/Qwen2.5-0.5B-Instruct"
```

Then, use the environment and client in Python:

```python
from trl import CodeAgentEnvironment, E2BExecuter, VLLMClientGenerationConfig, VLLMClient

client = VLLMClient()
gen_config = VLLMClientGenerationConfig(
    n=8,
    repetition_penalty=1.0,
    temperature=0.8,
    top_p=0.9,
    top_k=10,
    min_p=0.0,
    max_tokens=256,
)
tokenizer = ...  # Load your tokenizer here
code_executer = E2BExecuter(api_key="YOUR_E2B_TOKEN")
my_env = CodeAgentEnvironment(
    code_executer=code_executer,
    tokenizer=tokenizer,
    parsing_string="<code>",
    stop_string="</code>",
)
prompts = [...]  # List of prompt strings
responses = my_env.run_agent(
    vllm_client=client,
    generation_config=gen_config,
    prompts=prompts
)
```

---

## Writing a Custom Environment

To create your own environment, subclass `Environment` and implement a `generate()` method. This method receives the vLLM client, a generation config, and a list of prompts, and must return a list of tokenized completions (as lists of token IDs).

The `generate()` method is called by the GRPO training loop to perform rollouts.

**Example:**

```python
from trl import Environment, VLLMClientGenerationConfig

class MyCustomEnv(Environment):
    def __init__(self, ...):
        # Your custom initialization
        pass

    def generate(self, vllm_client, generation_config: VLLMClientGenerationConfig, prompts: list[str]) -> list:
        # Your custom rollout logic
        completion_ids = vllm_client.generate(
            prompts=prompts,
            **vars(generation_config)
        )
        return completion_ids
```

You can add any additional methods or logic needed for your task.

---

## Training with a Custom Environment

To use your environment for RL training, simply pass it to the `GRPOTrainer`:

```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    output_dir="qwen_0.5B-agent",
    use_vllm=True,
    # ... other config options ...
)

trainer = GRPOTrainer(
    model=...,                # Your model or model name
    reward_funcs=...,         # Your reward function(s)
    args=training_args,
    environment=my_env,       # Your custom or built-in environment
    # ... other trainer args ...
)
```

Then, start training as usual:

```python
trainer.train()
```

---

## Summary

- **Environments** let you fully customize how rollouts are performed during RL training with TRL and vLLM.
- Use built-in environments like `CodeAgentEnvironment` for code-executing agents, or implement your own by subclassing `Environment`.
- The `generate` method is the key entry point for custom rollout logic.
- Pass your environment to `GRPOTrainer` to use it during RL training.

For more advanced examples, see the [notebooks](../notebooks/) or the TRL documentation.
