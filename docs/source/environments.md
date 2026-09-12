# Environments

This module contains environments that plug into the [`GRPOTrainer`] through the `environment_factory` argument. An environment gives the model a set of **tools** to call during generation and drives the per-rollout lifecycle: `reset` is called at the start of each rollout (returning an instruction appended to the prompt), the environment's public methods are exposed as tools, and reward is provided by the trainer's `reward_funcs` (or, optionally, by the environment itself through a `get_reward` method).

<Tip warning={true}>

`environment_factory` is an experimental feature of the trainers. Its API may change.

</Tip>

## SandboxEnvironment

[[autodoc]] environments.SandboxEnvironment
