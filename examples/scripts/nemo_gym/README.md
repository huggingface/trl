# Post-training with NeMo Gym and TRL

This integration supports training language models in NeMo-Gym environments using TRL GRPO. Both single step and multi step tasks are supported, including multi-environment training. NeMo-Gym orchestrates rollouts, returning token ids and logprobs to TRL through the rollout function for training. Currently this integration is only supported through TRL's vllm server mode. 

Check out docs/source/nemo_gym_integration.md for a full integration guide!