# Maintained scripts

This folder shows multiple ways to use the objects from TRL such as `SFTTrainer`, `RewardTrainer`, `DDPOTrainer` and `PPOTrainer` in different scenarios. 

- `sft_trainer.py`: This script shows how to use the SFTTrainer to fine tune a model or adapters into a target dataset.
- `reward_trainer.py`: This script shows how to use the RewardTrainer to train a reward model on your own dataset.
- `sentiment_tuning.py`: This script shows how to use the PPOTrainer to fine-tune a sentiment analysis model using IMDB dataset
- `multi_adapter_rl.py`: This script shows how to use the PPOTrainer to train a single base model with multiple adapters. This scripts requires you to run the example script with the reward model training beforehand.
- `stable_diffusion_tuning_example.py`: This script shows to use DDPOTrainer to fine-tune a stable diffusion model using reinforcement learning.
