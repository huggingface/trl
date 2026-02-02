# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example of using SDPOTrainer for training with self-distillation.

This example demonstrates how to use SDPOTrainer to train a model using
self-distillation from high-reward trajectories.
"""

from datasets import load_dataset
from trl.experimental.sdpo import SDPOTrainer, SDPOConfig


# Define a simple reward function
def simple_reward_func(prompts, completions, **kwargs):
    """Simple reward function that rewards longer completions."""
    rewards = []
    for completion in completions:
        # Reward based on completion length (example only)
        reward = len(completion) / 100.0
        rewards.append(reward)
    return rewards


def main():
    # Load a dataset
    # For this example, we'll use a small subset of a dataset
    dataset = load_dataset("trl-lib/DeepMath-103K", split="train[:100]")

    # Configure SDPO training
    config = SDPOConfig(
        # General training parameters
        output_dir="./sdpo_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        bf16=True,  # Use bf16 if supported
        report_to="none",

        # Generation parameters
        max_completion_length=512,
        num_generations=8,
        temperature=1.0,

        # SDPO-specific parameters
        distillation_alpha=1.0,  # Reverse KL (recommended)
        distillation_topk=20,
        full_logit_distillation=False,
        distillation_is_clip=2.0,
        distillation_add_tail=False,
        dont_reprompt_on_self_success=True,
        ema_update_rate=0.01,
        max_reprompt_len=10240,
        distillation_weight=1.0,
        use_successful_as_teacher=True,

        # GRPO parameters (inherited)
        beta=0.0,  # No reference model
        loss_type="dapo",
    )

    # Initialize SDPO Trainer
    trainer = SDPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",  # Use a small model for testing
        reward_funcs=simple_reward_func,
        args=config,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model("./sdpo_output/final_model")


if __name__ == "__main__":
    main()