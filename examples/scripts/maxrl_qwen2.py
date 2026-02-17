#!/usr/bin/env python
"""Simple script to run MaxRL training on a small GPT model"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import MaxRLConfig, MaxRLTrainer
from trl.rewards import accuracy_reward

# Load a small instruct model (supports chat templates)
print("Loading Qwen/Qwen2.5-0.5B-Instruct model...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Load the dataset
print("Loading DeepMath-103K dataset...")
dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

# Configure the training
training_args = MaxRLConfig(
    output_dir="maxrl_gpt2_demo",
    num_generations=2,
    learning_rate=1e-6,
    max_completion_length=128,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    max_steps=10,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
)

# Initialize the trainer with the loaded model
print("Initializing MaxRL trainer...")
trainer = MaxRLTrainer(
    model=model,
    reward_funcs=accuracy_reward,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
print("Starting MaxRL training with GPT-2...")
trainer.train()

print("Training completed!")
print(f"Model saved to: {training_args.output_dir}")
