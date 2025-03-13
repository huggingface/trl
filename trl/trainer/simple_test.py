from datasets import Dataset
import pandas as pd

# Create a simple dataset with math and coding problems
data = {
    "prompt": [
        "Solve: 2 + 2 = ?",
        "Write a function to calculate factorial",
        "Solve: 5 * 3 = ?",
        "Write a function to check if a string is a palindrome"
    ],
    "task_type": ["math", "coding", "math", "coding"]
}

test_dataset = Dataset.from_pandas(pd.DataFrame(data))


def math_reward_func(prompts, completions, task_type, **kwargs):
    print(f"Processing batch with {len(prompts)} samples")
    rewards = []
    for i, (prompt, completion, task) in enumerate(zip(prompts, completions, task_type)):
        print(f"Sample {i}: Task type = {task}")
        if task == "math":
            # Simple reward for math: check if completion contains the correct answer
            if "4" in completion and prompt.startswith("Solve: 2 + 2"):
                reward = 1.0
            elif "15" in completion and prompt.startswith("Solve: 5 * 3"):
                reward = 1.0
            else:
                reward = -1.0
            print(f"  Math reward: {reward}")
            rewards.append(reward)
        else:
            # Return None for non-math tasks
            print(f"  Skipping (None)")
            rewards.append(None)
    return rewards

######passpass###################TEST####################################
from .grpo_trainer import GRPOTrainer
from .grpo_config import GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a small model for testing
model_name = "Qwen/Qwen2.5-0.5B"  # Use a small model for quick testing
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Initialize the trainer with minimal settings
trainer = GRPOTrainer(
    model=model,
    reward_funcs=math_reward_func,
    train_dataset=test_dataset,
    processing_class=tokenizer,
    args=GRPOConfig(
        output_dir="./output",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        max_steps=2,  # Just run a couple of steps for testing
        logging_steps=1,
        num_generations=2,  # Use a small number for testing
        log_completions=True,  # To see the generated completions
    )
)

# Run training
trainer.train()