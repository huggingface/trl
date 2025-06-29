from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.trainer import RLOOConfig, RLOOTrainer, RLOOConfig_NEW, RLOOTrainer_NEW


prompt = "what is the capital of canada?"
completion = "it is ottawa"

# Create a simple dataset
dataset = Dataset.from_dict({
    "prompt": [prompt] * 4,  # Repeat the same prompt 10 times
})

# Model and tokenizer
model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Simple reward function that counts unique characters
def reward_func(completions, **kwargs):
    """Reward function that counts unique characters in completion content."""
    rewards = []
    for completion in completions:
        # Extract the content from the completion
        content = completion[0]["content"]
        # Count unique characters in the content
        unique_chars = len(set(content))
        rewards.append(float(unique_chars))
    return rewards



test_completions = [[{"role": "assistant", "content": completion}]]
test_reward = reward_func(test_completions)


# New RLOO config
new_config = RLOOConfig_NEW(
    output_dir="new-rloo-test",
    per_device_train_batch_size=2,
    num_generations=2,
    max_completion_length=10,
    report_to="none",
)

# New RLOO trainer
new_trainer = RLOOTrainer_NEW(
    model=model_id,
    reward_funcs=reward_func,
    args=new_config,
    train_dataset=dataset,
)


print("\n=== Testing Old RLOO Implementation ===")

# Old RLOO config
old_config = RLOOConfig(
    exp_name="old-rloo-test",
    output_dir="old-rloo-test",
    per_device_train_batch_size=2,
    rloo_k=2,  # Number of generations per prompt
    max_new_tokens=10,
    report_to="none",
    num_train_epochs=1,
    total_episodes=2,  # Just run a few episodes
)

# Load model for old implementation
model = AutoModelForCausalLM.from_pretrained(model_id)
ref_model = AutoModelForCausalLM.from_pretrained(model_id)

# Old RLOO trainer
old_trainer = RLOOTrainer(
    config=old_config,
    processing_class=tokenizer,
    policy=model,
    ref_policy=ref_model,
    reward_model=reward_func,
    train_dataset=dataset,
)

print("Old RLOO trainer created successfully")

print("\n=== Summary ===")
print("Both implementations are ready for testing.")
print("The reward function should return the number of unique characters in the completion.")
print(f"For completion '{completion}', expected reward: {len(set(completion))}") 