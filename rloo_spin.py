# test_rloo.py
from datasets import load_dataset
from trl.trainer.rloo_new import RLOOConfig, RLOOTrainer

dataset = load_dataset("trl-lib/tldr", split="train[:10]")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = RLOOConfig(
    output_dir="Qwen2-0.5B-RLOO", 
    logging_steps=10,
    num_generations=2,
    per_device_train_batch_size=2,
    max_steps=10
)

trainer = RLOOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,  # Now using reward_funcs to support both functions and models
    args=training_args,
    train_dataset=dataset,
)
trainer.train()