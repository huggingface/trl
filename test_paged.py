from trl import GRPOTrainer
from datasets import load_dataset

# Dummy vision-language dataset
dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

# Dummy reward function: count the number of unique characters in the completions
def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c[0]["content"])) for c in completions]

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    reward_funcs=[reward_num_unique_chars],
    train_dataset=dataset,
)

trainer.train()