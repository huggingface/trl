from datasets import load_dataset
from trl.trainer import GRPOTrainer, GRPOConfig

dataset = load_dataset("trl-lib/tldr", split="train")


# Dummy reward function: count the number of unique characters in the completions
def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]


training_args = GRPOConfig(
    output_dir="mask_truncated_completions",
    use_vllm=False,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    mask_truncated_completions=True,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B",
    args=training_args,
    reward_funcs=reward_num_unique_chars,
    train_dataset=dataset,
    max_completion_length=50,
)
trainer.train()