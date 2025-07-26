from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from trl.trainer import RLOOConfig, RLOOTrainer


# Load models
policy_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
policy_ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", padding_side="left")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def tokenize_function(example):
    return dict(
        input_ids=tokenizer(
            example["prompt"],
            truncation=True,
            padding=False,
        )["input_ids"]
    )


# Simple dataset with just two prompts
dataset = Dataset.from_dict(
    {
        "prompt": ["The sky is", "The sun is"],
    }
)
dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)
print(dataset)

# Dummy reward function
def reward_func(completions, **kwargs):
    """Reward function that rewards longer completions."""
    return [float(len(completion)) for completion in completions]


# Config
training_args = RLOOConfig(
    output_dir="rloo-debug",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    rloo_k=2,
    report_to="none",
    total_episodes=1,
    num_train_epochs=1,
    max_steps=2,
    token_level_kl=True,
    num_ppo_epochs=8,
)

# Trainer
trainer = RLOOTrainer(
    config=training_args,
    policy=policy_model,
    ref_policy=policy_ref_model,
    reward_model=reward_func,
    processing_class=tokenizer,
    train_dataset=dataset,
    eval_dataset=dataset,
)

trainer.train()
