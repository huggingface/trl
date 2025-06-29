from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl.trainer import RLOOConfig, RLOOTrainer
from trl.data_utils import apply_chat_template

# Load models
policy_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
policy_ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", padding_side="left")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

def tokenize_function(example):
    return dict(input_ids=tokenizer.apply_chat_template(
        example["prompt"],
        add_generation_prompt=True,
    ))

#dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
})
dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)

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
