
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from trl.trainer.rloo_new import RLOOTrainer_NEW
from trl.trainer.rloo_new_config import RLOOConfig_NEW


#dataset = load_dataset("trl-lib/tldr", split="train[:100]")
# Simple dataset with just two prompts
dataset = Dataset.from_dict(
    {
        "prompt": ["The sky is", "The sun is"],
    }
)
print(dataset)
model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)


# Dummy reward function: count the number of unique characters in the completions
def reward_func(completions, **kwargs):
    """Reward function that rewards longer completions."""
    return [float(len(completion)) for completion in completions]



training_args = RLOOConfig_NEW(
    output_dir="new-rloo-debug",
    per_device_train_batch_size=4,
    num_generations=2,
    max_completion_length=8,
    report_to=[],
    num_iterations=10,
)

trainer = RLOOTrainer_NEW(
    model=model_id,
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
