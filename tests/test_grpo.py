from datasets import load_dataset
from transformers import  AutoTokenizer
from trl.trainer import GRPOConfig, GRPOTrainer



dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")


# Load dataset and tokenize
#dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
# Create a simple dataset

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def reward_func(completions, **kwargs):
    """Reward function that rewards longer completions."""
    return [float(len(completion)) for completion in completions]


training_args = GRPOConfig(
    output_dir="grpo-debug",
    per_device_train_batch_size=4,  # reduce the batch size to reduce memory usage
    num_generations=2,  # reduce the number of generations to reduce memory usage
    max_completion_length=10,  # reduce the completion length to reduce memory usage
    report_to="none",
    loss_type = "bnpo",
    beta=0.1,  
)

trainer = GRPOTrainer(
    model=model_id,
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()