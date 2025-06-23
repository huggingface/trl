from datasets import load_dataset
from transformers import  AutoTokenizer
from trl.trainer import RLOOConfig_NEW, RLOOTrainer_NEW
from datasets import Dataset


#dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
})

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)


def reward_func(completions, **kwargs):
    """Reward function that rewards longer completions."""
    return [float(len(completion)) for completion in completions]



training_args = RLOOConfig_NEW(
    output_dir="new-rloo-debug",
    per_device_train_batch_size=4,  
    num_generations=2, 
    max_completion_length=8,  
    report_to="none",
    beta=0.05, # KL coefficient in old, have it to have fair 
)

trainer = RLOOTrainer_NEW(
    model=model_id,
    reward_funcs=reward_func,  # Pass the function itself, not the result of calling it
    args=training_args,
    train_dataset=dataset,
)

trainer.train()