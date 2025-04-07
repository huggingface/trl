from datasets import load_dataset
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig


dataset = load_dataset("trl-lib/tldr", split="train[:200]")


def reward_func(completions, **kwargs):
    return [len(set(c)) for c in completions]

args = GRPOConfig(
    output_dir="mask_truncated_test2",
    use_vllm=False,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    mask_truncated_completions=True, 
    max_completion_length=10, 
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_generations=4,  
)


trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B",
    args=args,
    reward_funcs=reward_func,
    train_dataset=dataset,
)


trainer.train() 


EOS: 2
max_completion_length: 5

[1,1,1,1] #(1,4) # NO EOS buts still less than max_completion_length ---> NO TRUNCATION
[1,1,1,1,1] #(1,5) # NO EOS and equal to max_completion_length ---> NO TRUNCATION
[1,1,1,1,1,1] #(1,6) # NO EOS and greater than max_completion_length ---> TRUNCATION
[1,1,1,2] #(1,4) # EOS and less than max_completion_length ---> NO TRUNCATION
[1,1,1,2,2] #(1,5) # EOS and equal to max_completion_length ---> NO TRUNCATION








[]