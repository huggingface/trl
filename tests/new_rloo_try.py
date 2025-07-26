
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from trl.trainer.rloo_new import RLOOTrainer_NEW
from trl.trainer.rloo_new_config import RLOOConfig_NEW


dataset = load_dataset("trl-lib/tldr", split="train[:100]")
# Simple dataset with just two prompts
# dataset = Dataset.from_dict(
#     {
#         "prompt": ["The sky is", "The sun is"],
#     }
# )
print(dataset["prompt"][0])
model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)


# Dummy reward function: count the number of unique characters in the completions
def reward_func(completions, **kwargs):
    """Reward function that rewards longer completions."""
    return [float(len(completion)) for completion in completions]



# Parameter mapping from old to new config:
# rloo_k → num_generations
# kl_coef → beta  
# cliprange → epsilon
# normalize_reward → normalize_rewards
# normalize_advantage → normalize_advantages
# num_ppo_epochs → num_iterations
# total_episodes, num_train_epochs, max_steps → max_steps
training_args = RLOOConfig_NEW(
    output_dir="new-rloo-debug",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    num_generations=2,  # was rloo_k=2
    max_completion_length=8,
    report_to=[],  # was report_to="none" 
    max_steps=2,  # match old config
    token_level_kl=True,  # match old config
    num_iterations=8,  # was num_ppo_epochs=8
    beta=0.05,  # was kl_coef=0.05 (this is default anyway)
    epsilon=0.2,  # was cliprange=0.2 (this is default anyway)
)

trainer = RLOOTrainer_NEW(
    model=model_id,
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # add eval_dataset like old config
)

trainer.train()
