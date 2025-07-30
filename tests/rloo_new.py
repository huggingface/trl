from trl.trainer.rloo_trainer_final import RLOOTrainer_NEW
from trl.trainer.rloo_trainer_final_config import RLOOConfig_NEW
from transformers import AutoTokenizer
from datasets import Dataset


# Simple dataset with just two prompts
dataset = Dataset.from_dict(
    {
        "prompt": ["The sky is", "The sun is"],
    }
)


model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def reward_func(completions, **kwargs):
    return [float(len(set(completion))) for completion in completions]


training_args = RLOOConfig_NEW(
    output_dir="new-rloo-debug",
    per_device_train_batch_size=4,
    num_generations=4,
    per_device_eval_batch_size=2,
    report_to=[],
    max_steps=3,  # match old config
    beta=0.05,  # was kl_coef=0.05 (this is default anyway)
)


trainer = RLOOTrainer_NEW(
    model=model_id,
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # add eval_dataset like old config
)

trainer.train()

