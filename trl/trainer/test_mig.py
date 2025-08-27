from trl.trainer.rloo_config import RLOOConfig
from trl.trainer.rloo_trainer import RLOOTrainer
from datasets import load_dataset


dataset = load_dataset("trl-lib/tldr", split="train[:10]")


def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


config = RLOOConfig(
    rloo_k=4,
    kl_coef=0.1,
    missing_eos_penalty=1.0
)

trainer = RLOOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_model=reward_len,  # RIP RM
    args=config,
    train_dataset=dataset,
    )

trainer.train()
