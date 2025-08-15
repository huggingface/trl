from trl import BEMACallback
from trl import SFTTrainer
from datasets import load_dataset

dataset = load_dataset("trl-lib/Capybara", split="train[:100]")

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    callbacks=[BEMACallback(update_after=10, update_freq=18)],
)
trainer.train()