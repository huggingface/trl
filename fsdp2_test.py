from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig


# Dummy reward function: count the number of unique characters in the completions
def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]


def main():
    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_num_unique_chars,
        train_dataset=dataset,
        args=GRPOConfig(run_name="fsdp2_test2", use_vllm=True, vllm_mode="colocate"),
        peft_config=LoraConfig()
    )
    trainer.train()


if __name__ == "__main__":
    main()
