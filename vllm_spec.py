from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

dataset = load_dataset("trl-lib/tldr", split="train")

# Dummy reward function: count the number of unique characters in the completions
def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]

training_args = GRPOConfig(
    output_dir="spec",
    use_vllm=True,
    vllm_mode="server",
    vllm_server_port=8001,
    bf16=True,
    gradient_checkpointing=True,
    per_device_train_batch_size=2,
    num_generations=2,
    max_completion_length=32,
    max_steps=10,
)

trainer = GRPOTrainer(
    model="microsoft/DialoGPT-medium",
    args=training_args,
    reward_funcs=reward_num_unique_chars,
    train_dataset=dataset,
)

trainer.train()

# Test commands:
# 1. Start vLLM server with speculative decoding:
# python trl/scripts/vllm_serve.py --model microsoft/DialoGPT-medium --speculative_config '{"model": "microsoft/DialoGPT-small", "num_speculative_tokens": 5}' --port 8001
# 
# 2. Run training with speculative decoding:
# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch vllm_spec.py