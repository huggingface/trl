# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM
from accelerate import FullyShardedDataParallelPlugin, Accelerator
import torch
dataset = load_dataset("trl-lib/tldr", split="train")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", use_vllm=True, fsdp_config={
    "fsdp_auto_wrap_policy": "transformer_based_wrap",
    "fsdp_transformer_cls_names_to_wrap": ["Qwen2DecoderLayer"],
    "fsdp_version": 2,
    "fsdp_reshard_after_forward": True,
    "fsdp_state_dict_type": "full_state_dict",
    "fsdp_mixed_precision": "bf16",
    "fsdp_gradient_accumulation_steps": 1,
})
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
