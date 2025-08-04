"""Basic GRPO training script.

Adapted from https://huggingface.co/docs/trl/main/en/grpo_trainer

Assumes we will use 2 gpus for FSDP training and 2 gpus for TP vLLM server.

Must first run:
```
trl vllm-serve --model Qwen/Qwen3-14B --tensor-parallel-size 2
```
Example accelerate config:
```
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: "no"
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: "bf16"
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
Run this script with `CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config_file accelerate_config.yaml grpo_tldr.py`
"""

from datasets import Dataset, load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer


dataset = load_dataset("trl-lib/tldr", split="train")
assert isinstance(dataset, Dataset)
dataset = dataset.select(range(128))


# Dummy reward function: count the number of unique characters in the completions
def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]


training_args = GRPOConfig(
    output_dir="test",
    use_vllm=True,
    vllm_mode="server",
    bf16=True,
    gradient_checkpointing=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_generations=2,
    log_level="debug",
    logging_steps=1,
    model_init_kwargs={
        "torch_dtype": "bfloat16",
    },
    beta=0.1,
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",
    task_type="CAUSAL_LM"
)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-14B",
    args=training_args,
    reward_funcs=reward_num_unique_chars,  # type: ignore
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()
