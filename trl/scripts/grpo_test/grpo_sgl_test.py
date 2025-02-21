import os
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import tempfile
import torch

# Use a checkpoint directory within your project path
checkpoint_dir = os.path.join("/home/misc/jinpan/trl-jin", "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

dataset = load_dataset(
    "trl-internal-testing/zen", "standard_prompt_only", split="train"
)

# Configure training arguments for engine mode.
# (Notice we do NOT set sglang_server_url so that the trainer uses engine mode.)
training_args = GRPOConfig(
    output_dir=checkpoint_dir,
    learning_rate=1.0e-03,
    per_device_train_batch_size=3,
    num_generations=9,
    max_completion_length=32,
    report_to="none",
    use_sglang=True,
    sglang_device="cuda:7",
    sglang_gpu_memory_utilization=0.9,
    # Do not set sglang_server_url – the Engine will be launched internally.
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
    args=training_args,
    train_dataset=dataset,
)

# Save a checkpoint so that update_weights_from_disk can work later.
trainer.model.save_pretrained(checkpoint_dir)
training_args.checkpoint_path = (
    checkpoint_dir  # Set the checkpoint path for weight updates
)

# Optionally, clone initial trainable parameters for testing.
previous_trainable_params = {
    n: param.clone() for n, param in trainer.model.named_parameters()
}

trainer.train()

# Check that the parameters have changed.
for n, param in previous_trainable_params.items():
    new_param = trainer.model.get_parameter(n)
    assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
print("test over")
