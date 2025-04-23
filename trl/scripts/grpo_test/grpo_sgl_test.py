# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import tempfile
import torch
import wandb

wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="y1cunhui-yang-independent",
            # Set the wandb project where this run will be logged.
            project="SGLANG-GRPO-Qwen2.5-0.5B-Instruct",
            group="sglang-trl"
        )

dataset = load_dataset("trl-lib/tldr", split="train[:5%]")

checkpoint_dir = os.path.join("/sgl-workspace/ryang/trl", "checkpoints/sgl")
os.makedirs(checkpoint_dir, exist_ok=True)

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(
    output_dir=os.path.join(checkpoint_dir, "Qwen2.5_output"), 
    logging_steps=10, 
    report_to="wandb", 
    use_sglang=True,
    sglang_device="cuda:0",
    sglang_gpu_memory_utilization=0.9,
    sglang_server_url="http://127.0.0.1:30000")


trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
# # Save a checkpoint to initialize checkpoint_path locally
# trainer.model.save_pretrained(checkpoint_dir)
training_args.checkpoint_path = checkpoint_dir  # Set the checkpoint path for later use

# #Optionally, clone initial trainable parameters for testing.
# previous_trainable_params = {
#     n: param.clone() for n, param in trainer.model.named_parameters()
# }

trainer.train()
# import os
# from datasets import load_dataset
# from trl import GRPOConfig, GRPOTrainer
# import tempfile
# import torch
# import wandb

# wandb.init(
#             # Set the wandb entity where your project will be logged (generally your team name).
#             entity="y1cunhui-yang-independent",
#             # Set the wandb project where this run will be logged.
#             project="VLLM-GRPO-Qwen2.5-0.5B-Instruct",
#         )

# # Use a checkpoint directory within your project path
# checkpoint_dir = os.path.join("/sgl-workspace/ryang/trl", "checkpoints")
# os.makedirs(checkpoint_dir, exist_ok=True)

# dataset = load_dataset("trl-lib/tldr", split="train")
# dataset = dataset.train_test_split(test_size=0.01, seed=42)["test"]

# # Define the reward function, which rewards completions that are close to 20 characters
# def reward_len(completions, **kwargs):
#     return [-abs(20 - len(completion)) for completion in completions]



# training_args = GRPOConfig(
#     output_dir=checkpoint_dir,  # Set output directory here
#     learning_rate=1.0e-03,
#     per_device_train_batch_size=3,
#     num_generations=3,
#     max_completion_length=32,
#     report_to="wandb",
#     use_sglang=True,
#     logging_steps=10,
#     sglang_device="cuda:1",
#     sglang_gpu_memory_utilization=0.9,
#     sglang_server_url="http://127.0.0.1:30000",
# )

# trainer = GRPOTrainer(
#     model="Qwen/Qwen2.5-0.5B-Instruct",
#     reward_funcs=reward_len,
#     args=training_args,
#     train_dataset=dataset,
# )



# trainer.train()

# # Check that the parameters have changed.
# # for n, param in previous_trainable_params.items():
# #     new_param = trainer.model.get_parameter(n)
# #     assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
# # print("test over")
