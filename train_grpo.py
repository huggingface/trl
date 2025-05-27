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

from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer


dataset = load_dataset("open-r1/DAPO-Math-17k-Processed-R1-Distill-Qwen-Math-7B-v03.00-step-000008190-filter", split="train")
def make_conversation(example, prompt_column: str = "prompt"):
    prompt = []

    if prompt_column not in example:
        raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

    prompt.append({"role": "user", "content": example[prompt_column]})
    return {"prompt": prompt}

dataset = dataset.map(make_conversation)

if "messages" in dataset.column_names:
    dataset = dataset.remove_columns("messages")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


training_args = GRPOConfig(output_dir="data/Qwen/Qwen2.5-0.5B-Instruct", logging_steps=1, gradient_accumulation_steps=2, num_generations=4, 
                           max_completion_length=4000, max_steps=20, gradient_checkpointing=False, beta=0.0, per_device_train_batch_size=2, 
                           replay_buffer_class="SSRReplayBuffer", 
                           ssr_capacity_scalar=4
                           )
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
