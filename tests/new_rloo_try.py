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
from transformers import AutoTokenizer

from trl.trainer.rloo_new import RLOOTrainer_NEW
from trl.trainer.rloo_new_config import RLOOConfig_NEW


dataset = load_dataset("trl-lib/tldr", split="train[:100]")


model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)


# Dummy reward function: count the number of unique characters in the completions
def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]


training_args = RLOOConfig_NEW(
    output_dir="new-rloo-debug",
    per_device_train_batch_size=4,
    num_generations=2,
    max_completion_length=8,
    report_to="wandb",
    num_iterations=2,
    normalize_rewards=True,
    normalize_advantages=True,
)

trainer = RLOOTrainer_NEW(
    model=model_id,
    reward_funcs=reward_num_unique_chars,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
