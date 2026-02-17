# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

from trl.experimental.dppo import DPPOConfig, DPPOTrainer
from trl.rewards import accuracy_reward


def main():
    dataset = load_dataset("trl-lib/DeepMath-103K", split="train[:100]")

    config = DPPOConfig(
        output_dir="dppo-testing",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        # DPPO-specific trust-region settings (paper-inspired sanity run)
        divergence_type="topk_kl",
        divergence_topk=20,
        # use_vllm=True,
        # vllm_mode="colocate",
        vllm_max_model_length=512,
        epsilon=0.2,
        epsilon_high=0.28,
        clip_ratio_c=3.0,
        num_generations=2,
        max_completion_length=256,
        learning_rate=1e-6,
        bf16=True,
        # gradient_checkpointing=True,
        logging_steps=1,
        report_to="none",
        max_steps=1,
    )

    trainer = DPPOTrainer(
        model="Qwen/Qwen3-1.7B",
        reward_funcs=accuracy_reward,
        args=config,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
