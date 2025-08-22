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

import tempfile
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import RLOOConfig, RLOOTrainer


def reward_func(completions, **kwargs):
    """Reward function that rewards completions with more unique letters."""
    return [float(len(set(completion))) for completion in completions]


def main():
    model_id = "Qwen/Qwen3-0.6B"
    policy_model = AutoModelForCausalLM.from_pretrained(model_id)
    policy_ref_model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = RLOOConfig(
            output_dir=tmp_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            total_episodes=1,
            num_train_epochs=1,
            max_steps=2,
            rloo_k=2, 
            report_to="none",
            learning_rate=1e-6,  # Match GRPO default
            temperature=1.0,     # Match GRPO default
        )

        # Create a simple dataset
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        prompts = [tokenizer.apply_chat_template(dataset[i]["prompt"], tokenize=False) for i in range(len(dataset))]
        tokenized = tokenizer(prompts, padding=True, padding_side="right", return_tensors="pt")
        dummy_dataset = Dataset.from_dict({"input_ids": tokenized["input_ids"].tolist()})

        trainer = RLOOTrainer(
            config=training_args,
            policy=policy_model,
            reward_model=reward_func,
            ref_policy=policy_ref_model,
            processing_class=tokenizer,
            train_dataset=dummy_dataset,
            eval_dataset=dummy_dataset,
        )

        # # Check initial weights
        # initial_weights = policy_model.state_dict()
        # print("Sample weights before training:")
        # for name, param in list(initial_weights.items())[:3]:
        #     print(f"{name}: {param.flatten()[:5]}")

        # # Test that training completes without errors
        trainer.train()
        
        # # Check final weights
        # final_weights = policy_model.state_dict()
        # print("\nSample weights after training:")
        # for name, param in list(final_weights.items())[:3]:
        #     print(f"{name}: {param.flatten()[:5]}")
        
        # # Check if weights changed
        # weights_changed = False
        # for name in initial_weights:
        #     if not torch.equal(initial_weights[name], final_weights[name]):
        #         weights_changed = True
        #         break
        # print(f"\nWeights changed: {weights_changed}")


if __name__ == "__main__":
    main()
