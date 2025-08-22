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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

def reward_func(completions, **kwargs):
    """Reward function that rewards completions with more unique letters."""
    completion_contents = [completion[0]["content"] for completion in completions]
    return [float(len(set(content))) for content in completion_contents]


def main():
    model_id = "Qwen/Qwen3-0.6B"
    policy_model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")


    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = GRPOConfig(
            output_dir=tmp_dir,
            per_device_train_batch_size=2,
            learning_rate=1e-6,
            num_generations=2,
            report_to="none",
            beta=0.05,
            max_steps=2,
            loss_type="grpo",
            importance_sampling_level="sequence",
            log_completions=True,
            num_completions_to_print=2,
            shuffle_dataset=False,
            logging_steps=1,
            use_rloo=True,
            num_iterations=1,
            max_completion_length=11
        )
        
        trainer = GRPOTrainer(
            model=policy_model,
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        # Check initial weights
        initial_weights = policy_model.state_dict()
        print("Sample weights before training:")
        for name, param in list(initial_weights.items())[:3]:
            print(f"{name}: {param.flatten()[:5]}")
        
        trainer.train()
        
        # Check final weights
        final_weights = policy_model.state_dict()
        print("\nSample weights after training:")
        for name, param in list(final_weights.items())[:3]:
            print(f"{name}: {param.flatten()[:5]}")
        
        # Check if weights changed
        weights_changed = False
        for name in initial_weights:
            if not torch.equal(initial_weights[name], final_weights[name]):
                weights_changed = True
                break
        print(f"\nWeights changed: {weights_changed}")
        print("Training completed successfully!")


if __name__ == "__main__":
    main()
