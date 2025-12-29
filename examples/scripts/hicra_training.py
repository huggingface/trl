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

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
# HICRA Training Example

HICRA (Hierarchy-Aware Credit Assignment) is an extension of GRPO that focuses optimization
on strategic planning tokens, enabling LLMs to develop hierarchical reasoning capabilities.

This example demonstrates HICRA training on a math reasoning task using the DeepMath-103K dataset.

## Basic Usage

```bash
python examples/scripts/hicra_training.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name trl-lib/DeepMath-103K \
    --output_dir hicra_deepmath_qwen \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-6 \
    --hicra_alpha 0.2 \
    --use_hicra True
```

## Advanced Configuration

```bash
python examples/scripts/hicra_training.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name trl-lib/DeepMath-103K \
    --output_dir hicra_deepmath_qwen \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-6 \
    --hicra_alpha 0.2 \
    --use_hicra True \
    --use_planning_tokens True \
    --strategic_grams_path path/to/strategic_grams.json \
    --log_semantic_entropy True \
    --log_planning_token_ratio True \
    --report_to wandb \
    --push_to_hub True
```

## Key HICRA Parameters

- `hicra_alpha`: Amplification factor for planning/high-entropy tokens (default: 0.2)
- `use_hicra`: Enable/disable HICRA advantage modification (default: True)
- `hicra_entropy_topk`: Top-k percentile for entropy threshold (default: 0.3)
- `use_planning_tokens`: Whether to use Strategic Gram-based planning token identification (default: False)
- `strategic_grams_path`: Path to pre-computed Strategic Grams JSON file (optional)
- `strategic_grams`: Direct list of Strategic Grams (optional)
- `log_semantic_entropy`: Log strategic diversity metrics (default: True)
- `log_planning_token_ratio`: Log percentage of planning tokens (default: True)

## Comparison with GRPO

To compare HICRA with standard GRPO, simply set `use_hicra=False`:

```bash
python examples/scripts/hicra_training.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name trl-lib/DeepMath-103K \
    --output_dir grpo_baseline_deepmath_qwen \
    --use_hicra False
```

## References

- Paper: "Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning" (arXiv:2509.03646)
- VeRL Implementation: https://github.com/TIGER-AI-Lab/Hierarchical-Reasoner
"""

import os

from datasets import load_dataset

from trl import HICRAConfig, HICRATrainer, ModelConfig, ScriptArguments, TrlParser
from trl.rewards import accuracy_reward


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


def format_example(example):
    """
    Format DeepMath-103K examples for HICRA training.

    The dataset contains math problems with questions and answers.
    We format them as simple prompts for the model to solve.
    """
    question = example["question"]
    prompt = [{"role": "user", "content": f"Solve the following math problem:\n\n{question}"}]
    return {"prompt": prompt}


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, HICRAConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Load and format dataset
    dataset = load_dataset(script_args.dataset_name, split="train")
    dataset = dataset.map(format_example, remove_columns=["question"])

    # Split into train and eval if needed
    if training_args.eval_strategy != "no":
        dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    # Initialize HICRA trainer
    trainer = HICRATrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[accuracy_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train the model
    trainer.train()

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    print("✅ HICRA training completed!")
    print(f"💾 Model saved to {training_args.output_dir}")
    if training_args.push_to_hub:
        print("🤗 Model pushed to the Hub")
