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

# /// script
# dependencies = [
#     "trl[peft,quantization]",
#     "transformers>=5.3.0",
#     "trackio",
#     "mamba_ssm==2.2.5",
#     "causal_conv1d==1.5.2",
# ]
# ///

"""
Fine-tune NVIDIA Nemotron 3 models with SFT.

Prerequisites:

    pip install "transformers>=5.3.0"
    pip install --no-build-isolation mamba_ssm==2.2.5
    pip install --no-build-isolation causal_conv1d==1.5.2

Example:

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_nemotron_3.py \
    --dtype bfloat16 \
    --model_name_or_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --attn_implementation eager \
    --dataset_name HuggingFaceH4/Multilingual-Thinking \
    --max_length 128 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --optim paged_adamw_8bit \
    --logging_steps 10 \
    --output_dir nemotron-3-sft \
    --report_to trackio \
    --use_peft \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM

from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser, get_peft_config


def main(script_args, training_args, model_args):
    # NemotronH does not support gradient checkpointing
    training_args.gradient_checkpointing = False

    # Load model
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Load dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Merge thinking into message content using <think> tags and remove extra columns
    def merge_thinking_and_remove_key(example):
        new_messages = []
        for msg in example["messages"]:
            content = msg["content"]
            thinking = msg.get("thinking")
            if thinking and isinstance(thinking, str) and thinking.strip():
                content = f"<think>\n{thinking}\n</think>\n{content}"
            new_messages.append({"role": msg["role"], "content": content})
        example["messages"] = new_messages
        return example

    dataset = dataset.map(merge_thinking_and_remove_key)

    # Prepare eval dataset if needed
    eval_dataset = None
    if training_args.eval_strategy != "no" and script_args.dataset_test_split in dataset:
        eval_dataset = dataset[script_args.dataset_test_split]

    # Train model
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(script_args, training_args, model_args)
