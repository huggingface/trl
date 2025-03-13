# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
Train Gemma-3 on the Codeforces COTS dataset.

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml examples/scripts/sft_gemma3.py
"""

from datasets import load_dataset
from transformers import AutoModelForImageTextToText

from trl import SFTConfig, SFTTrainer


def main():
    # Load dataset
    train_dataset = load_dataset("open-r1/codeforces-cots", split="train")
    train_dataset = train_dataset.remove_columns("prompt")

    # Load model
    model_id = "google/gemma-3-12b-it"
    model = AutoModelForImageTextToText.from_pretrained(model_id, attn_implementation="eager")

    # Train model
    training_args = SFTConfig(
        output_dir=f"{model_id}-codeforces-SFT",
        logging_steps=10,
        bf16=True,
        use_liger_kernel=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=8192,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        dataset_num_proc=32,
        num_train_epochs=1,
    )
    trainer = SFTTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
    )
    trainer.train()

    # Push to hub
    trainer.push_to_hub(dataset_name="open-r1/codeforces-cots")


if __name__ == "__main__":
    main()
