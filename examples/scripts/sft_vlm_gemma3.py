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
Train Gemma-3 on the HuggingFaceH4/llava-instruct-mix-vsft dataset.

accelerate launch 
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm_gemma3.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path google/gemma-3-4b-it \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir gemma-3-4b-instruct-trl-sft-ChartQA \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --use_peft \
    --lora_target_modules down_proj, o_proj, k_proj, q_proj, gate_proj, up_proj, v_proj
"""

import torch

from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


def main():
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    def collate_fn(examples):
      texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
      images = [img.convert("RGB") if img.mode == "RGBA" else img for example in examples for img in example["images"]]

      # Tokenize the texts and process the images
      batch = processor(
          text=texts, images=images, return_tensors="pt", padding=True
      )  # Encode texts and images into tensors

      # The labels are the input_ids, and we mask the padding tokens in the loss computation
      labels = batch["input_ids"].clone()  # Clone input IDs for labels
      # Mask image tokens
      image_token_id = [
          processor.tokenizer.convert_tokens_to_ids(
              processor.tokenizer.special_tokens_map["boi_token"]
          )
      ]
      # Mask tokens for not being used in the loss computation
      labels[labels == processor.tokenizer.pad_token_id] = -100
      labels[labels == image_token_id] = -100
      labels[labels == 262144] = -100

      batch["labels"] = labels
      return batch  # Return the prepared batch

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor.tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    main()