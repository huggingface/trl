# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
pip install pillow

# Tested on 8x H100 GPUs
accelerate launch
    --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm_molmo.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path allenai/Molmo-7B-D-0924 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir sft-Molmo-7B-D-0924 \
    --bf16 \
    --torch_dtype bfloat16
"""

import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_quantization_config,
)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    model_config.trust_remote_code = True

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    processor.chat_template = processor.tokenizer.chat_template

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=64, lora_alpha=128, lora_dropout=0.1, target_modules="all-linear"
    )
    model = get_peft_model(model, peft_config)


    ################
    # Create a data collator to encode text and image pairs
    ################
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = []
        for example in examples:
            flattened_messages = []
            for message in example['messages']:
                content_texts = ''.join(content['text'] for content in message['content'] if content['text'] is not None)
                flattened_messages.append({'role': message['role'], 'content': content_texts})
            texts.append(processor.apply_chat_template(flattened_messages, tokenize=False))

        images = [example["images"] for example in examples]

        # Tokenize the texts and process the images
        # batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        batch = {
            'input_ids': [],
            'images': [],
            'image_input_idx': [],
            'image_masks': []
        }

        # https://huggingface.co/allenai/Molmo-7B-D-0924/blob/main/preprocessing_molmo.py#L115
        for text, image in zip(texts, images):
            processed = processor.process(text=text, images=image, return_tensors="pt", padding=True)
            batch['input_ids'].append(processed['input_ids'])
            batch['images'].append(processed['images'])
            batch['image_input_idx'].append(processed['image_input_idx'])
            batch['image_masks'].append(processed['image_masks'])

        batch['input_ids'] = torch.stack(batch['input_ids'])
        batch['images'] = torch.stack(batch['images'])
        batch['image_input_idx'] = torch.stack(batch['image_input_idx'])
        batch['image_masks'] = torch.stack(batch['image_masks'])

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels

        return batch

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
        processing_class=processor.tokenizer,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)
