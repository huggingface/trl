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

"""
Train Gemma-3 on the HuggingFaceH4/llava-instruct-mix-vsft dataset (single-image).

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm_gemma3.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path google/gemma-3-4b-it \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir gemma-3-4b-it-trl-sft-llava-instruct-mix-vsft \
    --bf16 \
    --torch_dtype bfloat16 \
    --use_peft \
    --lora_target_modules all-linear \
    --attn_implementation eager

Train Gemma-3 on the FanqingM/MMIU-Benchmark dataset (multi-image).

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm_gemma3.py \
    --dataset_name FanqingM/MMIU-Benchmark \
    --dataset_train_split test \
    --model_name_or_path google/gemma-3-4b-it \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir gemma-3-4b-it-trl-sft-MMIU-Benchmark \
    --bf16 \
    --torch_dtype bfloat16 \
    --use_peft \
    --lora_target_modules all-linear
    --attn_implementation eager
"""

import io
import os
import zipfile

import torch
from datasets import DatasetDict, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

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


# For multi-image example
def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                if image is not None:
                    image = Image.open(io.BytesIO(image["bytes"]))
                    image_inputs.append(image.convert("RGB"))
    return image_inputs


def format_data(samples: dict[str, any]) -> dict[str, list]:
    formatted_samples = {"messages": []}
    for cont in range(len(samples["question"])):
        images = []
        for img_path in samples["input_image_path"][cont]:
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append({"type": "image", "image": image})
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

        formatted_samples["messages"].append(
            [
                {"role": "system", "content": [{"type": "text", "text": samples["context"][cont]}]},
                {"role": "user", "content": images + [{"type": "text", "text": samples["question"][cont]}]},
                {"role": "assistant", "content": [{"type": "text", "text": samples["output"][cont]}]},
            ]
        )
    return formatted_samples


# For multi-image example
def prepare_dataset(dataset: DatasetDict, dataset_name: str, dataset_train_split: str) -> DatasetDict:
    all_files = list_repo_files(dataset_name, repo_type="dataset")
    zip_files = [f for f in all_files if f.endswith(".zip")]

    for zip_filename in zip_files:
        zip_path = hf_hub_download(repo_id=dataset_name, filename=zip_filename, repo_type="dataset")
        extract_folder = zip_filename.replace(".zip", "")
        os.makedirs(extract_folder, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

    dataset = dataset.map(format_data, batched=True, batch_size=4, num_proc=16)
    return dataset


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
    processor.tokenizer.padding_side = "right"

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    def collate_fn(examples):
        texts = [
            processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False).strip()
            for example in examples
        ]
        if "images" in examples[0]:  # single-image
            images = [[img.convert("RGB") for img in example["images"]] for example in examples]
        else:  # multi-image
            images = [process_vision_info(example["messages"]) for example in examples]

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        # Mask image tokens
        image_token_id = [
            processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.special_tokens_map["boi_token"])
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
    if script_args.dataset_name == "FanqingM/MMIU-Benchmark":
        dataset = prepare_dataset(dataset, script_args.dataset_name, script_args.dataset_train_split)

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
