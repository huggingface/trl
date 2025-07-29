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
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
# ]
# ///

"""
python examples/scripts/mpo_vlm.py \
    --dataset_name HuggingFaceH4/rlaif-v_formatted \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 8 \
    --dataset_num_proc 1 \
    --output_dir dpo_idefics_rlaif-v \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --use_peft \
    --lora_target_modules down_proj, o_proj, k_proj, q_proj, gate_proj, up_proj, v_proj \
    --loss_type sigmoid bco_pair sft \
    --loss_weights 0.8 0.2 1.0 \
    --bf16 True
"""

import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Model & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        trust_remote_code=model_args.trust_remote_code,
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForVision2Seq.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )
    else:
        ref_model = None
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    ################
    # Dataset
    ################
    dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        streaming=script_args.dataset_streaming,
    )
    train_dataset = dataset[script_args.dataset_train_split]
    test_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None

    def ensure_rgb(example):
        # Convert the image to RGB if it's not already
        image = example["images"][0]
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            example["images"] = [image]
        return example

    # Apply the transformation to the dataset (change num_proc depending on the available compute)
    train_dataset = train_dataset.map(ensure_rgb, num_proc=training_args.dataset_num_proc)
    if test_dataset is not None:
        test_dataset = test_dataset.map(ensure_rgb, num_proc=training_args.dataset_num_proc)

    ################
    # Training
    ################
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
