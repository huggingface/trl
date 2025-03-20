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
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
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

import torch

system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""


def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                # Get the image and convert to RGB
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs



def main():
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # Load dataset
    '''
    train_dataset = load_dataset("HuggingFaceM4/ChartQA", split="train")

    def format_data(sample):
      return [
          {
              "role": "system",
              "content": [{"type": "text", "text": system_message}],
          },
          {
              "role": "user",
              "content": [
                  {
                      "type": "image",
                      "image": sample["image"],
                  },
                  {
                      "type": "text",
                      "text": sample["query"],
                  },
              ],
          },
          {
              "role": "assistant",
              "content": [{"type": "text", "text": sample["label"][0]}],
          },
      ]

    train_dataset = [format_data(sample) for sample in train_dataset]
    '''

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model
    model_id = "google/gemma-3-4b-it"
    #model = AutoModelForImageTextToText.from_pretrained(model_id, attn_implementation="eager")
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16, quantization_config=bnb_config)
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

    
    # Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    # Apply PEFT model adaptation
    #peft_model = get_peft_model(model, peft_config)

    # Train model
    training_args = SFTConfig(
        output_dir="gemma-3-4b-instruct-trl-sft-ChartQA",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=True,
        report_to="tensorboard",
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    training_args.remove_unused_columns = False # important for collator

    def collate_fn(examples):
      print(examples)
      # Get the texts and images, and apply the chat template
      #texts = [
      #    processor.apply_chat_template(example, tokenize=False) for example in examples
      #]  # Prepare texts for processing
      texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
      #images = [example["images"] for example in examples]
      images = [img.convert("RGB") if img.mode == "RGBA" else img for example in examples for img in example["images"]]
      #image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

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

    '''
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        #peft_config=peft_config,
        peft_config=get_peft_config(model_args),
        processing_class=processor,
        data_collator=collate_fn,
    )
    '''

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

    # Push to hub
    trainer.push_to_hub(dataset_name="open-r1/codeforces-cots")


if __name__ == "__main__":
    main()