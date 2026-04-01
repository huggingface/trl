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
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
# Full training
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```

# LoRA
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```
"""
from transformers import TrainerCallback
import argparse
import os

from accelerate import logging
from datasets import load_dataset,concatenate_datasets,DatasetDict
from transformers import AutoConfig, AutoModelForCausalLM,AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")
# def merge_dataset(dataset):
#     def process_split(split):
#         original = dataset[split]
#         chosen = original.map(lambda x:{"completion":x["chosen"]},remove_columns=original.column_names)
#         rejected = original.map(lambda x:{"text":x["rejected"]},remove_columns = original.column_names)
#         merged = concatenate_datasets([chosen,rejected]) 
#         return merged
    
#     return DatasetDict({
#         split: process_split(split)
#         for split in dataset.keys()
#     })

# def preprocess(dataset):
    
#     def process(ex):
#         prompt = ex["prompt"]
#         completion = ex["chosen"][-1]["content"]
#         return {
#             "prompt" : prompt,
#             "completion": completion
#         }

#     def process_split(split):
#         origin = dataset[split]
#         origin = origin.map(process,remove_columns = origin.column_names)
#         return origin

#     return DatasetDict({
#         split: process_split(split)
#         for split in dataset.keys()
#     })
    
class PeekCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts):
        self.tokenizer = tokenizer
        self.prompts = prompts



    def on_log(self, args, state, control, model, **kwargs):
        # 仅在主进程打印
        if state.is_world_process_zero:
            model.eval()
            print(f"\n\033[33m[Step {state.global_step}] 中途采样调试:\033[0m")
            for p in self.prompts:
                # 注入 Qwen 模板
                messages = [{"role": "user", "content": p}]
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(model.device)
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.12,
                    no_repeat_ngram_size=4,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                response = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
                print(f"Q: {p}\nA: {response}\n" + "-"*30)
            
            model.train()



    

def main(script_args, training_args, model_args, dataset_args):
    import trl.trainer.sft_trainer as sft_trainer_module

    print(f"DEBUG SFTTrainer loaded from: {sft_trainer_module.__file__}", flush=True)

    ################
    # Model init kwargs
    ################
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    # Create model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Load the dataset
    if dataset_args.datasets and script_args.dataset_name:
        logger.warning(
            "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    # dataset = preprocess(dataset)
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Keep generation config aligned with tokenizer ids.
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    
    def preprocess_dataset(dataset):
        def process(example):
            def _extract_ids(x):
                if isinstance(x, dict):
                    return x.get("input_ids", [])
                return x

            prompt = example["chosen"][:-1]
            completion = [example["chosen"][-1]]
            prompt_ids = _extract_ids(tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=True,
                max_length=10000,
                truncation=True,
            ))

            input_ids = _extract_ids(tokenizer.apply_chat_template(
                prompt + completion,
                tokenize=True,
                max_length=10000,
                truncation=True,
            ))[:-1]

            completion_mask = [0] * len(prompt_ids) + [1] * (len(input_ids) - len(prompt_ids))


            if len(input_ids) == 0:
                return {"input_ids": [], "completion_mask": []}

            prompt_len = min(len(prompt_ids), len(input_ids))
            completion_mask = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)
            return {"input_ids": input_ids, "completion_mask": completion_mask}

        dataset = dataset.map(process, num_proc=4)
        dataset = dataset.select_columns(["input_ids","completion_mask"])
        dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0, num_proc=4)
        return dataset

            
            


    # 显式转换
    dataset = preprocess_dataset(dataset)
    for i in range(5):
        if i < len(dataset["train"]):
            print(f"数据{i}:{dataset['train'][i]}\n")




    # callback
    test_prompts = [
        "国庆节的日期是什么时候",
        "50字介绍监督微调"
    ]
    peek_callback = PeekCallback(tokenizer=tokenizer, prompts=test_prompts)

    # Initialize the SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=get_peft_config(model_args),
        callbacks=[peek_callback],

        
    )
    
    # Train the model
    trainer.train()

    # Log training complete
    trainer.accelerator.print("✅ Training completed.")

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: argparse._SubParsersAction | None = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
