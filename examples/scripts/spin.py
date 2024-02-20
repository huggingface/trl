# coding=utf-8
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
#
# Adapted from https://github.com/uclaml/SPIN/blob/main/spin/run_spin.py
"""
# zero-3:
accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml examples/scripts/spin.py \
    --model_name_or_path="alignment-handbook/zephyr-7b-sft-full" \
    --torch_dtype=bfloat16 \
    --report_to="wandb" \
    --learning_rate=5.0e-7 \
    --beta=0.1 \
    --bf16=true \
    --per_device_train_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --output_dir="spin-model" \
    --logging_steps=10 \
    --num_train_epochs=2 \
    --gradient_checkpointing \
    --max_length=1024 \
    --max_prompt_length=512 \
    --warmup_ratio=0.1 \
    --attn_implementation=flash_attention_2 \
    --push_to_hub=true \
    --hub_model_id="lewtun/zephyr-spin-iter0-v0" \
    --hub_private_repo=true \
    --max_steps=-1
    """
import re
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig, HfArgumentParser, set_seed

from trl import ModelConfig, SPINConfig, SPINTrainer, get_kbit_device_map, get_peft_config, get_quantization_config


def apply_chat_template(example, tokenizer, task, assistant_prefix="<|assistant|>\n"):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if all(k in example.keys() for k in ("real", "generated")):
        # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
        prompt_messages = [[msg for msg in example["real"] if msg["role"] == "user"][0]]
        # Insert system message
        if example["real"][0]["role"] != "system":
            prompt_messages.insert(0, {"role": "system", "content": ""})
        else:
            prompt_messages.insert(0, example["real"][0])

        real_messages = example["real"][1:]
        generated_messages = example["generated"][1:]
        example["text_real"] = tokenizer.apply_chat_template(real_messages, tokenize=False)
        example["text_generated"] = tokenizer.apply_chat_template(generated_messages, tokenize=False)
        example["text_prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        example["text_real"] = _strip_prefix(example["text_real"], assistant_prefix)
        example["text_generated"] = _strip_prefix(example["text_generated"], assistant_prefix)
    else:
        raise ValueError(f"Require `[real, generated]` keys but found {list(example.keys())}")
    return example


@dataclass
class ScriptArguments:
    dataset_name: str = field(default="HuggingFaceH4/spin-ultrachat-prompts", metadata={"help": "the dataset name"})
    max_seq_length: int = field(default=512, metadata={"help": "The maximum sequence length for SFT Trainer"})
    preprocessing_num_workers: int = field(
        default=12, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    temperature: float = field(default=1.0, metadata={"help": "The temperature for text generation"})
    max_new_tokens: int = field(default=64, metadata={"help": "The maximum number of tokens to generate"})
    do_generate: bool = field(default=False)


def main():
    parser = HfArgumentParser((ModelConfig, ScriptArguments, SPINConfig))
    model_args, script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=True)

    print(f"Model args: {model_args}\n\n")
    print(f"Script args: {script_args}\n\n")
    print(f"Training args: {training_args}\n\n")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = load_dataset(script_args.dataset_name)
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, revision=model_args.model_revision, truncation_side="left"
    )

    #####################
    # Apply chat template
    #####################
    processed_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "spin"},
        num_proc=script_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_real -> real and text_generated -> generated
    for split in ["train", "test"]:
        processed_datasets[split] = processed_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_real": "real", "text_generated": "generated"}
        )

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    #########################
    # Instantiate spin trainer
    #########################
    spin_trainer = SPINTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
    )

    if script_args.do_generate:
        prompt_train_ds = raw_datasets["train"].select(range(9))
        prompt_test_ds = raw_datasets["test"].select(range(3))
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=script_args.temperature,
            max_new_tokens=script_args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        # text_generation_callback = TextGenerationCallback(
        #     prompt_dataset=prompt_ds,
        #     prompt_column="prompt",
        #     generation_config=generation_config,
        # )
        # spin_trainer.add_callback(text_generation_callback)

    ###############
    # Training loop
    ###############
    if training_args.do_train:
        train_result = spin_trainer.train()
        metrics = train_result.metrics
        spin_trainer.log_metrics("train", metrics)
        spin_trainer.save_metrics("train", metrics)
        spin_trainer.save_state()
        spin_trainer.save_model(training_args.output_dir)
        # spin_trainer.push_to_hub()

    if script_args.do_generate:
        train_completions = spin_trainer.generate(prompt_train_ds, "prompt", generation_config)
        test_completions = spin_trainer.generate(prompt_test_ds, "prompt", generation_config)
        assert len(train_completions) == len(prompt_train_ds)
        assert len(test_completions) == len(prompt_test_ds)

    if spin_trainer.accelerator.is_main_process:

        def add_completion(x, completions):
            x["generated"] = [x["prompt"], {"role": "assistant", "content": completions.pop(0)}]
            return x

        train_ds = prompt_train_ds.map(add_completion, fn_kwargs={"completions": train_completions})
        test_ds = prompt_test_ds.map(add_completion, fn_kwargs={"completions": test_completions})
        train_ds.push_to_hub("HuggingFaceH4/spin-ultrachat-iter0", split="train", private=True)
        test_ds.push_to_hub("HuggingFaceH4/spin-ultrachat-iter0", split="test", private=True)


if __name__ == "__main__":
    main()
