"""
Run the BCO training script with the commands below. In general, the optimal configuration for BCO will be similar to that of KTO.

# Full training:
python examples/scripts/bco.py \
    --model_name_or_path=nnheui/stablelm-2-1_6b-sft-full \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --logging_steps 0.01 \
    --eval_steps 0.2 \
    --save_strategy no \
    --output_dir=bco-aligned-model \
    --logging_first_step \
    --max_length 2048 \
    --max_prompt_length 1536 \
    --max_completion_length 1024 \
    --no_remove_unused_columns \
    --warmup_ratio 0.1 \
    --bf16 \
    --loss_type bco \
    --report_to wandb

# QLoRA:
python examples/scripts/bco.py \
    --model_name_or_path=nnheui/stablelm-2-1_6b-sft-full \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --logging_steps 0.01 \
    --eval_steps 0.2 \
    --save_strategy no \
    --output_dir=bco-aligned-model-lora \
    --logging_first_step \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --max_length 2048 \
    --max_prompt_length 1536 \
    --max_completion_length 1024 \
    --no_remove_unused_columns \
    --warmup_ratio 0.1 \
    --bf16 \
    --loss_type bco \
    --use_peft \
    --load_in_4bit \
    --lora_target_modules=all-linear \
    --lora_r=16 \
    --lora_alpha=16
"""

import logging
from typing import Literal
from dataclasses import dataclass

from accelerate import PartialState
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, setup_chat_format


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the KTO training script.
    """

    llm_name: Literal["gpt-3.5-turbo", "llama-2-7b-chat", "llama-2-70b-chat"] = "gpt-3.5-turbo"


def build_helpfulness_dataset(llm_name: str) -> Dataset:
    """
    Filter `llm_name` completions and binarize given their helpfulness score.
    If helpfulness score is 5, it is desirable. Otherwise, it is undesirable.
    """
    def get_model_rating(example, metric: str, llm_name: str):
        try:
            model_index = example["models"].index(llm_name)
            return {metric: int(example["completions"][model_index]["annotations"][metric]["Rating"])}
        except ValueError as e:
            logging.warning(e)
            return -1

    def get_model_response(example, llm_name: str):
        try:
            model_index = example["models"].index(llm_name)
            return {"response": example["completions"][model_index]["response"]}
        except ValueError as e:
            logging.warning(e)
            return -1

    dataset = load_dataset("openbmb/UltraFeedback")["train"]

    ds = dataset.filter(lambda example: llm_name in example["models"], batched=False, num_proc=8)
    ds = ds.filter(lambda example: len(example["models"]) == len(example["completions"]), batched=False, num_proc=8)

    METRIC = "helpfulness"

    ds = ds.map(
        get_model_rating,
        batched=False,
        num_proc=8,
        fn_kwargs={"metric": METRIC, "llm_name": llm_name},
    )

    ds = ds.map(
        get_model_response,
        batched=False,
        num_proc=8,
        fn_kwargs={"llm_name": llm_name},
    )

    ds = ds.select_columns(["source", "instruction", "response", "helpfulness"])

    ds = ds.rename_columns({"instruction": "prompt", "response": "completion"})
    ds = ds.map(lambda example: {"label": example["helpfulness"] >= 5}, batched=False, num_proc=8)

    ds = ds.map(
        lambda example: {"prompt": [{"role": "user", "content": example["prompt"]}]},
        batched=False,
        num_proc=8,
    )
    dataset = ds.train_test_split(test_size=0.05, seed=42)

    return dataset


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    script_args, kto_args, model_args = parser.parse_args_into_dataclasses()

    kto_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    model_ref = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # If we are aligning a base model, we use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # Load the dataset
    dataset = build_helpfulness_dataset(script_args.llm_name)

    # Apply chat template
    def format_dataset(example):
        example["prompt"] = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)
        return example

    with PartialState().local_main_process_first():
        formatted_dataset = dataset.map(format_dataset, batched=False, num_proc=8)

    # Initialize the KTO trainer
    kto_trainer = KTOTrainer(
        model,
        model_ref,
        args=kto_args,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["test"],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    kto_trainer.train()
    kto_trainer.save_model(kto_args.output_dir)
