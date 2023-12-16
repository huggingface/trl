import os
import shutil
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForSequenceClassification, PeftConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)


shutil.disk_usage = lambda x: shutil._ntuple_diskusage(1, 1, 1)


@dataclass
class ScriptArguments:
    output_dir: Optional[str] = field(
        default="/home/toolkit/huggingface/openai_summarize_tldr_reward",
        metadata={"help": "output folder"},
    )
    model_name: Optional[str] = field(
        default="mnoukhov/pythia410m-tldr-sft-rm-adapter", metadata={"help": "the model name"}
    )
    new_column_name: Optional[str] = field(default="reward_baseline")
    dataset_name: Optional[str] = field(
        default="mnoukhov/openai_summarize_comparisons_tldrprompt", metadata={"help": "the dataset name"}
    )
    max_length: Optional[int] = field(default=560, metadata={"help": "maximum length for generation"})
    train_split: Optional[str] = field(default="train[:20]", metadata={"help": "the dataset name"})
    eval_split: Optional[str] = field(default=None, metadata={"help": "the dataset name"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    batch_size: Optional[int] = field(default=4)
    bf16: Optional[bool] = field(default=False)
    fp16: Optional[bool] = field(default=False)
    fp16_model: Optional[bool] = field(default=False)


def create_and_prepare_model(args):
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
        device_map = {"": Accelerator().local_process_index}
    else:
        device_map = None
        quantization_config = None

    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16_model:
        torch_dtype = torch.float16
    else:
        torch_dtype = None

    if "adapter" in args.model_name:
        model_cls = AutoPeftModelForSequenceClassification
        config = PeftConfig.from_pretrained(args.model_name)
        tokenizer_name = config.base_model_name_or_path
    else:
        model_cls = AutoModelForSequenceClassification
        tokenizer_name = args.model_name

    model = model_cls.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        num_labels=1,
        torch_dtype=torch_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def create_and_prepare_dataset(args, tokenizer, split, num_proc=2):
    dataset = load_dataset(args.dataset_name, split=split)

    def combine_and_tokenize(examples):
        if isinstance(examples["label"], str):
            texts = examples["prompt"] + examples["label"]
        else:
            texts = [prompt + label for prompt, label in zip(examples["prompt"], examples["label"])]

        return tokenizer(texts, truncation=True, padding=False, max_length=args.max_length)

    original_columns = dataset["train"].column_names

    dataset = dataset.map(
        combine_and_tokenize,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

    dataset.set_format("torch")
    return dataset


def strip_prompt(examples):
    examples["prompt"] = [prompt.strip() for prompt in examples["prompt"]]

    return examples


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model, tokenizer = create_and_prepare_model(script_args)

training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_eval_batch_size=script_args.batch_size,
    bf16=script_args.bf16,
    fp16=script_args.fp16,
)

if script_args.fp16:
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
else:
    data_collator = None

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

data_splits = {
    "train": script_args.train_split,
    "valid": script_args.eval_split,
}

original_datasets = create_and_prepare_dataset(script_args, tokenizer, split=data_splits)

augmented_dataset = load_dataset(script_args.dataset_name, split=data_splits)
augmented_dataset = augmented_dataset.map(strip_prompt, batched=True)

for key, dataset in original_datasets.items():
    preds = trainer.predict(dataset)
    reward_preds = preds[0].flatten()

    if trainer.accelerator.is_local_main_process:
        augmented_dataset[key] = augmented_dataset[key].add_column(script_args.new_column_name, reward_preds)

trainer.accelerator.wait_for_everyone()
if trainer.accelerator.is_main_process:
    # augmented_dataset.save_to_disk(script_args.output_dir)
    augmented_dataset.push_to_hub(os.path.basename(script_args.output_dir))
# trainer.accelerator.free_memro()
# if trainer.accelerator.is_local_main_process:
# trainer.model = gold_model
# trainer = Trainer(
#     model=gold_model,
#     args=training_args,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
# )

# original_datasets = create_and_prepare_dataset(script_args, tokenizer, split=data_splits)
# if trainer.accelerator.is_local_main_process:
#     import pdb
#
#     pdb.set_trace()

# for key, dataset in original_datasets.items():
#     preds = trainer.predict(dataset)
#     gold_reward_preds = preds[0].flatten()
#
#     if trainer.accelerator.is_local_main_process:
#         augmented_dataset[key] = augmented_dataset[key].add_column("gold_reward_baseline", gold_reward_preds)
