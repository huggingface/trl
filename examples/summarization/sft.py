from dataclasses import dataclass, field

import torch
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from tqdm.rich import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import ModelConfig, SFTTrainer


tqdm.pandas()


def get_kbit_device_map():
    # if is_xpu_available():
    #     return {"": f"xpu:{PartialState().local_process_index}"}
    if torch.cuda.is_available():
        return {"": PartialState().local_process_index}
    else:
        return None


def get_peft_config(model_config: ModelConfig):
    if model_config.use_peft is False:
        return None

    target_modules = model_config.lora_target_modules if model_config.lora_target_modules is not None else "all-linear"

    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        bias="none",
        task_type=model_config.lora_task_type,
        target_modules=target_modules,
        modules_to_save=model_config.lora_modules_to_save,
    )

    return peft_config


def get_quantization_config(model_config: ModelConfig):
    if model_config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_config.torch_dtype,  # For consistency with model weights, we use the same value as `torch_dtype`
            bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_config.use_bnb_nested_quant,
        )
    elif model_config.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def tldr_combine(examples):
    if isinstance(examples["label"], str):
        return examples["prompt"] + examples["label"]
    elif isinstance(examples["label"], list):
        return list(map(str.__add__, examples["prompt"], examples["label"]))
    else:
        raise Exception(f"weird input examples of type {type(examples)}")


@dataclass
class ScriptArguments:
    dataset_name: str = field(default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"})
    dataset_text_field: str = field(default=None, metadata={"help": "the text field of the dataset"})
    dataset_train_name: str = field(default="train", metadata={"help": "the name of the training set of the dataset"})
    dataset_test_name: str = field(default="test", metadata={"help": "the name of the training set of the dataset"})
    max_seq_length: int = field(default=512, metadata={"help": "The maximum sequence length for SFT Trainer"})
    packing: bool = field(default=False, metadata={"help": "Whether to apply data packing or not during training"})
    config: str = field(default=None, metadata={"help": "Path to the optional config file"})
    gradient_checkpointing_use_reentrant: bool = field(
        default=False, metadata={"help": "Whether to apply `use_reentrant` for gradient_checkpointing"}
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_into_dataclasses()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    train_dataset = load_dataset(args.dataset_name, split=args.dataset_train_name)
    eval_dataset = load_dataset(args.dataset_name, split=args.dataset_test_name)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field=args.dataset_text_field,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        packing=args.packing,
        formatting_func=tldr_combine,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
