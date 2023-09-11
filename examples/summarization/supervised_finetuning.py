import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    logging,
    set_seed,
)

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset


tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="EleutherAI/pythia-6.9b-deduped", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="CarperAI/openai_summarize_tldr", metadata={"help": "the dataset name"}
    )
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    streaming: Optional[bool] = field(default=False, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})

    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine")
    num_warmup_steps: Optional[int] = field(default=100)
    weight_decay: Optional[float] = field(default=0.05)
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    per_device_train_batch_size: Optional[int] = field(
        default=16, metadata={"help": "the per device train batch size"}
    )
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})

    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    # use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    bf16: Optional[bool] = field(default=True)

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the number of logging steps"})
    eval_steps: Optional[int] = field(default=1000, metadata={"help": "the number of logging steps"})
    save_steps: Optional[int] = field(default=10000, metadata={"help": "the number of logging steps"})
    seed: Optional[int] = field(default=0)
    just_eval: Optional[bool] = field(default=False, metadata={"help": "whether to use gradient checkpointing"})


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def prepare_sample_text(example):
    return example["prompt"] + example["label"]


def create_datasets(tokenizer, args):
    train_data = load_dataset(
        args.dataset_name,
        split="train",
        streaming=args.streaming,
    )

    if args.streaming:
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the train dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )

    # trlx does this, taking only 2000
    valid_data = load_dataset(
        args.dataset_name,
        split="valid[:2000]",
    )

    chars_per_token = chars_token_ratio(valid_data, tokenizer)

    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading the model")
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
    else:
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        token=True,
    )
    model.config.use_cache = False

    print("Loading dataset")
    print(args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name if args.tokenizer_name is None else args.tokenizer_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset, eval_dataset = create_datasets(tokenizer, args)

    # training_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     per_device_train_batch_size=args.per_device_train_batch_size,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     per_device_eval_batch_size=args.per_device_eval_batch_size,
    #     learning_rate=args.learning_rate,
    #     logging_steps=args.logging_steps,
    #     max_steps=args.max_steps,
    #     report_to=args.log_with,
    #     eval_steps=args.eval_steps,
    #     save_steps=args.save_steps,
    #     lr_scheduler_type=args.lr_scheduler_type,
    #     warmup_steps=args.num_warmup_steps,
    #     optim=args.optimizer_type,
    #     bf16=True,
    #     remove_unused_columns=False,
    #     run_name="sft_llama2",
    # )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        # num_train_epochs=args.num_train_epochs,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        report_to=args.log_with,
        optim=args.optimizer_type,
        remove_unused_columns=False,
        disable_tqdm=False,
        # find_unused_params is necessary for grad checkpointing
        ddp_find_unused_parameters=(not args.gradient_checkpointing),
    )

    if args.use_peft:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["query_key_value"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    train_dataset.start_iteration = 0

    print("Starting main loop")

    # TODO maybe switch to DataCollatorForCompletionOnlyLM
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        max_seq_length=args.seq_length,
        packing=True,
    )

    if args.use_peft:
        trainer.model.print_trainable_parameters()

    if args.just_eval:
        trainer.evaluate()
    else:
        print("Training...")
        trainer.train()

        print("Saving last checkpoint of the model")
        trainer.save_model(args.output_dir)

        output_dir = os.path.join(args.output_dir, "final_checkpoint")
        trainer.model.save_pretrained(output_dir)

        # Free memory for merging weights
        del model
        torch.cuda.empty_cache()

        model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()

        output_merged_dir = os.path.join(args.output_dir, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, safe_serialization=True)
