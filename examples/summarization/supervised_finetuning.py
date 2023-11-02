import os
from dataclasses import dataclass, field
from typing import Optional

import bitsandbytes as bnb
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM, LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPT2Model,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.pytorch_utils import Conv1D
from transformers.trainer_utils import get_last_checkpoint

from trl import DataCollatorForCompletionOnlyLM, SFTTrainer


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
    train_split: Optional[str] = field(
        default="train", metadata={"help": "the dataset split to evaluate on; default to 'none' (no evaluation)"}
    )
    eval_split: Optional[str] = field(
        default="valid[:2000]",
        metadata={"help": "the dataset split to evaluate on; default to 'none' (no evaluation)"},
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
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing"}
    )
    seq_length: Optional[int] = field(default=560, metadata={"help": "Input sequence length"})

    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    lora_all_linear: Optional[bool] = field(default=False, metadata={"help": "lora adapter on all linear layers"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    bf16: Optional[bool] = field(default=True)
    fp16_model: Optional[bool] = field(
        default=False,
        metadata={},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    train_completions: Optional[bool] = field(default=False)
    packing: Optional[bool] = field(default=True)

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the number of logging steps"})
    eval_steps: Optional[int] = field(default=1000, metadata={"help": "the number of steps to eval at"})
    save_steps: Optional[int] = field(default=1000, metadata={"help": "the number of steps to save at"})
    save_strategy: Optional[str] = field(default="steps")
    seed: Optional[int] = field(default=0)
    just_eval: Optional[bool] = field(default=False)
    resume_from_checkpoint: Optional[str] = field(default=None)


def find_all_linear_names(args, model):
    if isinstance(model.transformer, GPT2Model):
        cls = Conv1D
    else:
        cls = (
            bnb.nn.Linear4bit if args.load_in_4bit else (bnb.nn.Linear8bitLt if args.load_in_8bit else torch.nn.Linear)
        )

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    if "score" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("score")

    print(lora_module_names)

    return list(lora_module_names)


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


def prepare_sample_text(examples):
    if isinstance(examples["label"], str):
        return examples["prompt"] + examples["label"]
    elif isinstance(examples["label"], list):
        return list(map(str.__add__, examples["prompt"], examples["label"]))
    else:
        raise Exception(f"weird input examples of type {type(examples)}")


def create_datasets(args):
    train_data = load_dataset(
        args.dataset_name,
        split=args.train_split,
        streaming=args.streaming,
    )

    if args.streaming:
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    valid_data = load_dataset(
        args.dataset_name,
        split=args.eval_split,
    )
    return train_data, valid_data


def create_model(args):
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
    elif args.fp16_model:
        torch_dtype = torch.float16
    else:
        torch_dtype = None

    # n_gpus = torch.cuda.device_count()
    # max_memory = "32000MB"
    # max_memory = {i: max_memory for i in range(n_gpus)}
    if "t5" in args.model_name:
        model_cls = AutoModelForSeq2SeqLM
    else:
        model_cls = AutoModelForCausalLM

    model = model_cls.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        # max_memory=max_memory,
        token=True,
    )
    model.config.torch_dtype = torch_dtype
    model.config.use_cache = False

    print("Loading dataset")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name if args.tokenizer_name is None else args.tokenizer_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = create_model(args)

    train_dataset, eval_dataset = create_datasets(args)

    if args.train_completions:
        data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template="TL;DR:")
    else:
        data_collator = None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        fp16=args.fp16,
        weight_decay=args.weight_decay,
        report_to=args.log_with,
        optim=args.optimizer_type,
        remove_unused_columns=False,
        disable_tqdm=False,
        # find_unused_params is necessary for grad checkpointing
        ddp_find_unused_parameters=(args.gradient_checkpointing),
    )

    if args.use_peft:
        if args.lora_all_linear:
            target_modules = find_all_linear_names(args, model)
        else:
            target_modules = None

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    chars_per_token = chars_token_ratio(train_dataset, tokenizer)
    print(f"The character to token ratio of the train dataset is: {chars_per_token:.2f}")

    print("Starting main loop")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        max_seq_length=args.seq_length,
        formatting_func=prepare_sample_text,
        packing=args.packing,
        chars_per_token=chars_per_token,
        data_collator=data_collator,
    )

    if args.use_peft:
        trainer.model.print_trainable_parameters()

    if not args.just_eval:
        if args.resume_from_checkpoint is not None:
            last_checkpoint = args.resume_from_checkpoint
        else:
            # when job is interrupted and restarted
            last_checkpoint = get_last_checkpoint(args.output_dir)

        print("Training...")
        trainer.train(resume_from_checkpoint=last_checkpoint)

        trainer.evaluate()

        print("Saving last checkpoint of the model")
        output_dir = os.path.join(args.output_dir, "final_model")
        trainer.save_model(output_dir)

        if args.use_peft:
            output_dir = os.path.join(args.output_dir, "final_adapter_checkpoint")
            trainer.model.save_pretrained(output_dir)

            # Free memory for merging weights
            del model
            torch.cuda.empty_cache()

            if "t5" in args.model_name:
                model_cls = AutoPeftModelForSeq2SeqLM
            else:
                model_cls = AutoPeftModelForCausalLM

            model = model_cls.from_pretrained(
                output_dir, device_map="auto", torch_dtype=trainer.model.config.torch_dtype
            )
            model = model.merge_and_unload()

            output_merged_dir = os.path.join(args.output_dir, "final_merged_checkpoint")
            model.save_pretrained(output_merged_dir, safe_serialization=True)

    else:
        results = trainer.evaluate()
        print(results)
