"""python train-trl.py --dataset_name gs://osmos-production-us-south1-ml-training/autoclean_preference_dataset/huggingface_dataset --mode 1 --bf16 --optim adafactor --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --max_steps 20 --run_name_for_checkpoints orpo1 --gradient_checkpointing False --learning_rate 8e-6 --group_by_length False --num_dataset_proc_workers 1 --logging_steps 10 --eval_steps 10 --gradient_accumulation_steps 1
"""

from enum import Enum
import os
import platform
import torch
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, load_from_disk
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
    BitsAndBytesConfig,
    set_seed,
)
from loguru import logger
from trl import ORPOTrainer, ORPOConfig, DPOConfig, DPOTrainer, SFTTrainer, SFTConfig

import multiprocessing

import torch_xla.runtime as xr

seed = 42
set_seed(seed)


import random
random.seed(42)  
orpo_toy_dataset_dict = {
    "prompt": ["T" * random.randint(50, 5000) for _ in range(300)],
    "chosen": ["T" * random.randint(50, 5000) for _ in range(300)],
    "rejected": ["T" * random.randint(50, 5000) for _ in range(300)],
}


# enum for modes - orpo, dpo, orpo-dpo
class Mode(Enum):
    ORPO = 1
    DPO = 2
    SFT = 3


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"}
    )

    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    eval_accumulation_steps: Optional[int] = field(default=10)
    learning_rate: Optional[float] = field(default=2e-5)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=1024)
    max_prompt_length: Optional[int] = field(default=512)
    max_completion_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use. Use 'orpofakedataset' if you want to use the toy dataset."},
    )
    num_dataset_proc_workers: Optional[int] = field(
        default=multiprocessing.cpu_count(),
        metadata={"help": "Number of workers to use for dataset processing."},
    )
    output_dir: Optional[str] = field(
        default=".",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    use_bitsnbytes: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adafactor",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    max_steps: int = field(
        default=-1, metadata={"help": "How many optimizer update steps to take"}
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    group_by_length: bool = field(
        default=False,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(
        default=100, metadata={"help": "Save checkpoint every X updates steps."}
    )
    logging_steps: int = field(
        default=1, metadata={"help": "Log every X updates steps."}
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "epochs or steps"},
    )
    eval_steps: int = field(
        default=0.1,
        metadata={
            "help": "Number of update steps between two evaluations. Should be an integer or a float in range [0,1). If smaller than 1, will be interpreted as ratio of total training steps."
        },
    )
    run_name_for_checkpoints: Optional[str] = field(
        default="",
        metadata={"help": "Folder to write checkpoints and TF Events to."},
    )
    mode: Mode = field(
        default=Mode.ORPO,
        metadata={"help": "Mode of training"},
    )
    xla_cache_dir: str = field(
        default="/xla_cache",
        metadata={"help": "director for storing xla compilation cache"}, 
    )


def configure_for_tpu():
    global AutoModelForCausalLM
    global fsdp_v2
    # from optimum.tpu import AutoModelForCausalLM, fsdp_v2
    from optimum.tpu import fsdp_v2

    from transformers import AutoModelForCausalLM

    fsdp_v2.use_fsdp_v2()

    return "tpu"


def configure_for_gpu():
    global AutoModelForCausalLM
    from transformers import AutoModelForCausalLM

    # fsdp_training_args = None
    # return ("cuda", fsdp_training_args)
    return "cuda"


def configure_for_apple_silicon():
    global AutoModelForCausalLM
    from transformers import AutoModelForCausalLM

    # fsdp_training_args = None
    # return ("mps", fsdp_training_args)
    return "mps"


def configure_for_accelerators():
    if torch.backends.mps.is_available():
        logger.info("Configuring for Apple Silicon")
        return configure_for_apple_silicon()
    # check if cuda is available
    if torch.cuda.is_available():
        logger.info("Configuring for GPU")
        return configure_for_gpu()
    else:
        logger.info("Configuring for TPU")
        return configure_for_tpu()


def create_and_prepare_model(args, device):
    # set torch dtype
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    if args.use_bitsnbytes == True:
        print("Using bitsnbytes")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=not args.use_4bit,
            load_in_4bit=args.use_4bit,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )
    else:
        bnb_config = None

    if torch.cuda.is_available() and compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print(
                "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
            )
            print("=" * 80)

    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        # token=os.environ["HF_TOKEN"],
        # trust_remote_code=False,
        torch_dtype=torch_dtype,
        use_cache=True
    )

    # model.to(device)
    # model = model.eval()

    if torch.backends.mps.is_available():
        model = model.to("mps")
        print("Using MPS")

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        # trust_remote_code=False,
        # token=os.environ["HF_TOKEN"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return (
        model,
        peft_config,
        tokenizer,
    )


def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples["prompt"])):
        text = f"{examples['prompt'][i]}{examples['chosen'][i]}"
        output_texts.append(text)
    return output_texts


def get_orpo_trainer(training_arguments, model, dataset, peft_config, tokenizer):
    orpo_config = ORPOConfig(
        max_length=script_args.max_seq_length,
        max_prompt_length=script_args.max_prompt_length,
        max_completion_length=script_args.max_completion_length, 
        is_encoder_decoder=False,
        dataset_num_proc=script_args.num_dataset_proc_workers,
        **training_arguments.to_dict(),
    )

    return ORPOTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=orpo_config,
        # compute_metrics=compute_metrics,
    )


def get_dpo_trainer(training_arguments, model, dataset, peft_config, tokenizer):
    dpo_config = DPOConfig(
        max_length=script_args.max_seq_length,
        max_prompt_length=script_args.max_prompt_length,
        is_encoder_decoder=False,
        dataset_num_proc=script_args.num_dataset_proc_workers,
        **training_arguments.to_dict(),
    )

    return DPOTrainer(
        model,
        None,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=dpo_config,
        # compute_metrics=compute_metrics,
    )


def get_sft_trainer(training_arguments, model, dataset, peft_config, tokenizer):
    sft_config = SFTConfig(
        dataset_text_field="prompt",
        max_seq_length=script_args.max_seq_length,
        dataset_num_proc=script_args.num_dataset_proc_workers,
        formatting_func=formatting_prompts_func,
        **training_arguments.to_dict(),
    )

    return SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=sft_config,
        # compute_metrics=compute_metrics,
    )


if __name__ == "__main__":
    

    device = configure_for_accelerators()

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    print(script_args)

    xr.initialize_cache(script_args.xla_cache_dir, readonly=False)

    output_dir = script_args.output_dir + (
        "/results"
        if script_args.run_name_for_checkpoints == ""
        else f"/results/{script_args.run_name_for_checkpoints}"
    )

    if torch.backends.mps.is_available():
        half_precision_backend = "cpu_amp"
        print("Using MPS, using CPU AMP")
    else:
        half_precision_backend = None

    print("Loading model")
    model, peft_config, tokenizer = create_and_prepare_model(script_args, device)
    if device == "tpu":
        cls_to_wrap = "LlamaDecoderLayer"
        fsdp_training_args = {
            "fsdp": "full_shard",
            "fsdp_config": fsdp_v2.get_fsdp_config(cls_to_wrap),
        }
        tokenizer.pad_token = tokenizer.eos_token

    else:
        fsdp_training_args = {}

    training_arguments = TrainingArguments(
        output_dir,
        overwrite_output_dir=False,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optim=script_args.optim,
        save_steps=script_args.save_steps,
        logging_steps=script_args.logging_steps,
        learning_rate=script_args.learning_rate,
        fp16=script_args.fp16,
        bf16=script_args.bf16,
        max_steps=script_args.max_steps,
        warmup_ratio=script_args.warmup_ratio,
        group_by_length=script_args.group_by_length,
        lr_scheduler_type=script_args.lr_scheduler_type,
        gradient_checkpointing=script_args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=script_args.num_dataset_proc_workers,
        remove_unused_columns=False,
        num_train_epochs=script_args.num_train_epochs,
        eval_strategy=script_args.eval_strategy,
        eval_steps=script_args.eval_steps,
        eval_accumulation_steps=script_args.eval_accumulation_steps,
        resume_from_checkpoint=True,
        half_precision_backend=half_precision_backend,
        dataloader_drop_last=True,
        **fsdp_training_args,
    )

    if script_args.dataset_name == "orpofakedataset":
        dataset = Dataset.from_dict(orpo_toy_dataset_dict)
    else:
        dataset = load_dataset(script_args.dataset_name)

    dataset = dataset.train_test_split(test_size=0.1)

    if script_args.mode == Mode.ORPO.value:
        trainer = get_orpo_trainer(
            training_arguments,
            model,
            dataset,
            peft_config,
            tokenizer,
        )
    elif script_args.mode == Mode.DPO.value:
        trainer = get_dpo_trainer(
            training_arguments,
            model,
            dataset,
            peft_config,
            tokenizer,
        )
    elif script_args.mode == Mode.SFT.value:
        trainer = get_sft_trainer(
            training_arguments,
            model,
            dataset,
            peft_config,
            tokenizer,
        )

    print("Starting Training")
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    print("Training complete")
    lora_output_dir = output_dir + "/lora_model"
    print("Saving LORA model to: ", lora_output_dir)
    trainer.save_model(lora_output_dir)

    print("Merging LORA")
    merged_output_dir = output_dir + "/merged_model"
    model = trainer.model.merge_and_unload()
    print("Saving merged model to: ", merged_output_dir)

    is_main_process = (trainer.accelerator.is_main_process,)
    save_function = (trainer.accelerator.save,)

    model.save_pretrained(
        merged_output_dir, is_main_process=is_main_process, save_function=save_function
    )
    tokenizer.save_pretrained(merged_output_dir)
    tokenizer._tokenizer.model.save(merged_output_dir)