import shutil
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, DatasetInfo, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)


shutil.disk_usage = lambda x: shutil._ntuple_diskusage(1, 1, 1)


@dataclass
class ScriptArguments:
    output_dir: Optional[str] = field(
        default="/home/toolkit/huggingface/openai_summarize_comparison_pseudolabel",
        metadata={"help": "output folder"},
    )
    model_name: Optional[str] = field(default="EleutherAI/pythia-6.9b-deduped", metadata={"help": "the model name"})
    # tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(
        default="CarperAI/openai_summarize_comparisons", metadata={"help": "the dataset name"}
    )
    train_split: Optional[str] = field(default="train[:20]", metadata={"help": "the dataset name"})
    eval_split: Optional[str] = field(default="test[:20]", metadata={"help": "the dataset name"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    better_transformer: Optional[bool] = field(default=False)
    flash_attention: Optional[bool] = field(default=False)
    batch_size: Optional[int] = field(default=4)
    bf16: Optional[bool] = field(default=False)
    fp16: Optional[bool] = field(default=False)
    fp16_model: Optional[bool] = field(default=False)
    seq_length: Optional[int] = field(default=560, metadata={"help": "Input sequence length"})


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

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        num_labels=1,
        torch_dtype=torch_dtype,
    )

    if args.better_transformer:
        model.to_bettertransformer()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def preprocess_function(examples):
    str_chosen = []
    str_rejected = []

    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        str_chosen.append(prompt + "\n" + chosen)
        str_rejected.append(prompt + "\n" + rejected)

    tokenized_chosen = tokenizer(
        str_chosen, padding="max_length", truncation=True, max_length=script_args.seq_length, return_tensors="pt"
    )
    tokenized_rejected = tokenizer(
        str_rejected, padding="max_length", truncation=True, max_length=script_args.seq_length, return_tensors="pt"
    )

    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"],
    }


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model, tokenizer = create_and_prepare_model(script_args)
accelerator = Accelerator()

data_splits = [split for split in [script_args.train_split, script_args.eval_split] if split is not None]
relabel_dataset = DatasetDict()

for split in data_splits:
    dataset = load_dataset(script_args.dataset_name, split=split)

    dataloader = DataLoader(dataset, batch_size=script_args.batch_size)

    model, dataloader = accelerator.prepare(model, dataloader)

    model.eval()

    output_dataset = {"prompt": [], "chosen": [], "rejected": []}

    for examples in tqdm(dataloader):
        inputs = preprocess_function(examples)
        with torch.no_grad():
            # if script_args.flash_attention:
            #     with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            #         output = model(
            #             batch["input_ids"],
            #             attention_mask=batch["attention_mask"],
            #         )

            rewards_chosen = model(
                input_ids=inputs["input_ids_chosen"].to(accelerator.device),
                attention_mask=inputs["attention_mask_chosen"].to(accelerator.device),
            )[0]
            rewards_rejected = model(
                input_ids=inputs["input_ids_rejected"].to(accelerator.device),
                attention_mask=inputs["attention_mask_rejected"].to(accelerator.device),
            )[0]

            pseudolabels = torch.sign(rewards_chosen - rewards_rejected)

            pseudolabels = accelerator.gather(pseudolabels).cpu().numpy()

            for prompt, init_chosen, init_rejected, label in zip(
                examples["prompt"], examples["chosen"], examples["rejected"], pseudolabels
            ):
                output_dataset["prompt"].append(prompt)
                if label >= 0:
                    output_dataset["chosen"].append(init_chosen)
                    output_dataset["rejected"].append(init_rejected)
                else:
                    output_dataset["chosen"].append(init_rejected)
                    output_dataset["rejected"].append(init_chosen)

    ds_info = DatasetInfo("CarperAI/openai_summarize_comparisons relabelled with a finetuned Pythia 6.9B")
    relabel_dataset[split] = Dataset.from_dict(output_dataset, split=split, info=ds_info)

relabel_dataset.save_to_disk(script_args.output_dir)
