# [WIP]
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
)


@dataclass
class ScriptArguments:
    reward_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "the reward model name"})
    
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the HF data path"})
    save_dataset_path: Optional[str] = field(default=None, metadata={"help": "the HF data path"})
    prompt_column_name: Optional[str] = field(default="prompt")
    generation_column_name: Optional[str] = field(default="generated")
    
    bs: Optional[int] = field(default=16)
    max_length: Optional[int] = field(default=512)
    
    bf16: Optional[bool] = field(default=True if torch.cuda.get_device_capability()[0] == 8 else False, metadata={"help": "whether to use bf16."})
    fp16: Optional[bool] = field(default=True if not torch.cuda.get_device_capability()[0] == 8 else False, metadata={"help": "whether to use fp16."})


def main():

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    accelerator = Accelerator(
        mixed_precision= "bf16" if script_args.bf16 else "fp16" if script_args.fp16 else "no"
    )

    # Load model, reward model and tokenizer
    reward_model = AutoModelForSequenceClassification.from_pretrained(script_args.reward_model_name_or_path)
    reward_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load and extract the prompt in the dataset. We do not need this step if we already have a dataset with prompts separated from the answers.
    reward_dataset = load_dataset(script_args.dataset_name, split="train")

    def preprocess_function(sample):
        def extract_prompt(prompt, generation):
            return prompt + " " + generation

        prompts = []
        for chosen, rejected in zip(sample["prompt"], sample["generated"]):
            prompts.append(extract_prompt(chosen, rejected))
        model_inputs = tokenizer(prompts, max_length=script_args.max_length, truncation=True)

        return model_inputs

    reward_dataset = reward_dataset.map(preprocess_function, batched=True)

    reward_collator = DataCollatorWithPadding(tokenizer, padding=True, max_length=script_args.max_length)
    reward_dataloader = DataLoader(reward_dataset, batch_size=script_args.bs, shuffle=False, collate_fn=reward_collator)
    
    (
        model,
        reward_dataloader
    ) = accelerator.prepare(
        model,
        reward_dataloader
        )

    accelerator.wait_for_everyone()

    all_rewards = []
    for batch in reward_dataloader:
        with torch.no_grad():
            rewards = reward_model(**batch).logits
            rewards = accelerator.gather(rewards)
            all_rewards.extend(rewards)

    all_rewards = [reward.item() for reward in all_rewards][: len(reward_dataset)]
    
    reward_dataset = reward_dataset.add_column("rewards", all_rewards)

    accelerator.wait_for_everyone()
    
    reward_dataset.save_to_disk(os.path.join(script_args.save_dataset_path, "train"))

if __name__ == "__main__":
    main()
