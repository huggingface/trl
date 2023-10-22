# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional

from accelerate import Accelerator
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser
)

from trl.trainer.utils import generate


@dataclass
class ScriptArguments:
    # model parameters
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "the model name"})
    mixed_precision: Optional[str] = field(default="fp16", metadata={"help": "the model dtype"})
    # data parameters
    dataset_name: Optional[str] = field(default="Dahoas/full-hh-rlhf", metadata={"help": "the HF data path"})
    split: Optional[str] = field(default="train", metadata={"help": "the dataset split to use for generation"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the generation batch size"})
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    save_dataset_path: Optional[str] = field(default="sft_gen_dataset", metadata={"help": "the path for saving the generated dataset"})
    # generation parameters
    max_new_tokens: Optional[int] = field(
        default=128, metadata={"help": "the maximum number of tokens generated per sample"}
    )
    temperature: Optional[float] = field(default=1.0, metadata={"help": "the sampling temperature"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "top_p sampling argument"})
    top_k: Optional[float] = field(default=50, metadata={"help": "top_k sampling argument"})
    num_return_sequences: Optional[int] = field(default=64, metadata={"help": "the number of return sequences"})
    # instrumentation
    sanity_check: Optional[bool] = field(default=False)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(
        mixed_precision=script_args.mixed_precision
    )
    
    # load sft policy
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # for generation
    tokenizer.padding_side = "left"

    # define gen_kwargs
    generation_kwargs = {
        "top_k": script_args.top_k,
        "top_p": script_args.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "temperature": script_args.temperature,
        "max_new_tokens": script_args.max_new_tokens,
        "num_return_sequences": script_args.num_return_sequences,
    }

    # load and preprocess the dataset
    dataset = load_dataset(script_args.dataset_name)[script_args.split]

    if script_args.sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    def tokenize_fn(samples):
        model_inputs = tokenizer(samples["prompt"])

        return {
            **model_inputs,
        }

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=list(dataset.features))
    dataset = dataset.filter(lambda x: len(x["input_ids"])<script_args.max_prompt_length)

    data_collator = DataCollatorForSeq2Seq(tokenizer, max_length=script_args.max_prompt_length, pad_to_multiple_of=8)

    dataloader = DataLoader(dataset, batch_size=script_args.batch_size, shuffle=False, collate_fn=data_collator)

    model, dataloader = accelerator.prepare(model, dataloader)

    # generate responses from sft policy
    prompts, responses = generate(model, dataloader, tokenizer, accelerator, **generation_kwargs)
    
    generated_dataset = Dataset.from_dict({"prompt": prompts, "response": responses})
    
    # save the generated dataset
    generated_dataset.save_to_disk(script_args.save_dataset_path)