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

import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizerBase
)


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
    
    
@torch.no_grad()
def generate(
    model: PreTrainedModel,
    dataloader: DataLoader,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    **generation_kwargs,
) -> Dataset:
    
    all_predictions = []
    all_prompts = []
    pbar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)

    for batch in dataloader:
        sequence_length = batch["input_ids"].shape[1]

        all_tokens = accelerator.unwrap_model(model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **generation_kwargs,
        )

        generated_tokens = all_tokens[:, sequence_length:]
        prompt_tokens = all_tokens[:,:sequence_length]

        generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
        prompt_tokens = accelerator.pad_across_processes(prompt_tokens, dim=1, pad_index=tokenizer.pad_token_id)

        generated_tokens = accelerator.gather(generated_tokens)
        generated_tokens = generated_tokens.cpu()
        prompt_tokens = accelerator.gather(prompt_tokens)
        prompt_tokens = prompt_tokens.cpu()

        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
            prompt_tokens = prompt_tokens[0]

        all_predictions.extend(generated_tokens)
        all_prompts.extend(prompt_tokens)
        pbar.update(1)

    accelerator.wait_for_everyone()

    all_predictions = tokenizer.batch_decode(all_predictions, skip_special_tokens=True)
    all_prompts = tokenizer.batch_decode(all_prompts, skip_special_tokens=True)

    def filter_text(text):
        """
        Only extract the AI response. Useful in case the model was not trained with masked human responses.
        """
        return text.split("Human:")[0].strip()

    # postprocessing
    all_predictions = [filter_text(generated_text) for generated_text in all_predictions]
    
    generated_dataset = Dataset.from_dict({"prompt": all_prompts, "response": all_predictions})

    return generated_dataset


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
    generated_dataset = generate(model, dataloader, tokenizer, accelerator, **generation_kwargs)
    
    # save the generated dataset
    generated_dataset.save_to_disk(script_args.save_dataset_path)