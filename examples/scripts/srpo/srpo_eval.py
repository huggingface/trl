# flake8: noqa
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
import os

os.environ["HUGGINGFACE_CACHE"] = "/workspace/.cache/huggingface"
os.environ["DATA_DIR"] = "./data"
os.environ["HF_DATASETS_CACHE"] = f"{os.environ['HUGGINGFACE_CACHE']}/datasets"
os.environ["HF_HOME"] = f"{os.environ['HUGGINGFACE_CACHE']}/misc"
os.environ["TRANSFORMERS_CACHE"] = f"{os.environ['HUGGINGFACE_CACHE']}/transformers"
os.environ["WANDB_LOG_MODEL"] = "end"
"""
# pretrained:
python examples/scripts/srpo/srpo_eval.py \
    --model_name_or_path=./final_trained \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 3e-6 \
    --gradient_accumulation_steps 64 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="srpo_tldr" \
    --warmup_steps 150 \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

untrained:
python examples/scripts/srpo/srpo_eval.py \
    --model_name_or_path=EleutherAI/pythia-1b-deduped \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 3e-6 \
    --gradient_accumulation_steps 64 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="srpo_tldr" \
    --warmup_steps 150 \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns
"""

import logging
import multiprocessing
import os
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import SRPOScriptArguments, init_zero_verbose, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    SRPOConfig,
    SRPOTrainer,
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((SRPOScriptArguments, SRPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

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
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model_ref = None
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the DPOTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Dataset
    ################
    raw_datasets = load_dataset("trl-internal-testing/tldr-preference-sft-trl-style")
    test_dataset = raw_datasets["test"]

    if args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(50))

    def process(row):
        if row["prompt"].endswith("TL;DR:"):
            row["prompt"] = row["prompt"][:-6]
        # row["chosen"] = tokenizer.apply_chat_template(
        #     row["chosen"],
        #     padding=False,
        #     add_generation_prompt=True,
        #     tokenize=False
        # )
        # row["rejected"] = tokenizer.apply_chat_template(
        #     row["rejected"], 
        #     padding=False,
        #     add_generation_prompt=True,
        #     tokenize=False
        # )
        return row

    # train_dataset = train_dataset.map(
    test_dataset = test_dataset.select(range(1000)).map(
         process,
         num_proc=multiprocessing.cpu_count(),
    )

    ################
    # Training
    ################
    prefix_zero_prompt = """Below is a reddit POST and the corresponding SUBREDDIT and TITLE.
Write a both precise and concise summary of the contents of the POST."""
    prefix_n_prompt = """Below is a reddit POST and the corresponding SUBREDDIT, TITLE, and an EXAMPLE
SUMMARY. Write a both precise and concise summary of the contents of the POST."""

    post_revision_prompt = "TL;DR:"
    training_args.dataset_num_proc = multiprocessing.cpu_count()
    # with init_context:
    #     trainer = SRPOTrainer(
    #         model,
    #         model_ref,
    #         args=training_args,
    #         # train_dataset=train_dataset,
    #         eval_dataset=train_dataset,
    #         tokenizer=tokenizer,
    #         peft_config=get_peft_config(model_config),
    #         callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
    #     )

    print("BEFORE ITEMS")
    for item in test_dataset:
        zero_prompt = prefix_zero_prompt + "\n" + item["prompt"] + post_revision_prompt
        print("ITEM", zero_prompt)
        inputs = tokenizer(zero_prompt, return_tensors="pt")
        print("USING MODEL", model)
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=1024,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        import pdb; pdb.set_trace()
        print("HAS OUTPUT")
        decoded_output = tokenizer.batch_decode(output)

    # with save_context:
    #     trainer.save_model(training_args.output_dir)
