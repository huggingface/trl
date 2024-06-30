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

"""
# regular:
accelerate launch --config_file=examples/accelerate_configs/multi_gpu.yaml examples/scripts/nash_md.py \
    --dataset_name=trl-internal-testing/hh-rlhf-trl-style \
    --dataset_num_proc=4 \
    --model_name_or_path=Qwen/Qwen1.5-0.5B-Chat \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="scratch/dpo_anthropic_hh" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --sanity_check

# peft:
python examples/scripts/nash_md.py \
    --dataset_name=trl-internal-testing/hh-rlhf-trl-style \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="dpo_anthropic_hh" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""

import logging
import os
from contextlib import nullcontext

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    ModelConfig,
    NashMDConfig,
    NashMDTrainer,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.commands.cli_utils import DPOScriptArguments, TrlParser, init_zero_verbose
from trl.trainer import PairRMJudge


TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)


if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, NashMDConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ###################
    # Model & Tokenizer
    ###################
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
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, revision=model_config.model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Optional rich context managers
    ###############
    init_context = (
        nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the NashMDTrainer...")
    )
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Dataset
    ################
    ds = load_dataset(args.dataset_name)
    if args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(1_000))

    for key in ds:
        ds[key] = ds[key].remove_columns(["chosen", "rejected", "score_chosen", "score_rejected"])

    def process(row):
        row["prompt"] = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        return row

    ds = ds.map(
        process,
        num_proc=training_args.dataset_num_proc,
        load_from_cache_file=False,
    )
    train_dataset = ds[args.dataset_train_split]
    eval_dataset = ds[args.dataset_test_split]

    ################
    # Training
    ################
    with init_context:
        trainer = NashMDTrainer(
            model,
            model_ref,
            annotator_cls=PairRMJudge,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )
        # prompts_ds = load_dataset(args.dataset_name, split="test[:32]")
        # prompts_ds = prompts_ds.map(
        #     lambda x: {
        #         "prompt": tokenizer.apply_chat_template(x["chosen"][:-1], tokenize=False, add_generation_prompt=True)
        #     }
        # )
    #     win_rate_callback = WinRateCallback(
    #         prompts=prompts_ds["prompt"],
    #         judge=judge,
    #         generation_config=GenerationConfig(
    #             temperature=0.9,
    #             do_sample=True,
    #             num_return_sequences=1,
    #             pad_token_id=tokenizer.pad_token_id,
    #             eos_token_id=tokenizer.eos_token_id,
    #             max_new_tokens=512,
    #         ),
    #         trainer=trainer,
    #     )
    # trainer.add_callback(win_rate_callback)

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
