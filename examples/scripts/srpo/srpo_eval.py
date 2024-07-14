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
from tqdm import tqdm
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
import json

os.environ["HUGGINGFACE_CACHE"] = "/workspace/.cache/huggingface"
os.environ["DATA_DIR"] = "./data"
os.environ["HF_DATASETS_CACHE"] = f"{os.environ['HUGGINGFACE_CACHE']}/datasets"
os.environ["HF_HOME"] = f"{os.environ['HUGGINGFACE_CACHE']}/misc"
os.environ["TRANSFORMERS_CACHE"] = f"{os.environ['HUGGINGFACE_CACHE']}/transformers"
os.environ["WANDB_LOG_MODEL"] = "end"
"""
# SFT trained:
python examples/scripts/srpo/srpo_eval.py \
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

# RLHF trained:
python examples/scripts/srpo/srpo_eval.py \
    --model_name_or_path=./srpo_tldr \
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
import wandb
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
    # with wandb.init(entity="frasermince") as run:
    #     # Pass the name and version of Artifact
    #     artifact = run.use_artifact('unchart/huggingface/model-qltcdvjl:v1', type='model')

    #     # Download model weights to a folder and return the path
    #     model_dir = artifact.download()

    # print("***MODEL DIR", model_dir)
    # model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    untrained_model_name_or_path = "EleutherAI/pythia-1b-deduped"
    sft_model_name_or_path = "./srpo_sft_1"
    rlhf_model_name_or_path = "./srpo_tldr_peft_fix"
    rlhf_pretrained_model = "dpo_tldr"
    
    model_ref = None
    tokenizer = AutoTokenizer.from_pretrained(untrained_model_name_or_path)
    zero_instruction = """Below is a reddit POST and the corresponding SUBREDDIT and TITLE.
Write a both precise and concise summary of the contents of the POST.
"""
    n_instruction = """Below is a reddit POST and the corresponding SUBREDDIT and TITLE, and an EXAMPLE SUMMARY.
Write a both precise and concise summary of the contents of the POST.
"""

    tokenizer.chat_template = """Below is a reddit POST and the corresponding SUBREDDIT and TITLE{{", and an EXAMPLE SUMMARY." if example else "."}}
Write a both precise and concise summary of the contents of the POST.
{{messages}}
{%- if example %}
EXAMPLE SUMMARY: {{example + "\n"}}
{%- endif %}

TL;DR:
{%- if answer %}
{{answer}}
{%- endif %}
"""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # if args.sanity_check:
    #     for key in ds:
    #         ds[key] = ds[key].select(range(50))

    def process(row):
        if row["prompt"].endswith("TL;DR:"):
            row["prompt"] = row["prompt"][:-6]
        row["message"] = row["messages"][1]["content"]
        longest = len(tokenizer.apply_chat_template(row["prompt"], example=row["message"], padding=False)) + 30

        row["longest_length"] = longest
        return row

        return row

    # train_dataset = train_dataset.map(
    test_dataset = test_dataset.map(
         process,
         num_proc=multiprocessing.cpu_count(),
    )

    test_dataset = test_dataset.filter(lambda x: x["longest_length"] <= 700).select(range(10))
    
    model_sft = AutoModelForCausalLM.from_pretrained(sft_model_name_or_path, **model_kwargs)
    model_untrained = AutoModelForCausalLM.from_pretrained(untrained_model_name_or_path, **model_kwargs)
    model_rlhf = AutoModelForCausalLM.from_pretrained(rlhf_model_name_or_path, **model_kwargs)
    model_rlhf_pretrained = AutoModelForCausalLM.from_pretrained(rlhf_pretrained_model, **model_kwargs)

    model_untrained.cuda()
    model_sft.cuda()
    model_rlhf.cuda()
    model_rlhf_pretrained.cuda()
    model_untrained.eval()
    model_sft.eval()
    model_rlhf.eval()
    model_rlhf_pretrained.eval()

    annotator = PairwiseAutoAnnotator()
    model_names = ["RLHF Pretrained", "Untrained", "SFT", "RLHF"] + [f"RLHF Revision {i+1}" for i in range(5)]
    preferred = {}
    generations = []
    for m in model_names:
        preferred[m] = 0

    import pdb; pdb.set_trace()
    should_print = False

    total_alpaca_inputs = []
    prompts = []
    for item in tqdm(test_dataset):
        generation = {}
        post = item["post"]
        sft_summary = item["summary"] 
        zero_alpaca_farm_input_untrained = {"instruction": zero_instruction, "input": post, "output_1": sft_summary}
        zero_alpaca_farm_input_sft = {"instruction": zero_instruction, "input": post, "output_1": sft_summary}
        zero_alpaca_farm_input_rlhf = {"instruction": zero_instruction, "input": post, "output_1": sft_summary}
        zero_alpaca_farm_input_rlhf_pretrained = {"instruction": zero_instruction, "input": post, "output_1": sft_summary}
        n_alpaca_farm_input = {"instruction": n_instruction, "input": post, "output_1": sft_summary}
        templated_zero = tokenizer.apply_chat_template(item["prompt"], add_special_tokens=False, tokenize=False)
        rlhf_templated_zero = item["prompt"] + "TL;DR:"
        if should_print:
            print("******************************************************************")
            print("ITEM", templated_zero)
        inputs = tokenizer(templated_zero, return_tensors="pt")
        rlhf_inputs = tokenizer(templated_zero, return_tensors="pt")
        untrained_output = model_untrained.generate(
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            max_length=700,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        sft_output = model_sft.generate(
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            max_length=700,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        rlhf_output = model_rlhf.generate(
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            max_length=700,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        rlhf_pretrained_output = model_rlhf_pretrained.generate(
            input_ids=rlhf_inputs["input_ids"].cuda(),
            attention_mask=rlhf_inputs["attention_mask"].cuda(),
            max_length=700,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        untrained_decoded_output = tokenizer.batch_decode(untrained_output, skip_special_tokens=True)[0]
        sft_decoded_output = tokenizer.batch_decode(sft_output, skip_special_tokens=True)[0]
        rlhf_decoded_output = tokenizer.batch_decode(rlhf_output, skip_special_tokens=True)[0]
        rlhf_pretrained_decoded_output = tokenizer.batch_decode(rlhf_pretrained_output, skip_special_tokens=True)[0]

        if should_print:
            print(f"SFT: {sft_decoded_output[len(templated_zero):]}")
        untrained_tldr = untrained_decoded_output[len(templated_zero):]
        sft_tldr = sft_decoded_output[len(templated_zero):]
        current_tldr = rlhf_decoded_output[len(templated_zero):]
        rlhf_pretrained_tldr = rlhf_pretrained_decoded_output[len(templated_zero):]
        zero_alpaca_farm_input_untrained["output_2"] = untrained_tldr
        zero_alpaca_farm_input_sft["output_2"] = sft_tldr
        zero_alpaca_farm_input_rlhf["output_2"] = current_tldr
        zero_alpaca_farm_input_rlhf_pretrained["output_2"] = rlhf_pretrained_tldr
        import pdb; pdb.set_trace()
        if should_print:
            print(f"0 REVISION {current_tldr}")
        n_alpaca_farm_inputs = [n_alpaca_farm_input.copy() for a in range(5)]
        for n in range(5):

            templated_n = tokenizer.apply_chat_template(item["prompt"], example=current_tldr, add_special_tokens=False, tokenize=False)
            n_inputs = tokenizer(templated_n, return_tensors="pt")
            n_rlhf_output = model_rlhf.generate(
                input_ids=n_inputs["input_ids"].cuda(),
                attention_mask=n_inputs["attention_mask"].cuda(),
                max_length=700,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            n_rlhf_decoded_output = tokenizer.batch_decode(n_rlhf_output, skip_special_tokens=True)[0]
            current_tldr = n_rlhf_decoded_output[len(templated_n):]
            n_alpaca_farm_inputs[n]["output_2"] = current_tldr
            if should_print:
                print(f"{n + 1} REVISION {current_tldr}")
        if should_print:
            print("Pretrained RLHF", rlhf_pretrained_tldr)
            print("DATASET TLDR", sft_summary)
            print("******************************************************************")
        alpaca_inputs =  [zero_alpaca_farm_input_rlhf_pretrained] + [zero_alpaca_farm_input_untrained] + [zero_alpaca_farm_input_sft] + [zero_alpaca_farm_input_rlhf] + n_alpaca_farm_inputs

        total_alpaca_inputs.append(alpaca_inputs)
        
        prompts.append(post)
        
    for i, alpaca_inputs in enumerate(total_alpaca_inputs):
        result = annotator.annotate_pairs(alpaca_inputs)
        generation = {"prompt": prompts[i]}
        for j, res in enumerate(result):
                # generation[model_names[i]] = {"inputs": alpaca_inputs[i], "preference": res["preference"]}
                generation[model_names[j]] = {"inputs": alpaca_inputs[j], "preference": res["preference"]}
                if should_print:
                    print(f"Model: {model_names[j]}, Preference: {res['preference']}")
                # import pdb; pdb.set_trace()
                if res["preference"] == 2:
                    preferred[model_names[j]] += 1
        generations.append(generation)


    overall = {"generations": generations, "preferred": preferred}

    with open('dpo_generations.json', 'w') as f:
        json.dump(overall, f)
    