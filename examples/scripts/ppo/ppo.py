# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///
from datasets import Dataset
import os
import shutil
from trl.experimental.ppo import PPOConfig
import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ModelConfig, ScriptArguments, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.experimental.ppo import PPOConfig, PPOTrainer


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


"""
python -i examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir pythia-1b-deduped-descriptiveness-sentiment-trl-style-ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --missing_eos_penalty 1.0

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir pythia-1b-deduped-descriptiveness-sentiment-trl-style-ppo \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0
"""
class LeftPadToDatasetMaxCollator:
    """左填充到“整个数据集最大长度”"""

    def __init__(self, pad_token_id: int, max_length: int):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, features):
        batch_input_ids = []
        batch_attention_mask = []

        for f in features:
            ids = f["input_ids"][: self.max_length]
            pad_len = self.max_length - len(ids)
            padded = [self.pad_token_id] * pad_len + ids
            mask = [0] * pad_len + [1] * len(ids)

            batch_input_ids.append(padded)
            batch_attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }


def get_dataset_max_len(ds: Dataset, field: str = "input_ids") -> int:
    return max(len(x) for x in ds[field])


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    
    ################
    # Model & Tokenizer
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
        **model_kwargs,
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
        **model_kwargs,
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    ref_policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )


    ################
    # Dataset
    ################
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    BASE = "https://hf-mirror.com/datasets/openai/summarize_from_feedback/resolve/refs%2Fconvert%2Fparquet"

    dataset = load_dataset(
        "parquet",
        data_files={
            "train":      f"{BASE}/comparisons/train/0000.parquet",
            "validation": f"{BASE}/comparisons/validation/0000.parquet",
        },
    )
    dataset = dataset["train"]
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    dataset_text_field = "prompt"

    def prepare_dataset(dataset, tokenizer, add_special_tokens=True):
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        def build_example(ex):
            subreddit = ex["info"]["subreddit"]
            post = ex["info"]["post"]
            title = ex["info"]["title"]
            text1, text2 = ex["summaries"][0]['text'], ex["summaries"][1]['text']
            choice = ex["choice"]

            prompt = f"SUBREDDIT: {subreddit}\nTITLE: {title}\nPOST: {post}\nTL;DR: "
            if choice == 0:
                chosen_text, rejected_text = text1, text2
            else:
                chosen_text, rejected_text = text2, text1

            prompt_ids = tokenizer(prompt, add_special_tokens=add_special_tokens)["input_ids"]
            chosen_ids = tokenizer(chosen_text, add_special_tokens=add_special_tokens)["input_ids"]
            rejected_ids = tokenizer(rejected_text, add_special_tokens=add_special_tokens)["input_ids"]

            return {
                "prompt": prompt,
                "input_ids": prompt_ids,         # PPOTrainer 训练时会取 data["input_ids"]
                "chosen_text": chosen_text,
                "rejected_text": rejected_text,
                "chosen_ids": chosen_ids,
                "rejected_ids": rejected_ids,
            }

        dataset = dataset.map(build_example, num_proc=4)
        dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0, num_proc=4)
        return dataset



    
    


    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)


    max_len = get_dataset_max_len(train_dataset, "input_ids")
    data_collator = LeftPadToDatasetMaxCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=max_len,
    )
    training_args.report_to = "wandb"
    training_args.run_name = "ppo-tldr-exp1"
    training_args.stop_token = "eos"
    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.config.eos_token_id = tokenizer.eos_token_id
    policy.generation_config.pad_token_id = tokenizer.pad_token_id
    policy.generation_config.eos_token_id = tokenizer.eos_token_id
    ################
    # Training
    ################
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()
