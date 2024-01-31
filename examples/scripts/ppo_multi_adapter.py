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
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, HfArgumentParser

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available


input_min_text_length = 6
input_max_text_length = 12


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    model_name: Optional[str] = field(default="huggyllama/llama-7b", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the dataset name"})
    rm_adapter: Optional[str] = field(
        default="trl-lib/llama-7b-hh-rm-adapter", metadata={"help": "the rm adapter name"}
    )
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    use_safetensors: Optional[bool] = field(default=False, metadata={"help": "Use safetensors"})
    seed: Optional[int] = field(default=0, metadata={"help": "the random seed"})
    use_score_scaling: Optional[bool] = field(default=False, metadata={"help": "Use score scaling"})
    use_score_norm: Optional[bool] = field(
        default=False, metadata={"help": "Use score normalization. Only applicable if use_score_scaling is True"}
    )
    score_clip: Optional[float] = field(default=None, metadata={"help": "Score clipping"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


def create_and_prepare_dataset(tokenizer):
    dataset = load_dataset(script_args.dataset_name, split="train[:1%]")

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(example):
        text_size = input_size()
        example["input_ids"] = tokenizer.encode(example["chosen"])[:text_size]
        example["query"] = tokenizer.decode(example["input_ids"])
        return example

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format("torch")
    return dataset


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    script_args.model_name,
    device_map={"": "xpu:0"} if is_xpu_available() else {"": "npu:0"} if is_npu_available else {"": 0},
    peft_config=lora_config,
    quantization_config=nf4_config,
    reward_adapter=script_args.rm_adapter,
    use_safetensors=script_args.use_safetensors,
)
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

tokenizer.pad_token = tokenizer.eos_token

dataset = create_and_prepare_dataset(tokenizer)


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


config = PPOConfig(
    model_name=script_args.model_name,
    log_with=script_args.log_with,
    learning_rate=1e-5,
    batch_size=8,
    mini_batch_size=2,
    gradient_accumulation_steps=2,
    optimize_cuda_cache=True,
    seed=script_args.seed,
    use_score_scaling=script_args.use_score_scaling,
    use_score_norm=script_args.use_score_norm,
    score_clip=script_args.score_clip,
)

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
)

generation_kwargs = {
    "top_k": 0.0,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "max_new_tokens": 32,
}

for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    question_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute reward score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(ppo_trainer.accelerator.device)
    raw_rewards = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).compute_reward_score(**inputs)
    rewards = [raw_rewards[i, -1, 1] for i in range(len(raw_rewards))]  # take last token

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
