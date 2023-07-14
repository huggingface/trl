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
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import LlamaTokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler


rm_adapter_id = "trl-lib/llama-7b-hh-rm-adapter"
model_name = "huggyllama/llama-7b"
dataset_name = "Anthropic/hh-rlhf"

input_min_text_length = 6
input_max_text_length = 12


def create_and_prepare_dataset(tokenizer):
    dataset = load_dataset(dataset_name, split="train[:1%]")

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
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map={"": 0},
    peft_config=lora_config,
    reward_adapter=rm_adapter_id,
)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

dataset = create_and_prepare_dataset(tokenizer)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,
    batch_size=8,
    mini_batch_size=2,
    gradient_accumulation_steps=2,
    optimize_cuda_cache=True,
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
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
}

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
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
    raw_rewards = ppo_trainer.model.compute_reward_score(**inputs)
    rewards = [raw_rewards[i, -1, 1] for i in range(len(raw_rewards))]  # take last token

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
