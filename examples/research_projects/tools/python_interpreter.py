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
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, HfArgumentParser, load_tool

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, TextEnvironment


os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="bigcode/starcoderbase", metadata={"help": "the model name"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "max number of generated tokens per turn"})
    ppo_epochs: Optional[int] = field(default=1, metadata={"help": "max number of ppo epochs"})
    n_epochs: Optional[int] = field(default=32, metadata={"help": "max number of ppo epochs"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


def exact_match_reward(responses, answers=None):
    """Reward if generated response contains correct answer."""
    rewards = []
    pattern = r"Result\s*=\s*(-?\d+(?:\.\d+)?)\s*<submit>"  # generated by chatGPT
    for response, answer in zip(responses, answers):
        reward = 0.0
        try:
            predicted_number = None
            match_pattern = re.findall(pattern, response)
            if match_pattern:
                predicted_number = float(match_pattern[0])
            if predicted_number is not None:
                if np.abs(predicted_number - float(answer)) < 0.1:
                    reward += 1.0
        except Exception:
            pass
        rewards.append(torch.tensor(reward))
    return rewards


def evaluate(test_dataloader, text_env, ppo_trainer):
    test_rewards = []
    for test_batch in test_dataloader:
        _, _, _, rewards, _ = text_env.run(test_batch["query"], answers=test_batch["answer"])
        test_rewards.extend(rewards)
    test_rewards = ppo_trainer.accelerator.gather_for_metrics(
        torch.stack(test_rewards).to(ppo_trainer.accelerator.device)
    )
    return test_rewards.mean()


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_proj", "c_attn", "q_attn"],
)

# set up models
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    script_args.model_name,
    use_auth_token=True,
    load_in_4bit=True,
    peft_config=lora_config,
)
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("openai/gsm8k", "main", split="train")
ds = ds.rename_columns({"question": "query"})
ds = ds.map(lambda x: {"answer": x["answer"].split("#### ")[1]})
ds = ds.select(range(1, len(ds)))  # skip the first sample which is used in prompt

ds_test = load_dataset("openai/gsm8k", "main", split="test")
ds_test = ds_test.rename_columns({"question": "query"})
ds_test = ds_test.map(lambda x: {"answer": x["answer"].split("#### ")[1]})

test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size=script_args.batch_size)

# prompt
prompt = """\
Example of using a Python API to solve math questions.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

<request><PythonInterpreter>
def solution():
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result
print(solution())
<call>72<response>

Result = 72 <submit>

Q: """

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": -1,
    "max_new_tokens": script_args.max_new_tokens,
}

# trainer
ppo_config = PPOConfig(
    batch_size=script_args.batch_size,
    learning_rate=script_args.learning_rate,
    mini_batch_size=script_args.mini_batch_size,
    ppo_epochs=script_args.ppo_epochs,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    log_with="wandb",
    tracker_project_name="trl-gsm8k",
    remove_unused_columns=False,
    optimize_cuda_cache=True,
)

ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer, dataset=ds)
test_dataloader = ppo_trainer.accelerator.prepare(test_dataloader)

# text env
text_env = TextEnvironment(
    model,
    tokenizer,
    [load_tool("lvwerra/python-interpreter")],
    exact_match_reward,
    prompt,
    max_turns=2,
    generation_kwargs=generation_kwargs,
)

# main training loop
for epoch in range(script_args.n_epochs):
    for step, batch in enumerate(ppo_trainer.dataloader):
        if (step == 0) and (epoch % 4 == 0):  # evaluate every 4 epochs
            reward_mean_test = evaluate(test_dataloader, text_env, ppo_trainer)
        else:
            reward_mean_test = None

        queries, responses, masks, rewards, histories = text_env.run(batch["query"], answers=batch["answer"])
        train_stats = ppo_trainer.step(queries, responses, rewards, masks)

        # logging
        if reward_mean_test is not None:
            train_stats["env/reward_mean_test"] = reward_mean_test
        texts = {
            "query": batch["query"],
            "response": [tokenizer.decode(response) for response in responses],
            "answer": batch["answer"],
        }
        ppo_trainer.log_stats(train_stats, texts, rewards, columns_to_log=["query", "response", "answer"])

reward_mean_test = evaluate(test_dataloader, text_env, ppo_trainer)
ppo_trainer.save_pretrained(f"model/{script_args.model_name}-gsm8k")
