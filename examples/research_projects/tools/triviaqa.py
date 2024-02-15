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
from dataclasses import dataclass, field
from typing import Optional

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
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "max number of generated tokens per turn"})
    ppo_epochs: Optional[int] = field(default=1, metadata={"help": "max number of ppo epochs"})
    iterations: Optional[int] = field(default=1000, metadata={"help": "the number of iterations"})
    seed: Optional[int] = field(default=0, metadata={"help": "the random seed"})


parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

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
    args.model_name,
    use_auth_token=True,
    trust_remote_code=True,
    load_in_4bit=True,
    peft_config=lora_config,
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

# system prompt
prompt = """\
Answer the following question:

Q: In which branch of the arts is Patricia Neary famous?
A: Ballets
A2: <request><Wiki>Patricia Neary<call>Patricia Neary (born October 27, 1942) is an American ballerina, choreographer and ballet director, who has been particularly active in Switzerland. She has also been a highly successful ambassador for the Balanchine Trust, bringing George Balanchine's ballets to 60 cities around the globe.<response>
Result=Ballets<submit>

Q: Who won Super Bowl XX?
A: Chicago Bears
A2: <request><Wiki>Super Bowl XX<call>Super Bowl XX was an American football game between the National Football Conference (NFC) champion Chicago Bears and the American Football Conference (AFC) champion New England Patriots to decide the National Football League (NFL) champion for the 1985 season. The Bears defeated the Patriots by the score of 46–10, capturing their first NFL championship (and Chicago's first overall sports victory) since 1963, three years prior to the birth of the Super Bowl. Super Bowl XX was played on January 26, 1986 at the Louisiana Superdome in New Orleans.<response>
Result=Chicago Bears<submit>

Q: """

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": -1,
    "max_new_tokens": args.max_new_tokens,
}

# trainer
config = PPOConfig(
    batch_size=args.batch_size,
    model_name=args.model_name,
    learning_rate=args.learning_rate,
    log_with=args.log_with,
    mini_batch_size=args.mini_batch_size,
    ppo_epochs=args.ppo_epochs,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    seed=args.seed,
    optimize_cuda_cache=True,
)
ppo_trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer)
dataset = load_dataset("trivia_qa", "rc", split="train")
local_seed = args.seed + ppo_trainer.accelerator.process_index * 100003  # Prime
dataset = dataset.shuffle(local_seed)


def data_generator():
    for i in range(len(dataset)):
        yield dataset[i]["question"], list(dataset[i]["answer"]["normalized_aliases"])


gen = data_generator()
gen = iter(gen)


def generate_data(n):
    tasks, answers = [], []
    for _i in range(n):
        q, a = next(gen)
        tasks.append(q)
        answers.append(a)
    return tasks, answers


def exact_match_reward(responses, answers=None):
    """Reward if generated response contains correct answer."""
    rewards = []
    for response, answer in zip(responses, answers):
        reward = 0.0
        for a in answer:
            if a.lower() in response.lower():
                reward += 1.0
                break
        rewards.append(torch.tensor(reward))
    return rewards


def tool_fn(x):
    # limit the amount of tokens
    return tool(x).split("\n")[1][:600]


# text env
tool = load_tool("vwxyzjn/pyserini-wikipedia-kilt-doc")

text_env = TextEnvironment(
    model,
    tokenizer,
    {"Wiki": tool_fn},
    exact_match_reward,
    prompt,
    generation_kwargs=generation_kwargs,
    max_tool_reponse=400,
)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


print_trainable_parameters(model)
# main training loop
for i in range(args.iterations):
    tasks, answers = generate_data(config.batch_size)
    queries, responses, masks, rewards, histories = text_env.run(tasks, answers=answers)
    train_stats = ppo_trainer.step(queries, responses, rewards, masks)
    response_texts = [tokenizer.decode(response) for response in responses]
    query_texts = [tokenizer.decode(query) for query in queries]
    texts = {
        "query": [qt.split("<submit>")[-1].strip() for qt in query_texts],
        "response": response_texts,
        "answer": [", ".join(item) for item in answers],
    }
    all_rewards = ppo_trainer.accelerator.gather(torch.tensor(rewards, device=ppo_trainer.accelerator.device))
    ppo_trainer.log_stats(train_stats, texts, list(all_rewards), columns_to_log=["query", "response", "answer"])
    if i % 100 == 0:
        ppo_trainer.save_pretrained(f"models/{args.model_name}_{args.seed}_{i}_triviaqa")
