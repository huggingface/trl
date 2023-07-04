# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
import copy
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset
from multi_adapter_model_value import AutoModelForCausalLMWithMultiAdapterValueHead
from peft import LoraConfig
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler


tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_adapter_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    multi_adapter_value: Optional[bool] = field(default=False)
    separate_reward_model: Optional[str] = field(default="", metadata={"help": "the reward model name"})

    # EMA stuff
    ema_decay: Optional[float] = field(default=0.995, metadata={"help": "the ema decay rate"})
    reset_freq: Optional[int] = field(default=None, metadata={"help": "reset every n epochs"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
dataset_name = "lvwerra/stack-exchange-paired"
config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
    accelerator_kwargs={"kwargs_handlers": [DistributedDataParallelKwargs(find_unused_parameters=False)]},
)

train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train")
train_dataset = train_dataset.select(range(100000))

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(tokenizer, dataset_name="lvwerra/stack-exchange-paired"):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    # load imdb with datasets
    ds = load_dataset(dataset_name, data_dir="data/rl", split="train")
    original_columns = ds.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer)


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

if script_args.multi_adapter_value:
    model_cls = AutoModelForCausalLMWithMultiAdapterValueHead
else:
    model_cls = AutoModelForCausalLMWithValueHead


model = model_cls.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    device_map={"": current_device},
    peft_config=lora_config,
    reward_adapter=script_args.reward_adapter_name,
)

if script_args.reset_freq is not None:
    # only use the policy parameters that require grad
    ema = ExponentialMovingAverage(
        (
            param
            for name, param in model.named_parameters()
            if param.requires_grad and model.policy_adapter_name in name
        ),
        decay=script_args.ema_decay,
        use_num_updates=False,
    )
    initial_state_dict = copy.deepcopy(ema.state_dict())
    model.ema_model = ema
else:
    ema = None


# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
)

if script_args.separate_reward_model:
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=script_args.separate_reward_model,
        device_map={"": current_device},
        model_kwargs={"load_in_8bit": True},
        tokenizer=tokenizer,
        return_token_type_ids=False,
    )
    sent_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 16,
        "truncation": True,
    }
# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if epoch >= config.total_ppo_epochs:
        break

    question_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    reward_inputs = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False
    ).to(ppo_trainer.accelerator.device)

    if script_args.separate_reward_model:
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        raw_rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
    else:
        raw_rewards = ppo_trainer.compute_reward_score(**reward_inputs)
    rewards = [(raw_rewards[i] - script_args.reward_baseline) for i in range(len(raw_rewards))]

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if ema is not None:
        ema.update()

    if script_args.reset_freq and epoch and epoch % script_args.reset_freq == 0:
        ema.copy_to()
        ema.load_state_dict(initial_state_dict)
        ppo_trainer.accelerator.print("elastic reset")

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
