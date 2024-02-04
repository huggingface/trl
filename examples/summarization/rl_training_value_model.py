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
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import DistributedDataParallelKwargs
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.models.modeling_value_model import AutoModelForCausalLMWithValueModel


# import copy
# from torch_ema import ExponentialMovingAverage
# from transforers import pipeline

# tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    gold_reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    # tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(
        default="CarperAI/openai_summarize_tldr", metadata={"help": "the dataset name"}
    )
    train_split: Optional[str] = field(
        default="train", metadata={"help": "the dataset split to evaluate on; default to 'none' (no evaluation)"}
    )
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
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
    eval_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.05,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    # multi_adapter_value: Optional[bool] = field(default=False)
    # separate_reward_model: Optional[str] = field(default=None, metadata={"help": "the reward model name"})

    # Generation
    output_min_length: Optional[int] = field(default=24, metadata={"help": "the batch size"})
    output_max_length: Optional[int] = field(default=48, metadata={"help": "the batch size"})
    input_max_length: Optional[int] = field(default=512, metadata={"help": "maximum length for generation"})

    # Quantization
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    bf16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    fp16_model: Optional[bool] = field(
        default=False,
        metadata={},
    )

    # LoRA
    use_lora: Optional[bool] = field(
        default=True,
    )
    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_all_linear: Optional[bool] = field(default=False, metadata={"help": "lora adapter on all linear layers"})
    # # EMA stuff
    # ema_decay: Optional[float] = field(default=0.995, metadata={"help": "the ema decay rate"})
    # reset_freq: Optional[int] = field(default=None, metadata={"help": "reset every n epochs"})


def create_and_prepare_model(args):
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16_model:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLMWithValueModel.from_pretrained(
        args.model_name,
        args.reward_model_name,
        torch_dtype=torch_dtype,
    )

    # if script_args.ignore_bias_buffers:
    # torch distributed hack

    model.config.torch_dtype = torch_dtype
    model.config.use_cache = True
    # model.config.use_cache = not args.gradient_checkpointing

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def create_and_prepare_dataset(args, tokenizer, split, num_proc=2):
    dataset = load_dataset(args.dataset_name, split=split)

    dataset = dataset.rename_column("prompt", "query")
    original_columns = dataset.column_names
    original_columns.remove("query")

    dataset = dataset.map(
        lambda examples: tokenizer(examples["query"], truncation=True, max_length=args.input_max_length),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

    dataset.set_format("torch")
    return dataset


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
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

# set seed before initializing value head for deterministic eval
set_seed(config.seed)

model, tokenizer = create_and_prepare_model(script_args)
train_dataset = create_and_prepare_dataset(script_args, tokenizer, script_args.train_split)
# eval_dataset = create_and_prepare_dataset(script_args, tokenizer, script_args.eval_split)

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=train_dataset,
    data_collator=collator,
)

model.eval()

# if script_args.separate_reward_model:
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

reward_pipe = pipeline(
    "sentiment-analysis",
    model=script_args.reward_model_name,
    # device_map={"": Accelerator().local_process_index},
    tokenizer=tokenizer,
    return_token_type_ids=False,
)
if script_args.eval_freq is not None:
    gold_reward_pipe = pipeline(
        "sentiment-analysis",
        model=script_args.gold_reward_model_name,
        # device_map={"": Accelerator().local_process_index},
        # model_kwargs={"load_in_8bit": True},
        tokenizer=tokenizer,
        return_token_type_ids=False,
    )
sent_kwargs = {
    "top_k": None,
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
    "eos_token_id": tokenizer.eos_token_id,
}
output_length_sampler = LengthSampler(script_args.output_min_length, script_args.output_max_length)

for epoch, batch in tqdm(
    enumerate(ppo_trainer.dataloader),
    total=config.total_ppo_epochs,
    disable=not ppo_trainer.accelerator.is_local_main_process,
):
    if epoch >= config.total_ppo_epochs:
        break

    question_tensors = batch["input_ids"]

    query_response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=True,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    response_tensors = [tensor[len(question) :] for tensor, question in zip(query_response_tensors, question_tensors)]
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    reward_inputs = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False
    ).to(ppo_trainer.accelerator.device)

    # for tensor in reponse_tensors:

    for i, tensor in enumerate(query_response_tensors):
        if not torch.equal(tensor, reward_inputs["input_ids"][i][: reward_inputs["attention_mask"][i].sum()]):
            #TODO

    import pdb

    pdb.set_trace()
    # if script_args.reward_model_name:
    pipe_outputs = reward_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
    # else:
    #     raw_rewards = ppo_trainer.compute_reward_model_score(**reward_inputs)
    #     rewards = [(raw_rewards[i] - script_args.reward_baseline) for i in range(len(raw_rewards))]

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # if ema is not None:
    #     ema.update()
    #
    # if script_args.reset_freq and epoch and epoch % script_args.reset_freq == 0:
    #     ema.copy_to()
    #     ema.load_state_dict(initial_state_dict)
    #     ppo_trainer.accelerator.print("elastic reset")

    if script_args.eval_freq and epoch % script_args.eval_freq == 0:
        if ppo_trainer.accelerator.is_main_process:
            pipe_outputs = gold_reward_pipe(texts, **sent_kwargs)
            rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
            logs = {}
            logs["env/gold_reward_mean"] = torch.mean(rewards).cpu().numpy().item()
            logs["env/gold_reward_std"] = torch.std(rewards).cpu().numpy().item()
            logs["env/gold_reward_dist"] = rewards.cpu().numpy()
            ppo_trainer.accelerator.log(logs)
            print(logs)

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
