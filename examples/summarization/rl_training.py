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

import bitsandbytes as bnb
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    pipeline,
)

# from transformers.trainer_utils import get_last_checkpoint
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.models.modeling_value_adapter import AutoModelForCausalLMWithValueAdapter
from trl.trainer.utils import pad_to_length


# import copy
# from torch_ema import ExponentialMovingAverage
# from transforers import pipeline

tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    reward_adapter_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
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
    save_steps: Optional[int] = field(default=1000, metadata={"help": "the number of steps to save at"})
    save_strategy: Optional[str] = field(default="steps")
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    value_adapter: Optional[bool] = field(default=False)
    separate_reward_model: Optional[str] = field(default=None, metadata={"help": "the reward model name"})

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

    # LoRA
    use_lora: Optional[bool] = field(
        default=True,
    )
    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_all_linear: Optional[bool] = field(default=False, metadata={"help": "lora adapter on all linear layers"})

    # Gold Model
    eval_steps: Optional[int] = field(default=None)
    gold_model_name: Optional[str] = field(default=None, metadata={"help": "the reward model name"})
    gold_in_8bit: Optional[bool] = field(default=False, metadata={"help": "gold the model in 8 bits precision"})
    gold_in_4bit: Optional[bool] = field(default=False, metadata={"help": "gold the model in 4 bits precision"})
    gold_bf16: Optional[bool] = field(
        default=False,
    )
    gold_fp16: Optional[bool] = field(
        default=False,
    )
    gold_eval_greedy: Optional[bool] = field(default=False)
    # # EMA stuff
    # ema_decay: Optional[float] = field(default=0.995, metadata={"help": "the ema decay rate"})
    # reset_freq: Optional[int] = field(default=None, metadata={"help": "reset every n epochs"})
    input_ids_input: Optional[bool] = field(
        default=False,
    )
    strip_prompt: Optional[bool] = field(
        default=False,
    )


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.load_in_4bit else (bnb.nn.Linear8bitLt if args.load_in_8bit else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    if "score" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("score")

    return list(lora_module_names)


def create_and_prepare_model(args):
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
        device_map = {"": Accelerator().local_process_index}
    else:
        device_map = None
        quantization_config = None

    if args.bf16:
        torch_dtype = torch.bfloat16
    # elif args.fp16:
    #     torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if script_args.value_adapter:
        model_cls = AutoModelForCausalLMWithValueAdapter
    else:
        model_cls = AutoModelForCausalLMWithValueHead

    if args.use_lora:
        # we add `score` to the list of modules to save to
        # correctly save the score head.
        # if args.pretrained_adapter is not None:
        #     model = PeftModel.from_pretrained(model, args.pretrained_adapter)
        # else:
        if args.lora_all_linear:
            # hardcoded pythia
            # target_modules = find_all_linear_names(args, model)
            target_modules = ["dense_h_to_4h", "dense_4h_to_h", "query_key_value", "dense"]
        else:
            target_modules = None

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            modules_to_save=["score"],
        )

        # model = get_peft_model(model, peft_config)

        # TODO check
        # modules_to_save = ["score"]
        # for key, _ in model.named_modules():
        #     target_module_found = any(key.endswith(target_key) for target_key in modules_to_save)
        #     if target_module_found:
        #         model.get_submodule(key + ".original_module").requires_grad_(False)
        #
        # if torch_dtype == torch.bfloat16:
        #     for name, module in model.named_modules():
        #         if isinstance(module, LoraLayer):
        #             module = module.to(torch_dtype)
        #         if "norm" in name:
        #             module = module.to(torch.float32)
        #         if "score" in name or "embed_tokens" in name:
        #             if hasattr(module, "weight") and module.weight.dtype == torch.float32:
        #                 module = module.to(torch_dtype)
    else:
        peft_config = None

    model = model_cls.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        peft_config=peft_config,
        reward_adapter=script_args.reward_adapter_name,
    )

    # if script_args.ignore_bias_buffers:
    # torch distributed hack
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        args.gradient_checkpointing = False

    model.config.torch_dtype = torch_dtype
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

    def tokenize(queries):
        if args.strip_prompt:
            if isinstance(queries, list):
                queries = [q.strip() for q in queries]
            else:
                queries = queries.strip()

        return tokenizer(queries, truncation=True, max_length=args.input_max_length)

    dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=num_proc,
        input_columns="query",
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


# Gold Model
if script_args.gold_model_name is not None:
    if script_args.gold_in_8bit or script_args.gold_in_4bit:
        gold_quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.gold_in_8bit, load_in_4bit=script_args.gold_in_4bit
        )
        gold_device_map = {"": ppo_trainer.accelerator.local_process_index}
    else:
        gold_device_map = None
        gold_quantization_config = None

    if script_args.gold_bf16:
        torch_dtype = torch.bfloat16
    elif script_args.gold_fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    gold_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.gold_model_name,
        quantization_config=gold_quantization_config,
        torch_dtype=torch_dtype,
        device_map=gold_device_map,
    )

    gold_model = ppo_trainer.accelerator.prepare(gold_model)
    gold_model.eval()

model.eval()

if script_args.separate_reward_model:
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=script_args.separate_reward_model,
        device_map={"": Accelerator().local_process_index},
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

    full_response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=True,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )

    response_tensors = []
    for question, full_response in zip(question_tensors, full_response_tensors):
        response_tensors.append(full_response[len(question) :])

    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute sentiment score
    if script_args.input_ids_input:
        max_length = script_args.input_max_length + script_args.output_max_length
        default_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        full_response_mask = [torch.ones_like(element) for element in full_response_tensors]
        full_response_encoding = {"input_ids": full_response_tensors, "attention_mask": full_response_mask}
        policy_output = tokenizer.pad(
            full_response_encoding,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        tokenizer.padding_side = default_padding_side
    else:
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        policy_output = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False
        ).to(ppo_trainer.accelerator.device)

    if script_args.separate_reward_model:
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        raw_rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
    else:
        raw_rewards = ppo_trainer.compute_reward_model_score(**policy_output)
        rewards = [(raw_rewards[i] - script_args.reward_baseline) for i in range(len(raw_rewards))]

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)

    if script_args.eval_steps is not None and epoch % script_args.eval_steps == 0:
        if script_args.gold_eval_greedy:
            greedy_output = ppo_trainer.generate(
                question_tensors, do_sample=False, num_beams=1, max_new_tokens=script_args.output_max_length
            )
            max_length = script_args.input_max_length + script_args.output_max_length
            # policy_output = pad_to_length(greedy_output, max_length, tokenizer.pad_token_id)
            policy_output_decoded = tokenizer.batch_decode(greedy_output, skip_special_tokens=True)

        with torch.no_grad():
            gold_rewards = gold_model(**policy_output)[0]
    else:
        gold_rewards = None

    stats["epoch"] = epoch
    ppo_trainer.log_stats(stats, batch, rewards, gold_rewards)

    # if ema is not None:
    #     ema.update()
    #
    # if script_args.reset_freq and epoch and epoch % script_args.reset_freq == 0:
    #     ema.copy_to()
    #     ema.load_state_dict(initial_state_dict)
    #     ppo_trainer.accelerator.print("elastic reset")
    # ppo_trainer.accelerator.print(stats)

    if script_args.save_strategy != "no" and epoch > 0 and epoch % script_args.save_steps == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
