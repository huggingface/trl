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
from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_xpu_available


tqdm.pandas()


@dataclass
class ScriptArguments:
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            model_name="lvwerra/gpt2-imdb",
            query_dataset="imdb",
            reward_model="sentiment-analysis:lvwerra/distilbert-imdb",
            learning_rate=1.41e-5,
            log_with=None,
            mini_batch_size=128,
            batch_size=128,
            gradient_accumulation_steps=1,
            early_stopping=False,
            target_kl=6.0,
            kl_penalty="kl",
            seed=0,
            use_score_scaling=False,
            use_score_norm=False,
            score_clip=None,
        )
    )
    use_seq2seq: bool = False
    """whether to use seq2seq models"""
    use_peft: bool = False
    """whether to use peft"""
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})


args = tyro.cli(ScriptArguments)


# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, query_dataset, input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(query_dataset, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(args.ppo_config, args.ppo_config.query_dataset)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(args.ppo_config.seed)

# Now let's build the model, the reference model, and the tokenizer.
if not args.use_peft:
    ref_model = trl_model_class.from_pretrained(args.ppo_config.model_name, trust_remote_code=args.trust_remote_code)
    device_map = None
    peft_config = None
else:
    peft_config = args.peft_config
    ref_model = None
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}

model = trl_model_class.from_pretrained(
    args.ppo_config.model_name,
    trust_remote_code=args.trust_remote_code,
    device_map=device_map,
    peft_config=peft_config,
)


tokenizer = AutoTokenizer.from_pretrained(args.ppo_config.model_name)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.eos_token_id

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(args.ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    if is_xpu_available():
        device = "xpu:0"
    else:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
task, model_name = args.ppo_config.reward_model.split(":")
if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
    with ds_plugin.zero3_init_context_manager(enable=False):
        sentiment_pipe = pipeline(task, model=model_name, device=device)
else:
    sentiment_pipe = pipeline(task, model=model_name, device=device)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from gpt2
    response_tensors, ref_response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
    )
    batch["response"] = tokenizer.batch_decode(response_tensors)
    batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
    ref_pipe_outputs = sentiment_pipe(ref_texts, **sent_kwargs)
    ref_rewards = [torch.tensor(output[1]["score"]) for output in ref_pipe_outputs]
    batch["ref_rewards"] = ref_rewards

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
