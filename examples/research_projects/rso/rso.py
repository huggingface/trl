import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

@dataclass
class ScriptArguments:
    reward_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "the model name"})
    mixed_precision: Optional[str] = field(default="fp16", metadata={"help": "the model dtype"})
    # data parameters
    dataset_name: Optional[str] = field(default="Dahoas/full-hh-rlhf", metadata={"help": "the HF data path"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the generation batch size"})
    
    # rejection sampling 
    num_samples: Optional[int] = field(default=8, metadata={"help": "the number of samples to keep after rejection sampling"})
    beta: Optional[int] = field(default=.5, metadata={"help": "TO DO"})
    ranking_method: Optional[str] = field(default="first_round", metadata={"help": " or tournament TO DO"})
    
    # instrumentation
    sanity_check: Optional[bool] = field(default=False)
    
  
# taken from https://arxiv.org/pdf/2309.06657.pdf 
def conduct_rejection_sampling(
    response_candidates: List[str],
    response_rewards: List[float],
    num_samples: int,
    beta: float
):
    """Conducts rejection sampling guided by rewards.
    
    Args:
        response_candidates: response candidates from sft policy
        response_rewards: response rewards.
        num_samples: number of samples to sub-sample.
        beta: beta parameter in KL-constrained reward maximization objective.
        
    Returns:
        Rejection sampled sequences from the optimal policy.
    """
    candidates = {c: r for c, r in zip(response_candidates, response_rewards)}
    accepted = []
    while len(accepted) < num_samples:
        max_reward = max(candidates.values())
        to_remove = []
        for c, r in candidates.items():
            u = np.random.uniform()
            if u >= np.exp((r - max_reward) / beta):
                continue
            accepted.append(c)
            to_remove.append(c)
            if len(accepted) == num_samples:
                break
        for c in to_remove:
            candidates.pop(c)
    return accepted

      
@torch.no_grad()
def score(
    model: PreTrainedModel,
    dataloader: DataLoader,
    accelerator: Accelerator,
    dataset: Dataset
) -> Dataset:

    rewards = []
    pbar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)

    for batch in dataloader:
        scores = model(**batch).logits.squeeze(1)
        scores = accelerator.gather(scores)
        rewards.extend(scores)
        pbar.update(1)

    rewards = rewards[: len(dataset)]
    rewards = [reward.item() for reward in rewards]

    dataset = dataset.add_column("rewards", rewards)

    return dataset

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(
        mixed_precision=script_args.mixed_precision
    )
    
    # load reward model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(script_args.reward_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

    # load and preprocess the dataset
    dataset = load_from_disk(script_args.dataset_name)

    if script_args.sanity_check:
        dataset = dataset.dataset(range(min(len(dataset), 100)))

    def tokenize_fn(samples):
        # create the text column first
        text = [prompt + " " + response for prompt, response in zip(samples["prompt"], samples["response"])]
        model_inputs = tokenizer(text)

        return {
            **model_inputs,
        }

    reward_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=list(dataset.features))

    data_collator = DataCollatorWithPadding(tokenizer)

    dataloader = DataLoader(dataset, batch_size=script_args.batch_size, shuffle=False, collate_fn=data_collator)

    model, dataloader = accelerator.prepare(model, dataloader)
    
    dataset = score(model, dataloader, accelerator, dataset)
    
    # perform rejection sampling
    
    dataset = dataset.to_pandas()
    
    dataset = dataset.groupby("prompt").agg({'response':lambda x: list(x), 'scores':lambda x: list(x)}).reset_index()
    
    dataset["accepted"] = dataset.apply(
        lambda x: conduct_rejection_sampling(
            x["response"], 
            x["rewards"], 
            script_args.num_samples, 
            script_args.beta
        )
    )
    
    # sort the list first
    # then do tournament or first_round ranking
    
    
    
    # save the dataset for later finetuning with DPO
    
    
    
    
    
    
    

