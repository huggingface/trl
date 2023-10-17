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

# imports
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    HfArgumentParser, 
    PreTrainedModel,
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling
)

from datasets import load_dataset, Dataset
import torch
from accelerate import Accelerator
from typing import Optional, Dict
from dataclasses import field, dataclass
from torch.utils.data import DataLoader
from tqdm import tqdm


# Prompt formatting
PREAMBLE = """A good summary is a shorter piece of text that has the essence of the original. It tries to accomplish the same purpose and conveys the key information from the original post. Below we define four evaluation axes for summary quality: coherence, accuracy, coverage, and overall quality.
Coherence: This axis answers the question “how coherent is the summary on its own?” A summary is coherent if it’s easy to understand when read on its own and free of English errors. A summary is not coherent if it’s difficult to understand what the summary is trying to say. Generally, it’s more important that the summary is understandable than it being free of grammar errors. 
Accuracy: This axis answers the question “does the factual information in the summary accurately match the post?” A summary is accurate if it doesn’t say things that aren’t in the article, it doesn’t mix up people, and generally is not misleading. 
Coverage: This axis answers the question “how well does the summary cover the important information in the post?” A summary has good coverage if it mentions the main information from the post that’s important to understand the situation described in the post. A summary has poor coverage if someone reading only the summary would be missing several important pieces of information about the situation in the post. A summary with good coverage should also match the purpose of the original post (e.g. to ask for advice).
Overall quality: This axis answers the question “how good is the summary overall at representing the post?” This can encompass all of the above axes of quality, as well as others you feel are important. If it’s hard to find ways to make the summary better, the overall quality is good. If there are lots of different ways the summary can be made better, the overall quality is bad. 
You are an expert summary rater. Given a piece of text and two of its possible summaries, output 1 or 2 to indicate which summary best adheres to coherence, accuracy, coverage, and overall quality as defined above.
"""

ANNOTATION = """
Text - {src}
Summary 1 - {tgt1}
Summary 2 - {tgt2}
"""

ENDING = """
Consider the coherence, accuracy, coverage, and overall quality of each summary and explain which one is better.
Rationale:"""

TEXT = "[INST] " + PREAMBLE + ANNOTATION + ENDING + " [/INST]\n"

# main scoring function
def score(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, batch: Dict[str, torch.Tensor], max_length: int):
    """
    Two steps:
        1/ Generate the rationale
        2/ Concat the rationale with the prompt and compute the log probability of token ' 1' and ' 2'.
    """
    rationals = accelerator.unwrap_model(model).generate(**batch, do_sample=script_args.do_sample, max_new_tokens=script_args.max_new_tokens)
    # Decode so that we can pad to left again for a second generation step
    rationals = tokenizer.batch_decode(rationals, skip_special_tokens=True)
    rationals = [rationale + "\nPreferred Summary=" for rationale in rationals]
    model_inputs = tokenizer(rationals, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(model.device)
    
    logits = model(**model_inputs).logits[:, -1,:]
    choice_logits = torch.log_softmax(logits, dim=-1)
    
    token_1 = tokenizer.encode(" 1")[-1]
    token_2 = tokenizer.encode(" 2")[-1]
    
    logprobs_1 = choice_logits[:, token_1] 
    logprobs_2 = choice_logits[:, token_2] 
    
    return logprobs_1, logprobs_2


# define and parse arguments
@dataclass
class ScriptArguments:
    """
    The arguments for the RLAIF preference dataset creation script.
    """
    
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "Name of the LLM to use to create the preferential dataset"})
    
    dataset_name: Optional[str] = field(default="CarperAI/openai_summarize_comparisons", metadata={"help": "Name of the dataset with two completions per prompt"})
    save_dataset_path: Optional[str] = field(default="ai_feedback_dataset", metadata={"help": "Where to save your dataset"})
    max_prompt_length: Optional[int] = field(default=1024, metadta={"help": "maximum prompt length after applying the template"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "batch_size"})
    
    do_sample: Optional[bool] = field(default=False, metadata={"help": "whether or not to use sampling during generation"})
    max_new_tokens: Optional[int] = field(default=512, metadata={"help": "the maximum number of tokens to generate for the rationale"})
    
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only make inference on 100 samples"})
    split: Optional[str] = field(default="train", metadata={"help": "dataset split to make inference on"})
    
    
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

accelerator = Accelerator()

# load the dataset
dataset = load_dataset(script_args.dataset_name)[script_args.split]

# sanity check
if script_args.sanity_check:
    dataset = dataset.select(range(min(len(dataset), 100)))
   
# load the model and tokenizer 
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    device_map=accelerator.device,
    trust_remote_code=True,
    revision=script_args.revision,
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=True)

# set padding side to left and pad token id to eos token if need
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    
# preprocessing 
def preprocessing(samples, side):
    """
    Format and tokenize each sample so that they follow  the "OpenAI + COT 0-shot" prompt as in https://arxiv.org/pdf/2309.00267.pdf.
    We need to assess samples in first and second position in the prompt to avoid the impact of position bias.
    """
    if side=="left":
        samples = [TEXT.format(src=prompt, tgt1=left, tgt2=right) for prompt, left, right in zip(samples["prompt"],samples["chosen"], samples["rejected"])]
    else:
        samples = [TEXT.format(src=prompt, tgt1=left, tgt2=right) for prompt, left, right in zip(samples["prompt"],samples["rejected"], samples["chosen"])]
    
    model_inputs = tokenizer(samples, truncation=True, padding=False, max_length=script_args.max_prompt_length)
    
    return {
        **model_inputs
        }


left_samples = dataset["chosen"]
right_samples = dataset["rejected"]


# alternate the position of responses to avoid position bias
left_dataset = dataset.map(preprocessing, batched=True, fn_kwargs={"side": "left"}, remove_columns=list(dataset.features))
right_dataset = dataset.map(preprocessing, batched=True, fn_kwargs={"side": "right"}, remove_columns=list(dataset.features))

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
left_dataloader = DataLoader(left_dataset, batch_size=script_args.batch_size, shuffle=False, collate_fn=data_collator)
right_dataloader = DataLoader(right_dataset, batch_size=script_args.batch_size, shuffle=False, collate_fn=data_collator)

model, left_dataloader, right_dataloader = accelerator.prepare(model, left_dataloader, right_dataloader)


progress_bar = tqdm(total=len(left_dataloader), disable=not accelerator.is_local_main_process)
left_scores = []
right_scores = []
same_position_preferred = []
max_length = script_args.max_prompt_length+script_args.max_new_tokens


for left_batch, right_batch in zip(left_dataloader, right_dataloader):
    
    left_score_1, right_score_1 = score(model, tokenizer, left_batch, max_length)
    right_score_2, left_score_2  = score(model, tokenizer, right_batch, max_length)
    
    left_score = left_score_1 + left_score_2
    right_score = right_score_1 + right_score_2
    position_bias = ((left_score_1>right_score_1)!=(left_score_2>right_score_2)).float()
        
    # gather tensors
    left_score = accelerator.gather(left_score)
    right_score = accelerator.gather(right_score)
    position_bias = accelerator.gather(position_bias)
    
    left_scores.extend(left_score)
    right_scores.extend(right_score)
    same_position_preferred.extend(position_bias)
    
    progress_bar.update(1)
    
left_scores = left_scores[:len(dataset)]
right_scores = right_scores[:len(dataset)]
same_position_preferred = same_position_preferred[:len(dataset)]
    
dataset = Dataset.from_dict(
    {
        "left_sample": left_samples,
        "right_sample": right_samples,
        "left_score": left_scores,
        "right_score": right_scores,
        "same_position_preferred": same_position_preferred
    }
)


def postprocessing(samples):
    chosen = [samples[i]["left_sample"] if left_score>right_score else samples[i]["right_sample"] for i, left_score, right_score in enumerate(zip(samples["left_score"], samples["right_score"]))]
    rejected = [samples[i]["right_sample"] if left_score>right_score else samples[i]["left_sample"] for i, left_score, right_score in enumerate(zip(samples["left_score"], samples["right_score"]))]
    
    return {
        "chosen": chosen,
        "rejected": rejected,
        **samples
    }


dataset = dataset.map(postprocessing, batched=True)

dataset.save_to_disk(script_args.save_dataset_path)

