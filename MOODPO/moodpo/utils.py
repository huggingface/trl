import os
import gc
import copy
import shutil
from dataclasses import dataclass, field
from typing import Optional
from peft import PeftModel
from accelerate import Accelerator
from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk, disable_caching
import numpy as np
import pandas as pd
from tqdm import tqdm
from trl import AutoModelForCausalLMWithValueHead
disable_caching()

def clean_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class Instructions:
    response_split = "\n\nAssistant:"
    input_split = "\n\nHuman:"

    @staticmethod
    def get_input(query):
        before_response = Instructions.response_split.join(query.split(Instructions.response_split)[:-1])
        return before_response.rstrip() + ' ' + Instructions.response_split
        
    @staticmethod
    def get_response(response):
        return response.split(Instructions.response_split)[-1].strip()


class Instructions_summary():
    instruction_summary = "Generate a one-sentence summary of this post."
    response_split = "### Response:"
    input_split = "### Input:"
    instruction_split = "### Instruction:"

    @classmethod
    def prompt_input(self, input):
        # formulate the news
        return f"### Instruction: {Instructions_summary.instruction_summary} ### Input: {input} ### Response: "

    def get_prompt(self, query):
        before_response = self.response_split.join(query.split(self.response_split)[:-1])
        return before_response.rstrip() 

    def get_post(self, query):
        before_response = self.get_prompt(query)
        return before_response.split(self.input_split)[1].strip()

    def get_input(self, query):
        return self.get_prompt(query) + ' ' + self.response_split
        
    def get_response(self, response):
        return response.split(self.response_split)[-1].strip()


def build_dataset(path, tokenizer, rm_tokenizers_list, split='train', size=None):
    ds = load_dataset(path, split=split)
    if size is not None:
        ds = ds.select(range(size))
    
    def tokenize_new(sample, reject=False):
        if not reject:
            sample['text'] = sample['chosen'] 
        else:
            sample['text'] = sample['rejected'] 
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = split_text[0] + '\n\nAssistant:'
        sample["input_ids"] = tokenizer.encode(sample['prompt'])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        
        # Dynamically encode with all available tokenizers
        for i, rm_tokenizer in enumerate(rm_tokenizers_list):
            sample[f"rm_ids_{i}"] = rm_tokenizer.encode(sample['text'])
            
        return sample

    ds_concat = ds.map(tokenize_new, batched=False, fn_kwargs={"reject": False}, num_proc=30)
    
    # Create filter condition checking all tokenizers
    def filter_func(x):
        # Base condition for input_ids
        if not (len(x["input_ids"]) <= 256 and len(x["input_ids"]) >= 8):
            return False
            
        # Check all reward model tokenizers
        for i in range(len(rm_tokenizers_list)):
            field_name = f"rm_ids_{i}"
            if not (len(x[field_name]) <= 256 and len(x[field_name]) >= 8):
                return False
        return True
        
    ds_concat = ds_concat.filter(filter_func)
    
    # Create list of columns to remove
    columns_to_remove = ['rejected', 'chosen', 'text']
    for i in range(len(rm_tokenizers_list)):
        columns_to_remove.append(f"rm_ids_{i}")
        
    ds_concat = ds_concat.remove_columns(columns_to_remove)
    ds_concat = ds_concat.remove_columns([c for c in ds_concat.column_names if c != "input_ids"])
    # ds_concat.set_format(type="torch")
    return ds_concat


def build_dataset_summary(path, tokenizer, rm_tokenizer, split='train', size=None):
    ds = load_dataset(path, 'comparisons')
    ds = ds[split]
    ds = ds.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200, batched=False, num_proc=30)

    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = Instructions_summary.prompt_input(info_post)
        sample["prompt"] = prompt_summary
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False,  num_proc=30) 
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    remove_columns = ['info', 'summaries', 'choice', 'worker', 'batch', 'split', 'extra']
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds


def build_dataset_eval(path, tokenizer, rm_tokenizers_list, split='test', size=None):
    ds = load_dataset(path, split=split)
    if size is not None:
        ds = ds.select(range(size))
    ds = ds.select(range(0, len(ds), 4))  
    
    # Remove this line:
    # rm_tokenizer1, rm_tokenizer2 = rm_tokenizers_list[:2]
    
    def tokenize(sample):
        sample['text'] = sample['chosen'] 
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['response'] = split_text[-1].strip()
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        
        # Dynamically encode with all available tokenizers
        for i, rm_tokenizer in enumerate(rm_tokenizers_list):
            sample[f"input_ids_rm{i+1}"] = rm_tokenizer.encode(sample["prompt"])
        
        return sample

    ds_chosen = ds.map(tokenize, batched=False, num_proc=20)
    ds_concat = ds_chosen
    
    # Create a dynamic filter condition
    filter_condition = lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8
    for i in range(len(rm_tokenizers_list)):
        field_name = f"input_ids_rm{i+1}"
        filter_condition = lambda x, condition=filter_condition, field=field_name: condition(x) and len(x[field]) <= 512 and len(x[field]) >= 8
    
    ds_concat = ds_concat.filter(filter_condition)
    
    # Remove all the added columns
    columns_to_remove = ['chosen', 'rejected', 'text', 'prompt', 'response', 'query']
    for i in range(len(rm_tokenizers_list)):
        columns_to_remove.append(f"input_ids_rm{i+1}")
    
    ds_concat = ds_concat.remove_columns(columns_to_remove)
    ds_concat.set_format(type="torch")
    return ds_concat


def build_dataset_summary_eval(path, tokenizer, split='test', size=None):
    if split == 'test':
        split = 'validation'
    ds = load_dataset(path, 'comparisons')
    ds = ds[split]
    ds = ds.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200, batched=False, num_proc=30)

    # need to remove duplicated prompts for evaluation
    def remove_duplicate(duplicated_dataset):
        duplicated_dataset = duplicated_dataset.filter(lambda x: x['info']["id"] is not None)
        initial_list = duplicated_dataset.map(lambda x: {"id": x['info']["id"]})
        _ , unique_indices = np.unique(initial_list["id"], return_index=True, axis=0)
        filtered_dataset = duplicated_dataset.select(unique_indices.tolist())
        return filtered_dataset

    ds = remove_duplicate(ds)
    if size is not None:
        ds = ds.select(range(size))
    ds = ds.select(range(0, min(len(ds),2000))) # select 2000 data 

    def tokenize(sample):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = Instructions_summary.prompt_input(info_post)
        sample["prompt"] = prompt_summary
        sample["input_ids"] = tokenizer.encode(prompt_summary)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False,  num_proc=30) 
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    remove_columns = ['info', 'summaries', 'choice', 'worker', 'batch', 'split', 'extra']
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds

def check_lora_in_model_path(model, path):
    if os.path.exists(path):
        dirnames = os.listdir(path)
        if 'adapter_config.json' in dirnames:
            return True
        
        state_dict_keys = model.state_dict().keys()
        for key in state_dict_keys:
            if 'lora' in key:
                return True
    return False


def load_reward_model(reward_peft_path, gpu_id):
    num_labels = 2 if ('humor' in reward_peft_path or 'faithful' in reward_peft_path) else 1
    reward_model = AutoModelForSequenceClassification.from_pretrained(
                    reward_peft_path,
                    num_labels=num_labels, torch_dtype=torch.bfloat16,
                    device_map=gpu_id,
                    )
    if check_lora_in_model_path(reward_model, reward_peft_path):
        reward_model = PeftModel.from_pretrained(reward_model, reward_peft_path)
    if hasattr(reward_model, 'merge_and_unload'):
        reward_model = reward_model.merge_and_unload() # merge lora weights
    return reward_model.to(gpu_id)


def load_main_tokenizer(tokenier_name):
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>" 
    DEFAULT_UNK_TOKEN = "<unk>" 

    tokenizer = AutoTokenizer.from_pretrained(tokenier_name,padding_side="left", use_fast = False)
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
    return tokenizer


def get_rewards(reward_model, texts_for_rewards, reward_mean_std=None, sub_position=0):
    rewards = []
    # print('log: reward model forwarding ...')
    # remove the progress bar
    with torch.no_grad():
        for inputs in texts_for_rewards:
            if sub_position != 0: # for multiple output
                rewards.append(reward_model(**(inputs.to(reward_model.device))).logits[0][sub_position])
            else:
                rewards.append(reward_model(**(inputs.to(reward_model.device))).logits[0])
    
    if reward_mean_std is None:
        rewards = [r.cpu().detach().item() for r in rewards]
    else:
        mean_reward, std_reward = reward_mean_std
        rewards = [(r.cpu().detach().item() - mean_reward) / std_reward for r in rewards]
    return rewards



def save_configs(config, path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'training_config.txt'), 'w+') as f:
        if type(config) == dict:
            lines = [key + ' : ' + config[key] + '\n' for key in config.keys()]
            f.writelines(lines)
        else:
            f.writelines(str(config))


def get_average_state_dict(state_dicts, coefficients):
    i = 0
    for state_dict, coefficient in zip(state_dicts, coefficients):
        current_weights = state_dict
        for key in list(current_weights.keys()):
            if i == 0:
                state_dicts[0][key] = coefficient * current_weights[key]
            else :
                state_dicts[0][key] += coefficient * current_weights[key]
        i += 1
    return state_dicts[0]


def merge_weights_with_preference(base_model_names, preference, temp_save_path):
    models = []
    for base_model_name in base_model_names:
        model_tmp = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map='cpu',
        )
        models.append(model_tmp)
    state_dicts = [model_tmp.state_dict() for model_tmp in models]
    average_weights = get_average_state_dict(state_dicts, preference)
    model_1 = models[0]
    model_1.load_state_dict(average_weights, strict=False)
    if os.path.exists(temp_save_path):
        shutil.rmtree(temp_save_path, ignore_errors=True)
    model_1.save_pretrained(temp_save_path)

    while len(models):
        del models[0]
    while len(state_dicts):
        del state_dicts[0]
    del average_weights
    gc.collect()
    torch.cuda.empty_cache()


def merge_lora_weight(model, path):
    if check_lora_in_model_path(model, path):
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
    return model


def get_clean_data(full_responses, full_prompts, remove_bad=False):
    full_prompts_clean = []
    full_responses_clean = []
    for i, response in enumerate(full_responses):
        full_prompts[i] = full_prompts[i].strip('[PAD] ').strip('[PAD]').strip('<s>').strip('</s>').strip()
        response = response.strip('[PAD] ').strip('[PAD]').strip('<s>').strip('</s>')
        temp_resp = response.replace(full_prompts[i], '').strip().strip('\n\n----').strip('\n\n----- ').strip()
        if '</s>' in temp_resp:
            temp_resp = temp_resp[:temp_resp.rindex('</s>')]
        temp_resp = temp_resp.split('\n\nHuman:')[0].strip()
        temp_resp = temp_resp.split('\nHuman:')[0].strip()
        temp_resp = temp_resp.split('\n\nAssistant:')[0].strip()
        temp_resp = temp_resp.split('\nAssistant:')[0].strip()
        temp_resp = temp_resp.split('\n\n\n')[0].strip()
        clean_resp = full_prompts[i] + ' ' + temp_resp
        if remove_bad and (('.....' in clean_resp) or (clean_resp.count(':)') >= 3)):
            ## pass bad sample
            continue
        full_responses_clean.append(clean_resp)
        full_prompts_clean.append(full_prompts[i])
    return full_prompts_clean, full_responses_clean
