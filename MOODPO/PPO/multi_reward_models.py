from transformers import AutoTokenizer, pipeline
import torch
import numpy as np
import pandas as pd
from utils import load_reward_model, get_rewards

class RewardModels():
    def __init__(self, reward_model_path_list, rm_tokenizer_path_list, gpu_id_list, reward_stats_path=None):
        assert len(reward_model_path_list) == len(rm_tokenizer_path_list)
        self.reward_model_path_list = reward_model_path_list
        self.rm_tokenizer_path_list = rm_tokenizer_path_list
        self.num_rewards = len(reward_model_path_list)
        print(self.num_rewards)
        self.reward_stats = np.load(reward_stats_path) if reward_stats_path is not None else None
        self.reward_models = []
        self.rm_tokenizers = []
        if type(gpu_id_list) != list:
            gpu_id_list = [gpu_id_list, gpu_id_list, gpu_id_list]
    
        print('Loading reward models .....')
        for i in range(self.num_rewards):
            self.reward_models.append(load_reward_model(self.reward_model_path_list[i], gpu_id_list[i]))
            self.rm_tokenizers.append(AutoTokenizer.from_pretrained(self.rm_tokenizer_path_list[i]))
    
        
        # from transformers import pipeline
        # self.pipe = pipeline(model="Tristan/gpt2_reward_summarization", device=0, tokenizer=self.rm_tokenizers[0], function_to_apply="none")

    def get_reward_model_scores(self, queries_responses, summary_fun=None):
        texts_for_rewards = []
        for i in range(self.num_rewards):
            if i >= 1 and self.rm_tokenizer_path_list[i] == self.rm_tokenizer_path_list[i-1]:
                texts_for_rewards.append(texts_for_rewards[-1])
            elif 'faithful' in self.reward_model_path_list[i]:
                max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
                temp_encoded_texts = [self.rm_tokenizers[i](text=r, text_pair=summary_fun(q), return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
                texts_for_rewards.append(temp_encoded_texts)
            elif 'summary' in self.reward_model_path_list[i] or 'summarization' in self.reward_model_path_list[i]: # reverse prompt and response
                max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
                temp_encoded_texts = [self.rm_tokenizers[i](r + " " + self.rm_tokenizers[i].bos_token + " " + summary_fun(q), return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
                texts_for_rewards.append(temp_encoded_texts)
                ## For pipeline
                # posts = []
                # generated_summaries = []
                # for q, r in queries_responses:
                #     posts.append(q.split("### Input: ")[-1].split("### Response:")[0].strip())
                #     generated_summaries.append(r.split("### Response: ")[-1].strip())
                # bos_token = self.rm_tokenizers[i].bos_token
                # texts_for_rm = [generated_summaries[i] + ' ' + bos_token + ' ' + posts[i] for i in range(len(posts))]
            elif 'humor' in self.reward_model_path_list[i]: # use only response
                max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
                temp_encoded_texts = [self.rm_tokenizers[i](r, return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
                texts_for_rewards.append(temp_encoded_texts)
            else:
                max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
                temp_encoded_texts = [self.rm_tokenizers[i](q, r, return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
                texts_for_rewards.append(temp_encoded_texts)

        # normalize reward
        rewards = []
        for i in range(self.num_rewards):
            if self.reward_stats is not None:
                if type(self.reward_stats) == list or len(self.reward_stats) == 2 * self.num_rewards:
                    reward_mean_std = (self.reward_stats[2*i], self.reward_stats[2*i+1])
                else:
                    reward_mean_std = self.reward_stats[i]
            else:
                reward_mean_std = None

            if 'humor' in self.reward_model_path_list[i] or 'faithful' in self.reward_model_path_list[i]:
                temp_reward = get_rewards(self.reward_models[i], texts_for_rewards[i], reward_mean_std=reward_mean_std, sub_position=1)
            else:
                temp_reward = get_rewards(self.reward_models[i], texts_for_rewards[i], reward_mean_std=reward_mean_std)
                # temp_reward = [score['score'] for score in self.pipe(texts_for_rm)] # For pipeline
            rewards.append(temp_reward)
        return rewards
            
