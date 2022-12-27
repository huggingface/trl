# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from typing import Dict, List, Optional, Tuple
from accelerate import Accelerator
from datasets import load_dataset

from torch.optim import Adam
import torch.nn as nn
import torch
import time
import random
import wandb

from transformers import DataCollatorForLanguageModeling, AutoTokenizer, PreTrainedTokenizer

from trl.core import (logprobs_from_logits,
                      whiten,
                      clip_by_value,
                      entropy_from_logits,
                      flatten_dict,
                      stats_to_np,
                      stack_dicts,
                      WANDB_PADDING)
from trl.trainer import BaseTrainer, AdaptiveKLController, FixedKLController, LengthSampler
from trl import AutoModelForCausalLMWithValueHead


class AcceleratePPOTrainer(BaseTrainer):
    """
    The PPO_trainer uses Proximal Policy Optimization to optimise language models.
    """

    default_params = {
        "lr": 1.41e-5,
        "adap_kl_ctrl": True,
        "init_kl_coef":0.2,
        "target": 6,
        "horizon":10000,
        "gamma":1,
        "lam":0.95,
        "cliprange": .2,
        "cliprange_value":.2,
        "vf_coef":.1,
        "batch_size": 256,
        "forward_batch_size": 16,
        "ppo_epochs": 4,
    }

    def __init__(
        self,
        model: Optional[nn.Module]=None, 
        ref_model: Optional[nn.Module]=None, 
        tokenizer: Optional[PreTrainedTokenizer]=None, 
        **config
    ):
        """
        Initialize PPOTrainer.
        Args:
            model (torch.model): Hugging Face transformer GPT2 model with value head
            ref_model (torch.model): Hugging Face transformer GPT2 refrence model used for KL penalty
            tokenizer (tokenizer): Hugging Face tokenizer
            ppo_params (dict or None): PPO parameters for training. Can include following keys:
                'lr' (float): Adam learning rate, default: 1.41e-5
                'batch_size' (int): Number of samples per optimisation step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time, default: 16
                'ppo_epochs' (int): Number of optimisation epochs per batch of samples, default: 4
                'gamma' (float)): Gamma parameter for advantage calculation, default: 1.
                'lam' (float): Lambda parameter for advantage calcualation, default: 0.95
                'cliprange_value' (float): Range for clipping values in loss calculation, default: 0.2
                'cliprange' (float): Range for clipping in PPO policy gradient loss, default: 0.2
                'vf_coef' (float): Scaling factor for value loss, default: 0.1
                'adap_kl_ctrl' (bool): Use adaptive KL control, otherwise linear, default: True
                'init_kl_coef' (float): Initial KL penalty coefficient (used for adaptive and linear control), default: 0.2
                'target' (float): Target KL value for adaptive KL control, default: 6.0
                'horizon' (float): Horizon for adaptive KL control, default: 10000
        """
        super().__init__(self.default_params)

        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(log_with="wandb")
        self.config.update(config)

        # Step 2: Initialize model, tokenizer and dataset        
        self.model, self.ref_model, self.tokenizer = self._build_models_and_tokenizer(model, ref_model, tokenizer)
        self._build_dataset()


        # Step 3: Initialize optimizer and data collator        
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.optimizer = Adam(self.model.parameters(), lr=self.config['lr'])

        if self.config['adap_kl_ctrl']:
            self.kl_ctl = AdaptiveKLController(self.config['init_kl_coef'],
                                               self.config['target'],
                                               self.config['horizon'])
        else:
            self.kl_ctl = FixedKLController(self.config['init_kl_coef'])

        self.model, self.ref_model, self.optimizer, self.data_collator, self.dataloader = self.accelerator.prepare(self.model, self.ref_model, self.optimizer, self.data_collator, self.dataloader)

        # In a distributed setup, only logging needs to be performed on the main process
        # check: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        # or: https://discuss.pytorch.org/t/use-distributed-data-parallel-correctly/82500/11
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"

        # init wandb on the main process:
        if self.accelerator.is_main_process:
            wandb.init(name='run-42', project='gpt2-test', config=config)
            wandb.watch(self.model, log='all')

    def _build_dataset(self, dataset_name: str="imdb"):
        """
        Build dataset for training. This builds the dataset from `load_dataset`, one should 
        customize this function to train the model on its own dataset.
        
        Args:
            dataset_name (`str`): 
                The name of the dataset to be loaded.
        """
        # load imdb with datasets
        ds = load_dataset(dataset_name, split='train')
        ds = ds.rename_columns({'text': 'review', 'label': 'sentiment'})
        ds = ds.filter(lambda x: len(x["review"])>200, batched=False)


        self.input_size = LengthSampler(self.config["txt_in_min_len"], self.config["txt_in_max_len"])
        self.output_size = LengthSampler(self.config["txt_out_min_len"], self.config["txt_out_max_len"])

        def tokenize(sample):
            sample["tokens"] = self.tokenizer.encode(sample["review"])[:self.input_size()]
            sample["query"] = self.tokenizer.decode(sample["tokens"])
            return sample

        ds = ds.map(tokenize, batched=False)

        def collater(data):
            return dict((key, [d[key] for d in data]) for key in data[0])

        self.dataloader = torch.utils.data.DataLoader(ds, batch_size=self.config['batch_size'], collate_fn=collater)

    def generate(self, query_tensors: torch.Tensor, **gen_kwargs):
        """
        Generate response given query.


        Args:
            query_tensors (`torch.LongTensor`): 
                A tensor of shape (`batch_size`, `seq_len`) containing query tokens.
            gen_kwargs (dict[str, Any]): 
                Keyword arguments for generation.
        
        Returns: 
            response_tensors (`torch.LongTensor`): 
                A tensor of shape (`batch_size`, `gen_len`) containing response tokens. `gen_len` 
                is the length of the generated response that is sampled from `LengthSampler`.
        """
        response_tensors = []
        for i in range(self.config['batch_size']):
            gen_len = self.output_size()

            # In a multi-GPU setup the model is wrapped inside a DistributedDataParallel object.
            # this means that the model needs to be called with the module attribute.
            if self.accelerator.distributed_type == "MULTI_GPU":
                response = self.model.module.generate(query_tensors[i].unsqueeze(dim=0),
                                        max_new_tokens=gen_len, **gen_kwargs)
            else:
                response = self.model.generate(query_tensors[i].unsqueeze(dim=0),
                                            max_new_tokens=gen_len, **gen_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        return response_tensors

    def _build_models_and_tokenizer(
        self, 
        model: Optional[nn.Module] = None, 
        ref_model: Optional[nn.Module] = None , 
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        """
        Build models for training. This builds the reference model
        together with the model to be trained.

        Args:
            model (`nn.Module`, *optional*):
                The model to be trained. If `None` is passed, the model will be loaded from
                the pretrained model specified in the config.
            ref_model (`nn.Module`, *optional*): 
                The reference model. If `None` is passed, the model will be loaded from
                the pretrained model specified in the config. For mixed precision setup, 
                the reference model will be loaded in half precision according to the 
                `accelerator.mixed_precision` flag to save memory.
            tokenizer (`transformers.PreTrainedTokenizer`, *optional*): 
                The tokenizer to be used. If `None` is passed, the tokenizer will be loaded from
                the pretrained model specified in the config.
        """
        target_dtype_dict = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        target_dtype = torch.float32 if self.accelerator.mixed_precision not in target_dtype_dict else target_dtype_dict[self.accelerator.mixed_precision]

        # We don't cast the base model that is trained since autocast will do it under the hood for us
        # Check: https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372/4
        if model is None:
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.config['model_name'])
        else:
            self.model = model
        # But we can cast the reference model since the weights are not updated - we do that to save memory
        if ref_model is None:
            self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.config['model_name'], torch_dtype=target_dtype)
        else:
            self.ref_model = ref_model

        # tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
            # HACK: do we really need this?
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer


    def step(self, 
        queries: List[torch.LongTensor], 
        responses: List[torch.LongTensor], 
        scores: List[torch.FloatTensor]):
        """
        Run a PPO optimisation step.
        
        Args:
            queries (List[`torch.LongTensor`]): 
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]): 
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]): 
                List of tensors containing the scores of shape (`batch_size`)
        
        Returns:
            train_stats (dict[str, Any]): 
                a summary of the training statistics
        """

        bs = self.config['batch_size']
        assert bs == len(queries), f"Batch size ({bs}) does not match number of examples ({len(queries)})"

        timing = dict()
        t0 = time.time()

        response_lengths = [len(r) for r in responses]

        t = time.time()
        queries = [q.to(self.accelerator.device) for q in queries]
        responses = [r.to(self.accelerator.device) for r in responses]
        logprobs, ref_logprobs, values = self.batched_forward_pass(queries, responses)
        timing['time/ppo/forward_pass'] = time.time()-t

        t = time.time()
        rewards, non_score_reward = self.compute_rewards(scores, logprobs, ref_logprobs)
        timing['time/ppo/compute_rewards'] = time.time()-t

        t = time.time()
        all_stats = []
        idxs = list(range(bs))
        for _ in range(self.config['ppo_epochs']):
            random.shuffle(idxs)
            for i in range(bs):
                idx = idxs[i]
                train_stats = self.train_minibatch(logprobs[idx].unsqueeze(0), values[idx].unsqueeze(0),
                                                   rewards[idx].unsqueeze(0), queries[idx].unsqueeze(0),
                                                   responses[idx].unsqueeze(0),
                                                   torch.cat([queries[idx],responses[idx]]).unsqueeze(0))
                all_stats.append(train_stats)
        timing['time/ppo/optimize_step'] = time.time()-t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
        train_stats['policy/advantages'] = torch.nan_to_num(train_stats['policy/advantages'], WANDB_PADDING)
        train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)

        stats = self.record_step_stats(scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs,
                                       non_score_reward=non_score_reward, train_stats=train_stats,
                                       kl_coef=self.kl_ctl.value)
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing['time/ppo/calc_stats'] = time.time()-t

        self.kl_ctl.update(stats['objective/kl'], self.config['batch_size'])

        # Log the total ppo time
        timing['time/ppo/total'] = time.time()-t0
        stats.update(timing)
        return stats
    
    def gather_stats(self, stats):
        """
        Gather stats from all processes. Useful in the context of distributed training.

        Args:
            stats (dict[str, Any]): 
            a dictionary of stats to be gathered. The stats should contain torch tensors.
        
        Returns:
            stats (dict[str, Any]): 
                a dictionary of stats with the tensors gathered.
        """
        import torch.distributed as dist

        # Wait for all processes to finish
        dist.barrier()

        for k, v in stats.items():
            # We don't update the `'objective/kl'` since it is needed by each independent process
            # TODO: ask if it makes sense to average everything or just the objective (loss? reward?)
            if isinstance(v, torch.Tensor) and k != 'objective/kl':
                dist.all_reduce(v, dist.ReduceOp.SUM)
                v /= self.accelerator.num_processes
            stats[k] = v
        return stats

    def batched_forward_pass(self, queries: torch.Tensor, responses: torch.Tensor):
        """
        Calculate model outputs in multiple batches.
        
        Args:
            queries (`torch.LongTensor`): 
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`): 
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
        
        Returns:
            all_logprobs (`torch.FloatTensor`): 
                List of tensors containing the logprobs, shape (`batch_size`, `response_length`)
            all_ref_logprobs (`torch.FloatTensor`): 
                List of tensors containing the logprobs from the reference model, shape (`batch_size`, `response_length`)
            all_values (`torch.FloatTensor`): 
                List of tensors containing the output from the value head, shape (`batch_size`, `response_length`)

        """
        bs = self.config['batch_size']
        fbs = self.config['forward_batch_size']
        all_logprobs = []
        all_ref_logprobs = []
        all_values = []

        for i in range(int(bs/fbs)):
            query_batch = queries[i*fbs:(i+1)*fbs]
            response_batch = responses[i*fbs:(i+1)*fbs]
            input_ids = self.data_collator([torch.cat([q, r]) for q, r in zip(query_batch, response_batch)])["input_ids"]
            with torch.no_grad():
                logits, _, v = self.model(input_ids)
                ref_logits, _, _ = self.ref_model(input_ids)
            logprobs = logprobs_from_logits(logits[:,:-1,:], input_ids[:,1:])
            ref_logprobs = logprobs_from_logits(ref_logits[:,:-1,:], input_ids[:,1:])
            for j in range(fbs):
                start = len(query_batch[j])-1
                end = len(query_batch[j]) + len(response_batch[j])-1
                all_values.append(v[j, start-1:end-1])
                all_logprobs.append(logprobs[j, start:end])
                all_ref_logprobs.append(ref_logprobs[j, start:end])
        return all_logprobs, all_ref_logprobs, all_values

    def train_minibatch(
        self, 
        logprobs: torch.FloatTensor, 
        values: torch.FloatTensor, 
        rewards: torch.FloatTensor, 
        query: torch.LongTensor, 
        response: torch.LongTensor, 
        model_input: torch.LongTensor,
    ):
        """
        Train one PPO minibatch
        
        Args:
            logprobs (`torch.FloatTensor`): 
                Log probabilities of the model, shape [batch_size, response_length]
            values (`torch.FloatTensor`): 
                Values of the value head, shape [batch_size, response_length]
            rewards (`torch.FloatTensor`): 
                Rewards from the reward model, shape [batch_size, response_length]
            query (`torch.LongTensor`): 
                Encoded queries, shape [batch_size, query_length]
            response (`torch.LongTensor`): 
                Encoded responses, shape [batch_size, response_length]
            model_input (`torch.LongTensor`): 
                Concatenated queries and responses, shape [batch_size, query_length+response_length]
        
        Returns:
            train_stats (dict[str, `torch.Tensor`]): 
                Dictionary of training statistics
        """
        loss_p, loss_v, train_stats = self.loss(logprobs, values, rewards, query, response, model_input)
        loss = loss_p + loss_v
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        t = time.time()
        self.optimizer.step()
        train_stats['time/ppo/optimizer_step'] = torch.Tensor([time.time()-t]).to(self.accelerator.device)
        return train_stats

    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """Compute per token rewards from scores and KL-penalty."""
        rewards, non_score_rewards = [], []
        for score, logprob, ref_logprob in zip(scores, logprobs, ref_logprobs):
            kl = logprob - ref_logprob
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            reward[-1] += score
            rewards.append(reward)
        return rewards, non_score_rewards

    def loss(self, old_logprobs, values, rewards, query, response, model_input):
        """Calculate policy and value losses."""
        lastgaelam = 0
        advantages_reversed = []
        gen_len = response.shape[1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.config['gamma'] * nextvalues - values[:, t]
            lastgaelam = delta + self.config['gamma'] * self.config['lam'] * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = whiten(advantages)
        advantages = advantages.detach()

        logits, _, vpred = self.model(model_input)
        logprob = logprobs_from_logits(logits[:,:-1,:], model_input[:, 1:])

        #only the generation part of the values/logprobs is needed
        logprob, vpred = logprob[:, -gen_len:], vpred[:,-gen_len-1:-1]

        vpredclipped = clip_by_value(vpred,
                                     values - self.config["cliprange_value"],
                                     values + self.config["cliprange_value"])

        vf_losses1 = (vpred - returns)**2
        vf_losses2 = (vpredclipped - returns)**2
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac =  torch.mean(torch.gt(vf_losses2, vf_losses1).double())

        ratio = torch.exp(logprob - old_logprobs)

        if len(ratio.size()) != len(advantages.size()):
            ratio = ratio.unsqueeze(-1)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.config['cliprange'],
                                               1.0 + self.config['cliprange'])

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

        loss = pg_loss + self.config['vf_coef'] * vf_loss

        entropy = torch.mean(entropy_from_logits(logits))
        approxkl = .5 * torch.mean((logprob - old_logprobs)**2)
        policykl = torch.mean(logprob - old_logprobs)
        return_mean, return_var = torch.mean(returns), torch.var(returns)
        value_mean, value_var = torch.mean(values), torch.var(values)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl,policykl=policykl, clipfrac=pg_clipfrac,
                        advantages=advantages, advantages_mean=torch.mean(advantages), ratio=ratio),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(vpred=torch.mean(vpred), error=torch.mean((vpred - returns) ** 2),
                     clipfrac=vf_clipfrac, mean=value_mean, var=value_var),
        )
        return pg_loss, self.config['vf_coef'] * vf_loss, flatten_dict(stats)


    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        kl_list = [logprobs-ref_logprobs for logprobs, ref_logprobs in zip(data['logprobs'], data['ref_logprobs'])]
        mean_kl = torch.mean(torch.stack([torch.sum(kl) for kl in kl_list]))
        mean_entropy = torch.mean(torch.stack([torch.sum(-log_probs) for log_probs in data['logprobs']]))
        mean_non_score_reward =torch.mean(torch.stack([torch.sum(non_score_reward) for non_score_reward in data['non_score_reward']]))
        stats = {
            'objective/kl': mean_kl,
            'objective/kl_dist': kl_list,
            'objective/logprobs': data['logprobs'],
            'objective/ref_logprobs': data['ref_logprobs'],
            'objective/kl_coef': kl_coef,
            'objective/entropy': mean_entropy,
            'ppo/mean_non_score_reward': mean_non_score_reward,
        }

        for k, v in data['train_stats'].items():
            stats[f'ppo/{k}'] = torch.mean(v, axis=0)
        stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        return stats
    

    def log_stats(self, stats, timing, batch, rewards, t0, logs):
        """
        A function that logs all the training stats. Call it at the end of each epoch.

        Args:
            stats (dict[str, Any]): 
                A dictionary of training stats.
            timing (dict[str, Any]): 
                A dictionary of timing stats.
            batch (dict[str, Any]): 
                A dictionary of batch data, this containes the queries and responses.
            rewards (`torch.FloatTensor`): 
                A tensor of rewards.
            t0 (`float`): 
                The time at the start of the epoch.
            logs (dict[str, Any]): 
                A dictionary of logs.
        """
        # Log only if we are in the main process
        if self.accelerator.is_main_process:
            timing['time/epoch'] = time.time()-t0
            table_rows = [list(r) for r in zip(batch['query'], batch['response'], rewards.cpu().tolist())]
            logs.update({'game_log': wandb.Table(columns=['query', 'response', 'reward'], rows=table_rows)})
            logs.update(timing)
            logs.update(stats)
            logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
            logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
            logs['env/reward_dist'] = rewards.cpu().numpy()
            wandb.log(logs)
