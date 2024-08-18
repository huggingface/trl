import numpy as np

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizer, TrainerCallback, GenerationConfig
from datasets import Dataset
from accelerate import Accelerator

from .online_dpo_trainer import OnlineDPOTrainer
from .xpo_config import XPOConfig
from .utils import (
    batch_generation,
    get_reward,
    forward,
)
from ..models.utils import unwrap_model_for_generation


@staticmethod
def logits_to_logprobs(logits, response_ids, temperature=1.0):
    logits /= temperature + 1e-7
    logprobs = F.log_softmax(logits, dim=-1)
    return torch.gather(logprobs, -1, response_ids.unsqueeze(-1)).squeeze(-1)



class XPOTrainer(OnlineDPOTrainer):
    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_model = self.ref_model
        reward_model = self.reward_model
        tokenizer = self.tokenizer
        
        def repeat_generator():
            while True:
                yield from self.dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            min_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            ref_model.eval()
            model.eval()
            with torch.no_grad():
                all_queries = []
                all_chosen_responses = []
                all_rejected_responses = []
                all_ref_logprobs_chosen = []
                all_ref_logprobs_rejected = []
                all_ref_responses = []
                queries = data["input_ids"]
                all_queries.append(queries)

                # create batches
                for _ in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    context_length = queries.shape[1]

                    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
                        model_responses, _ =  batch_generation(
                            unwrapped_model,
                            queries,
                            args.local_rollout_forward_batch_size,
                            tokenizer.pad_token_id,
                            generation_config,
                        )

                    # reference model responses
                    with unwrap_model_for_generation(ref_model, accelerator) as unwrapped_model:
                        ref_responses, ref_logits = batch_generation(
                            ref_model,
                            queries,
                            args.local_rollout_forward_batch_size,
                            tokenizer.pad_token_id,
                            generation_config,
                        )
                        all_ref_responses.append(ref_responses)

                    # Compute rewards for both sets of responses
                    _, model_rewards, _ = get_reward(
                        reward_model, model_responses, tokenizer.pad_token_id, context_length
                    )
                    _, ref_rewards, _ = get_reward(
                        reward_model, ref_responses, tokenizer.pad_token_id, context_length
                    )
                    
                    # Create preference dataset
                    chosen_mask = model_rewards > ref_rewards
                    rejected_mask = ~chosen_mask

                    # chosen and rejected responses
                    chosen_responses = torch.where(chosen_mask.unsqueeze(1), model_responses, ref_responses)
                    rejected_responses = torch.where(rejected_mask.unsqueeze(1), model_responses, ref_responses)
                    all_chosen_responses.append(chosen_responses)
                    all_rejected_responses.append(rejected_responses)

                    ref_logprobs_model_responses = self.compute_logprobs(ref_model, model_responses, context_length)
                    ref_logprobs_ref_response = self.compute_logprobs(ref_model, ref_responses, context_length)

                    ref_logprobs_chosen = torch.where(chosen_mask.unsqueeze(1), ref_logprobs_model_responses, ref_logprobs_ref_response)
                    ref_logprobs_rejected = torch.where(rejected_mask.unsqueeze(1), ref_logprobs_model_responses, ref_logprobs_ref_response)
                    
                    all_ref_logprobs_chosen.append(ref_logprobs_chosen)
                    all_ref_logprobs_rejected.append(ref_logprobs_rejected)
                    
                # stack all the tensors
                all_queries = torch.cat(all_queries, dim=0)
                all_chosen_responses = torch.cat(all_chosen_responses, dim=0)
                all_rejected_responses = torch.cat(all_rejected_responses, dim=0)
                all_ref_logprobs_chosen = torch.cat(all_ref_logprobs_chosen, dim=0)
                all_ref_logprobs_rejected = torch.cat(all_ref_logprobs_rejected, dim=0)
                all_ref_responses = torch.cat(all_ref_responses, dim=0)
                torch.cuda.empty_cache()
                
            # Do multiple epochs of XPO training, with a fresh random shuffle in each epoch
            model.train()
            for epoch_idx in range(args.num_epochs):
                b_inds = np.random.permutation(args.local_batch_size // self.num_generation_per_prompt)
                minibatch_idx = 0
                for mini_batch_start in range(
                    0,
                    args.local_batch_size // self.num_generation_per_prompt,
                    args.local_mini_batch_size // self.num_generation_per_prompt,
                ):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size // self.num_generation_per_prompt
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(
                        0,
                        args.local_mini_batch_size // self.num_generation_per_prompt,
                        args.per_device_train_batch_size,
                    ):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                            ## context lengths
                            context_lengths = all_queries[micro_batch_inds].shape[1]

                            ## chosen
                            chosen_responses = all_chosen_responses[micro_batch_inds]
                            ## rejected
                            rejected_responses = all_rejected_responses[micro_batch_inds]

                            ## concated log_probs
                            concated_logprobs = self.compute_logprobs(
                                model, 
                                torch.cat([chosen_responses, rejected_responses], dim=0),
                                context_lengths,
                            )

                            (chosen_logprobs, rejected_logprobs) = torch.split(
                                concated_logprobs, [chosen_responses.shape[0], rejected_responses.shape[0]]
                            )

                            # ref logprobs
                            ref_logprobs_chosen = all_ref_logprobs_chosen[micro_batch_inds]
                            ref_logprobs_rejected = all_ref_logprobs_rejected[micro_batch_inds]

                            # log ratios
                            chosen_log_ratios = chosen_logprobs.sum(1) - ref_logprobs_chosen.sum(1)
                            rejected_log_ratios = rejected_logprobs.sum(1) - ref_logprobs_rejected.sum(1)
                            diff_log_ratios = chosen_log_ratios - rejected_log_ratios

                            # dpo losses
                            if self.loss_type == "sigmoid":
                                dpo_losses = -F.logsigmoid(self.beta * diff_log_ratios)
                            elif self.loss_type == "ipo":
                                losses = (diff_log_ratios - 1 / (2 * self.beta)) ** 2
                            else:
                                raise NotImplementedError(f"invalid loss type {self.loss_type}")

                            # xpo losses
                            model_logprobs_ref = self.compute_logprobs(
                                model, all_ref_responses[micro_batch_inds], context_lengths
                            )
                            xpo_losses = args.alpha * model_logprobs_ref.sum(1)

                            # total loss
                            loss = (dpo_losses + xpo_losses).mean()
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()            
                            
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    self.state.global_step += 1

                    self.lr_scheduler.step()
                    

    def compute_logprobs(self, model, responses, context_length):
        output = forward(model, responses, self.tokenizer.pad_token_id)
        logits = output.logits[:, context_length - 1 : -1]
        logits /= self.args.temperature + 1e-7
        logprobs = F.log_softmax(logits, dim=-1)
        target_ids = responses[:, context_length:]
        return torch.gather(logprobs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
