# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase, TrainerCallback
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_apex_available

from ..models.modeling_base import GeometricMixtureWrapper
from ..models.utils import unwrap_model_for_generation
from .nash_md_config import NashMDConfig
from .online_dpo_trainer import OnlineDPOTrainer
from .utils import (
    get_reward,
    truncate_right,
)


if is_apex_available():
    pass


class NashMDTrainer(OnlineDPOTrainer):
    r"""
    Initialize NashMDTrainer as a subclass of [`OnlineDPOConfig`].

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForCausalLM`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        reward_model (`transformers.PreTrainedModel`):
            The reward model to score completions with, preferably an `AutoModelForSequenceClassification`.
        judge (`BasePairwiseJudge`):
            The judge to use for pairwise comparison of model completions.
        args (`NashMDConfig`):
            The NashMD config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
    """

    _tag_names = ["trl", "nash-md"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        reward_model: Optional[nn.Module] = None,
        args: Optional[NashMDConfig] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.mixture_coeff = args.mixture_coeff

        # Overwrite the stats dictionary to include NashMD specific statistics
        self.stats = {
            # Remove "non_score_reward", "rlhf_reward", "scores"
            # Add "loss/dpo"
            "loss/dpo": [],
            "objective/kl": [],
            "objective/entropy": [],
            # Replace "scores" by "model_scores" and "ref_scores"
            "objective/model_scores": [],
            "objective/ref_scores": [],
            "objective/scores_margin": [],
            "rewards/chosen": [],
            "rewards/rejected": [],
            "rewards/accuracies": [],
            "rewards/margins": [],
            "logps/chosen": [],
            "logps/rejected": [],
            # Replace "contain_eos_token" by "model_contain_eos_token" and "ref_contain_eos_token"
            "val/model_contain_eos_token": [],
            "val/ref_contain_eos_token": [],
        }

    def _generate_completions(self, model, prompts):
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            model_output = unwrapped_model.generate(
                input_ids=prompts["input_ids"],
                attention_mask=prompts["attention_mask"],
                generation_config=self.generation_config,
            )

            with torch.no_grad(), unwrap_model_for_generation(self.ref_model, self.accelerator) as unwrapped_ref_model:
                mixture_model = GeometricMixtureWrapper(
                    model=unwrapped_model,
                    ref_model=unwrapped_ref_model,
                    generation_config=self.generation_config,
                    mixture_coeff=self.mixture_coeff,
                    device=self.accelerator.device,
                )

                mixture_output = mixture_model.generate(
                    input_ids=prompts["input_ids"],
                    attention_mask=prompts["attention_mask"],
                    generation_config=self.generation_config,
                )

        return model_output, mixture_output

    def _process_completions(self, model_output, ref_output, prompts):
        context_length = prompts["input_ids"].shape[1]

        # Process model completions
        model_completion_ids = model_output[:, context_length:]
        model_completion_ids, model_completion_mask = truncate_right(
            model_completion_ids, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        )
        model_data = {
            "input_ids": torch.cat((prompts["input_ids"], model_completion_ids), dim=1),
            "attention_mask": torch.cat((prompts["attention_mask"], model_completion_mask), dim=1),
        }

        # Process reference model completions
        ref_completion_ids = ref_output[:, context_length:]
        ref_completion_ids, ref_completion_mask = truncate_right(
            ref_completion_ids, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        )
        ref_data = {
            "input_ids": torch.cat((prompts["input_ids"], ref_completion_ids), dim=1),
            "attention_mask": torch.cat((prompts["attention_mask"], ref_completion_mask), dim=1),
        }

        return model_data, ref_data

    def _compute_rewards(self, model_data, ref_data, context_length):
        all_input_ids = torch.cat([model_data["input_ids"], ref_data["input_ids"]], dim=0)

        with torch.no_grad():
            _, all_scores, _ = get_reward(
                self.reward_model, all_input_ids, self.tokenizer.pad_token_id, context_length
            )

        model_scores, ref_scores = all_scores.chunk(2)

        # Apply EOS penalty if needed
        if self.args.missing_eos_penalty is not None:
            model_contain_eos = torch.any(model_data["input_ids"] == self.tokenizer.eos_token_id, dim=-1)
            ref_contain_eos = torch.any(ref_data["input_ids"] == self.tokenizer.eos_token_id, dim=-1)
            model_scores[~model_contain_eos] -= self.args.missing_eos_penalty
            ref_scores[~ref_contain_eos] -= self.args.missing_eos_penalty

        return model_scores, ref_scores

    def _compute_logprobs(self, model, model_data, mixture_data, context_length):
        def compute_logprobs_for_data(m, data, return_logits=False, model_logits=None):
            output = m(data["input_ids"], attention_mask=data["attention_mask"])
            logits = output.logits[:, context_length - 1 : -1]
            if model_logits is not None:
                # geometric mixture of logits
                logits = logits * self.mixture_coeff + model_logits * (1 - self.mixture_coeff)
            logprobs = F.log_softmax(logits, dim=-1)
            token_logprobs = torch.gather(logprobs, 2, data["input_ids"][:, context_length:].unsqueeze(-1)).squeeze(-1)
            return token_logprobs, logits if return_logits else token_logprobs

        # Compute logprobs for model completions
        model_logprobs_model_data, model_logits_model_data = compute_logprobs_for_data(
            model, model_data, return_logits=True
        )
        # Compute logprobs for model on reference completions (for XPO loss)
        model_logprobs_mixture_data, model_logits_mixture_data = compute_logprobs_for_data(
            model, mixture_data, return_logits=True
        )

        # Compute logprobs for reference model completions
        with torch.no_grad():
            mixture_logprobs_model_data = compute_logprobs_for_data(
                self.ref_model, model_data, model_logits=model_logits_model_data.detach()
            )
            mixture_logprobs_mixture_data = compute_logprobs_for_data(
                self.ref_model, mixture_data, model_logits=model_logits_mixture_data.detach()
            )

        # Mask padding tokens
        model_padding_mask = model_data["attention_mask"][:, context_length:] == 0
        ref_padding_mask = mixture_data["attention_mask"][:, context_length:] == 0
        model_logprobs_model_data = model_logprobs_model_data.masked_fill(model_padding_mask, 0.0)
        model_logprobs_mixture_data = model_logprobs_mixture_data.masked_fill(ref_padding_mask, 0.0)
        mixture_logprobs_mixture_data = mixture_logprobs_mixture_data.masked_fill(ref_padding_mask, 0.0)
        mixture_logprobs_model_data = mixture_logprobs_model_data.masked_fill(model_padding_mask, 0.0)

        return (
            model_logprobs_model_data,
            model_logprobs_mixture_data,
            mixture_logprobs_mixture_data,
            mixture_logprobs_model_data,
        )

    def _compute_losses(
        self,
        model_logprobs_model_data,
        model_logprobs_mixture_data,
        mixture_logprobs_mixture_data,
        mixture_logprobs_model_data,
        model_data_scores,
        mixture_data_scores,
    ):
        # Compute log probs
        model_logprobs_model_data_sum = model_logprobs_model_data.sum(1)
        model_logprobs_ref_data_sum = model_logprobs_mixture_data.sum(1)
        mixture_logprobs_mixture_data_sum = mixture_logprobs_mixture_data.sum(1)
        mixture_logprobs_model_data_sum = mixture_logprobs_model_data.sum(1)

        chosen_model_logprobs = torch.where(chosen_mask, model_logprobs_model_data_sum, model_logprobs_ref_data_sum)
        chosen_ref_logprobs = torch.where(chosen_mask, ref_logprobs_model_data_sum, ref_logprobs_ref_data_sum)
        chosen_log_ratios = chosen_model_logprobs - chosen_ref_logprobs

        rejected_model_logprobs = torch.where(~chosen_mask, model_logprobs_model_data_sum, model_logprobs_ref_data_sum)
        rejected_ref_logprobs = torch.where(~chosen_mask, ref_logprobs_model_data_sum, ref_logprobs_ref_data_sum)
        rejected_log_ratios = rejected_model_logprobs - rejected_ref_logprobs

        # Compute logits as the difference between chosen and rejected log ratios
        logits = chosen_log_ratios - rejected_log_ratios

        if self.args.loss_type == "sigmoid":
            dpo_losses = -F.logsigmoid(self.args.beta * logits)
        elif self.args.loss_type == "ipo":
            dpo_losses = (logits - 1 / (2 * self.args.beta)) ** 2
        else:
            raise NotImplementedError(f"invalid loss type {self.args.loss_type}")

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        self.ref_model.eval()

        # need the prompt_ only
        inputs = self._prepare_inputs(inputs)
        context_length = inputs["prompt_input_ids"].shape[1]
        prompts = {
            "input_ids": inputs["prompt_input_ids"],
            "attention_mask": inputs["prompt_attention_mask"],
        }
        del inputs

        # Sample completions from both the model and the reference model
        model_output, mixture_output = self._generate_completions(model, prompts)

        # Process model completions
        model_data, mixture_data = self._process_completions(model_output, mixture_output, prompts)

        # Compute rewards
        model_data_scores, mixture_data_scores = self._compute_rewards(model_data, mixture_data, context_length)

        # Compute logprobs
        (
            model_logprobs_model_data,
            model_logprobs_mixture_data,
            mixture_logprobs_mixture_data,
            mixture_logprobs_model_data,
        ) = self._compute_logprobs(model, model_data, mixture_data, context_length)

        # Compute loss
        loss = self._compute_losses(
            model_logprobs_model_data,
            model_logprobs_mixture_data,
            mixture_logprobs_mixture_data,
            mixture_logprobs_model_data,
            model_data_scores,
            mixture_data_scores,
        )

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model

        reward_model = self.reward_model
        tokenizer = self.tokenizer
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            min_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )
        ref_model = self.ref_model
        # Mixture model is needed only to generate responses
        mixture_model = GeometricMixtureWrapper(
            model=model,
            ref_model=self.ref_model,
            generation_config=generation_config,
            mixture_coeff=args.mixture_coeff,
            device=accelerator.device,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        loss_stats = torch.zeros(stats_shape, device=device)
        preference_loss_stats = torch.zeros(stats_shape, device=device)
        kl_stats = torch.zeros(stats_shape, device=device)
        preference_model_vs_mixture_stats = torch.zeros(stats_shape, device=device)
        model_logprobs_stats = torch.zeros(stats_shape, device=device)
        ref_logprobs_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_updates * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for update in range(1, args.num_updates + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)

            with torch.no_grad():
                all_queries = []
                all_responses = []
                all_query_responses = []
                all_postprocessed_query_responses = []
                all_sequence_lengths = []
                all_model_rewards = []  # rewards of the responses generated by model
                all_mixture_rewards = []  # rewards of the responses generated by ref mixture model
                all_preference_model_vs_mixture = []  # prefrence of the responses generated by model vs by ref mixture model
                all_ref_logprobs_model_response = []  # logprobs of model responses generated by ref mixture model

                queries = data["input_ids"]
                all_queries.append(queries)
                context_length = queries.shape[1]

                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    query_responses, logits_responses = batch_generation(
                        unwrapped_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        tokenizer.pad_token_id,
                        generation_config,
                    )

                with unwrap_model_for_generation(mixture_model, self.accelerator) as unwrapped_mixture_model:
                    mixture_query_responses, mixture_logits_responses = batch_generation(
                        unwrapped_mixture_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        tokenizer.pad_token_id,
                        generation_config,
                    )

                # To have a generality of the code
                all_query_responses.append(query_responses)

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    # get responses from the model and the reference mixture model
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    mixture_query_response = mixture_query_responses[i : i + args.local_rollout_forward_batch_size]
                    mixture_response = mixture_query_response[:, context_length:]

                    all_query_responses.append(query_response)
                    all_responses.append(response)

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    postprocessed_mixture_response = mixture_response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, tokenizer.pad_token_id, response
                        )
                        postprocessed_mixture_response = truncate_response(
                            args.stop_token_id, tokenizer.pad_token_id, mixture_response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    postprocessed_mixture_query_response = torch.cat((query, postprocessed_mixture_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                    all_postprocessed_query_responses.append(postprocessed_query_response)
                    all_sequence_lengths.append(sequence_length)

                    # Compute rewards for both sets of responses
                    _, model_reward, _ = get_reward(
                        reward_model, postprocessed_query_response, tokenizer.pad_token_id, context_length
                    )
                    _, mixture_reward, _ = get_reward(
                        reward_model, postprocessed_mixture_query_response, tokenizer.pad_token_id, context_length
                    )
                    # if the responses do not contain eos token then set the reward to penalty_reward_value
                    if args.non_eos_penalty:
                        model_resp_contains_eos = torch.any(
                            postprocessed_query_response == tokenizer.eos_token_id, dim=-1
                        )
                        model_reward = torch.where(model_resp_contains_eos, model_reward, args.penalty_reward_value)
                        ref_resp_contains_eos = torch.any(
                            postprocessed_mixture_query_response == tokenizer.eos_token_id, dim=-1
                        )
                        mixture_reward = torch.where(ref_resp_contains_eos, mixture_reward, args.penalty_reward_value)

                    # Save model rewards for logging
                    all_model_rewards.append(model_reward)
                    all_mixture_rewards.append(mixture_reward)

                    # Compute the preference between the model and the mixture model
                    # TODO: Replace it by a soft judge instead of BT model (P(model > ref) = sigmoid(model_reward - ref_reward))
                    preference_model_vs_mixture = F.sigmoid(model_reward - mixture_reward)
                    all_preference_model_vs_mixture.append(preference_model_vs_mixture)

                    # Compute the logprobs of the responses generated by the model, preserve all the disribution; shape [batch_size, response_length, vocab_size]
                    ref_logprobs_model_response = self.compute_logprobs(ref_model, query_response, context_length)
                    all_ref_logprobs_model_response.append(ref_logprobs_model_response)

                # stack all the tensors
                all_queries = torch.cat(all_queries, dim=0)
                all_responses = torch.cat(all_responses, dim=0)
                all_query_responses = torch.cat(all_query_responses, dim=0)
                all_postprocessed_query_responses = torch.cat(all_postprocessed_query_responses, dim=0)
                all_sequence_lengths = torch.cat(all_sequence_lengths, dim=0)
                all_model_rewards = torch.cat(all_model_rewards, dim=0)
                all_mixture_rewards = torch.cat(all_mixture_rewards, dim=0)
                all_preference_model_vs_mixture = torch.cat(all_preference_model_vs_mixture, dim=0)
                all_ref_logprobs_model_response = torch.cat(all_ref_logprobs_model_response, dim=0)

                del model_reward, mixture_reward, preference_model_vs_mixture, ref_logprobs_model_response
                torch.cuda.empty_cache()
                gc.collect()

            # Do multiple epochs of training, with a fresh random shuffle in each epoch
            for epoch_idx in range(args.num_epochs):
                b_inds = np.random.permutation(args.local_batch_size // args.num_generation_per_prompt)
                minibatch_idx = 0
                for mini_batch_start in range(
                    0,
                    args.local_batch_size // args.num_generation_per_prompt,
                    args.local_mini_batch_size // args.num_generation_per_prompt,
                ):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size // args.num_generation_per_prompt
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(
                        0,
                        args.local_mini_batch_size // args.num_generation_per_prompt,
                        args.per_device_train_batch_size,
                    ):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                            ## context lengths
                            context_lengths = all_queries[micro_batch_inds].shape[1]

                            ## response, query-responses and sequence lengths
                            responses = all_responses[micro_batch_inds]
                            query_responses = all_query_responses[micro_batch_inds]
                            sequence_lengths = all_sequence_lengths[micro_batch_inds]

                            ## padding masks
                            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(
                                responses.shape[0], 1
                            )
                            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)

                            ## compute model log_probs, shape [batch_size, response_length, vocab_size]
                            model_logprobs = self.compute_logprobs(model, query_responses, context_lengths)

                            ## pre-calcualted preference model vs mixture
                            preference_model_vs_mixture = all_preference_model_vs_mixture[micro_batch_inds]

                            ## pre-calculated ref logprobs for model response
                            ref_logprobs = all_ref_logprobs_model_response[micro_batch_inds]

                            ## Preference loss (sign `-` is because we want to maximize the preference instead of minimizing)
                            model_logprobs_over_gen = torch.gather(
                                model_logprobs, 2, query_responses[:, context_lengths:].unsqueeze(-1)
                            ).squeeze(-1)
                            model_logprobs_sum = torch.sum(model_logprobs_over_gen * ~padding_mask, dim=1)
                            preference_losses = -model_logprobs_sum * (
                                preference_model_vs_mixture - 0.5
                            )  # 0.5 is a control variate

                            ## KL loss
                            raw_kl_model_vs_ref = torch.sum(
                                torch.exp(model_logprobs) * (model_logprobs - ref_logprobs), dim=2
                            )
                            kl_model_vs_ref = torch.sum(raw_kl_model_vs_ref * ~padding_mask, dim=1)

                            # total loss
                            loss = (preference_losses + self.beta * kl_model_vs_ref).mean()

                            ## ref logprobs for logging
                            ref_logprobs_over_gen = torch.gather(
                                ref_logprobs, 2, query_responses[:, context_lengths:].unsqueeze(-1)
                            ).squeeze(-1)
                            ref_logprobs_sum = torch.sum(ref_logprobs_over_gen * ~padding_mask, dim=1)

                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                loss_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = loss
                                preference_loss_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    preference_losses.mean()
                                )
                                kl_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = kl_model_vs_ref.mean()
                                preference_model_vs_mixture_stats[
                                    epoch_idx, minibatch_idx, gradient_accumulation_idx
                                ] = preference_model_vs_mixture.mean()
                                model_logprobs_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    model_logprobs_sum.mean()
                                )
                                ref_logprobs_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    ref_logprobs_sum.mean()
                                )
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1

            with torch.no_grad():
                model_reward = all_model_rewards.mean()
                mixture_reward = all_mixture_rewards.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(kl_stats).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(-model_logprobs_stats).mean().item()
                metrics["objective/preference"] = (
                    self.accelerator.gather(preference_model_vs_mixture_stats).mean().item()
                )
                metrics["objective/model_rew"] = self.accelerator.gather(model_reward).mean().item()
                metrics["objective/mixture_rew"] = self.accelerator.gather(mixture_reward).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(loss_stats).mean().item()
                metrics["loss/pref_loss_avg"] = self.accelerator.gather(preference_loss_stats).mean().item()
                metrics["logps/model"] = self.accelerator.gather(model_logprobs_stats).mean().item()
                metrics["logps/ref"] = self.accelerator.gather(ref_logprobs_stats).mean().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.log(metrics)

            self.lr_scheduler.step()
            self.state.global_step += 1

            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def compute_logprobs(self, model, responses, context_length, ref_model=None, mixture_coeff=0):
        output = forward(model, responses, self.tokenizer.pad_token_id)
        logits = output.logits[:, context_length - 1 : -1]
        logits /= self.args.temperature + 1e-7

        if ref_model is not None:
            with torch.no_grad():
                ref_output = forward(ref_model, responses, self.tokenizer.pad_token_id)
                ref_logits = ref_output.logits[:, context_length - 1 : -1]
                ref_logits /= self.args.temperature + 1e-7
                logits = mixture_coeff * ref_logits + (1 - mixture_coeff) * logits

        # return the log of the distribution over tokens
        return F.log_softmax(logits, dim=-1)  # shape [batch_size, response_length, vocab_size]
