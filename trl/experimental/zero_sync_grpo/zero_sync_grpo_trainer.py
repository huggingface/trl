# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import atexit
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

import torch
from accelerate.utils import gather
from datasets import Dataset, IterableDataset
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from transformers.generation import ContinuousBatchingConfig

from ...data_utils import is_conversational
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import RepeatSampler, nanstd, pad
from .zero_sync_grpo_config import ZeroSyncGRPOConfig


# Functions with this signature can be used as reward functions
RewardFunc = Callable[[list, list], list[float]]


class ZeroSyncGRPOTrainer(_BaseTrainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method with zero-sync generation. Generation and
    training share ONE copy of the weights: a transformers continuous batching manager generates from the same
    parameter tensors the optimizer updates in place, so there is no second engine, no weight synchronization and no
    generation/training memory duplication. The engine's per-token logprobs are the exact behavior-policy logprobs of
    the completions and are used as the old policy in the clipped loss.

    Generation never stops: the trainer keeps `args.generation_ahead` batches of prompts in flight and trains on the
    oldest completed one, so the engine decodes continuously through reward computation, the backward pass and the
    optimizer step. Completions therefore lag the policy by up to `generation_ahead` optimizer steps; the measured
    logprob gap this introduces is small and concentrated in each completion's earliest tokens, and the clipped loss
    against the engine's own logprobs accounts for it. The first training batch is submitted `generation_ahead + 1`
    times to prime the pipeline.

    Example:

    ```python
    from datasets import load_dataset
    from trl.experimental.zero_sync_grpo import ZeroSyncGRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")


    def reward_num_unique_chars(completions, **kwargs):
        return [len(set(c)) for c in completions]


    trainer = ZeroSyncGRPOTrainer(
        model="Qwen/Qwen3-0.6B",
        reward_funcs=reward_num_unique_chars,
        train_dataset=dataset,
    )
    trainer.train()
    ```

    Args:
        model (`str` or [`~transformers.PreTrainedModel`]):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object.
        reward_funcs (`Callable` or `list[Callable]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. The functions are provided with the
            prompts, the generated completions, plus any additional columns in the dataset, and must return a list of
            floats (or `None` for samples the function does not apply to).
        args ([`~trl.experimental.zero_sync_grpo.ZeroSyncGRPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], *optional*):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None,
            None)`):
            A tuple containing the optimizer and the scheduler to use.
    """

    _tag_names = ["trl", "zero-sync-grpo"]
    _name = "Zero-Sync GRPO"

    def __init__(
        self,
        model: str | PreTrainedModel,
        reward_funcs: RewardFunc | list[RewardFunc],
        args: ZeroSyncGRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = ZeroSyncGRPOConfig(f"{model_name}-ZeroSyncGRPO")

        # Model
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `ZeroSyncGRPOConfig`, but your model is already "
                    "instantiated. This argument can only be used when the `model` argument is a string."
                )

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path)
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs
        self.reward_func_names = [func.__name__ for func in reward_funcs]

        # Training arguments
        self.num_generations = args.num_generations
        self.max_completion_length = args.max_completion_length
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self.epsilon = args.epsilon
        self.generation_ahead = args.generation_ahead

        def data_collator(features):
            # No data collation is needed in GRPO
            return features

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        if global_batch_size % self.num_generations != 0:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations})."
            )

        # Zero-sync generation state: the continuous batching manager is created lazily on the first generation (the
        # model must be on its final device first). The pipeline holds the submitted batches; results arrive
        # interleaved across in-flight batches and are parked in `_results` until their batch is trained on.
        self._manager = None
        self._pipeline = deque()
        self._results = {}

        # Metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.per_device_train_batch_size * self.accelerator.num_processes // self.num_generations,
            shuffle=True,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _init_manager(self):
        # The manager is attached to the unwrapped training model: decoding reads the same parameter tensors the
        # optimizer updates in place. warmup() must run before start() so the cuda graphs are captured on the main
        # thread (transformers#48312).
        model = self.accelerator.unwrap_model(self.model)
        generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            eos_token_id=self.processing_class.eos_token_id,
        )
        cb_kwargs = dict(self.args.continuous_batching_config or {})
        cb_kwargs["return_logprobs"] = True  # the engine's logprobs are the behavior-policy logprobs
        self._manager = model.init_continuous_batching(
            generation_config=generation_config,
            continuous_batching_config=ContinuousBatchingConfig(**cb_kwargs),
        )
        self._manager.warmup()
        self._manager.start()
        # The background thread is not a daemon: without an explicit stop the process never exits, including after an
        # exception in the training loop.
        atexit.register(self._manager.stop, block=False)

    def _submit(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        # Tokenize the batch's prompts (re-tokenizing the whole conversation through the chat template) and submit one
        # generation request per row. The sampler already repeated each prompt `num_generations` times, so requests
        # map one-to-one to completions.
        prompts = [x["prompt"] for x in inputs]
        if is_conversational(inputs[0]):
            prompt_ids = [
                self.processing_class.apply_chat_template(
                    prompt, add_generation_prompt=True, **self.chat_template_kwargs
                )["input_ids"]
                for prompt in prompts
            ]
        else:
            prompt_ids = [self.processing_class(prompt)["input_ids"] for prompt in prompts]
        request_ids = [
            self._manager.add_request(ids, max_new_tokens=self.max_completion_length) for ids in prompt_ids
        ]
        return {"inputs": inputs, "prompts": prompts, "prompt_ids": prompt_ids, "request_ids": request_ids}

    def _collect(self, batch: dict[str, Any]) -> tuple[list[list[int]], list[list[float]]]:
        # Wait for every request of the batch; the manager keeps decoding the other in-flight batches meanwhile, and
        # their results arrive interleaved, so completed requests are parked in `_results` until their batch is
        # consumed. A dead background thread never raises from `get_result` (it returns None forever,
        # transformers#48334), so poll `fatal_error` to fail fast instead of spinning.
        while any(rid not in self._results for rid in batch["request_ids"]):
            result = self._manager.get_result(timeout=1.0)
            if result is None:
                fatal_error = self._manager.background_thread_status.fatal_error
                if fatal_error is not None:
                    raise RuntimeError("The continuous batching background thread died.") from fatal_error
                continue
            self._results[result.request_id] = result
        results = [self._results.pop(rid) for rid in batch["request_ids"]]
        completion_ids = [list(result.generated_tokens) for result in results]
        logprobs = [list(result.logprobs[: len(result.generated_tokens)]) for result in results]
        return completion_ids, logprobs

    def _prepare_inputs(self, generation_batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | Any]:
        return self._generate_and_score_completions(generation_batch)

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        if self._manager is None:
            self._init_manager()

        if mode == "train":
            # Submit-ahead pipeline: keep `generation_ahead` batches in flight and train on the oldest one, so the
            # engine decodes continuously through reward computation, the backward pass and the optimizer step. The
            # first batch primes the pipeline (its prompts are generated and trained `generation_ahead` extra times).
            while len(self._pipeline) < self.generation_ahead:
                self._pipeline.append(self._submit(inputs))
            self._pipeline.append(self._submit(inputs))
            batch = self._pipeline.popleft()
        else:
            batch = self._submit(inputs)
        completion_ids_list, logprobs_list = self._collect(batch)
        inputs, prompts, prompt_ids_list = batch["inputs"], batch["prompts"], batch["prompt_ids"]

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids_list, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": content}] for content in completions_text]
        else:
            completions = completions_text
        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This
        # is important because rewards will be normalized per group, and completions are distributed. We will later
        # slice rewards_per_func to extract each process's subset.
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        keys = [key for key in inputs[0] if key not in ["prompt"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        for i, reward_func in enumerate(self.reward_funcs):
            output_reward_func = reward_func(
                prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
            )
            # Convert None values to NaN
            output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        rewards_per_func = gather(rewards_per_func)

        # A completion for which every reward function returned None is unscorable. nansum would collapse it to 0,
        # which both biases the per-group baseline and hands the completion a spurious advantage. Mark these rows NaN
        # so they're excluded from the (nan-aware) baseline below; their advantage is forced to 0 afterwards.
        unscorable_mask = torch.isnan(rewards_per_func).all(dim=1)

        rewards = rewards_per_func.nansum(dim=1)
        rewards[unscorable_mask] = torch.nan
        mean_grouped_rewards = torch.nanmean(rewards.view(-1, self.num_generations), dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        if self.num_generations > 1:
            std_rewards = nanstd(rewards.view(-1, self.num_generations), dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        else:  # doesn't occur during training, but could occur in eval
            std_rewards = torch.zeros_like(rewards)

        advantages = (rewards - mean_grouped_rewards) / (std_rewards + 1e-4)

        # Unscorable completions (every reward func returned None) carry no learning signal: their reward is NaN here,
        # so zero their advantage to keep them from moving the policy.
        advantages = torch.nan_to_num(advantages, nan=0.0)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(
                torch.nanmean(rewards_per_func[:, i]).item()
            )
        self._metrics[mode]["reward"].append(torch.nanmean(rewards).item())
        self._metrics[mode]["reward_std"].append(nanstd(rewards).item())
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids_list], dtype=torch.float32)
        agg_completion_lengths = gather(completion_lengths.to(device))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.mean().item())

        # Build the training tensors: prompt + completion concatenated, with a mask over the completion tokens and
        # the engine's behavior-policy logprobs aligned to them.
        input_ids = [torch.tensor(p + c) for p, c in zip(prompt_ids_list, completion_ids_list, strict=True)]
        completion_mask = [
            torch.cat([torch.zeros(len(p), dtype=torch.long), torch.ones(len(c), dtype=torch.long)])
            for p, c in zip(prompt_ids_list, completion_ids_list, strict=True)
        ]
        old_per_token_logps = [
            torch.cat([torch.zeros(len(p), dtype=torch.float32), torch.tensor(lp, dtype=torch.float32)])
            for p, lp in zip(prompt_ids_list, logprobs_list, strict=True)
        ]
        pad_token_id = self.processing_class.pad_token_id
        input_ids = pad(input_ids, padding_value=pad_token_id, padding_side="right").to(device)
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right").to(device)
        old_per_token_logps = pad(old_per_token_logps, padding_value=0.0, padding_side="right").to(device)
        attention_mask = pad(
            [torch.ones(len(p) + len(c), dtype=torch.long) for p, c in zip(prompt_ids_list, completion_ids_list, strict=True)],
            padding_value=0,
            padding_side="right",
        ).to(device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "advantages": advantages,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]
        advantages = inputs["advantages"]

        # Per-token logprobs of the sampled tokens under the current policy. The first position has no prediction
        # target, so logits are shifted by one.
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits = logits[:, :-1, :] / self.temperature
        targets = input_ids[:, 1:]
        per_token_logps = torch.gather(
            logits.log_softmax(dim=-1), dim=2, index=targets.unsqueeze(-1)
        ).squeeze(-1)
        mask = completion_mask[:, 1:].to(per_token_logps.dtype)

        # Clipped surrogate loss. The old policy is the generation engine itself: its logprobs are exact for the
        # sampled tokens, so no extra forward pass is needed to obtain them.
        old_per_token_logps = inputs["old_per_token_logps"][:, 1:]
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss = -torch.min(ratio * advantages.unsqueeze(1), clipped_ratio * advantages.unsqueeze(1))
        loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)

        # Log the metrics
        mode = "train" if self.model.training else "eval"
        is_clipped = ((ratio < 1 - self.epsilon) | (ratio > 1 + self.epsilon)) & (mask > 0)
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather(is_clipped.sum() / mask.sum().clamp(min=1.0)).mean().item()
        )
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()
