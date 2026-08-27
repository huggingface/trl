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
from datasets import Dataset, IterableDataset
from transformers import (
    AutoProcessor,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.generation import ContinuousBatchingConfig

from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import create_model_from_path, nanstd, pad
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

    Generation never stops. Prompts are submitted to the engine continuously, each completion is collected as it
    finishes, a group's advantages are computed as soon as its last completion lands, and a training batch is formed
    from whichever scored samples are ready first. A slow group never blocks a batch of fast ones; it simply lands in
    a later batch. Completions therefore lag the policy by a bounded number of optimizer steps; the measured logprob
    gap this introduces is small and concentrated in each completion's earliest tokens, and the clipped loss against
    the engine's own logprobs accounts for it.

    Groups are formed per process: each process generates and scores its own prompts, so advantages require no
    cross-process communication.

    Example:

    ```python
    from datasets import load_dataset
    from trl.experimental.zero_sync_grpo import ZeroSyncGRPOTrainer

    dataset = load_dataset("trl-lib/tldr-chat", split="train")


    def reward_num_unique_chars(completions, **kwargs):
        return [len(set(c[0]["content"])) for c in completions]


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
            functions with the prompts and completions and sum the rewards. The functions are provided with one
            group's prompts, completions and completion ids, plus any additional columns in the dataset, and must
            return a list of floats (or `None` for samples the function does not apply to).
        args ([`~trl.experimental.zero_sync_grpo.ZeroSyncGRPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"` in the
            [conversational](dataset_formats#conversational) format (a list of messages). Any additional columns are
            passed to the reward functions.
        eval_dataset ([`~datasets.Dataset`], *optional*):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's
            name with [`~transformers.AutoProcessor.from_pretrained`]. For vision-language models the processor's
            tokenizer is used; only text-only data is supported.
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
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = ZeroSyncGRPOConfig(f"{model_name}-ZeroSyncGRPO")

        # Model. The architecture (causal LM or vision-language model) is inferred from the checkpoint; VLMs are
        # supported with text-only data.
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            # The same weights both generate and train, so default to the checkpoint dtype rather than fp32
            model_init_kwargs.setdefault("dtype", "auto")
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `ZeroSyncGRPOConfig`, but your model is already "
                    "instantiated. This argument can only be used when the `model` argument is a string."
                )

        # Processing class. For VLMs this is a processor whose tokenizer handles the text-only conversations.
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model.config._name_or_path)
        if isinstance(processing_class, ProcessorMixin):
            self._tokenizer = processing_class.tokenizer
        else:
            self._tokenizer = processing_class
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

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

        if args.per_device_train_batch_size % self.num_generations != 0:
            raise ValueError(
                f"The per-device train batch size ({args.per_device_train_batch_size}) must be evenly divisible by "
                f"the number of generations per prompt ({self.num_generations}): each step consumes "
                "`per_device_train_batch_size` scored samples while the dataloader feeds "
                "`per_device_train_batch_size / num_generations` prompts."
            )

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

        # Zero-sync generation state. The continuous batching manager is created lazily on the first generation (the
        # model must be on its final device first). `_inflight` maps every pending request id to its group; a group's
        # advantages are computed as soon as its last completion lands, and its samples join `_ready`, from which
        # training batches are drawn.
        self._manager = None
        self._inflight = {}
        self._ready = deque()

        # Metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def get_train_dataloader(self):
        # Each step consumes `per_device_train_batch_size` scored samples and every prompt produces
        # `num_generations` of them, so the dataloader feeds `per_device_train_batch_size / num_generations`
        # prompts per step.
        return self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size // self.num_generations,
            sampler_fn=self._get_train_sampler,
            is_training=True,
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
            eos_token_id=self._tokenizer.eos_token_id,
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

    def _submit_group(self, example: dict[str, Any]) -> None:
        # Tokenize the prompt (the whole conversation goes through the chat template) and submit `num_generations`
        # requests for it. The group is scored as a unit once its last completion lands.
        prompt_ids = self.processing_class.apply_chat_template(
            example["prompt"], add_generation_prompt=True, **self.chat_template_kwargs
        )["input_ids"]
        group = {"example": example, "prompt_ids": prompt_ids, "scored": [], "size": self.num_generations}
        for _ in range(self.num_generations):
            request_id = self._manager.add_request(prompt_ids, max_new_tokens=self.max_completion_length)
            self._inflight[request_id] = group

    def _drain(self, timeout: float) -> None:
        # Collect and score every completion the engine has finished; a group's advantages are computed once its last
        # completion lands. A dead background thread never raises from `get_result` (it returns None forever,
        # transformers#48334), so poll `fatal_error` to fail fast instead of spinning.
        while True:
            result = self._manager.get_result(timeout=timeout)
            if result is None:
                fatal_error = self._manager.background_thread_status.fatal_error
                if fatal_error is not None:
                    raise RuntimeError("The continuous batching background thread died.") from fatal_error
                return
            group = self._inflight.pop(result.request_id)
            group["scored"].append(self._score_completion(group, result))
            if len(group["scored"]) == group["size"]:
                self._finalize_group(group)

    def _score_completion(self, group: dict[str, Any], result) -> dict[str, Any]:
        # Score one completion the moment it finishes, so reward computation overlaps generation. The reward
        # functions receive one-element lists, with the same signature as everywhere else in TRL.
        example = group["example"]
        completion_ids = list(result.generated_tokens)
        completion_text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        completions = [[{"role": "assistant", "content": completion_text}]]
        reward_kwargs = {key: [example[key]] for key in example if key not in ["prompt"]}

        rewards_per_func = torch.zeros(len(self.reward_funcs))
        for i, reward_func in enumerate(self.reward_funcs):
            output_reward_func = reward_func(
                prompts=[example["prompt"]], completions=completions, completion_ids=[completion_ids], **reward_kwargs
            )
            # Convert None values to NaN
            rewards_per_func[i] = torch.nan if output_reward_func[0] is None else output_reward_func[0]
        return {
            "completion_ids": completion_ids,
            "logprobs": list(result.logprobs[: len(completion_ids)]),
            "rewards_per_func": rewards_per_func,
        }

    def _finalize_group(self, group: dict[str, Any]) -> None:
        mode = "train" if self.model.training else "eval"
        rewards_per_func = torch.stack([scored["rewards_per_func"] for scored in group["scored"]])

        # A completion for which every reward function returned None is unscorable. nansum would collapse it to 0,
        # which both biases the group baseline and hands the completion a spurious advantage. Mark these rows NaN so
        # they're excluded from the (nan-aware) baseline below; their advantage is forced to 0 afterwards.
        unscorable_mask = torch.isnan(rewards_per_func).all(dim=1)
        rewards = rewards_per_func.nansum(dim=1)
        rewards[unscorable_mask] = torch.nan
        advantages = (rewards - torch.nanmean(rewards)) / (nanstd(rewards) + 1e-4)

        # Unscorable completions carry no learning signal: their reward is NaN here, so zero their advantage to keep
        # them from moving the policy.
        advantages = torch.nan_to_num(advantages, nan=0.0)

        # Metrics are per-process: groups are formed and scored locally, and `Trainer.log` reports process zero.
        for i, reward_func_name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(
                torch.nanmean(rewards_per_func[:, i]).item()
            )
        self._metrics[mode]["reward"].append(torch.nanmean(rewards).item())
        self._metrics[mode]["reward_std"].append(nanstd(rewards).item())
        completion_lengths = [len(scored["completion_ids"]) for scored in group["scored"]]
        self._metrics[mode]["completions/mean_length"].append(sum(completion_lengths) / len(completion_lengths))

        for scored, advantage in zip(group["scored"], advantages, strict=True):
            self._ready.append(
                {
                    "prompt_ids": group["prompt_ids"],
                    "completion_ids": scored["completion_ids"],
                    "logprobs": scored["logprobs"],
                    "advantage": advantage,
                }
            )

    def _prepare_inputs(self, generation_batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        if self._manager is None:
            self._init_manager()

        if mode == "train":
            # Keep the engine `generation_ahead` batches of prompts ahead of training: the first batch primes the
            # pipeline (its prompts are generated and trained `generation_ahead` extra times), after which submitting
            # one incoming batch per step maintains the depth.
            while len(self._inflight) < self.generation_ahead * len(generation_batch) * self.num_generations:
                for example in generation_batch:
                    self._submit_group(example)
            for example in generation_batch:
                self._submit_group(example)
            num_samples = len(generation_batch) * self.num_generations
        else:
            for example in generation_batch:
                self._submit_group(example)
            num_samples = len(generation_batch) * self.num_generations

        while len(self._ready) < num_samples:
            self._drain(timeout=1.0)
        samples = [self._ready.popleft() for _ in range(num_samples)]

        # Build the training tensors: prompt + completion concatenated, with a mask over the completion tokens and
        # the engine's behavior-policy logprobs aligned to them.
        input_ids = [torch.tensor(s["prompt_ids"] + s["completion_ids"]) for s in samples]
        completion_mask = [
            torch.cat(
                [
                    torch.zeros(len(s["prompt_ids"]), dtype=torch.long),
                    torch.ones(len(s["completion_ids"]), dtype=torch.long),
                ]
            )
            for s in samples
        ]
        old_per_token_logps = [
            torch.cat(
                [
                    torch.zeros(len(s["prompt_ids"]), dtype=torch.float32),
                    torch.tensor(s["logprobs"], dtype=torch.float32),
                ]
            )
            for s in samples
        ]
        attention_mask = [torch.ones(len(ids), dtype=torch.long) for ids in input_ids]
        pad_token_id = self._tokenizer.pad_token_id
        return {
            "input_ids": pad(input_ids, padding_value=pad_token_id, padding_side="right").to(device),
            "attention_mask": pad(attention_mask, padding_value=0, padding_side="right").to(device),
            "completion_mask": pad(completion_mask, padding_value=0, padding_side="right").to(device),
            "old_per_token_logps": pad(old_per_token_logps, padding_value=0.0, padding_side="right").to(device),
            "advantages": torch.stack([s["advantage"] for s in samples]).to(device),
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
        per_token_logps = torch.gather(logits.log_softmax(dim=-1), dim=2, index=targets.unsqueeze(-1)).squeeze(-1)
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
