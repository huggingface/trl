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
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
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

from ...chat_template_utils import add_response_schema, parse_response, supports_tool_calling
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import create_model_from_path, nanstd, pad, patch_chunked_lm_head
from .zero_sync_grpo_config import ZeroSyncGRPOConfig


# Functions with this signature can be used as reward functions
RewardFunc = Callable[[list, list], list[float]]


@dataclass(frozen=True)
class TurnRecord:
    """One generation call: the whole prompt this turn, the tokens the model produced, their logprobs."""

    prompt_ids: list[int]
    output_ids: list[int]
    output_log_probs: list[float]


def _chain_to_sequences(turns: list[TurnRecord]) -> tuple[list[dict[str, Any]], int]:
    """Reconcile one conversation's turns (in order) into training rows; fork when the tokens drift.

    Every turn renders the full conversation through the chat template, so a template that rewrites
    history (dropped reasoning, summarized turns) re-tokenizes previously generated tokens differently.
    A turn continues the current row only if its prompt still starts with every token held so far; a
    single changed token forks a new row, so trained tokens are never silently rewritten as context.
    Returns the rows and the number of forks.
    """
    rows: list[dict[str, Any]] = []
    forks = 0
    for turn in turns:
        if rows and turn.prompt_ids[: len(rows[-1]["input_ids"])] == rows[-1]["input_ids"]:
            row = rows[-1]
            context = turn.prompt_ids[len(row["input_ids"]) :]
        else:
            forks += bool(rows)
            row = {"input_ids": [], "completion_mask": [], "logprobs": []}
            rows.append(row)
            context = turn.prompt_ids
        row["input_ids"].extend(context + turn.output_ids)
        row["completion_mask"].extend([0] * len(context) + [1] * len(turn.output_ids))
        row["logprobs"].extend([0.0] * len(context) + turn.output_log_probs)
    return [row for row in rows if any(row["completion_mask"])], forks


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
        tools: list[Callable] | None = None,
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
        self.max_tool_calling_iterations = args.max_tool_calling_iterations

        # Tools. `add_response_schema` teaches the tokenizer how to parse tool calls back out of a
        # generation, unless the chat template already carries a response template.
        self.tools = tools
        self.tool_dict = {tool.__name__: tool for tool in tools} if tools else {}
        if tools:
            if not supports_tool_calling(processing_class):
                raise ValueError(
                    "The provided chat template does not support tool calling. The template must be able to render a "
                    "full tool-calling conversation (user -> assistant with tool_calls -> tool)."
                )
            if getattr(self._tokenizer, "response_template", None) is None:
                processing_class = add_response_schema(processing_class)
                self._tokenizer = getattr(processing_class, "tokenizer", processing_class)

        if args.per_device_train_batch_size % self.num_generations != 0:
            raise ValueError(
                f"The per-device train batch size ({args.per_device_train_batch_size}) must be evenly divisible by "
                f"the number of generations per prompt ({self.num_generations}): each step consumes "
                "`per_device_train_batch_size` scored samples while the dataloader feeds "
                "`per_device_train_batch_size / num_generations` prompts."
            )

        # Compute per-token logprobs without ever materializing the [batch, seq, vocab] logits: the lm_head runs in
        # chunks with an online logsumexp. Long completions make this the difference between training and an OOM.
        # The patch replaces `forward`, and the generation engine decodes through that same object, so the standard
        # forward is kept for calls without labels, which is what decoding does.
        generation_forward = model.forward
        patch_chunked_lm_head(model, chunk_size=8192, temperature=self.temperature)
        training_forward = model.forward

        def forward(*args, labels=None, **kwargs):
            if labels is None:
                return generation_forward(*args, **kwargs)
            return training_forward(*args, labels=labels, **kwargs)

        model.forward = forward

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
        self._primed = False
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
        # Generation and training share the device, so the KV cache must leave room for gradients, optimizer states
        # and activations; the engine's own default claims most of the free memory.
        cb_kwargs.setdefault("max_memory_percent", 0.2)
        self._manager = model.init_continuous_batching(
            generation_config=generation_config,
            continuous_batching_config=ContinuousBatchingConfig(**cb_kwargs),
        )
        self._manager.warmup()
        self._manager.start()
        # The background thread is not a daemon: without an explicit stop the process never exits, including after an
        # exception in the training loop.
        atexit.register(self._manager.stop, block=False)

    def _tokenize_conversation(self, messages: list[dict[str, Any]]) -> list[int]:
        # Re-tokenize the WHOLE conversation each turn: the reconciler in `_chain_to_sequences` catches template
        # rewrites of earlier turns precisely because the prompt is rebuilt from the message list instead of glued
        # onto held tokens.
        return self.processing_class.apply_chat_template(
            messages, add_generation_prompt=True, tools=self.tools or None, **self.chat_template_kwargs
        )["input_ids"]

    def _submit_group(self, example: dict[str, Any]) -> None:
        # Start `num_generations` rollouts of the prompt. Each rollout is a chain of turns: a completed turn that
        # calls tools spawns the next turn's request, and the rollout is scored when its final turn lands.
        group = {"example": example, "scored": [], "size": self.num_generations}
        for _ in range(self.num_generations):
            prompt_ids = self._tokenize_conversation(example["prompt"])
            rollout = {
                "group": group,
                "messages": list(example["prompt"]),
                "completion": [],
                "turns": [],
                "prompt_ids": prompt_ids,
                "iterations": 0,
                "tool_calls": 0,
            }
            request_id = self._manager.add_request(prompt_ids, max_new_tokens=self.max_completion_length)
            self._inflight[request_id] = rollout

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
            rollout = self._inflight.pop(result.request_id)
            self._advance_rollout(rollout, result)

    def _advance_rollout(self, rollout: dict[str, Any], result) -> None:
        # One turn just finished: record it, and either spawn the next turn (the model called tools) or score the
        # rollout (it produced a final answer). Scoring happens the moment the rollout finishes, so reward
        # computation overlaps generation.
        turn_ids = list(result.generated_tokens)
        rollout["turns"].append(TurnRecord(rollout["prompt_ids"], turn_ids, list(result.logprobs[: len(turn_ids)])))
        if self.tools:
            assistant_message = parse_response(self._tokenizer, turn_ids, prefix=rollout["prompt_ids"])
        else:
            assistant_message = {
                "role": "assistant",
                "content": self._tokenizer.decode(turn_ids, skip_special_tokens=True),
            }
        rollout["messages"].append(assistant_message)
        rollout["completion"].append(assistant_message)

        tool_calls = assistant_message.get("tool_calls")
        may_iterate = (
            self.max_tool_calling_iterations is None or rollout["iterations"] < self.max_tool_calling_iterations
        )
        if tool_calls and may_iterate:
            tool_messages = self._execute_tool_calls(tool_calls)
            rollout["tool_calls"] += len(tool_calls)
            rollout["messages"].extend(tool_messages)  # tool result goes back as a MESSAGE, re-tokenized next turn
            rollout["completion"].extend(tool_messages)
            rollout["iterations"] += 1
            rollout["prompt_ids"] = self._tokenize_conversation(rollout["messages"])
            request_id = self._manager.add_request(rollout["prompt_ids"], max_new_tokens=self.max_completion_length)
            self._inflight[request_id] = rollout
            return

        group = rollout["group"]
        group["scored"].append(self._score_rollout(group["example"], rollout))
        if len(group["scored"]) == group["size"]:
            self._finalize_group(group)

    def _execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, str]]:
        tool_messages = []
        for tool_call in tool_calls:
            function = tool_call["function"]
            name = function["name"]
            tool = self.tool_dict.get(name)
            if tool is None:
                # A hallucinated tool name is a policy error that should decay with training
                tool_messages.append({"role": "tool", "name": name, "content": str({"error": f"unknown tool {name}"})})
                continue
            try:
                result = tool(**function.get("arguments", {}))
            except Exception as error:
                result = {"error": str(error)}
            tool_messages.append({"role": "tool", "name": name, "content": str(result)})
        return tool_messages

    def _score_rollout(self, example: dict[str, Any], rollout: dict[str, Any]) -> dict[str, Any]:
        # The reward functions receive the whole conversation's completion (assistant and tool messages) as
        # one-element lists, with the same signature as everywhere else in TRL.
        completion_ids = [token for turn in rollout["turns"] for token in turn.output_ids]
        reward_kwargs = {key: [example[key]] for key in example if key not in ["prompt"]}

        rewards_per_func = torch.zeros(len(self.reward_funcs))
        for i, reward_func in enumerate(self.reward_funcs):
            output_reward_func = reward_func(
                prompts=[example["prompt"]],
                completions=[rollout["completion"]],
                completion_ids=[completion_ids],
                **reward_kwargs,
            )
            # Convert None values to NaN
            rewards_per_func[i] = torch.nan if output_reward_func[0] is None else output_reward_func[0]

        # Reconcile the turn chain into training rows: one row for a conversation the template only appended to,
        # several when a rewrite forked it.
        sequences, forks = _chain_to_sequences(rollout["turns"])
        return {
            "sequences": sequences,
            "num_completion_tokens": len(completion_ids),
            "tool_calls": rollout["tool_calls"],
            "forks": forks,
            "rewards_per_func": rewards_per_func,
        }

    def _finalize_group(self, group: dict[str, Any]) -> None:
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

        # A rollout produces one row, or several when the chat template rewrote history. They all carry the rollout's
        # advantage, and its metrics ride along on the first one so a fork isn't counted twice.
        for scored, reward, reward_per_func, advantage in zip(
            group["scored"], rewards, rewards_per_func, advantages, strict=True
        ):
            metrics = {
                "reward": reward,
                "rewards_per_func": reward_per_func,
                "completion_length": scored["num_completion_tokens"],
                "tool_calls": scored["tool_calls"],
                "forks": scored["forks"],
            }
            for row in scored["sequences"]:
                self._ready.append({**row, "advantage": advantage, "metrics": metrics})
                metrics = None  # only the rollout's first row carries them

    def _prepare_inputs(self, generation_batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        if self._manager is None:
            self._init_manager()

        # Submit this step's prompts, then train on the oldest scored samples. In training the very first batch is
        # submitted `generation_ahead` extra times to prime the pipeline; from then on one batch in per step keeps
        # that many batches in flight, so the engine always has work queued while the trainer computes its step.
        if mode == "train" and not self._primed:
            for _ in range(self.generation_ahead):
                for example in generation_batch:
                    self._submit_group(example)
            self._primed = True
        for example in generation_batch:
            self._submit_group(example)
        num_samples = len(generation_batch) * self.num_generations

        # Time spent waiting for the engine. Near zero means generation is fully hidden behind the training step; a
        # large value means the engine is the bottleneck, so raise `generation_ahead` (more requests in flight) or
        # give it a bigger KV pool.
        wait_start = time.perf_counter()
        while len(self._ready) < num_samples:
            self._drain(timeout=1.0)
        self._metrics[mode]["generation_wait_s"].append(time.perf_counter() - wait_start)
        samples = [self._ready.popleft() for _ in range(num_samples)]

        # Metrics of the rollouts this step trains on. They are per process: groups are formed and scored locally,
        # and `Trainer.log` reports process zero.
        rollout_metrics = [sample["metrics"] for sample in samples if sample["metrics"] is not None]
        rewards = torch.stack([metrics["reward"] for metrics in rollout_metrics])
        rewards_per_func = torch.stack([metrics["rewards_per_func"] for metrics in rollout_metrics])
        for i, reward_func_name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(
                torch.nanmean(rewards_per_func[:, i]).item()
            )
        self._metrics[mode]["reward"].append(torch.nanmean(rewards).item())
        self._metrics[mode]["reward_std"].append(nanstd(rewards).item())
        self._metrics[mode]["completions/mean_length"].append(
            sum(metrics["completion_length"] for metrics in rollout_metrics) / len(rollout_metrics)
        )
        if self.tools:
            self._metrics[mode]["tools/call_count"].append(
                sum(metrics["tool_calls"] for metrics in rollout_metrics) / len(rollout_metrics)
            )
            self._metrics[mode]["completions/forks"].append(
                sum(metrics["forks"] for metrics in rollout_metrics) / len(rollout_metrics)
            )

        # Build the training tensors. Each row already carries its full token sequence, the mask over the trained
        # (model-generated) tokens and the engine's behavior-policy logprobs aligned to them, 0.0 on context.
        input_ids = [torch.tensor(s["input_ids"]) for s in samples]
        completion_mask = [torch.tensor(s["completion_mask"], dtype=torch.long) for s in samples]
        old_per_token_logps = [torch.tensor(s["logprobs"], dtype=torch.float32) for s in samples]
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
        completion_mask = inputs["completion_mask"]
        advantages = inputs["advantages"]

        # Per-token logprobs of the sampled tokens under the current policy, computed by the chunked lm_head, so
        # only completion positions are scored and the full-vocab logits never materialize.
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["input_ids"],
            completion_mask=completion_mask,
        )
        per_token_logps = outputs["log_probs"]
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
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather((outputs["entropy"] * mask).sum() / mask.sum().clamp(min=1.0)).mean().item()
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
