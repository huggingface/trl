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

from __future__ import annotations

import asyncio
import copy
import math
import textwrap
from contextlib import nullcontext
from copy import copy as shallow_copy
from typing import TYPE_CHECKING, Any

import torch
import transformers
from accelerate.utils import gather_object
from datasets import Dataset, IterableDataset
from packaging.version import Version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedTokenizerBase, Trainer

from ...chat_template_utils import parse_response
from ...data_utils import (
    apply_chat_template,
    is_conversational,
    prepare_multimodal_messages,
)
from ...extras.profiling import profiling_context, profiling_decorator
from ...models import unwrap_model_for_generation
from ...models.utils import disable_gradient_checkpointing
from ...trainer.grpo_trainer import GRPOTrainer, RewardFunc
from ...trainer.utils import (
    entropy_from_logits,
    nanstd,
    pad,
    selective_log_softmax,
    use_adapter,
)
from .dppo_config import DPPOConfig


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin


SAFETY_CLAMP_MAX = 20


def _strip_padding(tensor: torch.Tensor, mask: torch.Tensor) -> list[list]:
    """Remove padding from a batched tensor using a mask, returning a ragged list-of-lists."""
    return [row[m].tolist() for row, m in zip(tensor, mask.bool(), strict=True)]


class DPPOTrainer(GRPOTrainer):
    """
    Trainer for Divergence Proximal Policy Optimization (DPPO).

    DPPO replaces PPO/GRPO's heuristic ratio-clipping with a principled trust region based on direct policy
    divergence estimates. PPO-style clipping masks tokens based on probability ratio π/μ, which over-penalizes
    low-probability tokens and under-penalizes high-probability tokens. In contrast, DPPO masks based on
    direct approximation of policy divergence (e.g TV or KL) ensuring updates stay within a theoretically
    grounded trust region.


    Four divergence approximations are supported:
    - `binary_tv`: Absolute probability difference |π(a) - μ(a)| (simplest)
    - `binary_kl`: Bernoulli KL divergence between old and new token probabilities
    - `topk_tv`: Total variation over the top-K tokens of the distribution
    - `topk_kl`: KL divergence over the top-K tokens of the distribution

    Args:
        model (`str` or [`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using `<ModelArchitecture>.from_pretrained` (where `<ModelArchitecture>` is derived from the model
              config) with the keyword arguments in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
            - A [`~peft.PeftModel`] object. Only causal language models are supported.
        reward_funcs (`RewardFunc | list[RewardFunc]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                   functions can be either synchronous or asynchronous and can also return `None` when the reward is
                   not applicable to those samples. This is useful for multi-task training where different reward
                   functions apply to different types of samples. When a reward function returns `None` for a sample,
                   that reward function is excluded from the reward calculation for that sample. For more details, see
                   [Using a custom reward
                  function](#using-a-custom-reward-function).

                  The trainer's state is also passed to the reward function. The trainer's state is an instance of
                  [`~transformers.TrainerState`] and can be accessed by accessing the `trainer_state` argument to the
                  reward function's signature.
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        reward_processing_classes ([`~transformers.PreTrainedTokenizerBase`] or `list[PreTrainedTokenizerBase]`, *optional*):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using
            [`~transformers.AutoTokenizer.from_pretrained`]. For elements in `reward_funcs` that are custom reward
            functions (not [`~transformers.PreTrainedModel`]), the corresponding entries in `reward_processing_classes`
            are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
            model and a scheduler given by [`~transformers.get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        tools (list of `Callable`, *optional*):
            A list of callable tool functions (sync or async) that the model can invoke during generation. Each tool
            should be a standard Python function with properly type-hinted arguments and return values, and a
            Google-style docstring describing its purpose, arguments, and return value. For more details, see:
            https://huggingface.co/docs/transformers/en/chat_extras#passing-tools. The model uses the function's name,
            type hints, and docstring to determine how to call it. Ensure that the model's chat template supports tool
            use and that it has been fine-tuned for tool calling.
        rollout_func (`RolloutFunc`, *optional*):
            Function to use for generating completions. It receives the list of prompts allocated to the current
            process and the trainer instance. It must return a dict with `"prompt_ids"`, `"completion_ids"`, and
            `"logprobs"` fields. Any other fields are forwarded to the reward functions. This feature is experimental
            and may change or be removed at any time without prior notice.
    """

    _tag_names = ["trl", "dppo"]
    _name = "DPPO"
    _paper = {
        "title": "Rethinking the Trust Region in LLM Reinforcement Learning",
        "id": "2602.04879",
        # docstyle-ignore
        "citation": textwrap.dedent(
            """\
            @misc{zhang2025rethinkingtrust,
                title        = {{Rethinking the Trust Region in LLM Reinforcement Learning}},
                author       = {Yan Zhang and others},
                year         = 2025,
                url          = {https://arxiv.org/abs/2602.04879},
                archivePrefix= {arXiv},
                eprint       = {2602.04879},
                primaryClass = {cs.LG}
            }"""
        ),
    }

    def __init__(
        self,
        model: str | PreTrainedModel,
        reward_funcs: RewardFunc | list[RewardFunc],
        args: DPPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = DPPOConfig(f"{model_name}-DPPO")

        self.divergence_type = args.divergence_type
        self.divergence_topk = args.divergence_topk
        self.clip_ratio_c = args.clip_ratio_c

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        if self.divergence_type.startswith("topk_") and self.use_vllm:
            self.vllm_generation.logprobs = self.divergence_topk

    def _generate_single_turn(self, prompts: list):
        """Generate completions, always extracting sampled token logprobs.

        Returns:
            6-tuple of (prompt_ids, completion_ids, logprobs, topk_logprobs, topk_token_ids, extra_fields).
            topk_logprobs and topk_token_ids are None when divergence_type is not topk.
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        needs_topk = self.divergence_type.startswith("topk_")
        K = self.divergence_topk

        if self.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                with profiling_context(self, "sync_weights"):
                    self.vllm_generation.sync_weights()
                self._last_loaded_step = self.state.global_step

            num_generations = self.num_generations if mode == "train" else self.num_generations_eval
            prompt_ids, completion_ids, logprobs, logprob_token_ids, extra_fields = self.vllm_generation.generate(
                prompts=prompts, num_generations=num_generations, profiler=profiling_context(self, "vLLM.generate")
            )

            if needs_topk:
                # vLLM returns up to K+1 entries sorted by rank (most probable first).
                # The sampled token is always included but may be at any position.
                # Truncate to K for topk; find the sampled token's logprob for the standard return.
                topk_logprobs = [[lp[:K] for lp in seq] for seq in logprobs]
                topk_token_ids = [[tid[:K] for tid in seq] for seq in logprob_token_ids]
                sampled_logprobs = []
                for seq_lps, seq_tids, seq_cids in zip(logprobs, logprob_token_ids, completion_ids, strict=True):
                    seq_sampled = []
                    for step_lps, step_tids, sampled_tid in zip(seq_lps, seq_tids, seq_cids, strict=True):
                        idx = step_tids.index(sampled_tid)
                        seq_sampled.append(step_lps[idx])
                    sampled_logprobs.append(seq_sampled)
            else:
                sampled_logprobs = [[step_lps[0] for step_lps in seq_lps] for seq_lps in logprobs]
                topk_logprobs = None
                topk_token_ids = None

            return prompt_ids, completion_ids, sampled_logprobs, topk_logprobs, topk_token_ids, extra_fields
        else:
            if is_conversational({"prompt": prompts[0]}):
                generate_inputs = self.processing_class.apply_chat_template(
                    conversation=prompts,
                    tools=self.tools,
                    chat_template=self.chat_template,
                    add_generation_prompt=True,
                    tokenize=True,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                    return_dict=True,
                    **self.chat_template_kwargs,
                )
            else:
                generate_inputs = self.processing_class(
                    text=prompts, padding=True, padding_side="left", return_tensors="pt"
                )
            generate_inputs = Trainer._prepare_inputs(self, generate_inputs)

            gen_config = shallow_copy(self.generation_config)
            gen_config.output_logits = True
            gen_config.return_dict_in_generate = True

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    generation_kwargs=self.generation_kwargs,
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                gen_output = unwrapped_model.generate(
                    **generate_inputs, generation_config=gen_config, disable_compile=True
                )

            prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
            prompt_length = prompt_ids.size(1)
            completion_ids = gen_output.sequences[:, prompt_length:]

            # Stack logits: tuple of (batch, vocab_size) per step -> (batch, seq_len, vocab_size)
            all_logits = torch.stack(gen_output.logits, dim=1)
            all_logits = all_logits / self.temperature
            all_log_probs = all_logits.log_softmax(dim=-1)

            sampled_logprobs = all_log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
            if needs_topk:
                topk_logps, topk_ids = torch.topk(all_log_probs, k=K, dim=-1)
            else:
                topk_logps, topk_ids = None, None

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            prompt_ids_out = _strip_padding(prompt_ids, prompt_mask)
            completion_ids_out = _strip_padding(completion_ids, completion_mask)
            logprobs_out = _strip_padding(sampled_logprobs, completion_mask)
            if needs_topk:
                topk_logprobs = _strip_padding(topk_logps, completion_mask)
                topk_token_ids = _strip_padding(topk_ids, completion_mask)
            else:
                topk_logprobs = None
                topk_token_ids = None

            extra_fields = {}
            return prompt_ids_out, completion_ids_out, logprobs_out, topk_logprobs, topk_token_ids, extra_fields

    def _tool_call_loop(
        self, prompts, prompt_ids, completion_ids, completions, logprobs, topk_logprobs, topk_token_ids
    ):
        """Tool execution loop that also threads top-K logprob data alongside logprobs.

        Mirrors GRPOTrainer._tool_call_loop but additionally concatenates topk_logprobs and topk_token_ids
        the same way logprobs is concatenated: real data for model-generated tokens, zero-padding for
        tool-result tokens. When topk data is None (binary divergence), behaves identically to the parent.
        """
        K = self.divergence_topk
        has_topk = topk_logprobs is not None

        tool_calls = [completion[0].get("tool_calls") for completion in completions]
        idxs_with_tool = [idx for idx, tool_call in enumerate(tool_calls) if tool_call]
        tool_calls = [tool_calls[idx] for idx in idxs_with_tool]
        tool_mask = [[1] * len(ids) for ids in completion_ids]
        tool_call_count = 0
        tool_failure_count = 0
        iteration_num = 0
        while idxs_with_tool and iteration_num < self.max_tool_calling_iterations:
            prompt_completion_tools = [prompts[i] for i in idxs_with_tool]

            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                tool_call_list = tool_calls[idx]
                prompt_completion_tool = prompt_completion_tools[idx]
                prompt_completion_tool.append(completions[idx_with_tool][-1])
                async_coros = []
                tool_call_results = []
                for tool_call in tool_call_list:
                    tool_call_count += 1
                    if tool_call["type"] == "function":
                        function = tool_call["function"]
                        name = function["name"]
                        if name in self._sync_tool_dict:
                            tool_call_results.append((name, self._sync_tool_dict[name](**function["arguments"])))
                        elif name in self._async_tool_dict:
                            async_coros.append((name, self._async_tool_dict[name](**function["arguments"])))
                        else:
                            tool_failure_count += 1
                            tool_call_results.append((name, {"error": f"Unknown tool: {name}"}))
                    else:
                        tool_failure_count += 1
                        name = tool_call.get("name", "unknown")
                        tool_call_results.append((name, {"error": f"Unsupported tool call type: {tool_call['type']}"}))

                if async_coros:

                    async def _run_async_tools(async_coros):
                        coros = [coro for _, coro in async_coros]
                        results = await asyncio.gather(*coros, return_exceptions=True)
                        return [(name, result) for (name, _), result in zip(async_coros, results, strict=False)]

                    async_results = asyncio.run_coroutine_threadsafe(
                        _run_async_tools(async_coros), self.async_loop
                    ).result()

                    for name, result in async_results:
                        if isinstance(result, Exception):
                            tool_failure_count += 1
                            tool_call_results.append((name, {"error": str(result)}))
                        else:
                            tool_call_results.append((name, result))

                for name, result in tool_call_results:
                    tool_message = {"role": "tool", "name": name, "content": str(result)}
                    prompt_completion_tool.append(tool_message)
                    completions[idx_with_tool].append(tool_message)

            # Tokenize and filter samples whose length exceeds max allowed length
            pct_ids = self.processing_class.apply_chat_template(
                prompt_completion_tools,
                tools=self.tools,
                chat_template=self.chat_template,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
                **self.chat_template_kwargs,
            )
            if self.use_vllm and self.vllm_mode == "colocate":
                max_model_len = self.llm.llm_engine.model_config.max_model_len
            elif not self.use_vllm:
                max_model_len = self.model.config.max_position_embeddings
            else:
                raise NotImplementedError(
                    f"Unsupported mode detected: use_vllm={self.use_vllm}, vllm_mode={self.vllm_mode}"
                )
            overlong = [len(pct) >= max_model_len for pct in pct_ids]
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                if overlong[idx]:
                    prompt_length = len(prompt_ids[idx_with_tool])
                    ct = pct_ids[idx][prompt_length : prompt_length + self.max_completion_length]
                    completion_ids[idx_with_tool] = ct
                    tool_mask[idx_with_tool] += [1] * (len(ct) - len(tool_mask[idx_with_tool]))
                    if logprobs is not None:
                        logprobs[idx_with_tool] += [0.0] * (len(ct) - len(logprobs[idx_with_tool]))
                    if has_topk:
                        topk_logprobs[idx_with_tool] += [[0.0] * K] * (len(ct) - len(topk_logprobs[idx_with_tool]))
                        topk_token_ids[idx_with_tool] += [[0] * K] * (len(ct) - len(topk_token_ids[idx_with_tool]))

            idxs_with_tool = [idx for idx, o in zip(idxs_with_tool, overlong, strict=True) if not o]
            prompt_completion_tools = [pct for pct, o in zip(prompt_completion_tools, overlong, strict=True) if not o]
            if not idxs_with_tool:
                break

            # Generate new completions after tool execution
            (
                prompt_completion_tool_ids,
                post_tool_ids,
                post_tool_logprobs,
                post_tool_topk_logprobs,
                post_tool_topk_token_ids,
                _,
            ) = self._generate_single_turn(prompt_completion_tools)

            # Sanity check: chat template must be prefix-preserving
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                pct = prompt_completion_tool_ids[idx]
                if prompt_ids[idx_with_tool] != pct[: len(prompt_ids[idx_with_tool])]:
                    raise ValueError(
                        "The chat template is not prefix-preserving. Please update it to use a prefix-preserving "
                        "format."
                    )

            # Truncate so that pct[len(prompt_ids[idx]):] + post_tool does not exceed max_completion_length
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_len = len(prompt_ids[idx_with_tool])
                completion_tool_ids = prompt_completion_tool_ids[idx][prompt_len:]
                excess_length = len(completion_tool_ids) + len(post_tool_ids[idx]) - self.max_completion_length
                if excess_length > 0:
                    post_tool_ids[idx] = post_tool_ids[idx][:-excess_length]
                    if logprobs is not None:
                        post_tool_logprobs[idx] = post_tool_logprobs[idx][:-excess_length]
                    if has_topk and post_tool_topk_logprobs is not None:
                        post_tool_topk_logprobs[idx] = post_tool_topk_logprobs[idx][:-excess_length]
                        post_tool_topk_token_ids[idx] = post_tool_topk_token_ids[idx][:-excess_length]
                    excess_length = len(completion_tool_ids) + len(post_tool_ids[idx]) - self.max_completion_length
                    if excess_length > 0:
                        prompt_completion_tool_ids[idx] = prompt_completion_tool_ids[idx][:-excess_length]

            # Update tool_mask and logprobs: tool result tokens get 0/0.0, post-tool model tokens get 1/real values
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_completion_tool_length = len(prompt_completion_tool_ids[idx])
                prompt_length = len(prompt_ids[idx_with_tool])
                completion_length = len(completion_ids[idx_with_tool])
                post_tool_length = len(post_tool_ids[idx])
                tool_length = prompt_completion_tool_length - prompt_length - completion_length
                tool_mask[idx_with_tool] += [0] * tool_length + [1] * post_tool_length
                if logprobs is not None:
                    logprobs[idx_with_tool] += [0.0] * tool_length + post_tool_logprobs[idx]
                if has_topk:
                    topk_pad = [[0.0] * K] * tool_length
                    tid_pad = [[0] * K] * tool_length
                    post_topk_lp = post_tool_topk_logprobs[idx] if post_tool_topk_logprobs is not None else []
                    post_topk_tid = post_tool_topk_token_ids[idx] if post_tool_topk_token_ids is not None else []
                    topk_logprobs[idx_with_tool] += topk_pad + post_topk_lp
                    topk_token_ids[idx_with_tool] += tid_pad + post_topk_tid

            # Update completion_ids with the new completions (after tool execution)
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_length = len(prompt_ids[idx_with_tool])
                pct = prompt_completion_tool_ids[idx]
                completion_ids[idx_with_tool] = pct[prompt_length:] + post_tool_ids[idx]

            # Decode post-tool completions
            post_tool_completions = [
                parse_response(self.processing_class, ids) if ids else {} for ids in post_tool_ids
            ]

            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                if post_tool_completions[idx]:
                    completions[idx_with_tool].append(post_tool_completions[idx])

            # Check for further tool calls
            tool_calls = [completion.get("tool_calls") for completion in post_tool_completions]
            idxs_with_tool = [idx for idx, tool_call in zip(idxs_with_tool, tool_calls, strict=True) if tool_call]
            tool_calls = [tool_call for tool_call in tool_calls if tool_call]
            iteration_num += 1

        return (
            tool_mask,
            completions,
            completion_ids,
            logprobs,
            topk_logprobs,
            topk_token_ids,
            tool_call_count,
            tool_failure_count,
        )

    def _generate(self, prompts: list):
        """Generate completions, handling tool calls, and thread top-K logprob data through the full pipeline.

        Returns:
            9-tuple of (prompt_ids, completion_ids, tool_mask, completions, total_completion_tokens,
            logprobs, topk_logprobs, topk_token_ids, extra_fields).
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = copy.deepcopy(prompts)

        prompt_ids, completion_ids, logprobs, topk_logprobs, topk_token_ids, extra_fields = self._generate_single_turn(
            prompts
        )

        # Decode completions
        if is_conversational({"prompt": prompts[0]}):
            if (
                Version(transformers.__version__) >= Version("5.0.0")
                and isinstance(self.processing_class, PreTrainedTokenizerBase)
                and hasattr(self.processing_class, "response_schema")
                and self.processing_class.response_schema is not None
            ):
                completions = [[parse_response(self.processing_class, ids)] for ids in completion_ids]
            else:
                contents = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                completions = [[{"role": "assistant", "content": content}] for content in contents]
        else:
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Extract tool calls from the completions and (possibly) execute them
        if self.tools:
            (
                tool_mask,
                completions,
                completion_ids,
                logprobs,
                topk_logprobs,
                topk_token_ids,
                tool_call_count,
                tool_failure_count,
            ) = self._tool_call_loop(
                prompts, prompt_ids, completion_ids, completions, logprobs, topk_logprobs, topk_token_ids
            )
        else:
            tool_mask = extra_fields.pop("env_mask", None)

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        if tool_mask is not None:
            completion_lengths = torch.tensor([sum(mask) for mask in tool_mask], device=device)
        else:
            completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()

        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        if self.tools:
            agg_tool_call_count = self.accelerator.gather(torch.tensor(tool_call_count, device=device)).sum()
            tool_call_frequency = (agg_tool_call_count / len(agg_prompt_lengths)).item()
            self._metrics[mode]["tools/call_frequency"].append(tool_call_frequency)
            agg_tool_failure_count = self.accelerator.gather(torch.tensor(tool_failure_count, device=device)).sum()
            failure_frequency = (
                (agg_tool_failure_count / agg_tool_call_count).item() if agg_tool_call_count > 0 else 0.0
            )
            self._metrics[mode]["tools/failure_frequency"].append(failure_frequency)

        return (
            prompt_ids,
            completion_ids,
            tool_mask,
            completions,
            total_completion_tokens,
            logprobs,
            topk_logprobs,
            topk_token_ids,
            extra_fields,
        )

    @profiling_decorator
    def _get_per_token_logps_with_topk(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        topk_token_ids,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Compute per-token log-probs, (optionally) entropies, and top-K log-probs in one forward pass.

        Evaluates the current policy's log-probs at the rollout's top-K token IDs from the same
        forward pass used for per_token_logps, avoiding an extra model call.

        Args:
            topk_token_ids: Rollout policy's top-K token IDs, shape (B, T, K). The current policy's
                log-probs are evaluated at these positions.

        Returns:
            Tuple of (per_token_logps, entropies, current_topk_logps).
        """
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []
        all_topk_logps = []

        for start in range(0, input_ids.size(0), batch_size):
            end = start + batch_size
            input_ids_batch = input_ids[start:end]
            attention_mask_batch = attention_mask[start:end]

            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[end].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[end]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start:end]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start:end]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start:end]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start:end]

            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False

            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

            with torch.no_grad():
                topk_logps = selective_log_softmax(logits, topk_token_ids[start:end])
            all_topk_logps.append(topk_logps)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        topk_logps = torch.cat(all_topk_logps, dim=0)
        return logps, entropies, topk_logps

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        if images is not None:
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            topk_logprobs_list,
            topk_token_ids_list,
            extra_fields,
        ) = self._generate(prompts)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
        sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        if tool_mask_list is not None:
            tool_mask = [torch.tensor(mask, device=device) for mask in tool_mask_list]
            tool_mask = pad(tool_mask, padding_value=1, padding_side="right")
        else:
            tool_mask = None
        if topk_logprobs_list is not None:
            sampling_topk_logps = [torch.tensor(lp, device=device) for lp in topk_logprobs_list]
            sampling_topk_logps = pad(sampling_topk_logps, padding_value=0.0, padding_side="right")
            sampling_topk_token_ids = [
                torch.tensor(tid, device=device, dtype=torch.long) for tid in topk_token_ids_list
            ]
            sampling_topk_token_ids = pad(sampling_topk_token_ids, padding_value=0, padding_side="right")
        else:
            sampling_topk_logps = None
            sampling_topk_token_ids = None

        # If mask_truncated_completions is enabled, zero out truncated completions for attention and loss masking
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            # Mask completion_mask for attention masking
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
            # Also mask tool_mask for consistency in multi-turn training
            if tool_mask is not None:
                tool_mask = tool_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs
        if images is not None:
            prompts_text = [
                apply_chat_template(
                    {"prompt": prompt}, self.processing_class, tools=self.tools, **self.chat_template_kwargs
                )["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        # When gradient checkpointing is enabled with use_reentrant=True (non default), calling the model inside a
        # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
        # Temporarily disable checkpointing to avoid this warning during inference.
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    # When training a PEFT adapter, how we obtain the reference depends on the setup:
                    # - New adapter: disabling adapters yields the base model.
                    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                    model = self.accelerator.unwrap_model(self.model)
                    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        if self.multi_objective_aggregation == "sum_then_normalize":
            # Apply weights to each reward function's output and sum
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            if self.scale_rewards in ["group", "none"]:
                # If self.scale_rewards = "none", we'll only use std_rewards to check for zero std for logging
                if num_generations > 1:
                    std_rewards = rewards.view(-1, num_generations).std(dim=1)
                    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
                else:  # doesn't occur during training, but could occur in eval when num_generations_eval=1
                    std_rewards = torch.zeros_like(rewards)
            elif self.scale_rewards == "batch":
                # Compute global std
                if rewards.numel() > 1:
                    std_rewards = rewards.std().expand_as(rewards)
                else:  # doesn't occur during training, but could occur in eval when num_generations_eval=batch_size=1
                    std_rewards = torch.zeros_like(rewards)
            else:
                raise ValueError(
                    f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                )

            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # for logging

        elif self.multi_objective_aggregation == "normalize_then_sum":
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
            std_k = nanstd(grouped, dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(mean_k)
            reward_k = (grouped - mean_k) / (std_k + 1e-4)
            reward_k = reward_k.view(-1, len(self.reward_funcs))
            rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            advantages = (rewards - rewards.mean()) / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # for logging

        else:
            raise ValueError(
                f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}. Must be "
                "'sum_then_normalize' or 'normalize_then_sum'."
            )

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        rewards = rewards_per_func.nansum(dim=1)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
            "sampling_per_token_logps": sampling_per_token_logps,
        }
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        if tool_mask is not None:
            output["tool_mask"] = tool_mask
        if sampling_topk_logps is not None:
            output["sampling_topk_logps"] = sampling_topk_logps
        if sampling_topk_token_ids is not None:
            output["sampling_topk_token_ids"] = sampling_topk_token_ids
        return output

    def _compute_divergence_mask(
        self,
        per_token_logps,
        sampling_per_token_logps,
        advantages,
        completion_mask,
        current_topk_logps=None,
        sampling_topk_logps=None,
    ):
        """
        Compute a per-token trust-region mask based on the configured divergence type. Tokens where the policy has
        diverged too far from the sampling distribution (in a direction that would increase the loss) are masked out.

        Args:
            per_token_logps (`torch.Tensor`):
                Log-probabilities of the current policy at the sampled tokens, shape `(B, T)`.
            sampling_per_token_logps (`torch.Tensor`):
                Log-probabilities of the sampling (rollout) policy at the sampled tokens, shape `(B, T)`.
            advantages (`torch.Tensor`):
                Per-token or per-sequence advantage estimates, broadcastable to `(B, T)`.
            completion_mask (`torch.Tensor`):
                Binary mask of shape `(B, T)` where `1` indicates valid completion tokens and `0` padding.
            current_topk_logps (`torch.Tensor` or `None`):
                Log-probabilities of the current policy at the rollout's top-K token IDs, shape `(B, T, K)`.
                Required when `divergence_type` is `"topk_tv"` or `"topk_kl"`.
            sampling_topk_logps (`torch.Tensor` or `None`):
                Log-probabilities of the sampling policy at the rollout's top-K token IDs, shape `(B, T, K)`.
                Required when `divergence_type` is `"topk_tv"` or `"topk_kl"`.

        Returns:
            `torch.Tensor`:
                Float mask of shape `(B, T)` where `1.0` indicates tokens to keep and `0.0` tokens to mask out.
        """
        prob = torch.exp(per_token_logps)
        sampling_prob = torch.exp(sampling_per_token_logps)

        delta_low = self.epsilon_low
        delta_high = self.epsilon_high

        if self.divergence_type == "binary_tv":
            # TV = |π - μ|
            divergence = (prob - sampling_prob).abs()
            # Mask tokens where divergence > threshold AND policy moves away from trust region
            invalid_pos = (divergence > delta_high) & (prob > sampling_prob)
            invalid_neg = (divergence > delta_low) & (prob < sampling_prob)
            mask = torch.where(advantages > 0, ~invalid_pos, ~invalid_neg)

        elif self.divergence_type == "binary_kl":
            # Bernoulli KL: D = μ log(μ/π) + (1-μ) log((1-μ)/(1-π))
            kl = sampling_prob * (sampling_per_token_logps - per_token_logps) + (1 - sampling_prob) * (
                torch.log1p(-sampling_prob.clamp(max=1 - 1e-7)) - torch.log1p(-prob.clamp(max=1 - 1e-7))
            )

            invalid_pos = (kl > delta_high) & (prob > sampling_prob)
            invalid_neg = (kl > delta_low) & (prob < sampling_prob)
            mask = torch.where(advantages > 0, ~invalid_pos, ~invalid_neg)

        elif self.divergence_type in ("topk_tv", "topk_kl"):
            current_topk_probs = torch.exp(current_topk_logps.float())
            rollout_topk_probs = torch.exp(sampling_topk_logps.float())

            # Aggregate remaining probability mass outside top-K into a single rest bucket.
            rollout_rest = (1.0 - rollout_topk_probs.sum(dim=-1)).clamp(min=1e-12)
            current_rest = (1.0 - current_topk_probs.sum(dim=-1)).clamp(min=1e-12)

            if self.divergence_type == "topk_tv":
                topk_tv = (current_topk_probs - rollout_topk_probs).abs().sum(dim=-1)
                rest_tv = (current_rest - rollout_rest).abs()
                divergence = (topk_tv + rest_tv) / 2.0
            else:
                topk_kl = (rollout_topk_probs * (sampling_topk_logps - current_topk_logps)).sum(dim=-1)
                rest_kl = rollout_rest * (rollout_rest.log() - current_rest.log())
                divergence = topk_kl + rest_kl

            invalid_pos = (divergence > delta_high) & (prob > sampling_prob)
            invalid_neg = (divergence > delta_low) & (prob < sampling_prob)
            mask = torch.where(advantages > 0, ~invalid_pos, ~invalid_neg)

        else:
            raise ValueError(f"Unknown divergence_type: {self.divergence_type}")

        return mask.float() * completion_mask

    def _compute_loss(self, model, inputs):
        # Compute per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        forward_kwargs = {
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "num_images": inputs.get("num_images"),
            "pixel_attention_mask": inputs.get("pixel_attention_mask"),
            "image_sizes": inputs.get("image_sizes"),
            "token_type_ids": inputs.get("token_type_ids"),
        }

        sampling_topk_token_ids = inputs.get("sampling_topk_token_ids")
        if self.divergence_type.startswith("topk_") and sampling_topk_token_ids is not None:
            per_token_logps, entropies, current_topk_logps = self._get_per_token_logps_with_topk(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                topk_token_ids=sampling_topk_token_ids,
                compute_entropy=True,
                **forward_kwargs,
            )
        else:
            per_token_logps, entropies = self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                compute_entropy=True,
                **forward_kwargs,
            )
            current_topk_logps = None

        sampling_per_token_logps = inputs["sampling_per_token_logps"]
        sampling_topk_logps = inputs.get("sampling_topk_logps")

        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        # DPPO: compute IS ratio (clamped, detached) and divergence mask
        log_ratio = per_token_logps - sampling_per_token_logps
        ratio = torch.exp(log_ratio.clamp(max=math.log(self.clip_ratio_c))).detach()
        divergence_mask = self._compute_divergence_mask(
            per_token_logps,
            sampling_per_token_logps,
            advantages,
            mask,
            current_topk_logps=current_topk_logps,
            sampling_topk_logps=sampling_topk_logps,
        )
        divergence_mask = divergence_mask.detach()

        # DPPO loss: -advantages * ratio * mask * log_prob
        per_token_loss = -advantages * ratio * divergence_mask * per_token_logps

        # KL divergence with reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mode = "train" if self.model.training else "eval"
        normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
        loss = (per_token_loss * mask).sum() / normalizer

        # Log metrics
        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        prob_diff = (torch.exp(per_token_logps) - torch.exp(sampling_per_token_logps)).abs()
        self._metrics[mode]["prob_diff/mean"].append(
            self.accelerator.gather(masked_batch_mean(prob_diff)).nanmean().item()
        )
        per_seq_max = prob_diff.masked_fill(mask == 0, float("-inf")).max(dim=1).values
        per_seq_min = prob_diff.masked_fill(mask == 0, float("inf")).min(dim=1).values
        self._metrics[mode]["prob_diff/max"].append(self.accelerator.gather(per_seq_max).max().item())
        self._metrics[mode]["prob_diff/min"].append(self.accelerator.gather(per_seq_min).min().item())

        self._metrics[mode]["advantages/mean"].append(advantages.mean().item())
        self._metrics[mode]["advantages/std"].append(advantages.std().item())

        # Log divergence mask statistics (analogous to clip_ratio in GRPO)
        is_masked = (divergence_mask == 0) & (mask > 0)
        is_masked_pos = is_masked & (advantages > 0)
        is_masked_neg = is_masked & (advantages < 0)

        mask_ratio_pos = masked_batch_mean(is_masked_pos.float())
        mask_ratio_neg = masked_batch_mean(is_masked_neg.float())
        mask_ratio = masked_batch_mean(is_masked.float())

        gathered_mask_ratio_neg = self.accelerator.gather(mask_ratio_neg)
        self._metrics[mode]["mask_ratio/negative_adv_mean"].append(gathered_mask_ratio_neg.nanmean().item())
        gathered_mask_ratio_pos = self.accelerator.gather(mask_ratio_pos)
        self._metrics[mode]["mask_ratio/positive_adv_mean"].append(gathered_mask_ratio_pos.nanmean().item())
        gathered_mask_ratio = self.accelerator.gather(mask_ratio)
        self._metrics[mode]["mask_ratio/overall_mean"].append(gathered_mask_ratio.nanmean().item())

        return loss
