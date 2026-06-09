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

import inspect
import re
import textwrap
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any

import datasets
import torch
from accelerate.utils import gather_object, is_peft_model
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_liger_kernel_available, is_peft_available, logging

from ...data_utils import apply_chat_template, is_conversational
from ...models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from ...models.utils import _ForwardRedirection
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import (
    RepeatSampler,
    create_model_from_path,
    disable_dropout_in_model,
    get_config_model_id,
    identity,
    pad,
    selective_log_softmax,
    split_tensor_dict,
    use_adapter,
)
from ..utils import prepare_peft_model
from .loss_utils import (
    add_tail_bucket,
    apply_importance_sampling_clipping,
    compute_divergence,
    compute_full_logit_self_distillation_loss,
    compute_sampled_token_self_distillation_loss,
    compute_topk_self_distillation_loss,
)
from .sdpo_config import SDPOConfig
from .teacher_sync import PEFTAdapterEMACallback, SyncTeacherModelCallback, is_pure_lora_training


logger = logging.get_logger(__name__)


if is_peft_available():
    from peft import PeftConfig

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss


TrainingBatch = dict[str, torch.Tensor | Any]


def build_teacher_request_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
) -> tuple[list[list[int]], list[int], list[int]]:
    """Trim padded batch tensors into per-sample (prompt+completion) sequences for teacher-server requests."""
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"input_ids and attention_mask must have the same shape, got {input_ids.shape} and {attention_mask.shape}."
        )

    input_ids_cpu = input_ids.detach().cpu()
    attention_mask_cpu = attention_mask.detach().cpu().bool()
    prompt_lengths = prompt_attention_mask.detach().cpu().sum(dim=1).to(torch.long)

    trimmed_input_ids: list[list[int]] = []
    prompt_lengths_list: list[int] = []
    completion_lengths_list: list[int] = []
    for row, mask, prompt_length in zip(input_ids_cpu, attention_mask_cpu, prompt_lengths, strict=True):
        trimmed_row = row[mask]
        prompt_len = int(prompt_length.item())
        if prompt_len < 0 or prompt_len > trimmed_row.numel():
            raise ValueError(
                f"Invalid prompt length {prompt_len} for trimmed sequence of length {trimmed_row.numel()}."
            )
        trimmed_input_ids.append(trimmed_row.tolist())
        prompt_lengths_list.append(prompt_len)
        completion_lengths_list.append(int(trimmed_row.numel()) - prompt_len)
    return trimmed_input_ids, prompt_lengths_list, completion_lengths_list


@dataclass
class DistillationLogits:
    """Aligned logits and masks used to compute a self-distillation objective."""

    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    loss_mask: torch.Tensor
    student_logits: torch.Tensor
    teacher_logits: torch.Tensor


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    """Extract the text content from the last user message in a conversational prompt."""
    last_message = messages[-1]
    if last_message.get("role") != "user":
        raise ValueError(
            f"Self-distillation teacher prompt construction expects the conversation to end with a user turn, "
            f"but the last message has role '{last_message.get('role')}'. "
            f"Prompts ending with assistant prefills or tool turns are not supported."
        )
    content = last_message.get("content", "")
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if part.get("type") == "text")
    return content


class SuccessfulRolloutTeacherContextBuilder:
    """Builds teacher contexts from successful rollouts"""

    def __init__(self, trainer):
        self.trainer = trainer
        self.last_metrics: dict[str, float] = {}

    def _build_reprompt_text(self, prompt_text: str, solution_text: str, feedback_text: str) -> str:
        return self.trainer.args.reprompt_template.format(
            prompt=prompt_text,
            solution=solution_text,
            feedback=feedback_text,
        )

    def _tokenize_teacher_messages(
        self, teacher_messages_list: list[str | list[dict[str, Any]]]
    ) -> dict[str, torch.Tensor]:
        device = self.trainer.accelerator.device
        teacher_prompt_ids_list = self.trainer._tokenize_prompts_untruncated(teacher_messages_list)
        teacher_prompt_ids = [
            torch.as_tensor(ids[-self.trainer.args.max_reprompt_len :], device=device)
            for ids in teacher_prompt_ids_list
        ]
        teacher_prompt_mask = [torch.ones(len(ids), dtype=torch.long, device=device) for ids in teacher_prompt_ids]
        return {
            "prompt_ids": pad(
                teacher_prompt_ids, padding_value=self.trainer._tokenizer.pad_token_id, padding_side="left"
            ),
            "prompt_mask": pad(teacher_prompt_mask, padding_value=0, padding_side="left"),
        }

    def build(
        self,
        output: dict[str, torch.Tensor | Any],
        prompts: list[Any],
        rewards: torch.Tensor,
        feedbacks: list[Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        device = self.trainer.accelerator.device
        mode = "train" if self.trainer.model.training else "eval"
        num_generations = self.trainer.num_generations if mode == "train" else self.trainer.num_generations_eval
        completion_ids = output["completion_ids"]
        completion_mask = output["completion_mask"]

        num_local = len(prompts)
        process_start = self.trainer.accelerator.process_index * num_local
        process_slice = slice(process_start, process_start + num_local)

        # Rewards arrive already locally sliced (per-process) from the rollout mixin; re-gather them so
        # the mining loop can find successful rollouts across all processes within each generation group.
        all_rewards = self.trainer.accelerator.gather(rewards)
        # Completion tensors are padded to the local max length per rank; align shapes before gathering.
        # Use separate variables so the original completion_ids/completion_mask stay unpadded for the
        # teacher concat (they must match the student's sequence length for logits_to_keep alignment).
        padded_completion_ids = self.trainer.accelerator.pad_across_processes(
            completion_ids, dim=1, pad_index=self.trainer._tokenizer.pad_token_id
        )
        all_completion_ids = self.trainer.accelerator.gather(padded_completion_ids)
        all_prompts = gather_object(prompts)
        total_samples = all_rewards.shape[0]
        all_feedbacks = gather_object(feedbacks) if feedbacks is not None else [None] * total_samples

        threshold = self.trainer.args.success_reward_threshold
        dont_reprompt_self = self.trainer.args.dont_reprompt_on_self_success
        feedback_only_without_solution = self.trainer.args.environment_feedback_only_without_solution
        self_distillation_mask = torch.zeros(total_samples, device=device)
        num_with_solution = 0
        num_with_feedback_available = 0
        num_with_feedback_used = 0
        success_group_count = 0
        successful_demo_indices: list[int | None] = [None] * total_samples
        use_feedback_flags: list[bool] = [False] * total_samples
        has_solution_flags: list[bool] = [False] * total_samples

        for i in range(total_samples):
            group_start = (i // num_generations) * num_generations
            group_end = group_start + num_generations

            successful = []
            if self.trainer.args.use_successful_as_teacher:
                for j in range(group_start, group_end):
                    if dont_reprompt_self and j == i:
                        continue
                    if all_rewards[j].item() >= threshold:
                        successful.append(j)

            if i % num_generations == 0:
                # Count groups with any successful rollout, ignoring self-exclusion which only
                # affects per-sample teacher assignment, not whether the group has successes.
                group_has_success = any(all_rewards[j].item() >= threshold for j in range(group_start, group_end))
                if group_has_success:
                    success_group_count += 1

            raw_feedback = all_feedbacks[i]
            has_feedback = isinstance(raw_feedback, str) and raw_feedback.strip() != ""
            if has_feedback:
                num_with_feedback_available += 1

            has_solution = len(successful) > 0
            has_solution_flags[i] = has_solution
            if has_solution:
                successful_demo_indices[i] = successful[0]
            use_feedback = (
                self.trainer.args.include_environment_feedback
                and has_feedback
                and (not feedback_only_without_solution or not has_solution)
            )
            use_feedback_flags[i] = use_feedback
            if use_feedback:
                num_with_feedback_used += 1
            if has_solution or use_feedback:
                self_distillation_mask[i] = 1.0
            if has_solution:
                num_with_solution += 1

        local_teacher_messages = []
        local_self_distillation_mask = self_distillation_mask[process_slice]
        for global_idx in range(process_start, process_start + num_local):
            original_prompt = all_prompts[global_idx]
            raw_feedback = all_feedbacks[global_idx]
            has_solution = has_solution_flags[global_idx]
            use_feedback = use_feedback_flags[global_idx]

            if not has_solution and not use_feedback:
                local_teacher_messages.append(original_prompt)
                continue

            solution_text = ""
            if has_solution:
                demo_idx = successful_demo_indices[global_idx]
                if demo_idx is None:
                    raise RuntimeError("Expected a successful demonstration index for an active SDPO teacher prompt.")
                demo_ids = all_completion_ids[demo_idx]
                demo_ids = demo_ids[demo_ids != self.trainer._tokenizer.pad_token_id]
                demo_text = self.trainer.processing_class.decode(demo_ids, skip_special_tokens=True)

                if self.trainer.args.remove_thinking_from_demonstration:
                    demo_text = re.sub(r"<think>.*?</think>", "", demo_text, flags=re.DOTALL).strip()

                solution_text = self.trainer.args.solution_template.format(successful_previous_attempt=demo_text)

            feedback_text = ""
            if use_feedback:
                feedback_text = self.trainer.args.feedback_template.format(feedback_raw=raw_feedback)

            if isinstance(original_prompt, list):
                system_messages = original_prompt[:-1]
                prompt_text = _extract_last_user_text(original_prompt)
                reprompt_text = self._build_reprompt_text(prompt_text, solution_text, feedback_text)
                local_teacher_messages.append(system_messages + [{"role": "user", "content": reprompt_text}])
            else:
                local_teacher_messages.append(self._build_reprompt_text(original_prompt, solution_text, feedback_text))

        teacher_batch = self._tokenize_teacher_messages(local_teacher_messages)
        teacher_input_ids = torch.cat([teacher_batch["prompt_ids"], completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_batch["prompt_mask"], completion_mask], dim=1)

        batch_size = total_samples if total_samples > 0 else 1
        num_groups = max(1, total_samples // max(1, num_generations))
        self.last_metrics = {
            "self_distillation/success_group_fraction": success_group_count / num_groups,
            "self_distillation/success_sample_fraction": num_with_solution / batch_size,
            "self_distillation/feedback_available_fraction": num_with_feedback_available / batch_size,
            "self_distillation/feedback_used_fraction": num_with_feedback_used / batch_size,
            "self_distillation/reprompt_sample_fraction": self_distillation_mask.float().mean().item(),
        }

        return {
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "self_distillation_mask": local_self_distillation_mask,
        }


class SDPOTrainer(_BaseTrainer):
    """
    Trainer for Self-Distillation Policy Optimization (SDPO).

    SDPO augments on-policy optimization with self-distillation from the model's own high-reward trajectories. It
    converts tokenized feedback into a dense learning signal without any external teacher or explicit reward model.
    SDPO treats the current model conditioned on feedback as a self-teacher and distills its feedback-informed
    next-token predictions back into the policy.
    """

    config_cls = SDPOConfig
    _tag_names = ["trl", "sdpo"]
    _name = "SDPO"
    # docstyle-ignore
    _paper = {
        "title": "Reinforcement Learning via Self-Distillation",
        "id": "2601.20802",
        "citation": textwrap.dedent("""\
            @article{hubotter2026sdpo,
                title        = {{Reinforcement Learning via Self-Distillation}},
                author       = {Jonas H\\"ubotter and Frederike L\\"ubeck and Lejs Behric and Anton Baumann and Marco Bagatella and Daniel Marta and Ido Hakimi and Idan Shenfeld and Thomas Kleine Buening and Carlos Guestrin and Andreas Krause},
                year         = 2026,
                eprint       = {arXiv:2601.20802}
            }"""),
    }

    def __init__(
        self,
        model: str | PreTrainedModel | nn.Module,
        reward_funcs: Any | list[Any] | None = None,
        args: SDPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config=None,
    ):
        if reward_funcs is None or (isinstance(reward_funcs, list) and len(reward_funcs) == 0):
            raise ValueError("`reward_funcs` is required for SDPOTrainer because SDPO must score rollouts.")
        if train_dataset is None:
            raise ValueError("`train_dataset` is required")

        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        elif args.model_init_kwargs is not None:
            logger.warning(
                "You passed `model_init_kwargs` to the self-distillation config, but `model` is already "
                "instantiated. The `model_init_kwargs` will be ignored."
            )

        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        if peft_config is None and getattr(model, "peft_config", None) is not None:
            logger.warning(
                "The provided self-distillation student model already contains a PEFT adapter. "
                "This setup is accepted but not directly supported. In particular, `teacher_model_kind='base'` "
                "may refer to the underlying base weights rather than the exact initially loaded student state "
                "including its adapter. For unambiguous teacher behavior, start from a merged/non-adapter model "
                "or manage separate adapters explicitly."
            )
        if peft_config is not None:
            if not is_peft_available():
                raise ImportError(
                    "You passed `peft_config` but the `peft` library is not installed. "
                    "Install it with `pip install trl[peft]`."
                )
            if not isinstance(peft_config, PeftConfig):
                raise TypeError(
                    f"`peft_config` must be a `peft.PeftConfig` instance (e.g. `peft.LoraConfig`), "
                    f"got {type(peft_config).__name__}."
                )
            if is_peft_model(model):
                raise ValueError(
                    "You passed a `PeftModel` instance together with a `peft_config`. Pass either a base "
                    "model with `peft_config`, or a pre-wrapped PEFT model."
                )
        if peft_config is not None or (is_peft_available() and getattr(model, "peft_config", None) is not None):
            model = prepare_peft_model(model, peft_config, args)

        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(
                get_config_model_id(model.config), truncation_side="left", padding_side="left"
            )

        if isinstance(processing_class, ProcessorMixin):
            self._tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            self._tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.num_generations_eval = args.num_generations_eval or args.num_generations
        self.num_iterations = args.num_iterations
        self.shuffle_dataset = args.shuffle_dataset
        self.loss_type = args.loss_type
        self.mask_truncated_completions = args.mask_truncated_completions
        self.temperature = args.temperature
        self.use_vllm = args.use_vllm
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self._step = 0
        self._buffered_inputs = None
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._diagnostic_counters = {
            "train": defaultdict(int),
            "eval": defaultdict(int),
        }

        self.generation_kwargs = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "pad_token_id": self._tokenizer.pad_token_id,
            "bos_token_id": self._tokenizer.bos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "repetition_penalty": args.repetition_penalty,
            "cache_implementation": args.cache_implementation,
        }
        if args.generation_kwargs is not None:
            self.generation_kwargs.update(args.generation_kwargs)
        self.generation_config = GenerationConfig(**self.generation_kwargs, disable_compile=True)

        if hasattr(model, "warnings_issued"):
            model.warnings_issued["estimate_tokens"] = True

        # Teacher logprobs from the running vLLM generation server (reuses the synced student weights) instead of a
        # local teacher forward. MVP: `live` teacher only — the generation server holds the current student weights.
        self.use_teacher_server = args.use_teacher_server
        if self.use_teacher_server:
            if not (args.use_vllm and args.vllm_mode == "server"):
                raise ValueError(
                    "`use_teacher_server=True` requires `use_vllm=True` and `vllm_mode='server'`: teacher logprobs are "
                    "served by the same vLLM server used for generation."
                )
            if args.teacher_model_kind != "live":
                raise ValueError(
                    "`use_teacher_server=True` only supports `teacher_model_kind='live'` (the generation server holds "
                    f"the current student weights), got {args.teacher_model_kind!r}."
                )
            if args.distillation_weight != 1.0:
                raise ValueError(
                    "`use_teacher_server=True` only supports pure self-distillation with `distillation_weight=1.0`, "
                    f"got {args.distillation_weight}. A convex blend with the policy loss needs the full-vocabulary "
                    "logits, which the server does not return."
                )
            if args.distillation_mode not in ("sampled_token", "topk_logits"):
                raise ValueError(
                    "`use_teacher_server=True` only supports `distillation_mode` in {'sampled_token', 'topk_logits'}, "
                    f"got {args.distillation_mode!r}. The server returns the teacher's top-k logprobs, not the full "
                    "vocabulary, so `full_logits` is unavailable. Note `topk_logits` distills over the teacher's own "
                    "top-k support (the server cannot score the student's top-k indices)."
                )
        # Liger fused JSD loss for `full_logits`: same generalized JSD as `compute_divergence`, so alpha maps to beta.
        self.use_liger_loss = False
        if args.use_liger_kernel:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `use_liger_kernel` as the self-distillation loss. Run "
                    "`pip install liger-kernel`."
                )
            if args.distillation_weight != 1.0:
                raise ValueError(
                    "`use_liger_kernel` only supports pure self-distillation with `distillation_weight=1.0`, got "
                    f"{args.distillation_weight}. A convex blend with the policy loss needs the policy-gradient term, "
                    "so the fused distillation kernel offers no benefit there."
                )
            if args.distillation_mode != "full_logits":
                raise ValueError(
                    "`use_liger_kernel` only supports `distillation_mode='full_logits'`, got "
                    f"{args.distillation_mode!r}. The fused JSD kernel operates on the full vocabulary and cannot "
                    "express the top-k support or sampled-token objectives."
                )
            if args.distillation_is_clip is not None:
                raise ValueError(
                    "`use_liger_kernel` is incompatible with `distillation_is_clip`: the fused kernel does not expose "
                    "per-token losses for importance-sampling clipping."
                )
            if args.loss_type != "bnpo":
                logger.warning(
                    "The Liger fused loss reduces with a token-level mean (equivalent to `loss_type='bnpo'`); the "
                    f"configured `loss_type={args.loss_type!r}` is ignored on the Liger path."
                )
            self.liger_jsd_loss = LigerFusedLinearJSDLoss(
                beta=args.distillation_alpha,
                ignore_index=-100,
                temperature=args.temperature,
                compiled=False,
                weight_hard_loss=0.0,
                weight_soft_loss=1.0,
            )
            self._forward_redirection = _ForwardRedirection()
            self.use_liger_loss = True

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            compute_loss_func="non-None value to disable scaling",
        )

        self._last_loaded_step = -1 if self.use_vllm else 0
        if self.use_vllm:
            from ...generation.vllm_generation import VLLMGeneration

            self.vllm_generation = VLLMGeneration(
                model=self.model,
                accelerator=self.accelerator,
                is_fsdp_enabled=self.is_fsdp_enabled,
                processing_class=self.processing_class,
                mode=args.vllm_mode,
                server_base_url=args.vllm_server_base_url,
                server_host=args.vllm_server_host,
                server_port=args.vllm_server_port,
                group_port=args.vllm_group_port,
                server_timeout=args.vllm_server_timeout,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_length=args.vllm_max_model_length,
                max_num_seqs=args.per_device_train_batch_size
                * args.vllm_tensor_parallel_size
                * args.steps_per_generation,
                enable_sleep_mode=args.vllm_enable_sleep_mode,
                model_impl=args.vllm_model_impl,
                repetition_penalty=args.repetition_penalty,
                temperature=self.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                max_completion_length=self.max_completion_length,
                logprobs=None,
                generation_kwargs=args.generation_kwargs,
            )

        # Per-rank read-only client to the same generation server for teacher scoring (weights are synced there by
        # `VLLMGeneration`; scoring needs no weight-update communicator). Mirrors the distillation trainer's
        # `teacher_client`.
        self.teacher_client = None
        if self.use_teacher_server:
            from ...generation.vllm_client import VLLMClient

            base_url = args.vllm_server_base_url or f"http://{args.vllm_server_host}:{args.vllm_server_port}"
            self.teacher_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        self._setup_teacher_model()
        self.model_accepts_loss_kwargs = False

        self.importance_sampling_level = args.importance_sampling_level
        self.scale_rewards = args.scale_rewards
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high
        self.beta = args.beta

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_model_init_kwargs = args.model_init_kwargs or {}
                if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                    reward_model_init_kwargs["device_map"] = None
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func,
                    num_labels=1,
                    **reward_model_init_kwargs,
                )
            if isinstance(reward_funcs[i], nn.Module):
                self.reward_func_names.append(get_config_model_id(reward_funcs[i].config).split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        if args.reward_weights is not None:
            if len(args.reward_weights) != len(self.reward_funcs):
                raise ValueError("Number of reward weights must match number of reward functions")
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(self.reward_funcs), dtype=torch.float32)

        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(self.reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        if len(reward_processing_classes) != len(self.reward_funcs):
            raise ValueError("Number of reward processing classes must match number of reward functions")

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, self.reward_funcs, strict=True)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(get_config_model_id(reward_func.config))
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                elif self.is_fsdp_enabled:
                    self.reward_funcs[i] = prepare_fsdp(reward_func, self.accelerator)
                else:
                    self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        self.teacher_context_builder = SuccessfulRolloutTeacherContextBuilder(self)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "privileged_context"]

    def _dispatch_self_distillation_callback(self, event_name: str, **payload) -> None:
        for callback in self.callback_handler.callbacks:
            callback_fn = getattr(callback, event_name, None)
            if callback_fn is not None:
                callback_fn(
                    args=self.args,
                    state=self.state,
                    control=self.control,
                    model=self.model,
                    processing_class=self.processing_class,
                    **payload,
                )

    def _setup_teacher_model(self) -> None:
        """Prepare teacher state according to the semantic teacher choice.

        Resolve `teacher_model_kind` × PEFT state into the effective teacher:

            - `"live"` (any model):
                Teacher is the student. No divergence, no callback.
            - `"base"` + PEFT model:
                Teacher reuses `self.model`; the base weights are recovered downstream by disabling the adapter via
                `use_adapter` during teacher forward.
            - `"base"` + non-PEFT model:
                Teacher is a frozen deepcopy of the initial student (falls through to the copy branch below).
            - `"ema"` + pure-LoRA training:
                Teacher reuses `self.model`; a dedicated `"teacher"` LoRA adapter is attached and updated by
                `PEFTAdapterEMACallback`. Teacher forward switches to that adapter downstream.
            - `"ema"` (otherwise):
                Teacher is a frozen deepcopy synchronized each step by `SyncTeacherModelCallback`.

        Must be called after `super().__init__` so that `self.callback_handler` is available.
        """

        teacher_model_kind = self.args.teacher_model_kind

        if teacher_model_kind == "live":
            self.teacher_model = self.model
            return

        if teacher_model_kind == "base" and is_peft_model(self.model):
            self.teacher_model = self.model
            return

        if self._use_peft_ema_teacher_adapter():
            # Must run after super().__init__ so self.callback_handler exists.
            self.add_callback(
                PEFTAdapterEMACallback(
                    model=self.model,
                    teacher_adapter_name="teacher",
                    update_rate=self.args.teacher_update_rate,
                    sync_steps=self.args.teacher_sync_steps,
                    accelerator=self.accelerator,
                )
            )
            self.teacher_model = self.model
            return

        if is_peft_model(self.model):
            raise ValueError(
                "`teacher_model_kind='ema'` with a non-pure-LoRA PEFT model is not supported: the separate EMA "
                "teacher cannot be parameter-matched to the PEFT student. Use pure-LoRA training, a non-PEFT model, "
                "or `teacher_model_kind` in {'live', 'base'}."
            )

        # Build the teacher from the model path (like the GRPO/DPO reference model) rather than deep-copying the
        # student: under ZeRO-3 the student params are already sharded, so a deep copy would clone empty shards.
        model_init_kwargs = self.args.model_init_kwargs or {}
        if self.args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
            model_init_kwargs["device_map"] = None
        self.teacher_model = create_model_from_path(get_config_model_id(self.model.config), **model_init_kwargs)
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()
        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(self.teacher_model, self.accelerator)
        elif self.is_fsdp_enabled:
            self.teacher_model = prepare_fsdp(self.teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(self.teacher_model, evaluation_mode=True)

        if teacher_model_kind == "ema":
            self.add_callback(SyncTeacherModelCallback(teacher_model=self.teacher_model, accelerator=self.accelerator))

    def _use_peft_ema_teacher_adapter(self) -> bool:
        return self.args.teacher_model_kind == "ema" and is_pure_lora_training(self.model, self.accelerator)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset=None) -> Sampler:
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations_eval,
            seed=self.args.seed,
        )

    def training_step(self, model, inputs, num_items_in_batch):
        # Gather spans forward+backward: the fused JSD computes the lm_head grad in backward.
        with self._get_liger_zero3_lm_head_gather_ctx(model):
            output = super().training_step(model, inputs, num_items_in_batch)
        self._step += 1
        return output

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if not isinstance(inputs, dict):
            inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None

    def _prepare_inputs(self, generation_batch):
        """Return the per-step training batch, regenerating rollouts and buffering them for reuse in train mode.

        In train mode, rollouts are generated once every `steps_per_generation * num_iterations` steps and split into
        per-step slices reused until the next regeneration. In eval mode, every batch is freshly prepared.
        """
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                buffered_batch = self._prepare_training_batch(generation_batch)
                self._buffered_inputs = split_tensor_dict(buffered_batch, self.args.steps_per_generation)
                self._dispatch_self_distillation_callback(
                    "on_generation_batch_built",
                    generate_every=generate_every,
                    steps_per_generation=self.args.steps_per_generation,
                )
            return self._buffered_inputs[self._step % self.args.steps_per_generation]
        return self._prepare_training_batch(generation_batch)

    def _prepare_training_batch(self, inputs: list[dict[str, Any]]) -> TrainingBatch:
        """Sample student rollouts, calculate advantage and construct teacher prompts"""
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Generate student rollouts and decode completions for reward functions
        batch = self.sample_rollouts(inputs)
        prompts = [example["prompt"] for example in inputs]
        privileged_contexts = [example.get("privileged_context") for example in inputs]
        completion_ids_list = self._get_completion_ids_list(batch)
        if is_conversational({"prompt": prompts[0]}):
            completions_text = self.processing_class.batch_decode(batch["completion_ids"], skip_special_tokens=True)
            completions = [[{"role": "assistant", "content": content}] for content in completions_text]
        else:
            completions = self.processing_class.batch_decode(batch["completion_ids"], skip_special_tokens=True)

        # Compute rewards over the globally gathered rollout batch
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        if rewards_per_func.numel() == 0:
            rewards = torch.zeros(self.accelerator.num_processes * len(prompts), device=device)
        else:
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Normalize rewards within generation groups to produce local policy advantages
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1).repeat_interleave(num_generations, dim=0)
        if self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            group_std_rewards = rewards.view(-1, num_generations).std(dim=1)
        elif self.scale_rewards == "none":
            std_rewards = torch.ones_like(rewards)
            group_std_rewards = torch.ones(rewards.numel() // num_generations, device=device, dtype=rewards.dtype)
        else:
            group_std_rewards = rewards.view(-1, num_generations).std(dim=1)
            std_rewards = group_std_rewards.repeat_interleave(num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_rewards + 1e-4)
        local_batch_size = batch["completion_ids"].size(0)
        process_start = self.accelerator.process_index * local_batch_size
        process_slice = slice(process_start, process_start + local_batch_size)
        local_rewards = rewards[process_slice]
        local_advantages = advantages[process_slice]

        self._record_reward_diagnostics(mode, rewards, rewards_per_func, group_std_rewards)
        self._record_completion_metrics(mode, batch)

        batch["rewards"] = local_rewards
        batch["advantages"] = local_advantages
        teacher_context = self.teacher_context_builder.build(
            batch,
            prompts,
            batch["rewards"],
            feedbacks=privileged_contexts,
        )

        for key, value in self.teacher_context_builder.last_metrics.items():
            self._metrics[mode][key].append(value)
        self._warn_on_inactive_self_distillation(mode)

        self._dispatch_self_distillation_callback(
            "on_teacher_context_built",
            teacher_input_ids=teacher_context["teacher_input_ids"],
            teacher_attention_mask=teacher_context["teacher_attention_mask"],
            completion_mask=batch["completion_mask"],
            self_distillation_mask=teacher_context["self_distillation_mask"],
        )

        batch.update(
            {
                "teacher_input_ids": teacher_context["teacher_input_ids"],
                "teacher_attention_mask": teacher_context["teacher_attention_mask"],
                "self_distillation_mask": teacher_context["self_distillation_mask"],
            }
        )

        self._dispatch_self_distillation_callback(
            "on_self_distillation_batch_prepared",
            old_per_token_logps=batch.get("old_per_token_logps"),
            prompt_ids=batch["prompt_ids"],
            completion_ids=batch["completion_ids"],
            teacher_input_ids=batch["teacher_input_ids"],
            teacher_attention_mask=batch["teacher_attention_mask"],
            self_distillation_mask=batch.get("self_distillation_mask"),
        )
        return batch

    def sample_rollouts(self, inputs: list[dict[str, Any]]) -> TrainingBatch:
        """Generate completions for a batch of prompts and assemble the training batch."""
        prompts = [example["prompt"] for example in inputs]
        prompt_ids = self._tokenize_prompts(prompts)
        self._dispatch_self_distillation_callback(
            "on_generation_prompts_selected",
            generation_prompts=prompts,
            generation_prompt_text=None,
        )

        prompt_ids_list, completion_ids_list = self._generate(prompt_ids)
        device = self.accelerator.device
        prompt_ids = [torch.tensor(ids) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self._tokenizer.pad_token_id, padding_side="left").to(device=device)
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left").to(device=device)

        completion_ids = [torch.tensor(ids) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self._tokenizer.pad_token_id, padding_side="right").to(
            device=device
        )
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right").to(device=device)

        if self.mask_truncated_completions:
            eos_and_pad = [self._tokenizer.eos_token_id, self._tokenizer.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        old_per_token_logps = self._compute_rollout_logps(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
        )
        batch: TrainingBatch = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "raw_completion_lengths": torch.tensor(
                [len(ids) for ids in completion_ids_list], device=device, dtype=torch.long
            ),
        }
        if old_per_token_logps is not None:
            batch["old_per_token_logps"] = old_per_token_logps
        return batch

    def _get_completion_ids_list(self, batch: TrainingBatch) -> list[list[int]]:
        raw_completion_lengths = batch["raw_completion_lengths"].detach().cpu().tolist()
        return [
            ids[:length].tolist()
            for ids, length in zip(batch["completion_ids"].detach().cpu(), raw_completion_lengths, strict=True)
        ]

    def _tokenize_prompts_untruncated(self, prompts: list[Any]) -> list[list[int]]:
        if is_conversational({"prompt": prompts[0]}):
            tokenized = self.processing_class.apply_chat_template(
                conversation=prompts,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **self.chat_template_kwargs,
            )
            prompt_ids = tokenized["input_ids"]
        else:
            prompt_ids = self.processing_class(text=prompts)["input_ids"]
        return prompt_ids

    def _tokenize_prompts(self, prompts: list[Any]) -> list[list[int]]:
        prompt_ids = self._tokenize_prompts_untruncated(prompts)
        if self.max_prompt_length is not None:
            prompt_ids = [ids[-self.max_prompt_length :] for ids in prompt_ids]
        return prompt_ids

    def _generate(self, prompt_ids: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
        if self.use_vllm:
            return self._generate_vllm(prompt_ids)
        return self._generate_transformers(prompt_ids)

    def _generate_vllm(self, prompt_ids: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
        if self.state.global_step != self._last_loaded_step:
            self.vllm_generation.sync_weights()
            self._last_loaded_step = self.state.global_step

        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        prompt_ids_out, completion_ids_list, _, _ = self.vllm_generation.generate(
            prompts=prompt_ids,
            images=None,
            num_generations=num_generations,
        )
        return prompt_ids_out, completion_ids_list

    def _generate_transformers(self, prompt_ids: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
        device = self.accelerator.device
        prompt_tensors = [torch.tensor(ids) for ids in prompt_ids]
        padded_ids = pad(prompt_tensors, padding_value=self._tokenizer.pad_token_id, padding_side="left").to(
            device=device
        )
        attention_mask = pad([torch.ones_like(t) for t in prompt_tensors], padding_value=0, padding_side="left").to(
            device=device
        )
        generate_inputs: dict[str, torch.Tensor | Any] = {"input_ids": padded_ids, "attention_mask": attention_mask}

        with (
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs, generation_config=self.generation_config
            )

        prompt_length = generate_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        is_eos = completion_ids == self._tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()
        completion_ids_list = [
            c[m].tolist() for c, m in zip(completion_ids.cpu(), completion_mask.bool().cpu(), strict=True)
        ]
        return prompt_ids, completion_ids_list

    def _compute_rollout_logps(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        generate_every = self.args.steps_per_generation * self.num_iterations
        old_per_token_logps = None

        if self.args.gradient_accumulation_steps % generate_every != 0:
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)
            with torch.no_grad():
                logits = self._forward_logits(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                )
                old_per_token_logps = selective_log_softmax(logits, completion_ids)

        return old_per_token_logps

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        if len(self.reward_funcs) == 0:
            return torch.zeros((len(prompts), 0), device=device)

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, strict=True)
        ):
            if isinstance(reward_func, nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions, strict=True)]
                    texts = [
                        apply_chat_template(x, reward_processing_class, **self.chat_template_kwargs)["text"]
                        for x in messages
                    ]
                else:
                    texts = [p + c for p, c in zip(prompts, completions, strict=True)]
                reward_inputs = reward_processing_class(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = _BaseTrainer._prepare_inputs(self, reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    completion_ids=completion_ids_list,
                    **reward_kwargs,
                )
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        return self.accelerator.gather(rewards_per_func)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The SDPOTrainer does not support returning outputs")

        if self.args.distillation_weight == 1.0:
            if self.use_teacher_server:
                loss = self._compute_server_distillation_loss(model, inputs)
            elif self.use_liger_loss:
                accumulation_scale = self.current_gradient_accumulation_steps if self.model.training else 1.0
                return self._compute_liger_loss(model, inputs) / accumulation_scale
            else:
                distillation_logits = self._compute_teacher_student_logits(model, self.teacher_model, inputs)
                loss = self._compute_self_distillation_loss(model, inputs, distillation_logits)
        elif self.args.distillation_weight == 0.0:
            student_input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
            student_attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
            student_logits = self._forward_logits(
                model=model,
                input_ids=student_input_ids,
                attention_mask=student_attention_mask,
                logits_to_keep=inputs["completion_ids"].size(1),
            )
            loss = self._compute_policy_loss(inputs, student_logits)
        else:
            distillation_logits = self._compute_teacher_student_logits(model, self.teacher_model, inputs)
            policy_loss = self._compute_policy_loss(inputs, distillation_logits.student_logits)
            distillation_loss = self._compute_self_distillation_loss(model, inputs, distillation_logits)
            loss = (
                1 - self.args.distillation_weight
            ) * policy_loss + self.args.distillation_weight * distillation_loss

        accumulation_scale = self.current_gradient_accumulation_steps if self.model.training else 1.0
        return loss / accumulation_scale

    def _compute_policy_loss(
        self,
        inputs,
        student_logits,
    ) -> torch.Tensor:
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        per_token_logps = selective_log_softmax(student_logits, completion_ids)
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "sequence":
            log_ratio = (log_ratio * completion_mask).sum(-1, keepdim=True) / completion_mask.sum(
                -1, keepdim=True
            ).clamp(min=1.0)
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)

        if self.loss_type == "grpo":
            loss = (per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            loss = loss.mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        elif self.loss_type == "dapo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["self_distillation/policy_loss"].append(
            self.accelerator.gather(loss.detach()).mean().item()
        )

        return loss

    def _compute_self_distillation_loss(
        self,
        model,
        inputs: TrainingBatch,
        distillation_logits: DistillationLogits,
    ) -> torch.Tensor:
        """Compute the per-token distillation loss and aggregate by normalizing over sequence length.

        Dispatches between three objectives based on `distillation_mode`:

            - `"topk_logits"`: top-k approximation of the divergence, optionally with a tail bucket for the remaining
              probability mass (`distillation_add_tail`).
            - `"full_logits"`: full-vocab divergence.
            - `"sampled_token"`: token-level (reverse-KL) distillation on sampled `completion_ids`.

        When `distillation_is_clip` is set and `old_per_token_logps` are available, the loss is corrected by a clipped
        importance-sampling ratio between the current student and the student at rollout time.
        """
        if distillation_logits.loss_mask.sum() == 0:
            mode = "train" if model.training else "eval"
            self._log_self_distillation_metric(mode, 0.0)
            # Keep the zero loss attached to the student graph so backward produces zero gradients instead of stopping.
            return distillation_logits.student_logits.sum() * 0.0

        if self.args.distillation_mode == "topk_logits":
            if self.args.distillation_topk is None:
                raise ValueError("`distillation_mode='topk_logits'` requires `distillation_topk` to be set.")
            per_token_loss = compute_topk_self_distillation_loss(
                distillation_logits.student_logits,
                distillation_logits.teacher_logits,
                distillation_topk=self.args.distillation_topk,
                distillation_alpha=self.args.distillation_alpha,
                distillation_add_tail=self.args.distillation_add_tail,
            )
        elif self.args.distillation_mode == "full_logits":
            per_token_loss = compute_full_logit_self_distillation_loss(
                distillation_logits.student_logits,
                distillation_logits.teacher_logits,
                distillation_alpha=self.args.distillation_alpha,
            )
        elif self.args.distillation_mode == "sampled_token":
            per_token_loss = compute_sampled_token_self_distillation_loss(
                distillation_logits.student_logits,
                distillation_logits.teacher_logits,
                distillation_logits.completion_ids,
                distillation_alpha=self.args.distillation_alpha,
            )
        else:
            raise ValueError(
                "distillation_mode must be one of: 'sampled_token', 'full_logits', 'topk_logits', "
                f"got {self.args.distillation_mode!r}"
            )

        old_per_token_logps = inputs.get("old_per_token_logps")
        if self.args.distillation_is_clip is not None and old_per_token_logps is not None:
            student_per_token_logps = selective_log_softmax(
                distillation_logits.student_logits,
                distillation_logits.completion_ids,
            )
            per_token_loss = apply_importance_sampling_clipping(
                per_token_loss,
                student_per_token_logps,
                old_per_token_logps,
                self.args.distillation_is_clip,
            )

        loss = (per_token_loss * distillation_logits.loss_mask).sum(-1) / distillation_logits.loss_mask.sum(-1).clamp(
            min=1.0
        )
        loss = loss.mean()

        mode = "train" if model.training else "eval"
        mean_distill_loss = (
            per_token_loss * distillation_logits.loss_mask
        ).sum() / distillation_logits.loss_mask.sum().clamp(min=1.0)
        self._log_self_distillation_metric(
            mode,
            self.accelerator.gather(mean_distill_loss).mean().item(),
        )
        return loss

    def _compute_server_distillation_loss(self, model, inputs: TrainingBatch) -> torch.Tensor:
        """Distillation loss with teacher logprobs served by the vLLM generation server (`teacher_model_kind='live'`).

        The student is forwarded locally (grad) for its logits; the teacher logprobs are fetched from the server (no
        local teacher forward). `sampled_token` distills the realized token (reverse KL); `topk_logits` distills over
        the teacher's own top-k support (the server cannot score the student's top-k indices).
        """
        # Buffered batches are reused across optimizer steps (`num_iterations > 1`), so the server weights can lag the
        # live student; re-sync before scoring.
        if self.state.global_step != self._last_loaded_step:
            self.vllm_generation.sync_weights()
            self._last_loaded_step = self.state.global_step

        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        logits_to_keep = completion_ids.size(1)

        self_distillation_mask = inputs.get("self_distillation_mask")
        if self_distillation_mask is None:
            loss_mask = completion_mask
        else:
            loss_mask = completion_mask * self_distillation_mask.unsqueeze(1)

        student_input_ids = torch.cat([inputs["prompt_ids"], completion_ids], dim=1)
        student_attention_mask = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)
        student_logits = self._forward_logits(
            model=model,
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            logits_to_keep=logits_to_keep,
        )

        mode = "train" if model.training else "eval"
        # No early return for an all-masked rank: the gathers below are collective, so every rank must run the full
        # path (the masked reductions resolve to a zero loss on such a rank).
        teacher = self._get_teacher_token_logprobs_from_server(inputs, logits_to_keep)
        # Padding positions come back as `-inf`; the masked-out positions are neutralized to finite values below so the
        # divergence does not leak `+inf` through them (mirrors the distillation trainer).
        required = loss_mask.bool()
        student_per_token_logps = selective_log_softmax(student_logits, completion_ids)

        teacher_per_token_logps = teacher["actual_logprobs"]
        if (required & ~torch.isfinite(teacher_per_token_logps)).any():
            raise ValueError("Teacher server returned no logprob for a required completion token.")
        teacher_per_token_logps = torch.where(
            required, teacher_per_token_logps, torch.zeros_like(teacher_per_token_logps)
        )

        # Mean realized-token logprob gap between the local student and the context-conditioned teacher on the server.
        # It shrinks as distillation progresses; a sudden jump points at a weight-sync or numerics problem.
        abs_diff = ((student_per_token_logps.detach() - teacher_per_token_logps).abs() * loss_mask).sum() / (
            loss_mask.sum().clamp(min=1.0)
        )
        self._metrics[mode]["self_distillation/server_logprob_abs_diff"].append(
            self.accelerator.gather(abs_diff).mean().item()
        )

        if self.args.distillation_mode == "sampled_token":
            # Reverse-KL token-level objective, matching `compute_sampled_token_self_distillation_loss`.
            per_token_loss = (student_per_token_logps - teacher_per_token_logps).detach() * student_per_token_logps
        else:
            teacher_topk_logps = teacher["topk_logprobs"]
            teacher_topk_ids = teacher["topk_token_ids"]
            # Project the student onto the teacher's top-k support, then renormalize (or add a tail bucket) before the
            # divergence, matching `compute_topk_self_distillation_loss` on the teacher's support.
            keep = required.unsqueeze(-1)
            student_topk_logps = torch.gather(
                torch.log_softmax(student_logits, dim=-1), dim=-1, index=teacher_topk_ids
            )
            student_topk_logps = torch.where(keep, student_topk_logps, torch.zeros_like(student_topk_logps))
            teacher_topk_logps = torch.where(keep, teacher_topk_logps, torch.zeros_like(teacher_topk_logps))
            if self.args.distillation_add_tail:
                student_topk_logps = add_tail_bucket(student_topk_logps)
                teacher_topk_logps = add_tail_bucket(teacher_topk_logps)
            else:
                student_topk_logps = student_topk_logps - torch.logsumexp(student_topk_logps, dim=-1, keepdim=True)
                teacher_topk_logps = teacher_topk_logps - torch.logsumexp(teacher_topk_logps, dim=-1, keepdim=True)
            per_token_loss = compute_divergence(student_topk_logps, teacher_topk_logps, self.args.distillation_alpha)

        old_per_token_logps = inputs.get("old_per_token_logps")
        if self.args.distillation_is_clip is not None and old_per_token_logps is not None:
            per_token_loss = apply_importance_sampling_clipping(
                per_token_loss, student_per_token_logps, old_per_token_logps, self.args.distillation_is_clip
            )

        loss = (per_token_loss * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)
        loss = loss.mean()
        mean_distill_loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        self._log_self_distillation_metric(mode, self.accelerator.gather(mean_distill_loss).mean().item())
        return loss

    def _get_teacher_token_logprobs_from_server(
        self, inputs: TrainingBatch, logits_to_keep: int
    ) -> dict[str, torch.Tensor]:
        """Per-token teacher logprobs from the vLLM server.

        Returns a dict with `actual_logprobs` (`(B, T)`, the realized-token logprob for reverse KL) and `topk_logprobs`
        / `topk_token_ids` (`(B, T, K)`, the teacher's top-k support for top-k logit distillation). Completions are
        right-padded, so server values fill the leading positions of each row and the trailing padding keeps the `-inf`
        / `0` sentinels (neutralized by the caller).
        """
        import numpy as np

        teacher_input_ids = inputs["teacher_input_ids"]
        teacher_attention_mask = inputs["teacher_attention_mask"]
        device = teacher_input_ids.device
        batch_size = teacher_input_ids.size(0)
        prompt_attention_mask = teacher_attention_mask[:, : teacher_attention_mask.size(1) - logits_to_keep]
        if self.args.distillation_mode == "topk_logits" and self.args.distillation_topk is None:
            raise ValueError("`distillation_mode='topk_logits'` requires `distillation_topk` to be set.")
        top_logprobs = self.args.distillation_topk if self.args.distillation_mode == "topk_logits" else 1

        sequences, prompt_lengths, _ = build_teacher_request_inputs(
            teacher_input_ids, teacher_attention_mask, prompt_attention_mask
        )
        result = self.teacher_client.get_sequence_logprobs(
            sequences=sequences,
            prompt_lengths=prompt_lengths,
            top_logprobs=top_logprobs,
            temperature=self.temperature,
        )

        actual = np.full((batch_size, logits_to_keep), float("-inf"), dtype=np.float32)
        topk = np.full((batch_size, logits_to_keep, top_logprobs), float("-inf"), dtype=np.float32)
        topk_ids = np.zeros((batch_size, logits_to_keep, top_logprobs), dtype=np.int64)
        for i in range(batch_size):
            seq_actual = result["actual_logprobs"][i]
            n = min(len(seq_actual), logits_to_keep)
            if n:
                actual[i, :n] = np.array(seq_actual, dtype=np.float32)[:n, 0]
                topk[i, :n] = np.array(result["logprobs"][i], dtype=np.float32)[:n]
                topk_ids[i, :n] = np.array(result["logprob_token_ids"][i], dtype=np.int64)[:n]
        return {
            "actual_logprobs": torch.from_numpy(actual).to(device),
            "topk_logprobs": torch.from_numpy(topk).to(device),
            "topk_token_ids": torch.from_numpy(topk_ids).to(device),
        }

    def _compute_teacher_student_logits(
        self,
        model,
        teacher_model,
        inputs: TrainingBatch,
    ) -> DistillationLogits:
        """Compute the per-token logits of the student and teacher over the completion tokens.

        The student is forwarded on its own input (original prompt plus the sampled completion) while the teacher is
        forwarded on its input (prompt, privileged context, and the same completion). Both sets of logits are aligned
        to the completion tokens so they can be compared position-by-position in the distillation loss.

        The teacher forward runs under `torch.no_grad()` and the context resolved by
        `_get_teacher_context_for_self_distillation`, which routes it to the correct weights.
        """
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        logits_to_keep = completion_ids.size(1)

        self_distillation_mask = inputs.get("self_distillation_mask")
        if self_distillation_mask is None:
            loss_mask = completion_mask
        else:
            loss_mask = completion_mask * self_distillation_mask.unsqueeze(1)
        student_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        student_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        student_logits = self._forward_logits(
            model=model,
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            logits_to_keep=logits_to_keep,
        )

        with torch.no_grad(), self._get_teacher_context_for_self_distillation():
            teacher_logits = self._forward_logits(
                model=teacher_model,
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
                logits_to_keep=logits_to_keep,
            )

        return DistillationLogits(
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            loss_mask=loss_mask,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
        )

    def _forward_logits(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        """Forward the model and return temperature-scaled logits aligned to the completion tokens."""
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        logits = model(**model_inputs).logits
        logits = logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        return logits / self.temperature

    def _compute_liger_loss(self, model, inputs: TrainingBatch) -> torch.Tensor:
        """`full_logits` distillation via the Liger fused JSD kernel: forwards the base models for hidden states and
        fuses the lm_head projection with the divergence, never materializing the full-vocab logits.

        Each model is forwarded through its own wrapper via `_forward_redirection` so FSDP2/DeepSpeed materialize the
        sharded params during the unwrapped base forward. The fused kernel needs both lm_head weights live at once, so
        the frozen teacher weight is captured while the teacher is materialized and handed to the student pass.
        """
        logits_to_keep = inputs["completion_ids"].size(1)
        self_distillation_mask = inputs.get("self_distillation_mask")
        if self_distillation_mask is None:
            loss_mask = inputs["completion_mask"]
        else:
            loss_mask = inputs["completion_mask"] * self_distillation_mask.unsqueeze(1)

        unwrapped_student = self.accelerator.unwrap_model(model)
        unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)

        with torch.no_grad(), self._get_teacher_context_for_self_distillation():
            teacher_hidden, teacher_weight, teacher_bias = self._forward_redirection(
                self.teacher_model,
                unwrapped_teacher,
                self._liger_teacher_side,
                unwrapped_teacher,
                inputs,
                logits_to_keep,
            )

        return self._forward_redirection(
            model,
            unwrapped_student,
            self._liger_student_loss,
            unwrapped_student,
            inputs,
            logits_to_keep,
            loss_mask,
            teacher_hidden,
            teacher_weight,
            teacher_bias,
        )

    def _liger_teacher_side(self, teacher, inputs: TrainingBatch, logits_to_keep: int):
        """Teacher hidden states + frozen lm_head weight, captured while the teacher params are materialized."""
        hidden = teacher.get_decoder()(
            input_ids=inputs["teacher_input_ids"],
            attention_mask=inputs["teacher_attention_mask"],
            use_cache=False,
        ).last_hidden_state
        hidden = hidden[:, :-1][:, -logits_to_keep:]
        head = teacher.get_output_embeddings()
        # Clone so the weight survives re-sharding once this forward context exits.
        weight = head.weight.detach().clone()
        bias = head.bias.detach().clone() if head.bias is not None else None
        return hidden, weight, bias

    def _liger_student_loss(
        self,
        student,
        inputs: TrainingBatch,
        logits_to_keep,
        loss_mask,
        teacher_hidden,
        teacher_weight,
        teacher_bias,
    ):
        student_input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        student_attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        student_hidden = student.get_decoder()(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            use_cache=False,
        ).last_hidden_state
        # Align hidden states to the completion-predicting positions, matching `_forward_logits`.
        student_hidden = student_hidden[:, :-1][:, -logits_to_keep:]

        # `ignore_index` masks non-response positions; the token values only feed the disabled hard-CE term.
        completion_ids = inputs["completion_ids"]
        true_labels = torch.where(loss_mask.bool(), completion_ids, torch.full_like(completion_ids, -100))

        student_head = student.get_output_embeddings()
        # Per-sequence then batch mean (grpo), matching the non-Liger path: the fused kernel reduces by total tokens
        # (bnpo), so we call it per sequence and average.
        seq_losses = [
            self.liger_jsd_loss(
                student_input=student_hidden[i],
                student_weight=student_head.weight,
                teacher_input=teacher_hidden[i],
                teacher_weight=teacher_weight,
                true_labels=true_labels[i],
                student_bias=student_head.bias,
                teacher_bias=teacher_bias,
            )
            for i in range(student_hidden.size(0))
        ]
        loss = torch.stack(seq_losses).mean()

        mode = "train" if student.training else "eval"
        self._log_self_distillation_metric(mode, self.accelerator.gather(loss.detach()).mean().item())
        return loss

    def _get_liger_zero3_lm_head_gather_ctx(self, model):
        """Gather the sharded student/teacher lm_head weights for the fused matmul under ZeRO-3. Liger reads
        `lm_head.weight` by attribute, so the gather hook never fires; the decoder forward gathers itself. No-op
        outside ZeRO-3."""
        if not self.use_liger_loss:
            return nullcontext()

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        if deepspeed_plugin is None or deepspeed_plugin.zero_stage != 3:
            return nullcontext()

        import deepspeed

        unwrapped_student = self.accelerator.unwrap_model(model)
        unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
        student_head = unwrapped_student.get_output_embeddings()
        teacher_head = unwrapped_teacher.get_output_embeddings()
        params = [student_head.weight, teacher_head.weight]
        if student_head.bias is not None:
            params.append(student_head.bias)
        if teacher_head.bias is not None:
            params.append(teacher_head.bias)
        return deepspeed.zero.GatheredParameters(params, modifier_rank=None)

    def _get_teacher_context_for_self_distillation(self):
        """Return the context manager that routes the teacher forward to the correct weights.

        For non-PEFT models this is a no-op. For PEFT models:

            - `teacher_model_kind == "base"`: disable the student adapter so the teacher forward uses the base weights.
            - `teacher_model_kind == "ema"` under pure-LoRA training: switch to the `"teacher"` LoRA adapter.
            - otherwise: no-op; the teacher is a separate deepcopy.
        """
        teacher_model_kind = self.args.teacher_model_kind
        if not is_peft_model(self.model):
            return nullcontext()

        target_model = self.accelerator.unwrap_model(self.teacher_model)

        if teacher_model_kind == "base":
            return use_adapter(target_model, adapter_name=None)
        if teacher_model_kind == "ema" and self._use_peft_ema_teacher_adapter():
            return use_adapter(target_model, adapter_name="teacher")
        return nullcontext()

    def _record_completion_metrics(self, mode: str, batch: TrainingBatch) -> None:
        device = self.accelerator.device
        completion_ids_list = self._get_completion_ids_list(batch)
        agg_completion_lengths = self.accelerator.gather(batch["raw_completion_lengths"])
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        eos_and_pad = [self._tokenizer.eos_token_id, self._tokenizer.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

    def _log_self_distillation_metric(self, mode: str, value: float) -> None:
        metric_prefix = self._name.lower().replace(" ", "_")
        self._metrics[mode]["self_distillation/distillation_loss"].append(value)
        self._metrics[mode][f"{metric_prefix}/distillation_loss"].append(value)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {k: sum(v) / len(v) for k, v in self._metrics[mode].items() if v}
        if mode == "eval":
            metrics = {f"eval_{k}": v for k, v in metrics.items()}
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

    def _record_reward_diagnostics(
        self,
        mode: str,
        rewards: torch.Tensor,
        rewards_per_func: torch.Tensor,
        group_std_rewards: torch.Tensor,
    ) -> None:
        tolerance = self.args.diagnostics_flat_tolerance

        reward_mean = rewards.mean() if rewards.numel() > 0 else torch.tensor(0.0, device=self.accelerator.device)
        reward_std = rewards.std() if rewards.numel() > 1 else torch.tensor(0.0, device=self.accelerator.device)
        reward_min = rewards.min() if rewards.numel() > 0 else torch.tensor(0.0, device=self.accelerator.device)
        reward_max = rewards.max() if rewards.numel() > 0 else torch.tensor(0.0, device=self.accelerator.device)
        flat_group_fraction = (
            (group_std_rewards <= tolerance).float().mean()
            if group_std_rewards.numel() > 0
            else torch.tensor(1.0, device=self.accelerator.device)
        )

        self._metrics[mode]["self_distillation/reward_mean"].append(self.accelerator.gather(reward_mean).mean().item())
        self._metrics[mode]["self_distillation/reward_std"].append(self.accelerator.gather(reward_std).mean().item())
        self._metrics[mode]["self_distillation/reward_min"].append(self.accelerator.gather(reward_min).min().item())
        self._metrics[mode]["self_distillation/reward_max"].append(self.accelerator.gather(reward_max).max().item())
        self._metrics[mode]["self_distillation/group_reward_std_mean"].append(
            self.accelerator.gather(group_std_rewards.mean() if group_std_rewards.numel() > 0 else reward_std)
            .mean()
            .item()
        )
        self._metrics[mode]["self_distillation/flat_group_fraction"].append(
            self.accelerator.gather(flat_group_fraction).mean().item()
        )

        if rewards_per_func.numel() > 0:
            reward_func_means = rewards_per_func.nanmean(dim=0)
            gathered_means = self.accelerator.gather(reward_func_means).view(-1, reward_func_means.numel()).mean(dim=0)
            for reward_name, reward_func_mean in zip(self.reward_func_names, gathered_means.tolist(), strict=True):
                self._metrics[mode][f"self_distillation/rewards/{reward_name}"].append(reward_func_mean)

        reward_is_flat = reward_std.item() <= tolerance
        grouped_rewards_are_flat = flat_group_fraction.item() >= 1.0 - tolerance
        if reward_is_flat and grouped_rewards_are_flat:
            self._warn_on_degenerate_diagnostics(
                mode=mode,
                counter_key="flat_rewards",
                message=(
                    "Observed flat SDPO rewards across all sampled generations. "
                    "Policy advantages will collapse to zero, and SDPO will not learn. "
                    "Check reward density, reward shaping, or `success_reward_threshold`."
                ),
            )
        else:
            self._diagnostic_counters[mode]["flat_rewards"] = 0

    def _warn_on_inactive_self_distillation(self, mode: str) -> None:
        metrics = self.teacher_context_builder.last_metrics
        tolerance = self.args.diagnostics_flat_tolerance

        reprompt_fraction = metrics.get("self_distillation/reprompt_sample_fraction", 0.0)
        success_fraction = metrics.get("self_distillation/success_group_fraction", 0.0)

        if reprompt_fraction <= tolerance:
            self._warn_on_degenerate_diagnostics(
                mode=mode,
                counter_key="inactive_self_distillation",
                message=(
                    "SDPO self-distillation is inactive because no reprompted samples were constructed. "
                    "This usually means no rollout exceeded `success_reward_threshold` and no usable privileged "
                    "feedback was available."
                ),
            )
        else:
            self._diagnostic_counters[mode]["inactive_self_distillation"] = 0

        if success_fraction <= tolerance:
            self._warn_on_degenerate_diagnostics(
                mode=mode,
                counter_key="no_successful_rollouts",
                message=(
                    "SDPO did not find any successful rollouts in the current generation groups. "
                    "If this persists, reduce task difficulty, adjust reward shaping, or lower "
                    "`success_reward_threshold`."
                ),
            )
        else:
            self._diagnostic_counters[mode]["no_successful_rollouts"] = 0

    def _warn_on_degenerate_diagnostics(self, mode: str, counter_key: str, message: str) -> None:
        interval = self.args.diagnostics_warning_interval
        if interval == 0:
            return

        self._diagnostic_counters[mode][counter_key] += 1
        count = self._diagnostic_counters[mode][counter_key]
        if count == 1 or count % interval == 0:
            logger.warning("%s Consecutive degenerate steps: %s.", message, count)
