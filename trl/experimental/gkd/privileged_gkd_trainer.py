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

import random
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from ...models.utils import unwrap_model_for_generation
from ...trainer.sft_trainer import SFTTrainer
from ...trainer.utils import empty_cache
from ..utils import DataCollatorForChatML
from .gkd_trainer import GKDTrainer
from .privileged_gkd_config import PrivilegedGKDConfig


if is_peft_available():
    from peft import PeftConfig


@dataclass
class DataCollatorForPrivilegedGKD(DataCollatorForChatML):
    privileged_key: str = "privileged_messages"
    max_privileged_length: int = 512

    @staticmethod
    def _is_message_dict(obj: Any) -> bool:
        return isinstance(obj, dict) and "role" in obj and "content" in obj

    def _format_privileged_value(self, privileged_value: Any, sample_index: int) -> str:
        if isinstance(privileged_value, str):
            text = privileged_value
        elif isinstance(privileged_value, list):
            if len(privileged_value) == 0:
                raise ValueError(f"`{self.privileged_key}` 不能为空（index={sample_index}）。")

            if all(self._is_message_dict(item) for item in privileged_value):
                # 单组对话
                text = self.tokenizer.apply_chat_template(
                    privileged_value, add_generation_prompt=False, tokenize=False
                )
            else:
                raise ValueError(
                    f"`{self.privileged_key}` 必须是字符串或消息列表（index={sample_index}）。"
                )
        else:
            raise ValueError(
                f"`{self.privileged_key}` 必须是字符串或消息列表，当前类型为 "
                f"{type(privileged_value)}（index={sample_index}）。"
            )

        if not text.strip():
            raise ValueError(f"`{self.privileged_key}` 格式化后为空（index={sample_index}）。")
        return text

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = super().__call__(examples)

        privileged_texts = []
        for i, example in enumerate(examples):
            if self.privileged_key not in example:
                raise ValueError(f"样本缺少 `{self.privileged_key}` 字段（index={i}）。")
            privileged_texts.append(self._format_privileged_value(example[self.privileged_key], i))

        tokenized_pi = self.tokenizer(
            privileged_texts,
            truncation=True,
            max_length=self.max_privileged_length,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        batch["pi_input_ids"] = tokenized_pi["input_ids"]
        batch["pi_attention_mask"] = tokenized_pi["attention_mask"]
        return batch


class PrivilegedSelfDistillTrainer(GKDTrainer):
    """
    Teacher 端可见特权信息（PI），Student 端不可见的自蒸馏 Trainer。
    """

    _tag_names = ["trl", "gkd", "privileged-self-distill"]
    _name = "PrivilegedSelfDistill"

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        teacher_model: PreTrainedModel | nn.Module | str | None = None,
        args: PrivilegedGKDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: "PeftConfig | None" = None,
        formatting_func: Callable | None = None,
    ):
        if args is None:
            model_name = model.split("/")[-1] if isinstance(model, str) else "model"
            args = PrivilegedGKDConfig(output_dir=f"{model_name}-PrivilegedGKD")

        if args.use_liger_kernel:
            raise ValueError("PrivilegedSelfDistillTrainer 当前不支持 use_liger_kernel=True。")
        if args.share_student_as_teacher and isinstance(model, str):
            warnings.warn(
                "当前使用字符串 model_id 初始化，并启用了 share_student_as_teacher。"
                "为避免初始化双份模型权重，建议传入已实例化的 model 对象。",
                stacklevel=2,
            )

        if data_collator is None:
            if processing_class is None:
                raise ValueError("当 data_collator 未提供时，processing_class 不能为空。")
            data_collator = DataCollatorForPrivilegedGKD(
                tokenizer=processing_class,
                max_length=args.max_length,
                privileged_key=args.privileged_key,
                max_privileged_length=args.max_privileged_length,
            )

        if teacher_model is None:
            teacher_model = args.teacher_model_name_or_path or model
        if teacher_model is None:
            raise ValueError(
                "teacher_model 不能为空，请显式传入 teacher_model 或配置 teacher_model_name_or_path。"
            )

        super().__init__(
            model=model,
            teacher_model=teacher_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        self.privileged_key = args.privileged_key
        self.max_privileged_length = args.max_privileged_length
        self.share_student_as_teacher = args.share_student_as_teacher
        self.rollout_log_steps = args.rollout_log_steps
        self.rollout_log_samples = args.rollout_log_samples
        self.rollout_log_max_new_tokens = args.rollout_log_max_new_tokens
        self.rollout_log_max_chars = args.rollout_log_max_chars
        self.debug_log_loss_steps = args.debug_log_loss_steps
        self.debug_log_grad_norm = args.debug_log_grad_norm
        self._last_rollout_log_step = -1
        if self.share_student_as_teacher:
            old_teacher_model = self.teacher_model
            self.teacher_model = self.model
            if old_teacher_model is not self.model:
                del old_teacher_model
                empty_cache()

    @staticmethod
    def _pad_variable_sequences(
        sequences: list[torch.Tensor], padding_value: int, batch_first: bool = True
    ) -> torch.Tensor:
        return pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)

    def _build_teacher_inputs(
        self, inputs: dict[str, torch.Tensor | Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.processing_class.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id 不能为空。")

        required_keys = {
            "input_ids",
            "attention_mask",
            "prompts",
            "prompt_attention_mask",
            "pi_input_ids",
            "pi_attention_mask",
        }
        missing = [key for key in required_keys if key not in inputs]
        if missing:
            raise ValueError(f"batch 缺少必要字段: {missing}")

        prompt_ids = inputs["prompts"]
        prompt_attention_mask = inputs["prompt_attention_mask"]
        student_input_ids = inputs["input_ids"]
        completion_mask = self._get_completion_mask(inputs)
        pi_input_ids = inputs["pi_input_ids"]
        pi_attention_mask = inputs["pi_attention_mask"]

        teacher_input_id_list = []
        teacher_attention_mask_list = []
        teacher_label_list = []

        batch_size = prompt_ids.size(0)
        for i in range(batch_size):
            prompt_tokens = prompt_ids[i][prompt_attention_mask[i].bool()]
            completion_tokens = student_input_ids[i][completion_mask[i]]
            pi_tokens = pi_input_ids[i][pi_attention_mask[i].bool()]

            if completion_tokens.numel() == 0:
                raise ValueError(f"第 {i} 个样本没有可用于蒸馏的 completion token。")

            teacher_tokens = torch.cat([prompt_tokens, pi_tokens, completion_tokens], dim=0)
            teacher_labels = torch.cat(
                [
                    torch.full(
                        (prompt_tokens.numel() + pi_tokens.numel(),),
                        -100,
                        dtype=completion_tokens.dtype,
                        device=completion_tokens.device,
                    ),
                    completion_tokens,
                ],
                dim=0,
            )
            teacher_attention_mask = torch.ones_like(teacher_tokens)

            teacher_input_id_list.append(teacher_tokens)
            teacher_attention_mask_list.append(teacher_attention_mask)
            teacher_label_list.append(teacher_labels)

        teacher_input_ids = self._pad_variable_sequences(
            teacher_input_id_list, padding_value=self.processing_class.pad_token_id
        )
        teacher_attention_mask = self._pad_variable_sequences(teacher_attention_mask_list, padding_value=0)
        teacher_labels = self._pad_variable_sequences(teacher_label_list, padding_value=-100)
        return teacher_input_ids, teacher_attention_mask, teacher_labels

    @staticmethod
    def _get_completion_mask(inputs: dict[str, torch.Tensor | Any]) -> torch.Tensor:
        required_keys = {"labels", "attention_mask", "prompts"}
        missing = [key for key in required_keys if key not in inputs]
        if missing:
            raise ValueError(f"batch 缺少必要字段: {missing}")

        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"].bool()
        prompt_length = inputs["prompts"].size(1)

        completion_mask = labels != -100

        prompt_token_mask = attention_mask[:, :prompt_length]
        prompt_labeled_mask = labels[:, :prompt_length] != -100
        is_on_policy = (prompt_token_mask & prompt_labeled_mask).any(dim=1)
        if torch.any(is_on_policy):
            on_policy_completion_mask = attention_mask.clone()
            on_policy_completion_mask[:, :prompt_length] = False
            completion_mask[is_on_policy] = on_policy_completion_mask[is_on_policy]

        return completion_mask

    @staticmethod
    def _extract_completion_logits(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        batch_size, _, vocab_size = shift_logits.shape
        completion_logits = []
        completion_labels = []
        max_length = 0

        for i in range(batch_size):
            valid_mask = shift_labels[i] != -100
            logits_i = shift_logits[i][valid_mask]
            labels_i = shift_labels[i][valid_mask]
            completion_logits.append(logits_i)
            completion_labels.append(labels_i)
            max_length = max(max_length, logits_i.size(0))

        if max_length == 0:
            raise ValueError("当前 batch 中不存在可用于蒸馏的 completion token。")

        padded_logits = shift_logits.new_zeros((batch_size, max_length, vocab_size))
        padded_labels = shift_labels.new_full((batch_size, max_length), -100)
        for i in range(batch_size):
            length_i = completion_logits[i].size(0)
            if length_i == 0:
                continue
            padded_logits[i, :length_i] = completion_logits[i]
            padded_labels[i, :length_i] = completion_labels[i]

        return padded_logits, padded_labels

    @staticmethod
    def _extract_completion_token_ids(
        input_ids: torch.Tensor, completion_mask: torch.Tensor
    ) -> list[torch.Tensor]:
        completion_token_ids = []
        for i in range(input_ids.size(0)):
            completion_token_ids.append(input_ids[i][completion_mask[i]])
        return completion_token_ids

    def _decode_token_ids(self, token_ids: torch.Tensor) -> str:
        if token_ids.numel() == 0:
            return "<空>"
        token_ids = token_ids.detach().cpu()
        if hasattr(self.processing_class, "decode"):
            decoded = self.processing_class.decode(token_ids, skip_special_tokens=True).strip()
            if decoded:
                return decoded
            # 如果全部是特殊 token，回退展示特殊 token 文本，避免日志中一直出现 "<空>"
            decoded_with_special = self.processing_class.decode(token_ids, skip_special_tokens=False).strip()
            if decoded_with_special:
                return f"<仅特殊token>{decoded_with_special}"
        return str(token_ids.tolist())

    def _truncate_for_log(self, text: str) -> str:
        text = text.replace("\n", "\\n")
        if len(text) <= self.rollout_log_max_chars:
            return text
        return f"{text[: self.rollout_log_max_chars]}...<截断>"

    def _should_log_rollouts(self) -> bool:
        if self.rollout_log_steps <= 0:
            return False
        if not self.accelerator.is_main_process:
            return False
        current_step = self.state.global_step + 1
        if current_step % self.rollout_log_steps != 0:
            return False
        if self._last_rollout_log_step == current_step:
            return False
        self._last_rollout_log_step = current_step
        return True

    def _should_log_training_debug(self) -> bool:
        if self.debug_log_loss_steps <= 0:
            return False
        if not self.accelerator.is_main_process:
            return False
        current_step = self.state.global_step + 1
        return current_step % self.debug_log_loss_steps == 0

    @staticmethod
    def _compute_grad_norm_and_count(model: nn.Module) -> tuple[float, int]:
        total_norm_sq = 0.0
        params_with_grad = 0
        for param in model.parameters():
            if param.grad is None:
                continue
            grad_norm = float(param.grad.detach().float().norm(2).item())
            total_norm_sq += grad_norm * grad_norm
            params_with_grad += 1
        return total_norm_sq**0.5, params_with_grad

    def _build_teacher_prompt_batch_for_logging(
        self, inputs: dict[str, torch.Tensor | Any], sample_count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.processing_class.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id 不能为空。")

        prompt_ids = inputs["prompts"]
        prompt_attention_mask = inputs["prompt_attention_mask"]
        pi_input_ids = inputs["pi_input_ids"]
        pi_attention_mask = inputs["pi_attention_mask"]

        teacher_prompts = []
        teacher_prompt_masks = []
        for i in range(sample_count):
            prompt_tokens = prompt_ids[i][prompt_attention_mask[i].bool()]
            pi_tokens = pi_input_ids[i][pi_attention_mask[i].bool()]
            teacher_prompt_tokens = torch.cat([prompt_tokens, pi_tokens], dim=0)
            teacher_prompts.append(teacher_prompt_tokens)
            teacher_prompt_masks.append(torch.ones_like(teacher_prompt_tokens))

        padded_teacher_prompts = self._pad_variable_sequences(
            teacher_prompts, padding_value=self.processing_class.pad_token_id
        )
        padded_teacher_prompt_masks = self._pad_variable_sequences(teacher_prompt_masks, padding_value=0)
        return padded_teacher_prompts, padded_teacher_prompt_masks

    def _build_rollout_log_generation_config(self) -> GenerationConfig:
        generation_config = GenerationConfig.from_dict(self.generation_config.to_dict())
        generation_config.max_new_tokens = self.rollout_log_max_new_tokens
        generation_config.min_new_tokens = 1
        return generation_config

    def _generate_teacher_rollout_for_logging(
        self,
        teacher_model_for_generation: nn.Module,
        teacher_prompts: torch.Tensor,
        teacher_prompt_attention_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        generation_inputs = {
            "prompts": teacher_prompts,
            "prompt_attention_mask": teacher_prompt_attention_mask,
        }
        rollout_generation_config = self._build_rollout_log_generation_config()
        with (
            unwrap_model_for_generation(
                teacher_model_for_generation,
                self.accelerator,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_teacher
        ):
            generated_ids, _, _ = self.generate_on_policy_outputs(
                unwrapped_teacher,
                generation_inputs,
                rollout_generation_config,
                self.processing_class.pad_token_id,
            )

        teacher_rollouts = []
        for i in range(generated_ids.size(0)):
            prompt_len = int(teacher_prompt_attention_mask[i].sum().item())
            teacher_completion = generated_ids[i, prompt_len:]
            if self.processing_class.pad_token_id is not None:
                teacher_completion = teacher_completion[teacher_completion != self.processing_class.pad_token_id]
            teacher_rollouts.append(teacher_completion)
        return teacher_rollouts

    def _log_student_teacher_rollouts(
        self,
        inputs: dict[str, torch.Tensor | Any],
        student_rollout_source: str,
        teacher_model_for_generation: nn.Module,
    ) -> None:
        batch_size = int(inputs["input_ids"].size(0))
        sample_count = min(self.rollout_log_samples, batch_size)
        if sample_count <= 0:
            return

        completion_mask = self._get_completion_mask(inputs)
        student_completions = self._extract_completion_token_ids(inputs["input_ids"], completion_mask)
        teacher_prompts, teacher_prompt_attention_mask = self._build_teacher_prompt_batch_for_logging(
            inputs, sample_count
        )
        teacher_completions = self._generate_teacher_rollout_for_logging(
            teacher_model_for_generation,
            teacher_prompts=teacher_prompts,
            teacher_prompt_attention_mask=teacher_prompt_attention_mask,
        )

        for i in range(sample_count):
            prompt_text = self._decode_token_ids(inputs["prompts"][i][inputs["prompt_attention_mask"][i].bool()])
            pi_text = self._decode_token_ids(inputs["pi_input_ids"][i][inputs["pi_attention_mask"][i].bool()])
            student_text = self._decode_token_ids(student_completions[i])
            teacher_text = self._decode_token_ids(teacher_completions[i])
            self.accelerator.print(

                    f"[PrivilegedRollout][step={self.state.global_step + 1}][sample={i}] "
                    f"student_source={student_rollout_source}\n"
                    f"prompt: {self._truncate_for_log(prompt_text)}\n"
                    f"pi: {self._truncate_for_log(pi_text)}\n"
                    f"student_rollout: {self._truncate_for_log(student_text)}\n"
                    f"teacher_rollout: {self._truncate_for_log(teacher_text)}"

            )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "pi_input_ids" not in inputs or "pi_attention_mask" not in inputs:
            raise ValueError(
                "缺少 PI 字段。请使用 DataCollatorForPrivilegedGKD，或确保 batch 中包含 "
                "`pi_input_ids` 和 `pi_attention_mask`。"
            )

        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        teacher_input_ids, teacher_attention_mask, teacher_labels = self._build_teacher_inputs(inputs)
        teacher_model = model if self.share_student_as_teacher else self.teacher_model
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
            )

        completion_mask = self._get_completion_mask(inputs)
        student_labels = inputs["input_ids"].clone()
        student_labels[~completion_mask] = -100
        student_completion_logits, student_completion_labels = self._extract_completion_logits(
            student_outputs.logits, student_labels
        )
        teacher_completion_logits, teacher_completion_labels = self._extract_completion_logits(
            teacher_outputs.logits, teacher_labels
        )

        if not torch.equal(student_completion_labels, teacher_completion_labels):
            raise ValueError("student 与 teacher 的 completion 标签未对齐，无法计算蒸馏损失。")

        loss = self.generalized_jsd_loss(
            student_logits=student_completion_logits,
            teacher_logits=teacher_completion_logits,
            labels=student_completion_labels,
            beta=self.beta,
            temperature=self.temperature,
        )
        empty_cache()
        return (loss, student_outputs) if return_outputs else loss

    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        student_rollout_source = "off-policy"

        if self.seq_kd:
            with (
                unwrap_model_for_generation(
                    self.teacher_model,
                    self.accelerator,
                    generation_kwargs=self.generation_kwargs,
                ) as unwrapped_model
            ):
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
            student_rollout_source = "teacher-seq-kd"

        if random.random() <= self.lmbda:
            with (
                unwrap_model_for_generation(
                    model,
                    self.accelerator,
                    generation_kwargs=self.generation_kwargs,
                ) as unwrapped_model
            ):
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
            student_rollout_source = "student-on-policy"

        if self._should_log_rollouts():
            teacher_model_for_generation = model if self.share_student_as_teacher else self.teacher_model
            self._log_student_teacher_rollouts(
                inputs,
                student_rollout_source=student_rollout_source,
                teacher_model_for_generation=teacher_model_for_generation,
            )

        loss = SFTTrainer.training_step(self, model, inputs, num_items_in_batch)
        if self._should_log_training_debug():
            log_message = f"[PrivilegedTrainDebug][step={self.state.global_step + 1}] loss={loss.item():.6f}"
            if self.debug_log_grad_norm:
                grad_norm, params_with_grad = self._compute_grad_norm_and_count(model)
                log_message += f" grad_norm={grad_norm:.6f} params_with_grad={params_with_grad}"
            self.accelerator.print(log_message)
        return loss
