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

import textwrap
from collections.abc import Callable
from typing import Any

import torch
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.utils import is_peft_available

from ...data_utils import is_conversational
from ...rewards import accuracy_reward
from ...trainer.utils import get_config_model_id, pad
from ..self_distillation.base_self_distillation_trainer import (
    BaseSelfDistillationTrainer,
    RolloutBatch,
    TrainingBatch,
)
from .sdzero_config import SDZeroConfig


if is_peft_available():
    from peft import PeftConfig


REPHRASE_PROMPT = "Let me rephrase the above solution."
RESTART_PROMPT = "Wait, this response is not correct, let me start over."


class SDZeroTrainer(BaseSelfDistillationTrainer):
    """
    On-policy self-distillation via revision feedback. See
    [Self-Distillation Zero](https://huggingface.co/papers/2604.12002).

    At each step, the student generates a response, a binary verifier judges it, and a control prompt is
    selected accordingly. The teacher provides a next-token distribution over the student's response,
    conditioned on the response and the control prompt. The student is updated via KL divergence to match
    the teacher's distribution.

    The dataset must contain two columns: `prompt` (the problem as a conversational list or plain string) and
    `answer` (the gold answer used by the binary verifier).

    Example:

    ```python
    from datasets import Dataset
    from trl.experimental.sdzero import SDZeroConfig, SDZeroTrainer

    dataset = Dataset.from_list([
        {"prompt": [{"role": "user", "content": "What is 2+2?"}], "answer": "4"},
    ])
    trainer = SDZeroTrainer(
        model="model-id-or-path",
        args=SDZeroConfig(output_dir="sdzero-model", max_steps=100),
        train_dataset=dataset,
    )
    trainer.train()
    ```

    Args:
        model (`str` or [`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
            Model to be trained. Can be a model id string, a local directory path, or a pre-instantiated
            model object.
        args ([`SDZeroConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`]):
            Training dataset. Must contain columns `prompt` and `answer`.
        eval_dataset ([`~datasets.Dataset`] or `dict[str, Dataset]`, *optional*):
            Evaluation dataset. Must meet the same column requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`], *optional*):
            Tokenizer or processor. If `None`, loaded from the model.
        reward_fn (`Callable`, *optional*):
            Binary reward function with signature `(completions, solution) -> list[float | None]`, where
            `completions` is a list of `[{"role": "assistant", "content": ...}]` lists and `solution` is a
            list of gold answer strings. Return values of `1.0` are treated as correct, anything else as
            incorrect. Defaults to [`~trl.rewards.accuracy_reward`], which parses `\\boxed{}` LaTeX format.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            Callbacks to customize the training loop.
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            Optimizer and scheduler.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "sdzero", "sd-zero"]
    _name = "SDZero"
    config_cls = SDZeroConfig
    # docstyle-ignore
    _paper = {
        "title": "Self-Distillation Zero: Self-Revision Turns Binary Rewards into Dense Supervision",
        "id": "2604.12002",
        "citation": textwrap.dedent("""\
            @article{sdzero2026,
                title        = {{Self-Distillation Zero: Self-Revision Turns Binary Rewards into Dense Supervision}},
                year         = 2026,
                eprint       = {arXiv:2604.12002}
            }"""),
    }

    def __init__(
        self,
        model: str | PreTrainedModel | nn.Module,
        args: SDZeroConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_fn: Callable | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: PeftConfig | None = None,
    ):
        if isinstance(train_dataset, IterableDataset):
            raise NotImplementedError("Iterable datasets are not yet supported in SDZeroTrainer.")
        if isinstance(eval_dataset, IterableDataset) or (
            isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
        ):
            raise NotImplementedError("Iterable eval datasets are not yet supported in SDZeroTrainer.")
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = SDZeroConfig(f"{model_name}-SDZero")

        self.reward_fn = reward_fn if reward_fn is not None else accuracy_reward

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "answer"]

    def finalize_batch(
        self,
        inputs: list[dict[str, Any]],
        rollout_batch: RolloutBatch,
    ) -> TrainingBatch:
        r"""
        Build the teacher context for the shared student rollout and assemble the training batch.

        For each example, the student's rollout `y_init` is scored by `reward_fn`. A control prompt
        `p_r` is selected from the outcome: the rephrase nudge when `y_init` verifies
        as correct, the restart nudge otherwise. The teacher input is then assembled as

        ```
        teacher_input_ids = T(x, y, p_r) ++ completion_ids
        ```

        with optional chat-template applied.
        """
        tokenizer = (
            self.processing_class.tokenizer
            if isinstance(self.processing_class, ProcessorMixin)
            else self.processing_class
        )

        # Decode student completions `y_init`
        completions = [
            tokenizer.decode(ids[mask.bool()], skip_special_tokens=True)
            for ids, mask in zip(rollout_batch.completion_ids, rollout_batch.completion_mask, strict=False)
        ]

        answers = [inp["answer"] for inp in inputs]
        chat_completions = [[{"role": "assistant", "content": c}] for c in completions]
        rewards = [r if r is not None else 0.0 for r in self.reward_fn(chat_completions, solution=answers)]
        control_prompts = [REPHRASE_PROMPT if r == 1.0 else RESTART_PROMPT for r in rewards]

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["sdzero/reward"].append(sum(rewards) / max(len(rewards), 1))

        # Build the teacher prompt
        prompts, _ = self._split_prompt_and_privileged_context(inputs)
        teacher_prompt_ids_list = []
        for prompt, y, control_prompt in zip(prompts, completions, control_prompts, strict=False):
            assistant_turn_prefix = self.args.assistant_turn_template.format(
                y=y,
                control_prompt=control_prompt,
            )
            if is_conversational({"prompt": prompt}):
                teacher_prompt_ids = tokenizer.apply_chat_template(
                    prompt + [{"role": "assistant", "content": assistant_turn_prefix}],
                    tokenize=True,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    **self.chat_template_kwargs,
                )
            else:
                teacher_prompt_ids = tokenizer(prompt + assistant_turn_prefix)["input_ids"]
            if self.max_prompt_length is not None:
                teacher_prompt_ids = teacher_prompt_ids[-self.max_prompt_length :]
            teacher_prompt_ids_list.append(teacher_prompt_ids)

        device = rollout_batch.completion_ids.device
        teacher_prompt_ids = [torch.tensor(ids) for ids in teacher_prompt_ids_list]
        teacher_prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in teacher_prompt_ids]
        teacher_prompt_ids = pad(teacher_prompt_ids, padding_value=self.pad_token_id, padding_side="left").to(
            device=device
        )
        teacher_prompt_mask = pad(teacher_prompt_mask, padding_value=0, padding_side="left").to(device=device)

        teacher_input_ids = torch.cat([teacher_prompt_ids, rollout_batch.completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, rollout_batch.completion_mask], dim=1)

        batch = rollout_batch.as_dict()
        batch["teacher_input_ids"] = teacher_input_ids
        batch["teacher_attention_mask"] = teacher_attention_mask
        return batch

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("SDZeroTrainer does not support returning outputs")

        distillation_logits = self._compute_teacher_student_logits(model, self.teacher_model, inputs)
        loss = self._compute_self_distillation_loss(model, inputs, distillation_logits)
        accumulation_scale = self.current_gradient_accumulation_steps if self.model.training else 1.0
        return loss / accumulation_scale
