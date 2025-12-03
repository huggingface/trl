# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import textwrap

import os
import time
import torch
import torch.nn as nn
from accelerate import PartialState, logging
import torch.nn.functional as F
from collections.abc import Callable
from datasets import Dataset, IterableDataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    BaseImageProcessor,
    FeatureExtractionMixin,
)
from transformers.trainer_callback import TrainerCallback
from transformers.utils import is_peft_available

from ...models import prepare_deepspeed, clone_chat_template
from ...trainer.grpo_trainer import GRPOTrainer, RewardFunc, RolloutFunc
from ...trainer.utils import (
    disable_dropout_in_model, 
    empty_cache, 
    get_config_model_id,
    remove_none_values
)
from .minillm_config import MiniLLMConfig
from ...data_utils import (
    is_conversational,
    is_conversational_from_value,
    maybe_convert_to_chatml,
)


if is_peft_available():
    from peft import PeftConfig

logger = logging.get_logger(__name__)

def dummy_reward_func(completions: list, **kwargs):
    # placeholder reward function when no reward function is provided
    return [1.0 for _ in completions]


def get_dataset_column_names(dataset: Dataset | IterableDataset) -> list[str]:
    return list(next(iter(dataset)).keys()) if dataset.column_names is None else dataset.column_names


class MiniLLMTrainer(GRPOTrainer):
    """
    Trainer for the Knowledge Distillation of Language Models (MiniLLM) method. This algorithm was initially proposed
    in the paper [Knowledge Distillation of Large Language Models](https://huggingface.co/papers/2306.08543).

    Example:

    ```python
    from datasets import load_dataset
    from trl.experimental.minillm import MiniLLMTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = MiniLLMTrainer(
        model="Qwen/Qwen3-0.6B",
        teacher_model="Qwen/Qwen3-1.7B",
        train_dataset=dataset,
    )
    trainer.train()
    ```

    Args:
        model (`str | PreTrainedModel`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        teacher_model (`PreTrainedModel | nn.Module | str`):
            Teacher model used for knowledge distillation. Instantiated similarly to `model`.
        reward_funcs (`RewardFunc | list[RewardFunc]`, *optional*):
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
                  functions can also return `None` when the reward is not applicable to those samples. This is useful
                  for multi-task training where different reward functions apply to different types of samples. When a
                  reward function returns `None` for a sample, that reward function is excluded from the reward
                  calculation for that sample. For more details, see [Using a custom reward
                  function](#using-a-custom-reward-function).

                  The trainer's state is also passed to the reward function. The trainer's state is an instance of
                  [`~transformers.TrainerState`] and can be accessed by accessing the `trainer_state` argument to the
                  reward function's signature.
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`experimental.minillm.MiniLLMConfig`], *optional*):
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
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        rollout_func (`RolloutFunc`, *optional*):
            Function to use for generating completions. It must take prompts, args, and processing_class as parameters
            and return a dict with `"prompt_ids"`, `"completion_ids"`, and `"logprobs"` fields. Any other fields that
            are forwarded to the reward functions. This feature is experimental and may change or be removed at any
            time without prior notice.
    """

    _tag_names = ["trl", "minillm"]
    _name = "MiniLLM"
    _paper = {
        "title": "MiniLLM: Knowledge Distillation of Large Language Models",
        "id": "2306.08543",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{
                gu2024minillm,
                title={{MiniLLM: Knowledge Distillation of Large Language Models}},
                author={Yuxian Gu and Li Dong and Furu Wei and Minlie Huang},
                booktitle={The Twelfth International Conference on Learning Representations},
                year={2024},
                url={https://openreview.net/forum?id=5h0qf7IBZZ}
            }"""),
    }

    def __init__(
        self,
        model: str | PreTrainedModel,
        teacher_model: PreTrainedModel | nn.Module | str,
        reward_funcs: RewardFunc | list[RewardFunc] | None = None,
        args: MiniLLMConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
        rollout_func: RolloutFunc | None = None,
        formatting_func: Callable[[dict], str] | None = None,
    ):
        if reward_funcs is None:
            reward_funcs = [dummy_reward_func]

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = MiniLLMConfig(f"{model_name}-MiniLLM")
        
        # We need old logprobs to compute the RKL
        args.always_track_old_logps = True

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(get_config_model_id(model.config))

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
            self._is_vlm = True
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
            self._is_vlm = False
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if args.eos_token is not None:
            eos_token = args.eos_token
            eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
            if eos_token_id is None:
                raise ValueError(
                    f"The specified `eos_token` ('{eos_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `eos_token` exists "
                    "in the vocabulary before using it as an EOS token."
                )
            tokenizer.eos_token_id = eos_token_id

        if args.chat_template_path is not None:
            if os.path.isfile(args.chat_template_path) and args.chat_template_path.endswith((".jinja", ".j2")):
                with open(args.chat_template_path, encoding="utf-8") as chat_template_file:
                    processing_class.chat_template = chat_template_file.read()
            else:
                model, processing_class, _ = clone_chat_template(
                    model, processing_class, args.chat_template_path
                )

        skip_prepare_dataset = (
            args.dataset_kwargs is not None
            and args.dataset_kwargs.get("skip_prepare_dataset", False)
        )
        if not skip_prepare_dataset:
            train_dataset = self._prepare_dataset(
                train_dataset, processing_class, args, formatting_func, "train"
            )
            if eval_dataset is not None:
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        key: self._prepare_dataset(dataset, processing_class, args, formatting_func, key)
                        for key, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_dataset(
                        eval_dataset, processing_class, args, formatting_func, "eval"
                    )

        super().__init__(
            model,
            reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            rollout_func=rollout_func,
        )

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the MiniLLMConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["dtype"] = (
                teacher_model_init_kwargs["dtype"]
                if teacher_model_init_kwargs["dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["dtype"])
            )

        if isinstance(teacher_model, str):
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.temperature = args.temperature
        self.kd_temperature = args.kd_temperature
        self.single_step_decomposition = args.single_step_decomposition
        self.rkl_advantage = args.rkl_advantage
        self.gamma = args.gamma
        self.length_normalization = args.length_normalization

        self._current_rkl_advantange_time = 0.0
        self._current_single_step_loss_time = 0.0
        self._current_rl_loss_time = 0.0

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin,
        args: MiniLLMConfig,
        formatting_func: Callable[[dict], str] | None,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        if isinstance(dataset, Dataset):  # IterableDataset does not support `with_transform`
            dataset = dataset.with_transform(remove_none_values)

        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = get_dataset_column_names(dataset)
        is_processed = "prompt" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None and is_processed:
                logger.warning(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                )

            if formatting_func is not None and not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"prompt": formatting_func(example)}

                dataset = dataset.map(_func, batched=False, **map_kwargs)

            if not is_processed:
                first_example = next(iter(dataset))
                if is_conversational_from_value(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
                    column_names = get_dataset_column_names(dataset)
                    dataset = dataset.map(
                        maybe_convert_to_chatml,
                        remove_columns="conversations" if "conversations" in column_names else None,
                        **map_kwargs,
                    )

                first_example = next(iter(dataset))
                if is_conversational(first_example):
                    def convert_conversation_to_prompt(
                            example: dict[str, list[dict[str, str]]],
                        ) -> dict[str, str]:
                        last_role = example["messages"][-1]["role"]
                        if last_role == "assistant":
                            messages = example["messages"][:-1]
                            completion = example["messages"][-1]["content"]
                        else:
                            messages = example["messages"]
                            completion = ""
                        prompt = processing_class.apply_chat_template(
                            messages,
                            continue_final_message=False,
                            tokenize=False,
                            add_generation_prompt=True,
                            **example.get("chat_template_kwargs", {}),
                        )

                        output = {
                            "prompt": prompt,
                            "completion": completion,
                        }
                        return output

                    dataset = dataset.map(
                        convert_conversation_to_prompt,
                        remove_columns=["messages"],
                        **map_kwargs,
                    )
            
            if PartialState().is_main_process:
                # print data example
                first_example = dataset[0]
                print("\n\n########## First example in dataset: ##########")
                print("Inputs:", first_example["prompt"])
                print("#" * 50 + "\n\n")
        
        return dataset


    def _single_step_decomposition_loss(
        self,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str = "batchmean",
    ):
        """
        Compute the MiniLLM loss for knowledge distillation using F.kl_div. See Eq. (1) of
        https://huggingface.co/papers/2306.08543 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature:
                Softmax temperature (default: 1.0)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """
        reg_loss = F.kl_div(
            teacher_log_probs, student_log_probs, reduction="none", log_target=True
        )  # (batch_size, sequence_length)

        # Masking
        if mask is not None:
            reg_loss = reg_loss[mask]

        # Apply reduction
        if reduction == "batchmean":
            return reg_loss.sum() / mask.sum() if mask is not None else reg_loss.sum() / reg_loss.size(0)
        elif reduction == "sum":
            return reg_loss.sum()
        elif reduction == "mean":
            return reg_loss.mean()
        else:
            return reg_loss

    @torch.no_grad()
    def _compute_advantage(
        self,
        student_log_probs_on_labels: torch.Tensor,
        teacher_log_probs_on_labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Compute the advantage for Reverse KL Divergence.

        Mostly following [this
        implementation](https://github.com/microsoft/LMOps/blob/e210d2c026b9958617887762400778ace81172e6/minillm/minillm/losses.py#L37-L49).

        $$ \text{rewards}_t = \text{teacher\_log\_probs\_on\_labels}_t - \text{student\_log\_probs\_on\_labels}_t $$

        If length normalization is enabled:

        $$ \text{lengths}_t = \sum_{i=t}^{T} \gamma^{i-t} $$

        $$ \text{advantages}_t = \frac{\sum_{i=t}^{T} \gamma^{i-t} \text{rewards}_i}{\text{lengths}_t} $$

        Otherwise:

        $$ \text{advantages}_t = \sum_{i=t}^{T} \gamma^{i-t} \text{rewards}_i $$

        Args:
            student_log_probs_on_labels: Log probabilities of the student model on the labels.
                Shape: (batch_size, sequence_length)
            teacher_log_probs_on_labels: Log probabilities of the teacher model on the labels.
                Shape: (batch_size, sequence_length)
            mask: Optional mask to apply to the log probabilities. Shape: (batch_size, sequence_length)
        Returns:
            advantage: Computed advantage. Shape: (batch_size, sequence_length)
        """
        response_length = student_log_probs_on_labels.size(1)
        if mask is None:
            mask = torch.ones_like(student_log_probs_on_labels)
        mask = mask.float()
        student_log_probs_on_labels = student_log_probs_on_labels * mask
        teacher_log_probs_on_labels = teacher_log_probs_on_labels * mask

        rewards = teacher_log_probs_on_labels - student_log_probs_on_labels  # (batch_size, sequence_length)

        if self.gamma > 0.0:
            gamma_pow = torch.pow(self.gamma, torch.arange(response_length, device=rewards.device))

            rewards = rewards * gamma_pow
            rewards = rewards.flip(1).cumsum(dim=1).flip(1)

            if self.length_normalization:
                mask = torch.where(mask < 0.5, 1e-4, mask)
                lengths = mask * gamma_pow
                lengths = lengths.flip(1).cumsum(dim=1).flip(1)
                rewards = rewards / lengths

        # if self.args.minillm_grpo:
        #     mean_grouped_rewards = rewards.view(-1, self.num_generations, response_length).mean(dim=1)
        #     mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        #     advantages = rewards - mean_grouped_rewards
        #     if self.args.minillm_scale_rewards in ["group", "none"]:
        #         # If self.scale_rewards = "none", we'll still log group level std
        #         std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        #         std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        #     elif self.args.minillm_scale_rewards == "batch":
        #         # Compute global std
        #         std_rewards = rewards.std().expand_as(rewards)
        #     else:
        #         raise ValueError(
        #             f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
        #         )
        #     raise NotImplementedError("MiniLLM GRPO is not implemented yet.")
        # else:
        advantages = rewards

        return advantages

    def get_rev_kl(self, log_p: torch.Tensor, log_q: torch.Tensor, mask: torch.Tensor):
        log_ratio = (log_p - log_q) * mask
        kl = log_ratio.float().exp() - 1 - log_ratio
        return kl

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Compute student output
        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Compute teacher output in eval mode
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Slice the logits for the generated tokens using the inputs["prompts"] lengths
        prompt_lengths = inputs["prompt_ids"].shape[1]
        student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]
        teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]
        shifted_labels = input_ids[:, prompt_lengths:]

        # Apply temperature scaling
        student_logits = student_logits / self.kd_temperature
        teacher_logits = teacher_logits / self.kd_temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if self.args.on_policy_logq:
            student_log_probs_on_labels = torch.gather(
                student_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)
            ).squeeze(-1)
        else:
            student_log_probs_on_labels = inputs["old_per_token_logps"]

        teacher_log_probs_on_labels = torch.gather(
            teacher_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)
        ).squeeze(-1)

        mask = shifted_labels != -100

        # Logging Meitrcs
        mode = "train" if self.model.training else "eval"

        # Compute advantage for Reverse KL
        if self.rkl_advantage:
            rkl_advantage_time = time.perf_counter()
            reverse_kl_advantage = self._compute_advantage(
                student_log_probs_on_labels=student_log_probs_on_labels,
                teacher_log_probs_on_labels=teacher_log_probs_on_labels,
                mask=mask,
            )

            inputs["advantages"] = inputs["advantages"].unsqueeze(1) + reverse_kl_advantage
            mean_advantage = (inputs["advantages"] * mask).sum() / mask.sum()
            self._metrics[mode]["minillm/mean_advantage"].append(mean_advantage.item())
            rkl_advantage_time_after = time.perf_counter()
            self._current_rkl_advantange_time += rkl_advantage_time_after - rkl_advantage_time
            if self._step % self.current_gradient_accumulation_steps == 0:
                self._metrics[mode]["rkl_advantage_time"].append(self._current_rkl_advantange_time)
                self._current_rkl_advantange_time = 0.0

        # Compute GRPO loss on verifiable reward
        rl_loss_time = time.perf_counter()
        loss = self._compute_loss(model, inputs)
        rl_loss_time_after = time.perf_counter()
        self._current_rl_loss_time += rl_loss_time_after - rl_loss_time
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics[mode]["rl_loss_time"].append(self._current_rl_loss_time)
            self._current_rl_loss_time = 0.0

        # Compute loss
        if self.single_step_decomposition:
            single_step_loss_time = time.perf_counter()
            single_step_decomposition_loss = self._single_step_decomposition_loss(
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                mask=mask,
            )

            self._metrics[mode]["minillm/single_step_decomposition_loss"].append(single_step_decomposition_loss.item())
            loss += single_step_decomposition_loss
            single_step_loss_time_after = time.perf_counter()
            self._current_single_step_loss_time += single_step_loss_time_after - single_step_loss_time
            if self._step % self.current_gradient_accumulation_steps == 0:
                self._metrics[mode]["single_step_loss_time"].append(self._current_single_step_loss_time)
                self._current_single_step_loss_time = 0.0

        # Compute Reverse KL for logging
        with torch.no_grad():
            reverse_kl = self.get_rev_kl(
                log_p=teacher_log_probs_on_labels,
                log_q=inputs["old_per_token_logps"],
                mask=mask,
            )
            reverse_kl = reverse_kl.sum() / mask.sum()
            self._metrics[mode]["minillm/reverse_kl"].append(reverse_kl.item())

        # Empty cache
        empty_cache()

        # Return loss
        return (loss, student_outputs) if return_outputs else loss
