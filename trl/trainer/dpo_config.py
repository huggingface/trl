# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union

from transformers import TrainingArguments


class FDivergenceType(Enum):
    REVERSE_KL = "reverse_kl"
    JS_DIVERGENCE = "js_divergence"
    ALPHA_DIVERGENCE = "alpha_divergence"


class FDivergenceConstants:
    ALPHA_DIVERGENCE_COEF_KEY = "alpha_divergence_coef"
    ALPHA_DIVERGENCE_COEF_DEFAULT = 1.0


@dataclass
class DPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`DPOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the
            [`DPOTrainer`] is provided as a string.
        ref_model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `ref_model` argument of the
            [`DPOTrainer`] is provided as a string.
        model_adapter_name (`str` or `None`, *optional*, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`str` or `None`, *optional*, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
        force_use_ref_model (`bool`, *optional*, defaults to `False`):
            If you provide a PEFT model as the active model and wish to use a different model for the `ref_model`, set
            this flag to `True`.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model and reference model.
        use_logits_to_keep (`bool`, *optional*, defaults to `False`):
            If `True`, only a specified number of logits are computed in the forward pass. This can be useful for
            saving memory and speeding up training by not computing the logits for all tokens, especially in
            scenarios when working with very long prompts where labels are ignored (-100).

        > Parameters that control the data preprocessing

        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        padding_value (`int` or `None`, *optional*, defaults to `None`):
            Padding value to use. If `None`, the padding value of the tokenizer is used.
        label_pad_token_id (`int`, *optional*, defaults to `-100`):
            Padding value to use for labels.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt.
        max_completion_length (`int` or `None`, *optional*, defaults to `None`):
            Maximum length of the completion.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the full sequence (prompt + completion).
        truncation_mode (`str`, *optional*, defaults to `"keep_end"`):
            Truncation mode to use when the sequence exceeds `max_length`. Possible values are `"keep_end"` and
            `"keep_start"`.
        padding_free (`bool`, *optional*, defaults to `False`):
            Whether to perform forward passes without padding by flattening all sequences in the batch into a single
            continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only
            supported with the `flash_attention_2` attention implementation, which can efficiently handle the flattened
            batch structure.
        precompute_ref_log_probs (`bool`, *optional*, defaults to `False`):
            Whether to precompute the log probabilities from the reference model. Setting this to `True` allows
            training without needing the reference model during training, which can help reduce GPU memory usage. If
            set to `False` (default), the reference model will be used during training to compute log probabilities
            on-the-fly.
        precompute_ref_batch_size (`int` or `None`, *optional*, defaults to `None`):
            Batch size to use when precomputing reference model log probabilities. This can be set higher than the
            training batch size to speed up preprocessing. If `None`, defaults to `per_device_train_batch_size` for
            training and `per_device_eval_batch_size` for evaluation.
        tools (`Optional[list[Union[dict, Callable]]]`, *optional*, defaults to `None`):
            List of tools (callable functions) that will be accessible to the model.
            If the template does not support function calling, this argument will have no effect.

        > Parameters that control the training

        learning_rate (`float`, *optional*, defaults to `1e-6`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
        loss_type (`str`, *optional*, defaults to `"sigmoid"`):
            Type of loss to use. Possible values are:

                - `"sigmoid"`: sigmoid loss from the original [DPO](https://huggingface.co/papers/2305.18290) paper.
                - `"hinge"`: hinge loss on the normalized likelihood from the [SLiC](https://huggingface.co/papers/2305.10425) paper.
                - `"ipo"`: IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.
                - `"exo_pair"`: pairwise EXO loss from the [EXO](https://huggingface.co/papers/2402.00856) paper.
                - `"nca_pair"`: pairwise NCA loss from the [NCA](https://huggingface.co/papers/2402.05369) paper.
                - `"robust"`: unbiased estimate of the DPO loss that is robust to preference noise from the [Robust DPO](https://huggingface.co/papers/2403.00409) paper.
                - `"bco_pair"`: pairwise BCO loss from the [BCO](https://huggingface.co/papers/2404.04656) paper.
                - `"sppo_hard"`: SPPO loss with hard label from the [SPPO](https://huggingface.co/papers/2405.00675) paper.
                - `"aot"`: AOT loss for paired datasets from the [AOT](https://huggingface.co/papers/2406.05882) paper.
                - `"aot_pair"`: AOT loss for unpaired datasets from the [AOT](https://huggingface.co/papers/2406.05882) paper.
                - `"discopop"`: DiscoPOP (a.k.a Log-Ratio Modulated Loss, LRML) loss from the [DiscoPOP](https://huggingface.co/papers/2406.08414) paper.
                - `"apo_zero"`: APO-zero loss from the [APO](https://huggingface.co/papers/2408.06266) paper.
                - `"apo_down"`: APO-down loss from the [APO](https://huggingface.co/papers/2408.06266) paper.

        beta (`float`, *optional*, defaults to `0.1`):
            Parameter controlling the deviation from the reference model. Higher β means less deviation from the
            reference model. For the IPO loss (`loss_type="ipo"`), β is the regularization parameter denoted by τ in
            the [paper](https://huggingface.co/papers/2310.12036).
        f_divergence_type (`str`, *optional*, defaults to `FDivergenceType.REVERSE_KL`):
            Type of f-divergence regularization function to compute divergence between policy and reference model.
        f_alpha_divergence_coef (`float`, *optional*, defaults to `1.0`):
            α coefficient in the α-divergence u^-α regularization function for DPO loss.
        reference_free (`bool`, *optional*, defaults to `False`):
            Whether to ignore the provided reference model and implicitly use a reference model that assigns equal
            probability to all responses.
        label_smoothing (`float`, *optional*, defaults to `0.0`):
            Robust DPO label smoothing parameter from the [cDPO](https://ericmitchell.ai/cdpo.pdf) report and
            [Robust DPO](https://huggingface.co/papers/2403.00409) paper that should be between `0.0` and `0.5`.
        use_weighting (`bool`, *optional*, defaults to `False`):
            Whether to weight the loss as done in the [WPO](https://huggingface.co/papers/2406.11827) paper.
        rpo_alpha (`float`, *optional*, defaults to `None`):
            α parameter from the [RPO](https://huggingface.co/papers/2404.19733) paper (v3), which controls the
            weighting of the NLL term in the loss. If `None`, no weighting is applied and the loss is the same as the
            DPO loss. The paper recommends `rpo_alpha=1.0`.
        discopop_tau (`float`, *optional*, defaults to `0.05`):
            τ/temperature parameter from the [DiscoPOP](https://huggingface.co/papers/2406.08414) paper, which controls
            the shape of log ratio modulated loss. The paper recommends the default value `discopop_tau=0.05`.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
            the `ref_model_mixup_alpha` parameter. This synchronization originites from the
            [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.6`):
            α parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which controls the mix
            between the current policy and the previous reference policy during updates. The reference policy is
            updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you
            must set `sync_ref_model=True`.
        ref_model_sync_steps (`int`, *optional*, defaults to `512`):
            τ parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which determines how
            frequently the current policy is synchronized with the reference policy. To use this parameter, you must
            set `sync_ref_model=True`.

        > Parameters that control the logging

        generate_during_eval (`bool`, *optional*, defaults to `False`):
            Whether to generate and log completions from both the model and the reference model to W&B or Comet during
            evaluation.
    """

    # Parameters that control the model and reference model
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `DPOTrainer` is provided as a string."
        },
    )
    ref_model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `ref_model` argument "
            "of the `DPOTrainer` is provided as a string."
        },
    )
    model_adapter_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the train target PEFT adapter, when using LoRA with multiple adapters."},
    )
    ref_adapter_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the reference PEFT adapter, when using LoRA with multiple adapters."},
    )
    force_use_ref_model: bool = field(
        default=False,
        metadata={
            "help": "If you provide a PEFT model as the active model and wish to use a different model for the "
            "`ref_model`, set this flag to `True`."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model and reference model."},
    )
    use_logits_to_keep: bool = field(
        default=False,
        metadata={
            "help": "If `True`, only a specified number of logits are computed in the forward pass. This can be "
            "useful for saving memory and speeding up training by not computing the logits for all tokens, especially "
            "in scenarios when working with very long prompts where labels are ignored (-100)."
        },
    )

    # Parameters that control the data preprocessing
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    padding_value: Optional[int] = field(
        default=None,
        metadata={"help": "Padding value to use. If `None`, the padding value of the tokenizer is used."},
    )
    label_pad_token_id: int = field(
        default=-100,
        metadata={"help": "Padding value to use for labels."},
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum length of the prompt."},
    )
    max_completion_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum length of the completion."},
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum length of the full sequence (prompt + completion)."},
    )
    truncation_mode: str = field(
        default="keep_end",
        metadata={
            "help": "Truncation mode to use when the sequence exceeds `max_length`. Possible values are `'keep_end'` "
            "and `'keep_start'`.",
            "choices": ["keep_end", "keep_start"],
        },
    )
    padding_free: bool = field(
        default=False,
        metadata={
            "help": "Whether to perform forward passes without padding by flattening all sequences in the batch into "
            "a single continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, "
            "this is only supported with the `flash_attention_2` attention implementation, which can efficiently "
            "handle the flattened batch structure."
        },
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={
            "help": "Whether to precompute the log probabilities from the reference model. Setting this to `True` "
            "allows training without needing the reference model during training, which can help reduce GPU memory "
            "usage. If set to `False` (default), the reference model will be used during training to compute log "
            "probabilities on-the-fly."
        },
    )
    precompute_ref_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size to use when precomputing reference model log probabilities. This can be set higher "
            "than the training batch size to speed up preprocessing. If `None`, defaults to "
            "`per_device_train_batch_size` for training and `per_device_eval_batch_size` for evaluation."
        },
    )
    tools: Optional[list[Union[dict, Callable]]] = field(
        default=None,
        metadata={
            "help": "List of tools (callable functions) that will be accessible to the model. If the template does "
            "not support function calling, this argument will have no effect."
        },
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    loss_type: str = field(
        default="sigmoid",
        metadata={
            "help": "Type of loss to use.",
            "choices": [
                "sigmoid",
                "hinge",
                "ipo",
                "exo_pair",
                "nca_pair",
                "robust",
                "bco_pair",
                "sppo_hard",
                "aot",
                "aot_pair",
                "discopop",
                "apo_zero",
                "apo_down",
            ],
        },
    )
    beta: float = field(
        default=0.1,
        metadata={
            "help": "Parameter controlling the deviation from the reference model. "
            "Higher β means less deviation from the reference model."
        },
    )
    f_divergence_type: FDivergenceType = field(
        default=FDivergenceType.REVERSE_KL,
        metadata={
            "help": "Type of f-divergence regularization function to compute divergence between policy and reference "
            "model."
        },
    )
    f_alpha_divergence_coef: float = field(
        default=1.0,
        metadata={"help": "α coefficient in the α-divergence u^-α regularization function for DPO loss."},
    )
    reference_free: bool = field(
        default=False,
        metadata={
            "help": "Whether to ignore the provided reference model and implicitly use a reference model that assigns "
            "equal probability to all responses."
        },
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={
            "help": "Robust DPO label smoothing parameter from the cDPO report and Robust DPO paper that should "
            "be between `0.0` and `0.5`."
        },
    )
    use_weighting: bool = field(
        default=False,
        metadata={"help": "Whether to weight the loss as done in the WPO paper."},
    )
    rpo_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": "α parameter from the RPO paper (v3), which controls the weighting of the NLL term in the loss. "
            "If `None`, no weighting is applied and the loss is the same as the DPO loss. The paper recommends "
            "`rpo_alpha=1.0`."
        },
    )
    discopop_tau: float = field(
        default=0.05,
        metadata={
            "help": "τ/temperature parameter from the DiscoPOP paper, which controls the shape of log ratio modulated "
            "loss. The paper recommends the default value `discopop_tau=0.05`."
        },
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int = field(
        default=512,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )

    # Parameters that control the logging
    generate_during_eval: bool = field(
        default=False,
        metadata={
            "help": "Whether to generate and log completions from both the model and the reference model to W&B or "
            "Comet during evaluation."
        },
    )

    # Deprecated parameters
    use_num_logits_to_keep: Optional[bool] = field(
        default=None,
        metadata={"help": "Deprecated. Use `use_logits_to_keep` instead."},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.use_num_logits_to_keep is not None:
            warnings.warn(
                "`use_num_logits_to_keep` is deprecated and will be remove in version 0.17.0. Use "
                "`use_logits_to_keep` instead.",
                DeprecationWarning,
            )
            self.use_logits_to_keep = self.use_num_logits_to_keep
