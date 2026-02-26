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

from dataclasses import dataclass, field
from typing import Any

from transformers import TrainingArguments

from .base_config import BaseConfig


@dataclass
class DPOConfig(BaseConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`DPOTrainer`].

    This class includes only the parameters that are specific to DPO training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`DPOTrainer`] is provided as a string.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model and reference model.

        > Parameters that control the data preprocessing

        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        pad_token (`str`, *optional*):
            Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that is also `None`,
            it falls back to `processing_class.eos_token`.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the left or
            right depending on the `truncation_mode`. If `None`, no truncation is applied.
        truncation_mode (`str`, *optional*, defaults to `"keep_start"`):
            Truncation mode to use when the sequence exceeds `max_length`. Possible values are `"keep_end"` and
            `"keep_start"`.
        padding_free (`bool`, *optional*, defaults to `False`):
            Whether to perform forward passes without padding by flattening all sequences in the batch into a single
            continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only
            supported with the FlashAttention 2 or 3, which can efficiently handle the flattened batch structure.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        precompute_ref_log_probs (`bool`, *optional*, defaults to `False`):
            Whether to precompute the reference model log probabilities for the entire training dataset before
            training. This allows to save memory during training, as the reference model does not need to be kept in
            memory.
        precompute_ref_batch_size (`int`, *optional*):
            Batch size to use when precomputing reference model log probabilities. This can be set higher than the
            training batch size to speed up preprocessing. If `None`, defaults to `per_device_train_batch_size` for
            training and `per_device_eval_batch_size` for evaluation.

        > Parameters that control the training

        loss_type (`list[str]`, *optional*, defaults to `["sigmoid"]`):
            Type of loss to use. Possible values are: `'sigmoid'`, `'hinge'`, `'ipo'`, `'exo_pair'`, `'nca_pair'`,
            `'robust'`, `'bco_pair'`, `'sppo_hard'`, `'aot'`, `'aot_unpaired'`, `'apo_zero'`, `'apo_down'`,
            `'discopop'`, `'sft'`. If multiple loss types are provided, they will be combined using the weights
            specified in `loss_weights`.
        loss_weights (`list[float]`, *optional*):
            List of loss weights for multi-loss combinations. Used when combining multiple loss types. Example: `[0.8,
            0.2, 1.0]` for MPO. If not provided, defaults to equal weights (`1.0`) for all loss types.
        ld_alpha (`float`, *optional*):
            α parameter from the LD-DPO paper, which controls the weighting of the verbose token log-probabilities in
            responses. If `None`, no weighting is applied to the verbose part, and the loss is equivalent to the
            standard DPO loss. Must be in [0.0, 1.0]: `ld_alpha=1.0` applies no weighting, and `ld_alpha=0.0` masks
            tokens beyond shared lengths.
        f_divergence_type (`str`, *optional*, defaults to `"reverse_kl"`):
            f-divergence regularizer between policy and reference (f-DPO paper). Possible values are: `reverse_kl`
            (default), `forward_kl`, `js_divergence`, `alpha_divergence`.
        f_alpha_divergence_coef (`float`, *optional*, defaults to `0.5`):
            α coefficient for the α-divergence u^-α regularizer, used only when `f_divergence_type='alpha_divergence'`.
        label_smoothing (`float`, *optional*, defaults to `0.0`):
            Label smoothing parameter used in Robust DPO and EXO. In Robust DPO, it is interpreted as the probability
            that a preference label is flipped and must lie in [0.0, 0.5); a typical value recommended by the Robust
            DPO paper is 0.1. In EXO, it corresponds to the ε label smoothing parameter, for which the paper recommends
            a typical value of 1e-3.
        beta (`float`, *optional*, defaults to `0.1`):
            Parameter controlling the deviation from the reference model. Higher β means less deviation from the
            reference model. For the IPO loss (`loss_type='ipo'`), this value is the regularization parameter denoted
            by τ in the [paper](https://huggingface.co/papers/2310.12036).
        use_weighting (`bool`, *optional*, defaults to `False`):
            Whether to apply WPO-style weighting (https://huggingface.co/papers/2406.11827) to preference pairs using
            the policy's length-normalized sequence probabilities.
        discopop_tau (`float`, *optional*, defaults to `0.05`):
            τ/temperature parameter from the DiscoPOP paper, which controls the shape of the log-ratio modulated loss
            when using `loss_type='discopop'`. The paper recommends the default value `discopop_tau=0.05`.
        activation_offloading (`bool`, *optional*, defaults to `False`):
            Whether to offload the activations to the CPU.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
            the `ref_model_mixup_alpha` parameter. This synchronization originates from the
            [TR-DPO](https://huggingface.co/papers/2404.09656) paper. `sync_ref_model=True` is not yet compatible with
            PEFT or `precompute_ref_log_probs=True`.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.6`):
            α parameter from the TR-DPO paper, which controls the mix between the current policy and the previous
            reference policy during updates. The reference policy is updated according to the equation: `π_ref = α *
            π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`.
        ref_model_sync_steps (`int`, *optional*, defaults to `512`):
            τ parameter from the TR-DPO paper, which determines how frequently the current policy is synchronized with
            the reference policy. To use this parameter, you must set `sync_ref_model=True`.

    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `10` instead of `500`.
    > - `gradient_checkpointing`: Defaults to `True` instead of `False`.
    > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
    > - `learning_rate`: Defaults to `1e-6` instead of `5e-5`.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )

    # Parameters that control the model
    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `DPOTrainer` is provided as a string."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model and reference model."},
    )

    # Parameters that control the data preprocessing
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    pad_token: str | None = field(
        default=None,
        metadata={
            "help": "Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that "
            "is also `None`, it falls back to `processing_class.eos_token`."
        },
    )
    max_length: int | None = field(
        default=1024,
        metadata={
            "help": "Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from "
            "the left or right depending on the `truncation_mode`. If `None`, no truncation is applied."
        },
    )
    truncation_mode: str = field(
        default="keep_start",
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
            "a single continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this "
            "is only supported with the FlashAttention 2 or 3, which can efficiently handle the flattened batch "
            "structure."
        },
    )
    pad_to_multiple_of: int | None = field(
        default=None,
        metadata={"help": "If set, the sequences will be padded to a multiple of this value."},
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={
            "help": "Whether to precompute the reference model log probabilities for the entire training dataset "
            "before training. This allows to save memory during training, as the reference model does not need to be "
            "kept in memory."
        },
    )
    precompute_ref_batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Batch size to use when precomputing reference model log probabilities. This can be set higher "
            "than the training batch size to speed up preprocessing. If `None`, defaults to "
            "`per_device_train_batch_size` for training and `per_device_eval_batch_size` for evaluation."
        },
    )

    # Parameters that control the training
    loss_type: list[str] = field(
        default_factory=lambda: ["sigmoid"],
        metadata={
            "help": "Type of loss to use. Possible values are: `'sigmoid'`, `'hinge'`, `'ipo'`, `'exo_pair'`, "
            "`'nca_pair'`, `'robust'`, `'bco_pair'`, `'sppo_hard'`, `'aot'`, `'aot_unpaired'`, `'apo_zero'`, "
            "`'apo_down'`, `'discopop'`, `'sft'`. If multiple loss types are provided, they will be combined using "
            "the weights specified in `loss_weights`.",
        },
    )
    loss_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "List of loss weights for multi-loss combinations. Used when combining multiple loss types. "
            "Example: `[0.8, 0.2, 1.0]` for MPO. If not provided, defaults to equal weights (`1.0`) for all loss "
            "types."
        },
    )
    ld_alpha: float | None = field(
        default=None,
        metadata={
            "help": "α parameter from the LD-DPO paper, which controls the weighting of the verbose token "
            "log-probabilities in responses. If `None`, no weighting is applied to the verbose part, and the loss is "
            "equivalent to the standard DPO loss. Must be in [0.0, 1.0]: `ld_alpha=1.0` applies no weighting, and "
            "`ld_alpha=0.0` masks tokens beyond shared lengths.",
        },
    )
    f_divergence_type: str = field(
        default="reverse_kl",
        metadata={
            "help": "f-divergence regularizer between policy and reference (f-DPO paper). Possible values are: "
            "`reverse_kl` (default), `forward_kl`, `js_divergence`, `alpha_divergence`.",
        },
    )
    f_alpha_divergence_coef: float = field(
        default=0.5,
        metadata={
            "help": "α coefficient for the α-divergence u^-α regularizer, used only when "
            "`f_divergence_type='alpha_divergence'`."
        },
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={
            "help": "Label smoothing parameter used in Robust DPO and EXO. In Robust DPO, it is interpreted as the "
            "probability that a preference label is flipped and must lie in [0.0, 0.5); a typical value recommended "
            "by the Robust DPO paper is 0.1. In EXO, it corresponds to the ε label smoothing parameter, for which the "
            "paper recommends a typical value of 1e-3."
        },
    )
    beta: float = field(
        default=0.1,
        metadata={
            "help": "Parameter controlling the deviation from the reference model. Higher β means less deviation from "
            "the reference model. For the IPO loss (`loss_type='ipo'`), this value is the regularization parameter "
            "denoted by τ in the [paper](https://huggingface.co/papers/2310.12036)."
        },
    )
    use_weighting: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply WPO-style weighting (https://huggingface.co/papers/2406.11827) to preference "
            "pairs using the policy's length-normalized sequence probabilities."
        },
    )
    discopop_tau: float = field(
        default=0.05,
        metadata={
            "help": "τ/temperature parameter from the DiscoPOP paper, which controls the shape of the log-ratio "
            "modulated loss when using `loss_type='discopop'`. The paper recommends the default value "
            "`discopop_tau=0.05`."
        },
    )
    activation_offloading: bool = field(
        default=False,
        metadata={"help": "Whether to offload the activations to the CPU."},
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter. This synchronization originates from the "
            "[TR-DPO](https://huggingface.co/papers/2404.09656) paper. `sync_ref_model=True` is not yet compatible "
            "with PEFT or `precompute_ref_log_probs=True`."
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

    def __post_init__(self):
        if isinstance(self.loss_type, str):
            self.loss_type = [self.loss_type]
        if self.loss_weights is not None and len(self.loss_weights) != len(self.loss_type):
            raise ValueError(
                "`loss_weights` must have the same length as `loss_type` when combining multiple losses. "
                f"Got {len(self.loss_weights)} weights for {len(self.loss_type)} loss types."
            )

        super().__post_init__()
