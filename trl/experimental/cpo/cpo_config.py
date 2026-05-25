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

import warnings
from dataclasses import dataclass, field
from typing import Any

from ...trainer.base_config import _BaseConfig


@dataclass
class CPOConfig(_BaseConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`experimental.cpo.CPOTrainer`].

    This class includes only the parameters that are specific to CPO training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`CPOTrainer`] is provided as a string.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.

        > Parameters that control the data preprocessing

        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the left or
            right depending on the `truncation_mode`. If `None`, no truncation is applied.
        truncation_mode (`str`, *optional*, defaults to `"keep_start"`):
            Truncation mode to use when the sequence exceeds `max_length`. The only supported value is
            `"keep_start"`. The `"keep_end"` value is deprecated and will be removed in v2.0.0.
        padding_free (`bool`, *optional*, defaults to `False`):
            Whether to perform forward passes without padding by flattening all sequences in the batch into a single
            continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only
            supported with the FlashAttention 2 or 3, which can efficiently handle the flattened batch structure.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.

        > Parameters that control the training

        loss_type (`list[str]`, *optional*, defaults to `["sigmoid"]`):
            Type of loss to use. Possible values are: `'sigmoid'`, `'hinge'`, `'ipo'`. If multiple loss types are
            provided, they will be combined using the weights specified in `loss_weights`.
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
        beta (`float`, *optional*, defaults to `0.1`):
            Parameter controlling the deviation from the reference model. Higher β means less deviation from the
            reference model. For the IPO loss (`loss_type='ipo'`), this value is the regularization parameter denoted
            by τ in the [paper](https://huggingface.co/papers/2310.12036).
        use_weighting (`bool`, *optional*, defaults to `False`):
            Whether to apply WPO-style weighting (https://huggingface.co/papers/2406.11827) to preference pairs using
            the policy's length-normalized sequence probabilities.
        activation_offloading (`bool`, *optional*, defaults to `False`):
            Whether to offload the activations to the CPU.
        cpo_alpha (`float`, *optional*, defaults to `1.0`):
            α parameter from the CPO paper, which controls the weight of the NLL term on chosen completions added to
            the preference loss. Setting `cpo_alpha=0` recovers the pure preference loss.

    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `10` instead of `500`.
    > - `gradient_checkpointing`: Defaults to `True` instead of `False`.
    > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
    > - `learning_rate`: Defaults to `1e-6` instead of `5e-5`.
    """

    _VALID_DICT_FIELDS = _BaseConfig._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )

    # Parameters that control the model
    model_init_kwargs: dict[str, Any] | str | None = field(
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
            "help": "Truncation mode to use when the sequence exceeds `max_length`. The only supported value is "
            "`'keep_start'`. The `'keep_end'` value is deprecated and will be removed in v2.0.0.",
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

    # Parameters that control the training
    loss_type: list[str] = field(
        default_factory=lambda: ["sigmoid"],
        metadata={
            "help": "Type of loss to use. Possible values are: `'sigmoid'`, `'hinge'`, `'ipo'`. If multiple loss types "
            "are provided, they will be combined using the weights specified in `loss_weights`.",
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
    activation_offloading: bool = field(
        default=False,
        metadata={"help": "Whether to offload the activations to the CPU."},
    )
    cpo_alpha: float = field(
        default=1.0,
        metadata={
            "help": "α parameter from the CPO paper, which controls the weight of the NLL term on chosen completions "
            "added to the preference loss. Setting `cpo_alpha=0` recovers the pure preference loss."
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
        if self.truncation_mode == "keep_end":
            warnings.warn(
                "The `'keep_end'` truncation mode is deprecated and will be removed in v2.0.0. "
                "Use `truncation_mode='keep_start'` (the default) instead.",
                FutureWarning,
                stacklevel=3,
            )

        super().__post_init__()
