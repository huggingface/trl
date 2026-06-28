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

from .base_config import _BaseConfig


@dataclass
class SFTConfig(_BaseConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`SFTTrainer`].

    This class includes only the parameters that are specific to SFT training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for the model's `from_pretrained` method, used when the `model` argument of the
            [`SFTTrainer`] is provided as a string. Causal language models are loaded with
            [`~transformers.AutoModelForCausalLM`]; encoder-decoder models are loaded with
            [`~transformers.AutoModelForSeq2SeqLM`].
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to
            the model's `from_pretrained` method and [`~transformers.AutoProcessor.from_pretrained`].
        router_aux_loss_coef (`float`, *optional*, defaults to `0.001`):
            Coefficient of the load-balancing auxiliary loss. Only has an effect when training a Mixture-of-Experts
            (MoE) model; for other models it does nothing. The auxiliary loss is added to the training loss with this
            weight. Set to `0.0` to disable it.
        chat_template_path (`str`, *optional*):
            If specified, sets the model's chat template. This can either be the path to a tokenizer (local directory
            or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, you must
            ensure that any special tokens referenced in the template are added to the tokenizer and that the model's
            embedding layer is resized accordingly.

        > Parameters that control the data preprocessing

        dataset_text_field (`str`, *optional*, defaults to `"text"`):
            Name of the column that contains text data in the dataset.
        dataset_kwargs (`dict[str, Any]`, *optional*):
            Dictionary of optional keyword arguments for the dataset preparation. The only supported key is
            `skip_prepare_dataset`. When the model is a VLM, `skip_prepare_dataset` is automatically treated as `True`
            regardless of the provided value, since preprocessing is done on the fly.
        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        eos_token (`str`, *optional*):
            Token used to indicate the end of a turn or sequence. If `None`, it defaults to
            `processing_class.eos_token`.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the left
            or right depending on `truncation_mode`. If `None`, no truncation is applied. When packing is enabled,
            this value sets the sequence length. For seq2seq models, this is the encoder sequence length.
        max_target_length (`int` or `None`, *optional*):
            Maximum decoder target length for seq2seq models. Defaults to `max_length` when unset. This setting has no
            effect for causal language models.
        truncation_mode (`str`, *optional*, defaults to `"keep_start"`):
            Truncation mode to use when the sequence exceeds `max_length`. The only supported value is
            `"keep_start"`. The `"keep_end"` value is deprecated and will be removed in v2.0.0.
        shuffle_dataset (`bool`, *optional*, defaults to `False`):
            Whether to shuffle the dataset.
        packing (`bool`, *optional*, defaults to `False`):
            Whether to group multiple sequences into fixed-length blocks to improve computational efficiency and reduce
            padding. Uses `max_length` to define sequence length. T5-family seq2seq models use `max_length` for the
            encoder stream and `max_target_length` for the decoder stream.
        packing_strategy (`str`, *optional*, defaults to `"bfd"`):
            Strategy for packing sequences. Can be `"bfd"` (best-fit decreasing, truncates overflow), `"bfd_split"`
            (best-fit decreasing, splits overflow sequences), or `"wrapped"` (aggressive, cuts mid-sequence).
        padding_free (`bool`, *optional*, defaults to `False`):
            Whether to perform forward passes without padding by flattening all sequences in the batch into a single
            continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only
            supported with the FlashAttention 2 or 3, which can efficiently handle the flattened batch structure. When
            packing is enabled with strategy `"bfd"`, padding-free is enabled, regardless of the value of this
            parameter. Only supported for causal language models.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        eval_packing (`bool`, *optional*):
            Whether to pack the eval dataset. If `None`, uses the same value as `packing` for causal models and
            defaults to `False` for seq2seq models. Packed seq2seq evaluation is not supported.

        > Parameters that control the training

        completion_only_loss (`bool`, *optional*):
            Whether to compute loss only on the completion part of the sequence. If set to `True`, loss is computed
            only on the completion, which is supported only for [prompt-completion](#prompt-completion) datasets. If
            `False`, loss is computed on the entire sequence. If `None` (default), the behavior depends on the dataset:
            loss is computed on the completion for [prompt-completion](#prompt-completion) datasets, and on the full
            sequence for [language modeling](#language-modeling) datasets. For encoder-decoder models, prompt tokens
            are used as encoder inputs and completion tokens are used as decoder labels, so the loss is computed on the
            completion side.
        assistant_only_loss (`bool`, *optional*, defaults to `False`):
            Whether to compute loss only on the assistant part of the sequence. If set to `True`, loss is computed only
            on the assistant responses, which is supported only for [conversational](#conversational) datasets. If
            `False`, loss is computed on the entire sequence.
        loss_type (`str`, *optional*, defaults to `"chunked_nll"`):
            Type of loss to use. When left unset, it defaults to `"chunked_nll"`, except when `use_liger_kernel=True`,
            in which case it defaults to `"nll"`. T5-family seq2seq models support `"chunked_nll"`; other
            encoder-decoder models use `"nll"`. Possible values are:

            - `"nll"`: standard negative log-likelihood.
            - `"dft"`: Dynamic Fine-Tuning, as described in
              [this paper](https://huggingface.co/papers/2508.05629).
            - `"chunked_nll"`: same math as `"nll"`, but the `lm_head` projection is computed on non-ignored tokens
              only (positions with `labels == -100` are dropped before the matmul) and the cross-entropy is processed
              in chunks of tokens to reduce peak activation memory. Supported for causal and T5-family models and not
              compatible with `use_liger_kernel`.

        activation_offloading (`bool`, *optional*, defaults to `False`):
            Whether to offload the activations to the CPU.

        > Deprecated parameters

        pad_token:

            <Deprecated version="1.1.0">

            Parameter `pad_token` is deprecated and will be removed in version v2.0.0. Set `tokenizer.pad_token`
            directly and pass it as `processing_class` to the trainer instead.

            </Deprecated>

    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `10` instead of `500`.
    > - `gradient_checkpointing`: Defaults to `True` instead of `False`.
    > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
    > - `learning_rate`: Defaults to `2e-5` instead of `5e-5`.
    """

    _VALID_DICT_FIELDS = _BaseConfig._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW."},
    )

    # Parameters that control the model
    model_init_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for the model's `from_pretrained` method, used when the `model` argument of "
            "the `SFTTrainer` is provided as a string."
        },
    )
    router_aux_loss_coef: float = field(
        default=0.001,
        metadata={
            "help": "Coefficient of the load-balancing auxiliary loss. Only has an effect when training a "
            "Mixture-of-Experts (MoE) model; for other models it does nothing. The auxiliary loss is added to the "
            "training loss with this weight. Set to `0.0` to disable it."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow loading models and tokenizers that ship custom Python code from the Hub. "
            "Forwarded to the model's `from_pretrained` method and `AutoProcessor.from_pretrained`."
        },
    )
    chat_template_path: str | None = field(
        default=None,
        metadata={
            "help": "If specified, sets the model's chat template. This can either be the path to a tokenizer (local "
            "directory or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, "
            "you must ensure that any special tokens referenced in the template are added to the tokenizer and "
            "that the model's embedding layer is resized accordingly."
        },
    )

    # Parameters that control the data preprocessing
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "Name of the column that contains text data in the dataset."},
    )
    dataset_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Dictionary of optional keyword arguments for the dataset preparation. The only supported key is "
            "`skip_prepare_dataset`. If the model is a VLM, `skip_prepare_dataset` value is ignored. When the model "
            "is a VLM, `skip_prepare_dataset` is automatically treated as `True` regardless of the provided value, "
            "since preprocessing is done on the fly."
        },
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    eos_token: str | None = field(
        default=None,
        metadata={
            "help": "Token used to indicate the end of a turn or sequence. If `None`, it defaults to `processing_class.eos_token`."
        },
    )
    max_length: int | None = field(
        default=1024,
        metadata={
            "help": "Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from "
            "the left or right depending on the `truncation_mode`. If `None`, no truncation is applied. When packing "
            "is enabled, this value sets the sequence length."
        },
    )
    max_target_length: int | None = field(
        default=None,
        metadata={"help": "Maximum decoder target length for seq2seq models. Defaults to `max_length` when unset."},
    )
    truncation_mode: str = field(
        default="keep_start",
        metadata={
            "help": "Truncation mode to use when the sequence exceeds `max_length`. The only supported value is "
            "`'keep_start'`. The `'keep_end'` value is deprecated and will be removed in v2.0.0.",
            "choices": ["keep_end", "keep_start"],
        },
    )
    shuffle_dataset: bool = field(
        default=False,
        metadata={"help": "Whether to shuffle the dataset."},
    )
    packing: bool = field(
        default=False,
        metadata={
            "help": "Whether to group multiple sequences into fixed-length blocks to improve computational efficiency "
            "and reduce padding. Uses `max_length` to define sequence length. T5-family seq2seq models use "
            "`max_length` for the encoder stream and `max_target_length` for the decoder stream."
        },
    )
    packing_strategy: str = field(
        default="bfd",
        metadata={
            "help": "Strategy for packing sequences. Can be `'bfd'` (best-fit decreasing, truncates overflow), "
            "`'bfd_split'` (best-fit decreasing, splits overflow sequences), or `'wrapped'` (aggressive, cuts "
            "mid-sequence).",
            "choices": ["bfd", "bfd_split", "wrapped"],
        },
    )
    padding_free: bool = field(
        default=False,
        metadata={
            "help": "Whether to perform forward passes without padding by flattening all sequences in the batch into "
            "a single continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this "
            "is only supported with the FlashAttention 2 or 3, which can efficiently handle the flattened batch "
            "structure. When packing is enabled with strategy `'bfd'`, padding-free is enabled, regardless of the "
            "value of this parameter. Only supported for causal language models."
        },
    )
    pad_to_multiple_of: int | None = field(
        default=None,
        metadata={"help": "If set, the sequences will be padded to a multiple of this value."},
    )
    eval_packing: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to pack the eval dataset. If `None`, uses the same value as `packing` for causal "
            "models and defaults to `False` for seq2seq models. Packed seq2seq evaluation is not supported."
        },
    )

    # Parameters that control the training
    completion_only_loss: bool | None = field(
        default=None,
        metadata={
            "help": (
                "Whether to compute loss only on the completion part of the sequence. If set to `True`, loss is "
                "computed only on the completion, which is supported only for prompt-completion datasets. If `False`, "
                "loss is computed on the entire sequence. If `None` (default), the behavior depends on the dataset: "
                "loss is computed on the completion for prompt-completion datasets, and on the full sequence for "
                "language modeling datasets. For encoder-decoder models, prompt tokens are used as encoder inputs "
                "and completion tokens are used as decoder labels."
            )
        },
    )
    assistant_only_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to compute loss only on the assistant part of the sequence. If set to `True`, loss is "
                "computed only on the assistant responses, which is supported only for conversational datasets. If `False`, "
                "loss is computed on the entire sequence."
            )
        },
    )
    loss_type: str | None = field(
        default=None,
        metadata={
            "help": "Type of loss to use. When left unset, it defaults to `'chunked_nll'`, except when "
            "`use_liger_kernel=True`, in which case it defaults to `'nll'`. T5-family seq2seq models support "
            "`'chunked_nll'`; other encoder-decoder models use `'nll'`. "
            "Possible values are `'nll'` (standard negative log-likelihood), `'dft'` (Dynamic Fine-Tuning, "
            "https://huggingface.co/papers/2508.05629), and `'chunked_nll'` (same math as `'nll'`, but the "
            "`lm_head` projection is computed on non-ignored tokens only — positions with `labels == -100` are "
            "dropped before the matmul — and the cross-entropy is processed in chunks of tokens to reduce peak "
            "activation memory; not compatible with `use_liger_kernel`; the patched `lm_head` path covers standard "
            "causal LMs, T5-family models, and VLMs whose language model exposes a top-level `lm_head`; "
            "architectures with a non-standard head are not supported)."
        },
    )
    activation_offloading: bool = field(
        default=False,
        metadata={"help": "Whether to offload the activations to the CPU."},
    )

    # Deprecated parameters
    pad_token: str | None = field(
        default=None,
        metadata={
            "help": "Deprecated. Set `tokenizer.pad_token` directly and pass it as `processing_class` to the trainer instead."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if self.pad_token is not None:
            warnings.warn(
                "`pad_token` is deprecated and will be removed in v2.0.0. "
                "Set `tokenizer.pad_token` directly and pass it as `processing_class` to the trainer instead.",
                FutureWarning,
                stacklevel=3,
            )
        if self.truncation_mode == "keep_end":
            warnings.warn(
                "The `'keep_end'` truncation mode is deprecated and will be removed in v2.0.0. "
                "Use `truncation_mode='keep_start'` (the default) instead.",
                FutureWarning,
                stacklevel=3,
            )
        if self.packing_strategy == "bfd-requeue":
            warnings.warn(
                "The `bfd-requeue` packing strategy has been renamed to `bfd_split`. Please update your configuration accordingly. "
                "The `bfd-requeue` strategy is deprecated and will be removed in v2.0.0.",
                FutureWarning,
                stacklevel=3,
            )
            self.packing_strategy = "bfd_split"

        # When unset, default to "chunked_nll" unless `use_liger_kernel=True`, in which case default to "nll".
        if self.loss_type is None:
            self.loss_type = "nll" if self.use_liger_kernel else "chunked_nll"
