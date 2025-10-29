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

from dataclasses import dataclass, field
from typing import Any, Optional

from transformers import TrainingArguments


@dataclass
class GOLDConfig(TrainingArguments):
    r"""
    Configuration class for the [`GOLDTrainer`].

    This class includes only the parameters that are specific to GOLD training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`SFTTrainer`] is provided as a string. If you're training a MoE architecture and want to
            include the load balancing/auxilliary loss as a part of the final loss, remember to set
            `output_router_logits=True` in this dictionary.
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
        pad_token (`str`, *optional*):
            Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that is also `None`,
            it falls back to `processing_class.eos_token`.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the right.
            If `None`, no truncation is applied. When packing is enabled, this value sets the sequence length.
        packing (`bool`, *optional*, defaults to `False`):
            Whether to group multiple sequences into fixed-length blocks to improve computational efficiency and reduce
            padding. Uses `max_length` to define sequence length.
        packing_strategy (`str`, *optional*, defaults to `"bfd"`):
            Strategy for packing sequences. Can be either `"bfd"` (best-fit decreasing, default), or `"wrapped"`.
        padding_free (`bool`, *optional*, defaults to `False`):
            Whether to perform forward passes without padding by flattening all sequences in the batch into a single
            continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only
            supported with the FlashAttention 2 or 3, which can efficiently handle the flattened batch structure. When
            packing is enabled with strategy `"bfd"`, padding-free is enabled, regardless of the value of this
            parameter.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        eval_packing (`bool`, *optional*):
            Whether to pack the eval dataset. If `None`, uses the same value as `packing`.

        > Parameters that control generation

        generation_batch_size: (`int`, *optional*):
            Batch size to use for generation. If `None`, it defaults to the effective training batch size:
            `per_device_train_batch_size * num_processes * steps_per_generation`. In other words, there is one
            generation batch processed per optimization step. Mutually exclusive with `steps_per_generation`.
        steps_per_generation: (`int`, *optional*):
            Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`. Mutually exclusive
            with `generation_batch_size`.
        temperature (`float`, defaults to `1.0`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int`, *optional*):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, top-k-filtering is
            disabled and all tokens are considered.
        min_p (`float`, *optional*):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat
            tokens.
        use_transformers_paged (`bool`, *optional*, defaults to `False`):
            Whether to use the `transformers` paged implementation for generation. If set to `True`, the `transformers`
            paged implementation will be used for generation instead of the default padded implementation. This
            parameter is only effective when `use_vllm` is set to `False`.
        cache_implementation (`str`, *optional*):
            Implementation of the cache method for faster generation when `use_vllm` is set to `False`.
        generation_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to [`~transformers.GenerationConfig`] (if using transformers) or
            `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the
            generation behavior, such as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that conflict
            with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them.

            
        > Parameters that control the training

        completion_only_loss (`bool`, *optional*):
            Whether to compute loss only on the completion part of the sequence. If set to `True`, loss is computed
            only on the completion, which is supported only for [prompt-completion](#prompt-completion) datasets. If
            `False`, loss is computed on the entire sequence. If `None` (default), the behavior depends on the dataset:
            loss is computed on the completion for [prompt-completion](#prompt-completion) datasets, and on the full
            sequence for [language modeling](#language-modeling) datasets.
        assistant_only_loss (`bool`, *optional*, defaults to `False`):
            Whether to compute loss only on the assistant part of the sequence. If set to `True`, loss is computed only
            on the assistant responses, which is supported only for [conversational](#conversational) datasets. If
            `False`, loss is computed on the entire sequence.
        loss_type (`str`, *optional*, defaults to `"nll"`):
            Type of loss to use. Possible values are `"nll"` (negative log-likelihood, default) and `"dft"` (Dynamic
            Fine-Tuning, as described in [this paper](https://huggingface.co/papers/2508.05629)).
        activation_offloading (`bool`, *optional*, defaults to `False`):
            Whether to offload the activations to the CPU.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs", "teacher_model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    bf16: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if "
            "`fp16` is not set."
        },
    )

    # Parameters that control the model
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `SFTTrainer` is provided as a string. If you're training a MoE architecture and want to include the "
            "load balancing/auxilliary loss as a part of the final loss, remember to set `output_router_logits=True` "
            "in this dictionary."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropouts in `model`."},
    )

    # Parameters that control the data preprocessing
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    eos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Token used to indicate the end of a turn or sequence. If `None`, it defaults to `processing_class.eos_token`."
        },
    )
    pad_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that "
            "is also `None`, it falls back to `processing_class.eos_token`."
        },
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from"
            "the right. If `None`, no truncation is applied. When packing is enabled, this value sets the "
            "sequence length."
        },
    )
    pad_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={"help": "If set, the sequences will be padded to a multiple of this value."},
    )

    # Parameters that control generation
    lmbda: float = field(
        default=0.5,
        metadata={
            "help": "Lambda parameter that controls the student data fraction (i.e., the proportion of on-policy "
            "student-generated outputs)."
        },
    )
    max_completion_length: int = field(
        default=128,
        metadata={"help": "Maximum number of tokens to generate per completion."},
    )
    temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    top_p: float = field(
        default=0.95,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled and all tokens are considered."
        },
    )


    # Parameters that control the training
    beta: float = field(
        default=0.5,
        metadata={
            "help": "Interpolation coefficient between `0.0` and `1.0` of the Generalized Jensen-Shannon Divergence "
            "loss. When beta is `0.0`, the loss is the KL divergence. When beta is `1.0`, the loss is the Inverse KL "
            "Divergence."
        },
    )
    activation_offloading: bool = field(
        default=False,
        metadata={"help": "Whether to offload the activations to the CPU."},
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )
    num_completions_to_print: Optional[int] = field(
        default=None,
        metadata={"help": "Number of completions to print with `rich`. If `None`, all completions are logged."},
    )
    wandb_log_unique_prompts: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to log unique prompts in wandb. If `True`, only unique prompts are logged. If `False`, "
            "all prompts are logged."
        },
    )



    # --------------------------------
    
    steps_per_generation: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of optimization steps per generation. If `None`, it defaults to gradient_accumulation_steps."
        },
    )

    # ULD Loss parameters
    use_uld_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Universal Logit Distillation (ULD) loss instead of Generalized Jensen-Shannon Divergence loss."
        },
    )
    use_extended_uld: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to enable extended ULD alignment that uses tokenizers to align and merge token "
                "probabilities across student and teacher tokenizations. When True, the trainer will compute "
                "token mappings and merge probabilities for split tokens; when False, ULD will use simple "
                "positional truncation like in the original ULD paper."
            )
        },
    )
    uld_use_hybrid_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use a hybrid loss that combines ULD loss and JSD loss. When True, the final loss is a "
                "a combination of JSD for known token mappings and ULD for unknown token mappings."
            )
        },
    )
    uld_hybrid_matched_weight: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Weight for the matched token loss component when using hybrid ULD + JSD loss. This weight scales "
                "the JSD loss computed over tokens that have a direct mapping between student and teacher "
                "tokenizations. If None, uses adaptive weighting based on vocabulary overlap. Must be set together "
                "with uld_hybrid_unmatched_weight (both None or both float)."
            )
        },
    )
    uld_hybrid_unmatched_weight: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Weight for the unmatched token loss component when using hybrid ULD + JSD loss. This weight scales "
                "the ULD loss computed over tokens that do not have a direct mapping between student and teacher "
                "tokenizations. If None, uses adaptive weighting based on vocabulary overlap. Must be set together "
                "with uld_hybrid_matched_weight (both None or both float)."
            )
        },
    )
    uld_crossentropy_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for the cross-entropy loss component in ULD loss."},
    )
    uld_distillation_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the distillation loss component in ULD loss."},
    )
    uld_student_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for student logits in ULD loss computation."},
    )
    uld_teacher_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for teacher logits in ULD loss computation."},
    )

    uld_skip_student_eos: bool = field(
        default=True,
        metadata={"help": "Whether to skip EOS token for student in ULD loss computation."},
    )
    uld_skip_teacher_eos: bool = field(
        default=True,
        metadata={"help": "Whether to skip EOS token for teacher in ULD loss computation."},
    )

    # transformers paged attention
    use_transformers_paged: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the `transformers` paged implementation for generation. If set to `True`, the "
            "`transformers` paged implementation will be used for generation instead of the default padded "
            "implementation."
        },
    )

    # vLLM parameters
    use_vllm: bool = field(
        default=False,
        metadata={"help": "Whether to use vLLM for generating completions. Requires `vllm` to be installed."},
    )
    vllm_mode: str = field(
        default="server",
        metadata={
            "help": 'Mode for vLLM integration. Either "server" (connect to a running TRL vLLM server) or "colocate" (run vLLM in the same process).'
        },
    )
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": 'Host of the vLLM server when `vllm_mode="server"`.'},
    )
    vllm_server_port: int = field(
        default=8001,
        metadata={"help": 'Port of the vLLM server when `vllm_mode="server"`.'},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={"help": 'Timeout (in seconds) for connecting to the vLLM server when `vllm_mode="server"`.'},
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": 'GPU memory utilization for the colocated vLLM engine when `vllm_mode="colocate"`. Lower values reduce contention when sharing a device with the student/teacher models.'
        },
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": 'Tensor parallel size for the colocated vLLM engine when `vllm_mode="colocate"`.'},
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={"help": "Regex pattern used for vLLM guided decoding (optional)."},
    )
    vllm_sync_frequency: int = field(
        default=1,
        metadata={
            "help": "Frequency (in training steps) to synchronize model weights to the vLLM engine. Set to 1 to sync after every step."
        },
    )
    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable sleep mode for the colocated vLLM engine. When `True`, the engine sleeps during the optimizer step and wakes for weight sync and generation."
        },
    )


    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        # Check lmbda and beta are in the range [0, 1]
        if self.lmbda < 0.0 or self.lmbda > 1.0:
            raise ValueError("lmbda must be in the range [0.0, 1.0].")
        if self.beta < 0.0 or self.beta > 1.0:
            raise ValueError("beta must be in the range [0.0, 1.0].")

        # Validate that max_length is sufficient for max_completion_length
        if self.max_length is not None and self.max_completion_length >= self.max_length:
            raise ValueError(
                f"max_completion_length ({self.max_completion_length}) must be smaller than max_length "
                f"({self.max_length}) to leave room for the prompt. Consider increasing max_length or reducing "
                "max_completion_length."
            )

        if self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps

        # Validate ULD parameters
        if self.use_uld_loss:
            if self.uld_crossentropy_weight < 0.0:
                raise ValueError("uld_crossentropy_weight must be non-negative.")
            if self.uld_distillation_weight < 0.0:
                raise ValueError("uld_distillation_weight must be non-negative.")
            if self.uld_student_temperature <= 0.0:
                raise ValueError("uld_student_temperature must be positive.")
            if self.uld_teacher_temperature <= 0.0:
                raise ValueError("uld_teacher_temperature must be positive.")

            # Validate hybrid loss weights - both must be None or both must be set
            if self.uld_use_hybrid_loss:
                if (self.uld_hybrid_matched_weight is None) != (self.uld_hybrid_unmatched_weight is None):
                    raise ValueError(
                        "uld_hybrid_matched_weight and uld_hybrid_unmatched_weight must both be None (for adaptive "
                        "weighting) or both be set to numeric values. Got uld_hybrid_matched_weight="
                        f"{self.uld_hybrid_matched_weight} and uld_hybrid_unmatched_weight="
                        f"{self.uld_hybrid_unmatched_weight}."
                    )
                if self.uld_hybrid_matched_weight is not None:
                    if self.uld_hybrid_matched_weight < 0.0:
                        raise ValueError("uld_hybrid_matched_weight must be non-negative.")
                    if self.uld_hybrid_unmatched_weight < 0.0:
                        raise ValueError("uld_hybrid_unmatched_weight must be non-negative.")
