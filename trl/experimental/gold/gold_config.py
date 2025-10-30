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
            argument of the [`GOLDTrainer`] is provided as a string.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropouts in `model`.

        > Parameters that control the data preprocessing

        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        eos_token (`str`, *optional*):
            Token used to indicate the end of a turn or sequence. If `None`, it defaults to
            `processing_class.eos_token`.
        pad_token (`str`, *optional*):
            Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that is also `None`,
            it falls back to `processing_class.eos_token`.
        max_length (`int`, *optional*, defaults to `1024`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the right.
            If `None`, no truncation is applied.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.

        > Parameters that control generation

        lmbda (`float`, *optional*, defaults to `0.5`):
            Lambda parameter that controls the student data fraction (i.e., the proportion of on-policy
            student-generated outputs).
        steps_per_generation (`int`, *optional*):
            Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`.
        max_completion_length (`int`, *optional*, defaults to `128`):
            Maximum number of tokens to generate per completion.
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `0.95`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int`, *optional*):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, top-k-filtering is
            disabled and all tokens are considered.
        use_transformers_paged (`bool`, *optional*, defaults to `False`):
            Whether to use the `transformers` paged implementation for generation. If set to `True`, the `transformers`
            paged implementation will be used for generation instead of the default padded implementation. This
            parameter is only effective when `use_vllm` is set to `False`.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for generation
            instead of the default model.generate(). Requires `vllm` to be installed.
        vllm_mode (`str`, *optional*, defaults to `"server"`):
            Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `"server"` or
            `"colocate"`.

            - `"server"`: The trainer will send generation requests to a separate vLLM server. Make sure a TRL vLLM
              server is running (start with `trl vllm-serve`).
            - `"colocate"`: vLLM will run in the same process and share the training GPUs. This avoids the need for a
              separate server but may cause resource contention with training.
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation to use for vLLM. Must be one of `"transformers"` or `"vllm"`. `"transformers"`: Use
            the `transformers` backend for model implementation. `"vllm"`: Use the `vllm` library for model
            implementation.
        vllm_guided_decoding_regex (`str`, *optional*):
            Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled.

        > Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)

        vllm_server_base_url (`str`, *optional*):
            Base URL for the vLLM server (e.g., `"http://localhost:8000"`). If provided, `vllm_server_host` and
            `vllm_server_port` are ignored.
        vllm_server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the vLLM server to connect to. Ignored if `vllm_server_base_url` is provided.
        vllm_server_port (`int`, *optional*, defaults to `8000`):
            Port of the vLLM server to connect to. Ignored if `vllm_server_base_url` is provided.
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.

        > Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)

        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.3`):
            Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to
            `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_gpu_memory_utilization` flag.
        vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to
            `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_tensor_parallel_size` flag.
        vllm_enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Whether to enable sleep mode for vLLM. If `True`, vLLM will sleep during the optimization step and woken
            for weight sync and generation.

        > Parameters that control the training
        beta (`float`, *optional*, defaults to `0.5`):
            Interpolation coefficient between `0.0` and `1.0` of the Generalized Jensen-Shannon Divergence loss. When
            beta is `0.0`, the loss is the KL divergence. When beta is `1.0`, the loss is the Inverse KL Divergence.
        use_uld_loss (`bool`, *optional*, defaults to `False`):
            Whether to use Universal Logit Distillation (ULD) loss instead of Generalized Jensen-Shannon Divergence
            loss.
        use_extended_uld (`bool`, *optional*, defaults to `True`):
            Whether to enable extended ULD alignment that uses tokenizers to align and merge token probabilities across
            student and teacher tokenizations. When True, the trainer will compute token mappings and merge
            probabilities for split tokens; when False, ULD will use simple positional truncation like in the original
            ULD paper.
        uld_use_hybrid_loss (`bool`, *optional*, defaults to `False`):
            Whether to use a hybrid loss that combines ULD loss and JSD loss. When True, the final loss is a
            combination of JSD for known token mappings and ULD for unknown token mappings.
        uld_hybrid_matched_weight (`float`, *optional*):
            Weight for the matched token loss component when using hybrid ULD + JSD loss. This weight scales the JSD
            loss computed over tokens that have a direct mapping between student and teacher tokenizations. If None,
            uses adaptive weighting based on vocabulary overlap. Must be set together with
            `uld_hybrid_unmatched_weight` (both None or both float).
        uld_hybrid_unmatched_weight (`float`, *optional*):
            Weight for the unmatched token loss component when using hybrid ULD + JSD loss. This weight scales the ULD
            loss computed over tokens that do not have a direct mapping between student and teacher tokenizations. If
            None, uses adaptive weighting based on vocabulary overlap. Must be set together with
            `uld_hybrid_matched_weight` (both None or both float).
        uld_crossentropy_weight (`float`, *optional*, defaults to `0.0`):
            Weight for the cross-entropy loss component in ULD loss.
        uld_distillation_weight (`float`, *optional*, defaults to `1.0`):
            Weight for the distillation loss component in ULD loss.
        uld_student_temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for student logits in ULD loss computation.
        uld_teacher_temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for teacher logits in ULD loss computation.
        uld_skip_student_eos (`bool`, *optional*, defaults to `True`):
            Whether to skip EOS token for student in ULD loss computation.
        uld_skip_teacher_eos (`bool`, *optional*, defaults to `True`):
            Whether to skip EOS token for teacher in ULD loss computation.
        activation_offloading (`bool`, *optional*, defaults to `False`):
            Whether to offload the activations to the CPU.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is installed,
            it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`.
        num_completions_to_print (`int`, *optional*):
            Number of completions to print with `rich`. If `None`, all completions are logged.
        wandb_log_unique_prompts (`bool`, *optional*, defaults to `False`):
            Whether to log unique prompts in wandb. If `True`, only unique prompts are logged. If `False`, all prompts
            are logged.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-7,
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
            "the `GOLDTrainer` is provided as a string."
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
            "help": "Token used to indicate the end of a turn or sequence. If `None`, it defaults to "
            "`processing_class.eos_token`."
        },
    )
    pad_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that is "
            "also `None`, it falls back to `processing_class.eos_token`."
        },
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from "
            "the right. If `None`, no truncation is applied."
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
    steps_per_generation: Optional[int] = field(
        default=None,
        metadata={"help": "Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`."},
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
    use_transformers_paged: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the `transformers` paged implementation for generation. If set to `True`, the "
            "`transformers` paged implementation will be used for generation instead of the default padded "
            "implementation. This parameter is only effective when `use_vllm` is set to `False`."
        },
    )

    # Parameters that control generation acceleration powered by vLLM
    use_vllm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for "
            "generation instead of the default `model.generate()`. Requires `vllm` to be installed."
        },
    )
    vllm_mode: str = field(
        default="server",
        metadata={
            "help": "Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `'server'` or "
            "`'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure "
            "a TRL vLLM server is running (start with `trl vllm-serve`). `'colocate'`: vLLM will run in the same "
            "process and share the training GPUs. This avoids the need for a separate server but may cause resource "
            "contention with training."
        },
    )
    vllm_model_impl: str = field(
        default="vllm",
        metadata={
            "help": "Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: "
            "Use the `transformers` backend for model implementation. `vllm`: Use the `vllm` library for "
            "model implementation."
        },
    )
    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable sleep mode for vLLM. If `True`, vLLM will sleep during the optimization step "
            "and be woken for weight sync and generation."
        },
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={"help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."},
    )

    # Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)
    vllm_server_base_url: Optional[str] = field(
        default=None,
        metadata={
            "help": "Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` "
            "and `vllm_server_port` are ignored."
        },
    )
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host of the vLLM server to connect to. Ignored if vllm_server_base_url is provided."},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "Port of the vLLM server to connect to. Ignored if vllm_server_base_url is provided."},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={
            "help": "Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up "
            "after the timeout, a `ConnectionError` is raised."
        },
    )

    # Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)
    vllm_gpu_memory_utilization: float = field(
        default=0.3,
        metadata={
            "help": "Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_gpu_memory_utilization` flag."
        },
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_tensor_parallel_size` flag."
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
                "Whether to enable extended ULD alignment that uses tokenizers to align and merge token probabilities "
                "across student and teacher tokenizations. When True, the trainer will compute token mappings and "
                "merge probabilities for split tokens; when False, ULD will use simple positional truncation like in "
                "the original ULD paper."
            )
        },
    )
    uld_use_hybrid_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use a hybrid loss that combines ULD loss and JSD loss. When True, the final loss is a "
                "combination of JSD for known token mappings and ULD for unknown token mappings."
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
                "with `uld_hybrid_unmatched_weight` (both None or both float)."
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
                "with `uld_hybrid_matched_weight` (both None or both float)."
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
    wandb_log_unique_prompts: bool = field(
        default=False,
        metadata={
            "help": "Whether to log unique prompts in wandb. If `True`, only unique prompts are logged. If `False`, "
            "all prompts are logged."
        },
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        super().__post_init__()

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
