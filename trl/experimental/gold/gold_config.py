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
from typing import Any

from transformers import TrainingArguments

from ...trainer.sft_config import SFTConfig


@dataclass
class GOLDConfig(SFTConfig):
    r"""
    Configuration class for [`GOLDTrainer`].

    This class includes only the parameters that are specific to GOLD training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] and [`SFTConfig`] documentation.

    Args:
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        lmbda (`float`, *optional*, defaults to `0.5`):
            Lambda parameter that controls the student data fraction (i.e., the proportion of on-policy
            student-generated outputs).
        beta (`float`, *optional*, defaults to `0.5`):
            Interpolation coefficient between `0.0` and `1.0` of the Generalized Jensen-Shannon Divergence loss. When
            beta is `0.0`, the loss is the KL divergence. When beta is `1.0`, the loss is the Inverse KL Divergence.
        max_completion_length (`int`, *optional*, defaults to `128`):
            Maximum number of tokens to generate per completion.
        teacher_model_name_or_path (`str` or `None`, *optional*, defaults to `None`):
            Model name or path of the teacher model. If `None`, the teacher model will be the same as the model being
            trained.
        teacher_model_init_kwargs (`dict[str, Any]]` or `None`, *optional*, defaults to `None`):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the teacher model
            from a string.
        teacher_tokenizer_name_or_path (`str` or `None`, *optional*, defaults to `None`):
            Tokenizer name or path for the teacher model. If None when using ULD loss, will use the same tokenizer as
            the student model (not recommended for cross-tokenizer distillation).
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        seq_kd (`bool`, *optional*, defaults to `False`):
            Seq_kd parameter that controls whether to perform Sequence-Level KD (can be viewed as supervised FT on
            teacher-generated output).
        use_uld_loss (`bool`, *optional*, defaults to `False`):
            Whether to use Universal Logit Distillation (ULD) loss instead of Generalized Jensen-Shannon Divergence
            loss.
        uld_crossentropy_weight (`float`, *optional*, defaults to `0.0`):
            Weight for the cross-entropy loss component in ULD loss. If 0, only ULD distillation loss is used.
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
        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions from the student model. Requires `vllm` to be installed.
        vllm_mode (`str`, *optional*, defaults to `"server"`):
            Mode for student vLLM integration. Either `"server"` (connect to a running TRL vLLM server) or `"colocate"`
            (run vLLM in the same process).
        vllm_server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the vLLM server for the student model (if `vllm_mode="server"`).
        vllm_server_port (`int`, *optional*, defaults to `8001`):
            Port of the vLLM server for the student model (if `vllm_mode="server"`).
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Timeout for connecting to the student vLLM server (if `vllm_mode="server"`).
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            GPU memory utilization for the colocated student vLLM engine (if `vllm_mode="colocate"`). It is recommended
            to set this to a low value if the student and teacher models share the same GPU.
        vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Tensor parallel size for the colocated student vLLM engine (if `vllm_mode="colocate"`).
        vllm_guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
            Regex for vLLM guided decoding for the student model.
        vllm_sync_frequency (`int`, *optional*, defaults to `1`):
            Frequency (in training steps) to synchronize student model weights to vLLM engine. Set to 1 to sync after
            every step.
        vllm_enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Enable vLLM sleep mode to offload student weights/cache during the optimizer step. Keeps GPU memory usage
            low, but waking the engine adds host–device transfer latency.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["teacher_model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-7,
        metadata={"help": "The initial learning rate for AdamW."},
    )

    # GOLD-specific parameters
    temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    top_p: float = field(
        default=0.95,
        metadata={
            "help": "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to "
            "`top_p` or higher are kept for generation."
        },
    )
    top_k: int = field(
        default=0,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."},
    )
    lmbda: float = field(
        default=0.5,
        metadata={
            "help": "Lambda parameter that controls the student data fraction (i.e., the proportion of on-policy "
            "student-generated outputs)."
        },
    )
    beta: float = field(
        default=0.5,
        metadata={
            "help": "Interpolation coefficient between `0.0` and `1.0` of the Generalized Jensen-Shannon Divergence "
            "loss. When beta is `0.0`, the loss is the KL divergence. When beta is `1.0`, the loss is the Inverse KL "
            "Divergence."
        },
    )
    max_completion_length: int = field(
        default=128,
        metadata={"help": "Maximum number of tokens to generate per completion."},
    )
    student_model_revision: str = field(
        default="main",
        metadata={
            "help": "Revision of the student model to use. If not specified, the default revision of the model will be used."
        },
    )
    teacher_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Model name or path of the teacher model. If `None`, the teacher model will be the same as the "
            "model being trained."
        },
    )
    teacher_model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the "
            "teacher model from a string."
        },
    )
    teacher_tokenizer_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Tokenizer name or path for the teacher model. If None when using ULD loss, will use the same "
            "tokenizer as the student model (not recommended for cross-tokenizer distillation)."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropouts in `model`."},
    )
    seq_kd: bool = field(
        default=False,
        metadata={
            "help": "Seq_kd parameter that controls whether to perform Sequence-Level KD (can be viewed as supervised "
            "FT on teacher-generated output)."
        },
    )
    steps_per_generation: int | None = field(
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
    uld_hybrid_matched_weight: float | None = field(
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
    uld_hybrid_unmatched_weight: float | None = field(
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
    vllm_guided_decoding_regex: str | None = field(
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
            "help": "Enable vLLM sleep mode to offload student weights/cache during the optimizer step. Keeps GPU "
            "memory usage low, but waking the engine adds host–device transfer latency."
        },
    )
    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )
    log_completions_steps: int = field(
        default=100,
        metadata={
            "help": "Number of steps between logging (prompt, completion) pairs. Only used if `log_completions` is "
            "set to `True`."
        },
    )
    num_completions_to_print: int | None = field(
        default=None,
        metadata={"help": "Number of completions to print with `rich`. If `None`, all completions are logged."},
    )
    wandb_entity: str | None = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: str | None = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: str | None = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )
    wandb_log_unique_prompts: bool = field(
        default=True,
        metadata={
            "help": ("Whether to log the unique prompts to wandb. This will create a new run for each unique prompt.")
        },
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    hub_model_revision: str | None = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    num_completions_to_print: int = field(default=5, metadata={"help": "Number of completions to print."})
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    trl_project: str = field(
        default="smollm3",
        metadata={
            "help": "The TRL project to use for evaluation. This is used to determine the path to the evaluation script."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # check lmbda and beta are in the range [0, 1]
        if self.lmbda < 0.0 or self.lmbda > 1.0:
            raise ValueError("lmbda must be in the range [0.0, 1.0].")
        if self.beta < 0.0 or self.beta > 1.0:
            raise ValueError("beta must be in the range [0.0, 1.0].")

        # Validate that max_length is sufficient for max_completion_length
        if self.max_length is not None and self.max_completion_length >= self.max_length:
            raise ValueError(
                f"max_completion_length ({self.max_completion_length}) must be smaller than max_length ({self.max_length}) "
                f"to leave room for the prompt. Consider increasing max_length or reducing max_completion_length."
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
