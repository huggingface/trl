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
from typing import Any, Literal

from transformers import TrainingArguments

from ...trainer.base_config import _BaseConfig


@dataclass
class OPSDConfig(_BaseConfig):
    r"""
    Configuration class for the [`OPSDTrainer`].

    Parameters:
        > Parameters that control the OPSD loss

        distillation_alpha (`float`, *optional*, defaults to `0.0`):
            Divergence interpolation coefficient. `0.0` is forward KL (the official OPSD setting), `1.0` is reverse KL,
            and intermediate values give the generalized JSD mixture.
        distillation_mode (`Literal["sampled_token", "full_logits", "topk_logits", "dopd"]`, *optional*, defaults to `"full_logits"`):
            Distillation objective mode. OPSD defaults to the full-vocabulary divergence of the paper. `topk_logits`
            restricts the divergence to the teacher's top-k support and `sampled_token` uses the token-level reverse KL
            (requires `distillation_alpha=1.0`). `dopd` routes each token between four regimes based on the
            teacher/student advantage gap, following DOPD (https://huggingface.co/papers/2606.30626); see the `distillation_dopd_*` parameters.
        distillation_topk (`int`, *optional*):
            Number of top tokens used by the `topk_logits` objective. Must be set when `distillation_mode=topk_logits`.
        distillation_kl_clip (`float`, *optional*, defaults to `0.05`):
            Pointwise per-vocabulary-entry clip applied to the divergence before it is summed over the vocabulary.
            Prevents high-divergence style tokens from dominating the training signal. `None` disables clipping. Only
            supported for the `full_logits` and `topk_logits` modes.
        distillation_is_clip (`float`, *optional*, defaults to `2.0`):
            Clipping coefficient for importance sampling in self-distillation. `None` disables clipping.
        distillation_add_tail (`bool`, *optional*, defaults to `False`):
            Whether to add a tail bucket for non-top-k probability mass.
        distillation_dopd_gap_threshold (`float`, *optional*, defaults to `2.0`):
            Advantage-gap threshold (in nats, on the realized token's log-probability difference) separating the
            "low gap" and "high gap" regimes. Only used when `distillation_mode="dopd"`.
        distillation_dopd_confidence_threshold (`float`, *optional*, defaults to `0.5`):
            Per-token max-probability threshold separating "confident" from "unsure" for both teacher and student.
            Only used when `distillation_mode="dopd"`.
        distillation_dopd_light_topk (`int`, *optional*, defaults to `8`):
            Top-k support size used by the DOPD regimes that apply a "light" distillation signal (low-gap /
            high-gap-student-confident). Only used when `distillation_mode="dopd"`.
        distillation_dopd_self_reg_weight (`float`, *optional*, defaults to `0.01`):
            Weight of the stop-gradient self-regularization term applied to low-confidence tokens. Only used when
            `distillation_mode="dopd"`.
        distillation_dopd_student_consistency_weight (`float`, *optional*, defaults to `0.1`):
            Weight of the stop-gradient student-consistency term applied to high-gap, student-confident tokens. Only
            used when `distillation_mode="dopd"`.

        > Parameters that control the teacher

        teacher_model_kind (`str`, *optional*, defaults to `"base"`):
            Semantic teacher choice for OPSD. `base` uses the initial student (the official OPSD setting), `live` uses
            the current student, and `ema` uses an exponentially averaged teacher.
        teacher_update_rate (`float`, *optional*, defaults to `0.05`):
            EMA update rate used when `teacher_model_kind="ema"`. A value of `1.0` reduces the update to a hard
            overwrite, periodically resyncing the teacher to the current student weights.
        teacher_sync_steps (`int`, *optional*, defaults to `1`):
            Number of optimizer steps between teacher updates.

        > Parameters that control the teacher prompt

        teacher_prompt_template (`str`, *optional*):
            Template used to combine the student prompt and the privileged ground-truth solution into the teacher
            prompt. Must contain the `{prompt}` and `{privileged_context}` placeholders. Defaults to the official OPSD
            wording, which wraps the solution in reference markers followed by a transition instruction.
        teacher_chat_template_kwargs (`dict[str, Any]`, *optional*):
            Extra kwargs forwarded to `apply_chat_template` when building the teacher prompt (for example
            `{"enable_thinking": True}` to pair a thinking teacher with a non-thinking student).

        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` argument
            of the `OPSDTrainer` is provided as a string.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the student and teacher models.

        > Parameters that control data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the columns required by the trainer in the dataset. Keep this to `False` if you
            provide extra columns (such as `privileged_context`) that the trainer needs.
        max_prompt_length (`int`, *optional*, defaults to `512`):
            Maximum prompt length. Longer prompts are truncated from the left.
        shuffle_dataset (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training dataset.

        > Parameters that control generation

        num_generations (`int`, *optional*, defaults to `1`):
            Number of generations to sample. The effective batch size (num_processes * per_device_batch_size *
            gradient_accumulation_steps) must be evenly divisible by this value.
        num_generations_eval (`int`, *optional*):
            Number of generations to sample during evaluation. This allows using fewer generations during evaluation to
            save computation. If `None`, uses the value of `num_generations`.
        max_completion_length (`int`, *optional*, defaults to `1024`):
            Maximum length of the generated completion.
        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1.0
            to consider all tokens.
        top_k (`int`, *optional*, defaults to `0`):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, top-k-filtering is
            disabled and all tokens are considered.
        min_p (`float`, *optional*):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model to repeat
            tokens.
        cache_implementation (`str`, *optional*):
            Implementation of the cache method for faster generation when use_vllm is set to False.
        generation_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or `SamplingParams` (if
            using vLLM) when sampling completions. This can be used to further customize the generation behavior, such
            as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that conflict with the other generation
            parameters (like `min_p`, `top_p`, etc.), they will override them.
        chat_template_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the `apply_chat_template` function when generating completions.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for generation
            instead of the default model.generate(). Requires `vllm` to be installed.
        vllm_mode (`str`, *optional*, defaults to `"colocate"`):
            Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `'server'` or
            `'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure a
            TRL vLLM server is running (start with `trl vllm-serve`). `'colocate'`: vLLM will run in the same process
            and share the training GPUs. This avoids the need for a separate server but may cause resource contention
            with training.
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: Use the
            `transformers` backend for model implementation. `vllm`: Use the `vllm` library for model implementation.
        vllm_enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory usage low, but
            waking the engine adds host–device transfer latency.
        vllm_server_base_url (`str`, *optional*):
            Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` and
            `vllm_server_port` are ignored.
        vllm_server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the vLLM server to connect to. Ignored if vllm_server_base_url is provided.
        vllm_server_port (`int`, *optional*, defaults to `8000`):
            Port of the vLLM server to connect to. Ignored if vllm_server_base_url is provided.
        vllm_group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group. This is used to communicate with the vLLM server. Unless the port
            is occupied, there is no need to change it.
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.
        vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to
            `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_tensor_parallel_size` flag.
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.3`):
            Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to
            `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_gpu_memory_utilization` flag.
        vllm_max_model_length (`int`, *optional*):
            Context window for vLLM. Set it to at least the maximum prompt length in the dataset plus
            `max_completion_length`; if omitted, it is inferred from the model config.

        > Parameters that control the training

        loss_type (`str`, *optional*, defaults to `"grpo"`):
            Policy loss aggregation. Supported: `grpo`, `bnpo`, `dr_grpo`, `dapo`.
        num_iterations (`int`, *optional*, defaults to `1`):
            Number of iterations per batch (denoted as μ in the algorithm).
        generation_batch_size (`int`, *optional*):
            Batch size to use for generation. If `None`, it defaults to the effective training batch size:
            `per_device_train_batch_size * num_processes * steps_per_generation`.
        steps_per_generation (`int`, *optional*):
            Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the `OPSDTrainer` is provided as a string."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the student and teacher models."},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "Whether to only keep the columns required by the trainer in the dataset. Keep this to `False` if you provide extra columns (such as `privileged_context`) that the trainer needs."
        },
    )
    max_prompt_length: int | None = field(
        default=512,
        metadata={"help": "Maximum prompt length. Longer prompts are truncated from the left."},
    )
    num_generations: int = field(
        default=1,
        metadata={
            "help": "Number of generations to sample. The effective batch size (num_processes * per_device_batch_size * gradient_accumulation_steps) must be evenly divisible by this value."
        },
    )
    num_generations_eval: int | None = field(
        default=None,
        metadata={
            "help": "Number of generations to sample during evaluation. This allows using fewer generations during evaluation to save computation. If `None`, uses the value of `num_generations`."
        },
    )
    max_completion_length: int | None = field(
        default=1024,
        metadata={"help": "Maximum length of the generated completion."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation, improving generation speed. However, disabling this option allows training models that exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible with vLLM generation."
        },
    )
    shuffle_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )
    generation_batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Batch size to use for generation. If `None`, it defaults to the effective training batch size: `per_device_train_batch_size * num_processes * steps_per_generation`."
        },
    )
    steps_per_generation: int | None = field(
        default=None,
        metadata={"help": "Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1.0 to consider all tokens."
        },
    )
    top_k: int = field(
        default=0,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, top-k-filtering is disabled and all tokens are considered."
        },
    )
    min_p: float | None = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
    generation_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the generation behavior, such as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that conflict with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them."
        },
    )
    chat_template_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to the `apply_chat_template` function when generating completions."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model to repeat tokens."
        },
    )
    cache_implementation: str | None = field(
        default=None,
        metadata={"help": "Implementation of the cache method for faster generation when use_vllm is set to False."},
    )
    use_vllm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for generation instead of the default model.generate(). Requires `vllm` to be installed."
        },
    )
    use_teacher_server: bool = field(
        default=False,
        metadata={
            "help": "Compute teacher logprobs from the running vLLM generation server instead of a local teacher "
            "forward. Only supported for `teacher_model_kind='live'` with `use_vllm=True` and `vllm_mode='server'`, "
            "and `distillation_mode` in {'sampled_token', 'topk_logits'} (the server returns the teacher's top-k "
            "logprobs, not the full vocabulary; `topk_logits` distills over the teacher's own top-k support)."
        },
    )
    vllm_mode: str = field(
        default="colocate",
        metadata={
            "help": "Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `'server'` or `'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure a TRL vLLM server is running (start with `trl vllm-serve`). `'colocate'`: vLLM will run in the same process and share the training GPUs. This avoids the need for a separate server but may cause resource contention with training."
        },
    )
    vllm_model_impl: str = field(
        default="vllm",
        metadata={
            "help": "Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: Use the `transformers` backend for model implementation. `vllm`: Use the `vllm` library for model implementation."
        },
    )
    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={
            "help": "Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory usage low, but waking the engine adds host–device transfer latency."
        },
    )
    vllm_server_base_url: str | None = field(
        default=None,
        metadata={
            "help": "Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` and `vllm_server_port` are ignored."
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
    vllm_group_port: int = field(
        default=51216,
        metadata={
            "help": "Port number for the weight update group. This is used to communicate with the vLLM server. Unless the port is occupied, there is no need to change it."
        },
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={
            "help": "Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the timeout, a `ConnectionError` is raised."
        },
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when launching the vLLM server via the `--vllm_tensor_parallel_size` flag."
        },
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.3,
        metadata={
            "help": "Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when launching the vLLM server via the `--vllm_gpu_memory_utilization` flag."
        },
    )
    vllm_max_model_length: int | None = field(
        default=None,
        metadata={
            "help": "Context window for vLLM. Set it to at least the maximum prompt length in the dataset plus `max_completion_length`; if omitted, it is inferred from the model config."
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},
    )
    teacher_model_kind: str = field(
        default="base",
        metadata={
            "help": "Semantic teacher choice for OPSD. `base` uses the initial student, `live` uses the current "
            "student, and `ema` uses an exponentially averaged teacher."
        },
    )
    teacher_update_rate: float = field(
        default=0.05,
        metadata={
            "help": 'EMA update rate used when `teacher_model_kind="ema"`. A value of `1.0` reduces the update '
            "to a hard overwrite, periodically resyncing the teacher to the current student weights."
        },
    )
    teacher_sync_steps: int = field(
        default=1,
        metadata={"help": "Number of optimizer steps between teacher updates."},
    )
    distillation_alpha: float = field(
        default=0.0,
        metadata={
            "help": "Divergence interpolation coefficient. `0.0` is forward KL (the official OPSD setting), `1.0` is "
            "reverse KL, and intermediate values give the generalized JSD mixture."
        },
    )
    distillation_mode: Literal["sampled_token", "full_logits", "topk_logits", "dopd"] = field(
        default="full_logits",
        metadata={"help": "Distillation objective mode. OPSD defaults to the full-vocabulary divergence."},
    )
    distillation_topk: int | None = field(
        default=None,
        metadata={"help": "Number of top tokens used by the `topk_logits` objective."},
    )
    distillation_kl_clip: float | None = field(
        default=0.05,
        metadata={
            "help": "Pointwise per-vocabulary-entry clip applied to the divergence before it is summed over the "
            "vocabulary. `None` disables clipping. Only supported for `full_logits` and `topk_logits`."
        },
    )
    distillation_is_clip: float | None = field(
        default=2.0,
        metadata={
            "help": "Clipping coefficient for importance sampling in self-distillation. `None` disables clipping."
        },
    )
    distillation_add_tail: bool = field(
        default=False,
        metadata={"help": "Whether to add a tail bucket for non-top-k probability mass."},
    )
    distillation_dopd_gap_threshold: float = field(
        default=2.0,
        metadata={
            "help": "Advantage-gap threshold (nats) separating the 'low gap' and 'high gap' DOPD regimes. Only "
            "used when `distillation_mode='dopd'`."
        },
    )
    distillation_dopd_confidence_threshold: float = field(
        default=0.5,
        metadata={
            "help": "Per-token max-probability threshold separating 'confident' from 'unsure' teacher/student. "
            "Only used when `distillation_mode='dopd'`."
        },
    )
    distillation_dopd_light_topk: int = field(
        default=8,
        metadata={
            "help": "Top-k support size for the DOPD regimes that apply a light distillation signal. Only used "
            "when `distillation_mode='dopd'`."
        },
    )
    distillation_dopd_self_reg_weight: float = field(
        default=0.01,
        metadata={
            "help": "Weight of the stop-gradient self-regularization term for low-confidence tokens. Only used "
            "when `distillation_mode='dopd'`."
        },
    )
    distillation_dopd_student_consistency_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight of the stop-gradient student-consistency term for high-gap, student-confident tokens. "
            "Only used when `distillation_mode='dopd'`."
        },
    )
    teacher_prompt_template: str = field(
        default=(
            "{prompt}\n\n"
            "Here is a reference solution to this problem:\n"
            "=== Reference Solution Begin ===\n{privileged_context}\n=== Reference Solution End ===\n\n"
            "After reading the reference solution above, make sure you truly understand the reasoning behind each "
            "step, do not copy or paraphrase it. Now, using your own words and independent reasoning, derive the "
            "same final answer to the problem above. Think step by step, explore different approaches, and don't "
            "be afraid to backtrack or reconsider if something doesn't work out:"
        ),
        metadata={
            "help": "Template used to combine the student prompt and the ground-truth solution into the teacher "
            "prompt. Must contain the `{prompt}` and `{privileged_context}` placeholders."
        },
    )
    teacher_chat_template_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={"help": "Extra kwargs forwarded to `apply_chat_template` when building the teacher prompt."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.distillation_mode == "sampled_token" and self.distillation_alpha != 1.0:
            raise ValueError(
                "`distillation_mode='sampled_token'` only supports reverse KL, so it requires "
                f"`distillation_alpha=1.0`, got {self.distillation_alpha}."
            )
        if self.distillation_kl_clip is not None:
            if self.distillation_kl_clip <= 0:
                raise ValueError(f"`distillation_kl_clip` must be positive, got {self.distillation_kl_clip}.")
            if self.distillation_mode == "sampled_token":
                raise ValueError(
                    "`distillation_kl_clip` only supports `distillation_mode` in {'full_logits', 'topk_logits'}: the "
                    "pointwise clip applies to per-vocabulary-entry divergences, which `sampled_token` does not "
                    "compute. Set `distillation_kl_clip=None`."
                )
        if (
            "{prompt}" not in self.teacher_prompt_template
            or "{privileged_context}" not in self.teacher_prompt_template
        ):
            raise ValueError(
                "teacher_prompt_template must contain both `{prompt}` and `{privileged_context}` placeholders"
            )
        num_processes = self.world_size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            global_batch_size = self.per_device_train_batch_size * num_processes
            if self.generation_batch_size % global_batch_size != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size ({global_batch_size})."
                )
            self.steps_per_generation = self.generation_batch_size // global_batch_size
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        else:
            raise ValueError("'generation_batch_size' and 'steps_per_generation' can not both be configured")

        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations ({self.num_generations})."
            )

        if self.do_eval and self.eval_strategy != "no":
            num_generations_eval = self.num_generations_eval or self.num_generations
            if (self.per_device_eval_batch_size * num_processes) % num_generations_eval != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by the number of generations used for evaluation ({num_generations_eval})."
                )
