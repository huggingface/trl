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

from .base_config import _BaseConfig


@dataclass
class DistillationConfig(_BaseConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`DistillationTrainer`].

    This class includes only the parameters that are specific to distillation training. For a full list of training
    arguments, please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this
    class may differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and teacher model

        model_init_kwargs (`str`, `dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`DistillationTrainer`] is provided as a string.
        teacher_model_init_kwargs (`str`, `dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `teacher_model`
            argument of the [`DistillationTrainer`] is provided as a string.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to
            [`~transformers.AutoModelForCausalLM.from_pretrained`] and
            [`~transformers.AutoProcessor.from_pretrained`]. Also applied to the teacher model load.
        router_aux_loss_coef (`float`, *optional*, defaults to `0.001`):
            Coefficient of the load-balancing auxiliary loss. Only has an effect when training a Mixture-of-Experts
            (MoE) model; for other models it does nothing. The auxiliary loss is added to the training loss with this
            weight. Set to `0.0` to disable it.
        disable_dropout (`bool`, *optional*, defaults to `False`):
            Whether to disable dropout in the model. This is useful for training with a teacher model, as it prevents
            the model from generating different logprobs for the same input.

        > Parameters that control the data preprocessing

        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.
        shuffle_dataset (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training dataset.
        pad_to_multiple_of (`int`, *optional*):
            If set, the prompts ids and completions ids will be padded to a multiple of this value.

        > Parameters that control generation

        steps_per_generation (`int`, *optional*):
            Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`. The generation
            batch size is derived from it: `per_device_train_batch_size * num_processes * steps_per_generation`.
        temperature (`float`, defaults to `1.0`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int`, *optional*, defaults to `0`):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, top-k-filtering is
            disabled and all tokens are considered.
        min_p (`float`, *optional*):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
        generation_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to [`~transformers.GenerationConfig`] (if using transformers) or
            `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the
            generation behavior, such as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that conflict
            with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them.
        chat_template_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the `apply_chat_template` function when generating completions.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat
            tokens.
        cache_implementation (`str`, *optional*):
            Implementation of the cache method for faster generation when `use_vllm` is set to `False`.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for generation
            instead of the default model.generate(). Requires `vllm` to be installed.
        vllm_mode (`str`, *optional*, defaults to `"colocate"`):
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
        vllm_structured_outputs_regex (`str`, *optional*):
            Regex for vLLM structured outputs. If `None` (default), structured outputs is disabled.

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
        vllm_group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group. This is used to communicate with the vLLM server. Unless the port
            is occupied, there is no need to change it.

        > Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)

        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.3`):
            Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to
            `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_gpu_memory_utilization` flag.
        vllm_max_model_length (`int`, *optional*):
            Context window for vLLM. Set it to at least the maximum prompt length in the dataset plus
            `max_completion_length`; if omitted, it is inferred from the model config.
        vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to
            `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_tensor_parallel_size` flag.
        vllm_enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory usage low, but
            waking the engine adds host–device transfer latency.

        > Parameters that control generation acceleration powered by transformers continuous batching

        use_transformers_continuous_batching (`bool`, *optional*, defaults to `False`):
            Whether to use transformers' continuous batching engine for generating completions. Requires
            `transformers>=5.8.0`.
        transformers_continuous_batching_config (`dict`, *optional*):
            Keyword arguments for [`~transformers.generation.ContinuousBatchingConfig`].

        > Parameters that control the training

        beta (`float`, *optional*, defaults to `1.0`):
            Interpolation coefficient of the generalized Jensen-Shannon divergence, in `[0.0, 1.0]`. `0.0` is the
            forward KL `KL(teacher || student)`, used by Qwen3. `1.0` (default) is the reverse KL
            `KL(student || teacher)`, the mode-seeking objective used by DeepSeek-V4 and GLM-5. Values in between
            interpolate between the two, giving the generalized JSD of the [GKD
            paper](https://huggingface.co/papers/2306.13649), whose default is `0.5`. Note that, unlike in
            [`GRPOConfig`], this is *not* a KL penalty coefficient: it selects which divergence is minimized.
        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is installed,
            it prints the sample. If `wandb` and/or `trackio` logging is enabled, it logs it to `wandb` and/or
            `trackio`.
        log_multimodal (`bool`, *optional*, defaults to `True`):
            Whether to log multimodal content (images, videos, etc.) together with completions. Disable this to reduce
            log size when using high-resolution multimodal data.
        num_completions_to_print (`int`, *optional*):
            Number of completions to print with `rich`. If `None`, all completions are logged.
        log_unique_prompts (`bool`, *optional*, defaults to `False`):
            Whether to log unique prompts. If `True`, only unique prompts are logged. If `False`, all prompts are
            logged.
        log_completions_hub_repo (`str`, *optional*):
            Hugging Face Hub repository to save the completions. Should be a complete repository name like
            `'username/reponame'` or `'orgname/reponame'`, or just `'reponame'` in which case the repository will be
            created in the currently-logged-in Hugging Face user's namespace. Note that this repository will be public
            unless you set `hub_private_repo=True` or your organization's default is to create private repositories."

    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `10` instead of `500`.
    > - `gradient_checkpointing`: Defaults to `True` instead of `False`.
    > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
    > - `learning_rate`: Defaults to `1e-6` instead of `5e-5`.
    """

    _VALID_DICT_FIELDS = _BaseConfig._VALID_DICT_FIELDS + [
        "model_init_kwargs",
        "transformers_continuous_batching_config",
    ]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )

    # Parameters that control the model and teacher model
    model_init_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `DistillationTrainer` is provided as a string."
        },
    )
    teacher_model_init_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the "
            "`teacher_model` argument of the `DistillationTrainer` is provided as a string."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow loading models and tokenizers that ship custom Python code from the Hub. "
            "Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoProcessor.from_pretrained`. Also applied to "
            "the teacher model load."
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
    disable_dropout: bool = field(
        default=False,
        metadata={
            "help": "Whether to disable dropout in the model. This is useful for training with a teacher model, as "
            "it prevents the model from generating different logprobs for the same input."
        },
    )
    # Parameters that control the data preprocessing
    max_completion_length: int | None = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
        },
    )
    shuffle_dataset: bool | None = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )
    pad_to_multiple_of: int | None = field(
        default=None,
        metadata={"help": "If set, the prompts ids and completions ids will be padded to a multiple of this value."},
    )

    # Parameters that control generation
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
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: int = field(
        default=0,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, "
            "top-k-filtering is disabled and all tokens are considered."
        },
    )
    min_p: float | None = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
    generation_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or "
            "`SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the "
            "generation behavior, such as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that "
            "conflict with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them."
        },
    )
    chat_template_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to the `apply_chat_template` function when generating "
            "completions."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )
    cache_implementation: str | None = field(
        default=None,
        metadata={"help": "Implementation of the cache method for faster generation when use_vllm is set to False."},
    )

    # Parameters that control generation acceleration powered by vLLM
    use_vllm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for "
            "generation instead of the default model.generate(). Requires `vllm` to be installed."
        },
    )
    vllm_mode: str = field(
        default="colocate",
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
            "help": "Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory "
            "usage low, but waking the engine adds host–device transfer latency."
        },
    )
    vllm_structured_outputs_regex: str | None = field(
        default=None,
        metadata={"help": "Regex for vLLM structured outputs. If `None` (default), structured outputs is disabled."},
    )

    # Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)
    vllm_server_base_url: str | None = field(
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
    vllm_group_port: int = field(
        default=51216,
        metadata={
            "help": "Port number for the weight update group. This is used to communicate with the vLLM server. "
            "Unless the port is occupied, there is no need to change it.",
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
    vllm_max_model_length: int | None = field(
        default=None,
        metadata={
            "help": "Context window for vLLM. Set it to at least the maximum prompt length in the dataset plus "
            "`max_completion_length`; if omitted, it is inferred from the model config."
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
        default=1.0,
        metadata={
            "help": "Interpolation coefficient of the generalized Jensen-Shannon divergence, in `[0.0, 1.0]`. `0.0` is "
            "the forward KL `KL(teacher || student)`, used by Qwen3. `1.0` (default) is the reverse KL "
            "`KL(student || teacher)`, the mode-seeking objective used by DeepSeek-V4 and GLM-5. Values in between "
            "interpolate between the two, giving the generalized JSD of the [GKD "
            "paper](https://huggingface.co/papers/2306.13649), whose default is `0.5`."
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
    log_multimodal: bool = field(
        default=True,
        metadata={
            "help": "Whether to log multimodal content (images, videos, etc.) together with completions. Disable this "
            "to reduce log size when using high-resolution multimodal data."
        },
    )
    num_completions_to_print: int | None = field(
        default=None,
        metadata={"help": "Number of completions to print with `rich`. If `None`, all completions are logged."},
    )
    log_unique_prompts: bool = field(
        default=False,
        metadata={
            "help": "Whether to log unique prompts. If `True`, only unique prompts are logged. If `False`, all "
            "prompts are logged."
        },
    )
    log_completions_hub_repo: str | None = field(
        default=None,
        metadata={
            "help": "Hugging Face Hub repository to save the completions. Should be a complete repository name like "
            "`'username/reponame'` or `'orgname/reponame'`, or just `'reponame'` in which case the repository will "
            "be created in the currently-logged-in Hugging Face user's namespace. Note that this repository will be "
            "public unless you set `hub_private_repo=True` or your organization's default is to create private "
            "repositories."
        },
    )

    # Parameters that control generation acceleration powered by transformers continuous batching
    use_transformers_continuous_batching: bool = field(
        default=False,
        metadata={
            "help": "Whether to use transformers' continuous batching engine for generating completions. Requires "
            "transformers>=5.8.0."
        },
    )
    transformers_continuous_batching_config: dict | None = field(
        default=None,
        metadata={"help": "Keyword arguments for `transformers.generation.ContinuousBatchingConfig`."},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.parallelism_config is not None and (
            self.parallelism_config.cp_enabled or self.parallelism_config.sp_enabled
        ):
            raise ValueError(
                "DistillationTrainer does not support sequence-dim parallelism (`parallelism_config.cp_size > 1` or "
                "`parallelism_config.sp_size > 1`) yet. The trainer builds model inputs after generation, so "
                "Transformers' context-parallel / Ulysses sequence-parallel input sharding cannot be applied to the "
                "raw generation batch. Set both `cp_size=1` and `sp_size=1`, or disable `parallelism_config`."
            )

        if self.log_completions_hub_repo is not None and not self.log_completions:
            raise ValueError(
                "log_completions_hub_repo is set, but log_completions is False. Enable log_completions to upload "
                "completions to the Hub, or unset log_completions_hub_repo."
            )

        # The generation batch is consumed over `steps_per_generation` optimization steps, so its size is fully
        # determined by them. It is derived here rather than exposed, since the two must always agree.
        if self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
        self.generation_batch_size = self.per_device_train_batch_size * self.world_size * self.steps_per_generation
