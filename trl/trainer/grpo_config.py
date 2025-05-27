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
from typing import Optional, Union

import transformers
from packaging import version
from transformers import TrainingArguments


@dataclass
class GRPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`GRPOTrainer`].

    Only the parameters specific to GRPO training are listed here. For details on other parameters, refer to the
    [`~transformers.TrainingArguments`] documentation.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`str`, `dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`GRPOTrainer`] is provided as a string.
        disable_dropout (`bool`, *optional*, defaults to `False`):
            Whether to disable dropout in the model. This is useful for training with a reference model, as it
            prevents the model from generating different logprobs for the same input.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
        num_generations (`int` or `None`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. The effective batch size (num_processes *
            per_device_batch_size * gradient_accumulation_steps) must be evenly divisible by this value.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.
        shuffle_dataset (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training dataset.

        > Parameters that control generation

        generation_batch_size: (`int` or `None`, *optional*, defaults to `None`):
            Batch size to use for generation. If `None`, it defaults to the effective training batch size:
            `per_device_train_batch_size * num_processes * gradient_accumulation_steps`.
        steps_per_generations: (`int` or `None`, *optional*, defaults to `None`):
            Number of optimization steps per generation. If `None`, it defaults to gradient_accumulation_steps.
        temperature (`float`, defaults to `1.0`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int` or `None`, *optional*, defaults to `None`):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, top-k-filtering is
            disabled and all tokens are considered.
        min_p (`float` or `None`, *optional*, defaults to `None`):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat
            tokens.
        cache_implementation (`str` or `None`, *optional*, defaults to `None`):
            Implementation of the cache method for faster generation when use_vllm is set to False.

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
        vllm_guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
            Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled.

        > Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)
        vllm_server_base_url (`str` or `None`, *optional*, defaults to `None`):
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

        > Parameters that control the training

        learning_rate (`float`, *optional*, defaults to `1e-6`):
            Initial learning rate for [`AdamW`] optimizer. The default value replaces that of
            [`~transformers.TrainingArguments`].
        beta (`float`, *optional*, defaults to `0.04`):
            KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving training
            speed, but may be numerically unstable for long training runs.
        num_iterations (`int`, *optional*, defaults to `1`):
            Number of iterations per batch (denoted as μ in the algorithm).
        epsilon (`float`, *optional*, defaults to `0.2`):
            Epsilon value for clipping.
        delta: (`float` or `None`, *optional*, defaults to `None`):
            Enables the upper clipping bound in two-sided GRPO loss when set to a float. If `None` (default), standard
            GRPO clipping is used. Recommended to be greater than `1 + ε` when enabled. This method is introduced in
            the [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291).
        epsilon_high (`float` or `None`, *optional*, defaults to `None`):
            Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound
            specified in argument `epsilon`. Paper [DAPO](https://huggingface.co/papers/2503.14476) recommends `0.28`.
        reward_weights (`list[float]` or `None`, *optional*, defaults to `None`):
            Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are
            weighted equally with weight `1.0`.
        scale_rewards (`bool`, *optional*, defaults to `True`):
            Whether to scale the rewards by dividing them by their standard deviation. If `True` (default), the rewards
            are normalized by the standard deviation, ensuring they have unit variance. If `False`, no scaling is
            applied. The [Dr. GRPO paper](https://huggingface.co/papers/2503.20783) recommends not scaling the rewards,
            as scaling by the standard deviation introduces a question-level difficulty bias.
        loss_type (`str`, *optional*, defaults to `"bnpo"`):
            Specifies the loss formulation to use. Supported values are:

            - `"grpo"`: Aggregates token-level losses by normalizing over sequence length. Not recommended due to
                length bias—this approach tends to prefer shorter completions with positive advantages and longer ones
                with negative advantages.
            - `"bnpo"`: Aggregates token-level losses by normalizing number of active token in the local batch.
                Note that normalization is performed over the local batch only, so results may slightly vary depending
                on the local batch size, despite a constant effective batch size. When using
                `per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss.
            - `"dr_grpo"`: Aggregates token-level losses by normalizing with a global constant. This method was
                introduced in the [Dr. GRPO paper](https://huggingface.co/papers/2503.20783) to eliminate length bias.
                The value of the constant corresponds to `max_completion_length`.
        mask_truncated_completions (`bool`, *optional*, defaults to `False`):
            When enabled, truncated completions are excluded from the loss calculation, preventing them from being
            incorrectly penalized and introducing noise during training. According to the
            [DAPO](https://huggingface.co/papers/2503.14476) paper, this is a good practice for training stability.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
            the `ref_model_mixup_alpha` parameter. This synchronization originates from the
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
        use_liger_loss (`bool`, *optional*, defaults to `False`):
            Whether to use the Liger GRPO loss.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is
            installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`.
        num_completions_to_print (`int` or `None`, *optional*, defaults to `None`):
            Number of completions to print with `rich`. If `None`, all completions are logged.
        wandb_log_unique_prompts (`bool`, *optional*, defaults to `False`):
            Whether to log unique prompts in wandb. If `True`, only unique prompts are logged. If `False`, all
            prompts are logged.
    """

    if version.parse(transformers.__version__) <= version.parse("4.50.3"):
        from transformers.training_args import _VALID_DICT_FIELDS

        _VALID_DICT_FIELDS.append("model_init_kwargs")
    else:
        _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters that control the model and reference model
    model_init_kwargs: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )
    disable_dropout: bool = field(
        default=False,
        metadata={
            "help": "Whether to disable dropout in the model. This is useful for training with a reference model, as "
            "it prevents the model from generating different logprobs for the same input."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The effective batch size (num_processes * per_device_batch_size "
            "* gradient_accumulation_steps) must be evenly divisible by this value."
        },
    )
    max_completion_length: Optional[int] = field(
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
    shuffle_dataset: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )

    # Parameters that control generation
    generation_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size to use for generation. If `None`, it defaults to the effective training batch size: "
            "`per_device_train_batch_size * num_processes * gradient_accumulation_steps`."
        },
    )
    steps_per_generation: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of optimization steps per generation. If `None`, it defaults to gradient_accumulation_steps."
        },
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
    top_k: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled and all tokens are considered."
        },
    )
    min_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
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
    cache_implementation: Optional[str] = field(
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
    vllm_server_base_url: Optional[str] = field(
        default=None,
        metadata={
            "help": "Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` "
            "and `vllm_server_port` are ignored."
        },
    )
    vllm_mode: str = field(
        default="server",
        metadata={
            "help": "Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `server` or "
            "`'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure a "
            "TRL vLLM server is running (start with `trl vllm-serve`). `'colocate'`: vLLM will run in the same "
            "process and share the training GPUs. This avoids the need for a separate server but may cause resource "
            "contention with training."
        },
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={"help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."},
    )

    # Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)
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
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    delta: Optional[float] = field(
        default=None,
        metadata={
            "help": "Enables the upper clipping bound in two-sided GRPO loss when set to a float. If `None` "
            "(default), standard GRPO clipping is used. Recommended to be greater than `1 + ε` when enabled. This "
            "method is introduced in the [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291)."
        },
    )
    epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`."
        },
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
    scale_rewards: bool = field(
        default=True,
        metadata={
            "help": "Whether to scale the rewards by dividing them by their standard deviation. If `True` (default), "
            "the rewards are normalized by the standard deviation, ensuring they have unit variance. If `False`, no "
            "scaling is applied. The Dr. GRPO paper recommends not scaling the rewards, as scaling by the standard "
            "deviation introduces a question-level difficulty bias."
        },
    )
    loss_type: str = field(
        default="bnpo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are `grpo`, `bnpo`, and `dr_grpo`. "
            "`'grpo'`: Aggregates token-level losses by normalizing over sequence length. Not recommended due to "
            "length bias—this approach tends to prefer shorter completions with positive advantages and longer ones "
            "with negative advantages. "
            "`'bnpo'`: Aggregates token-level losses by normalizing number of active token in the local batch. "
            "Note that normalization is performed over the local batch only, so results may slightly vary depending "
            "on the local batch size, despite a constant effective batch size. When using "
            "`per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss. "
            "`'dr_grpo'`: Aggregates token-level losses by normalizing with a global constant. This method was "
            "introduced in the Dr. GRPO paper to eliminate length bias. The value of the constant corresponds to "
            "`max_completion_length`."
        },
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={
            "help": "When enabled, truncated completions are excluded from the loss calculation, preventing them from "
            "being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is "
            "a good practice for training stability."
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
    use_liger_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the Liger GRPO loss."},
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

    def __post_init__(self):
        super().__post_init__()

        num_processes = self.world_size
        # The current default effective batch size
        if self.generation_batch_size is not None and self.steps_per_generation is not None:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        if self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps

        if self.generation_batch_size is None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation

        if self.generation_batch_size % self.per_device_train_batch_size * num_processes != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                f"({self.per_device_train_batch_size * num_processes})."
            )

        self.steps_per_generation = self.generation_batch_size // (self.per_device_train_batch_size * num_processes)

        # Check if the effective batch size can be divided by the number of generations
        if self.num_generations < 2:
            raise ValueError(
                "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
            )
        possible_values = [
            n_gen for n_gen in range(2, self.generation_batch_size + 1) if (self.generation_batch_size) % n_gen == 0
        ]

        if self.num_generations not in possible_values:
            raise ValueError(
                f"The effective train batch size ({num_processes} x {self.per_device_train_batch_size} x "
                f"{self.steps_per_generation}) must be evenly divisible by the number of generations per "
                f"prompt ({self.num_generations}). Given the current effective train batch size, the valid values for "
                f"the number of generations are: {possible_values}."
            )
        if self.eval_strategy != "no":
            global_eval_batch_size = self.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen for n_gen in range(2, global_eval_batch_size + 1) if (global_eval_batch_size) % n_gen == 0
            ]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {self.per_device_eval_batch_size}) must be "
                    f"evenly divisible by the number of generations per prompt ({self.num_generations}). Given the "
                    "current global eval batch size, the valid values for the number of generations are: "
                    f"{possible_values}."
                )
        if self.delta is not None and self.use_liger_loss:
            raise ValueError("Liger loss does not support two-sided GRPO loss yet.")
