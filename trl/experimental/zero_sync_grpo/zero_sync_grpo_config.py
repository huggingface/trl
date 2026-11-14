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

from ...trainer.base_config import _BaseConfig


@dataclass
class ZeroSyncGRPOConfig(_BaseConfig):
    r"""
    Configuration class for the [`experimental.zero_sync_grpo.ZeroSyncGRPOTrainer`].

    This class includes only the parameters that are specific to zero-sync GRPO training. For a full list of training
    arguments, please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this
    class may differ from those in [`~transformers.TrainingArguments`].

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]` or `str`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when instantiating the
            model from a path. The `revision` value is also used when loading the processing class.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to
            [`~transformers.AutoModelForCausalLM.from_pretrained`] and [`~transformers.AutoTokenizer.from_pretrained`].
        router_aux_loss_coef (`float`, *optional*):
            Coefficient of the load-balancing auxiliary loss for Mixture-of-Experts (MoE) models, added to the training
            loss with this weight. When not set, the value declared by the model config is used. Set to `0.0` to
            disable it. Fails when used with a non-MoE model.

        > Parameters that control generation

        num_generations (`int`, *optional*, defaults to `8`):
            Number of generations per prompt to sample.
        max_completion_length (`int`, *optional*, defaults to `2048`):
            Maximum number of tokens generated per engine request, so per turn in a multi-turn rollout.
        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1.0
            to consider all tokens.
        top_k (`int`, *optional*, defaults to `0`):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, top-k-filtering is
            disabled and all tokens are considered.
        chat_template_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the `apply_chat_template` function when generating completions.
        max_tool_calling_iterations (`int`, *optional*):
            Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and generation
            stops when the model generates a response turn with no tool calls.

        > Parameters that control the generation engine

        continuous_batching_config (`dict`, *optional*):
            Keyword arguments for [`~transformers.generation.ContinuousBatchingConfig`]. `return_logprobs` is always
            forced to `True`: the engine's logprobs are the behavior-policy logprobs of the completions.
        kv_cache_during_step (`str`, *optional*, defaults to `"release"`):
            What happens to the KV cache's GPU memory during the training step. Supported values are:

            - `"keep"`: the cache stays on the device, for small models or short completions, where memory is not the
              limit.
            - `"release"` (default): the cache waits in host memory and comes back before generation resumes, so the
              rollouts in flight carry on where they were, and the forward and backward get its memory. This copies the
              cache to the host and back at every step.
            - `"lend"`: the cache's memory is lent to the step with `sublet`. The step takes the cache's free blocks
              first, then free GPU memory, then the blocks of rollouts in flight, which are copied to the host and
              back. Only what the step overwrites moves. Needs `pip install 'sublet>=0.2'`.
        tp_size (`int`, *optional*, defaults to `1`):
            Number of processes the model is split across (tensor parallelism). The processes it is not split across
            hold replicas of it, which train on their own prompts and average their gradients.

        > Parameters that control the training

        epsilon (`float`, *optional*, defaults to `0.2`):
            Epsilon value for clipping.
        epsilon_high (`float`, *optional*):
            Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound
            specified in argument `epsilon`. Paper [DAPO](https://huggingface.co/papers/2503.14476) recommends `0.28`.
        importance_sampling_clip_min (`float`, *optional*, defaults to `0.5`):
            Lower bound C_min of the importance ratio between the policy and the engine that sampled. Tokens whose
            ratio falls outside [`importance_sampling_clip_min`, `importance_sampling_clip_max`] are dropped from the
            loss (masked importance sampling).
        importance_sampling_clip_max (`float`, *optional*, defaults to `5.0`):
            Upper bound C_max of the importance ratio.

        > Parameters that control the rollout loop

        max_concurrent_rollouts (`int`, *optional*, defaults to `32`):
            Maximum number of rollouts in flight at once. A rollout that finishes is replaced immediately by the next
            one, so the decode batch stays full instead of emptying and refilling once per optimizer step. The cost of
            a decode step is mostly reading the weights and the per-layer all-reduces, which do not depend on how many
            sequences are decoded, so a fuller batch is close to free throughput. Set it as high as the KV cache holds:
            past that, rollouts crowd the cache and decode slows. Completions lag the policy by roughly this many
            rollouts divided by the batch trained on per step.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps.
        num_completions_to_print (`int`, *optional*):
            Number of completions to print with `rich`. If `None`, all completions are logged.

    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `1` instead of `500`.
    > - `gradient_checkpointing`: Defaults to `True` instead of `False`.
    > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
    > - `learning_rate`: Defaults to `1e-6` instead of `5e-5`.
    > - `ignore_data_skip`: Defaults to `True` instead of `False`; the base Trainer's skip-and-replay loop does not apply to the rollout stream.
    > - `remove_unused_columns`: Defaults to `False` instead of `True`, and must stay `False`: every dataset column is passed to the reward functions.
    """

    _VALID_DICT_FIELDS = _BaseConfig._VALID_DICT_FIELDS + [
        "model_init_kwargs",
        "chat_template_kwargs",
        "continuous_batching_config",
    ]

    # Parameters that control the model
    model_init_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when instantiating "
            "the model from a path. The `revision` value is also used when loading the processing class."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow loading models and tokenizers that ship custom Python code from the Hub. "
            "Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`."
        },
    )
    router_aux_loss_coef: float | None = field(
        default=None,
        metadata={
            "help": "Coefficient of the load-balancing auxiliary loss for Mixture-of-Experts (MoE) models, added to "
            "the training loss with this weight. When not set, the value declared by the model config is used. Set "
            "to `0.0` to disable it. Fails when used with a non-MoE model."
        },
    )

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    logging_steps: float = field(
        default=1,
        metadata={
            "help": "Log every X update steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )
    ignore_data_skip: bool = field(
        default=True,
        metadata={
            "help": "Always `True` for ZeroSyncGRPO; the base Trainer's skip-and-replay loop does not apply to a live "
            "rollout stream."
        },
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "Must be `False` for ZeroSyncGRPO: every dataset column is passed to the reward functions. Remove "
            "the columns you don't need from the dataset instead."
        },
    )

    # Parameters that control generation
    num_generations: int = field(
        default=8,
        metadata={"help": "Number of generations per prompt to sample."},
    )
    max_completion_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum number of tokens generated per engine request, so per turn in a multi-turn rollout."
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
    top_k: int = field(
        default=0,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, "
            "top-k-filtering is disabled and all tokens are considered."
        },
    )
    chat_template_kwargs: dict | str | None = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to the `apply_chat_template` function when generating "
            "completions."
        },
    )
    max_tool_calling_iterations: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and "
            "generation stops when the model generates a response turn with no tool calls."
        },
    )

    # Parameters that control the generation engine
    continuous_batching_config: dict | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.generation.ContinuousBatchingConfig`. `return_logprobs` is "
            "always forced to `True`: the engine's logprobs are the behavior-policy logprobs of the completions."
        },
    )

    kv_cache_during_step: str = field(
        default="release",
        metadata={
            "help": "What happens to the KV cache's GPU memory during the training step. 'keep': the cache stays on "
            "the device, for small models or short completions, where memory is not the limit. 'release' (default): "
            "the cache waits in host memory and comes back before generation resumes, so the rollouts in flight carry "
            "on where they were, and the forward and backward get its memory. This copies the cache to the host and "
            "back at every step. 'lend': the cache's memory is lent to the step with `sublet`. The step takes the "
            "cache's free blocks first, then free GPU memory, then the blocks of rollouts in flight, which are copied "
            "to the host and back. Only what the step overwrites moves. Needs `pip install 'sublet>=0.2'`.",
            "choices": ["keep", "release", "lend"],
        },
    )
    tp_size: int = field(
        default=1,
        metadata={
            "help": "Number of processes the model is split across (tensor parallelism). The processes it is not "
            "split across hold replicas of it, which train on their own prompts and average their gradients."
        },
    )

    # Parameters that control the training
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    epsilon_high: float | None = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`."
        },
    )
    importance_sampling_clip_min: float = field(
        default=0.5,
        metadata={
            "help": "Lower bound C_min of the importance ratio between the policy and the engine that sampled. "
            "Tokens whose ratio falls outside [`importance_sampling_clip_min`, `importance_sampling_clip_max`] are "
            "dropped from the loss."
        },
    )
    importance_sampling_clip_max: float = field(
        default=5.0,
        metadata={"help": "Upper bound C_max of the importance ratio."},
    )

    # Parameters that control the rollout loop
    max_concurrent_rollouts: int = field(
        default=32,
        metadata={
            "help": "Maximum number of rollouts in flight at once. A rollout that finishes is replaced immediately by "
            "the next one, so the decode batch stays full instead of emptying and refilling once per optimizer step. "
            "Set it as high as the KV cache holds: past that, rollouts crowd the cache and decode slows."
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
    num_completions_to_print: int | None = field(
        default=None,
        metadata={"help": "Number of completions to print with `rich`. If `None`, all completions are logged."},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.remove_unused_columns:
            raise ValueError(
                "`remove_unused_columns=True` is not supported: ZeroSyncGRPO passes every dataset column to the reward "
                "functions. Remove the columns you don't need from the dataset before passing it to the trainer, for "
                "example with `dataset.remove_columns(...)`."
            )

        if self.num_generations < 2:
            raise ValueError(
                "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
            )

        if self.parallelism_config is not None and (
            self.parallelism_config.cp_enabled or self.parallelism_config.sp_enabled
        ):
            raise ValueError(
                "ZeroSyncGRPOTrainer does not support sequence-dim parallelism (`parallelism_config.cp_size > 1` "
                "or `parallelism_config.sp_size > 1`) yet. GRPO builds model inputs after generation "
                "inside the trainer, so Transformers' context-parallel / Ulysses sequence-parallel input "
                "sharding cannot be applied to the raw generation batch. Set both `cp_size=1` and `sp_size=1`, "
                "or disable `parallelism_config`."
            )
