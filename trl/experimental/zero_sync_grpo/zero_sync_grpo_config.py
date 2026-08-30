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
    Configuration class for the [`ZeroSyncGRPOTrainer`].

    This class includes only the parameters that are specific to zero-sync GRPO training. For a full list of training
    arguments, please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this
    class may differ from those in [`~transformers.TrainingArguments`].

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]` or `str`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`ZeroSyncGRPOTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.

        > Parameters that control generation

        num_generations (`int`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. The effective batch size (num_processes *
            per_device_batch_size * gradient_accumulation_steps) must be evenly divisible by this value.
        max_completion_length (`int`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int`, *optional*):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, top-k-filtering is
            disabled and all tokens are considered.
        chat_template_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the `apply_chat_template` function when generating completions.
        max_tool_calling_iterations (`int`, *optional*):
            Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and generation
            stops when the model generates a response turn with no tool calls or when the total response length
            reaches `max_completion_length`.
        rollouts_in_flight (`int`, *optional*, defaults to `32`):
            Number of rollouts the engine keeps generating at any moment. A rollout that finishes is replaced
            immediately by the next one, so the decode batch stays full instead of emptying and refilling once per
            optimizer step. The cost of a decode step is mostly reading the weights and the per-layer all-reduces,
            which do not depend on how many sequences are decoded, so a fuller batch is close to free throughput. Set
            it as high as the KV cache holds: past that, rollouts crowd the cache and decode slows. Completions lag
            the policy by roughly this many rollouts divided by the batch trained on per step.
        continuous_batching_config (`dict`, *optional*):
            Keyword arguments for [`~transformers.generation.ContinuousBatchingConfig`]. `return_logprobs` is always
            forced to `True`: the engine's logprobs are the behavior-policy logprobs of the completions.

        > Parameters that control the training

        packed_training (`bool`, *optional*, defaults to `False`):
            Pack the training samples back to back instead of padding them to the longest in the batch. Position ids
            restart at each sample and no attention mask is passed, so the model builds the block-diagonal mask
            itself; with a block-sparse attention implementation this removes the pad-token compute entirely, so
            unless `model_init_kwargs` sets an `attn_implementation`, packed training defaults it to
            `flex_attention` (flash attention for models with linear attention layers, whose kernels read the
            sample boundaries from the varlen kwargs and take the samples in one flat row). The loss is
            unchanged: each token keeps its own sample's advantage and behavior logprob.
        epsilon (`float`, *optional*, defaults to `0.2`):
            Epsilon value for clipping.

    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `10` instead of `500`.
    > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
    > - `learning_rate`: Defaults to `1e-6` instead of `5e-5`.
    """

    _VALID_DICT_FIELDS = _BaseConfig._VALID_DICT_FIELDS + [
        "model_init_kwargs",
        "continuous_batching_config",
        "chat_template_kwargs",
    ]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )
    bf16: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if "
            "`fp16` is not set."
        },
    )

    # Parameters that control the model
    model_init_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `ZeroSyncGRPOTrainer` is provided as a string."
        },
    )

    # Parameters that control the data preprocessing
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )

    # Parameters that control generation
    num_generations: int = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The effective batch size (num_processes * per_device_batch_size "
            "* gradient_accumulation_steps) must be evenly divisible by this value."
        },
    )
    max_completion_length: int = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
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
    top_k: int | None = field(
        default=None,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled and all tokens are considered."
        },
    )
    chat_template_kwargs: dict[str, Any] | None = field(
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
            "generation stops when the model generates a response turn with no tool calls or when the total response "
            "length reaches `max_completion_length`."
        },
    )
    rollouts_in_flight: int = field(
        default=32,
        metadata={
            "help": "Number of rollouts the engine keeps generating at any moment. A rollout that finishes is "
            "replaced immediately by the next one, so the decode batch stays full instead of emptying and refilling "
            "once per optimizer step. Set it as high as the KV cache holds: past that, rollouts crowd the cache and "
            "decode slows."
        },
    )
    continuous_batching_config: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.generation.ContinuousBatchingConfig`. `return_logprobs` is "
            "always forced to `True`: the engine's logprobs are the behavior-policy logprobs of the completions."
        },
    )

    tp_size: int = field(
        default=1,
        metadata={
            "help": "Number of processes the model is split across (tensor parallelism). With more than one, the "
            "trainer drives generation itself, between its own steps, rather than letting the engine run in its "
            "background thread: every rank then issues its collectives in the same order, which is what NCCL "
            "requires."
        },
    )
    packed_training: bool = field(
        default=False,
        metadata={
            "help": "Pack the training samples back to back instead of padding them to the longest in the batch. "
            "Position ids restart at each sample and no attention mask is passed, so the model builds the "
            "block-diagonal mask itself; with a block-sparse attention implementation this removes the pad-token "
            "compute entirely, so unless `model_init_kwargs` sets an `attn_implementation`, packed training "
            "defaults it to `flex_attention` (flash attention for models with linear attention layers, whose "
            "kernels read the sample boundaries from the varlen kwargs and take the samples in one flat row). "
            "The loss is unchanged: each token keeps its own sample's advantage and behavior logprob."
        },
    )
    release_kv_cache_during_step: bool = field(
        default=False,
        metadata={
            "help": "Free the generation KV cache while the training step runs, so the forward and backward can use "
            "that memory, and reallocate it before generating again. Only with `tp_size > 1`, where the trainer owns "
            "the engine and generation is quiescent during the step. Live rollouts are copied to host memory and back "
            "each step, so this trades throughput for the room to train on longer completions or larger batches. "
            "Requires `use_async_batching=False` in `continuous_batching_config`."
        },
    )

    # Parameters that control the training
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16
        super().__post_init__()
