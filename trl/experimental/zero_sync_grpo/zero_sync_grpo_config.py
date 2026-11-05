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

        model_init_kwargs (`str` or `dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`ZeroSyncGRPOTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.

        > Parameters that control generation

        num_generations (`int`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. They form the group the advantages are computed over.
        max_completion_length (`int`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int`, *optional*, defaults to `0`):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, top-k-filtering is
            disabled and all tokens are considered.
        chat_template_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the `apply_chat_template` function when generating completions.
        max_tool_calling_iterations (`int`, *optional*):
            Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and generation
            stops when the model generates a response turn with no tool calls or when the total response length
            reaches `max_completion_length`.
        fork_threshold_tokens (`int`, *optional*, defaults to `1024`):
            A multi-turn conversation is turned into training rows by re-tokenizing the whole conversation every turn
            and reconciling the result against the tokens held so far: a clean append stays one row, a rewrite (dropped
            reasoning, summarized history) forks a new row. When a turn's re-tokenized prompt drifts inside the last
            generated answer, the decision is made on the **drift size** — how many previously-trained tokens the
            realign would mask to context. A drift smaller than this many tokens is treated as a re-tokenization wobble
            (realigned as context); a larger drift — e.g. a long reasoning block dropped by the template — forks a new
            row so those trained tokens keep their training signal instead of being silently masked.
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

        epsilon (`float`, *optional*, defaults to `0.2`):
            Epsilon value for clipping.
        epsilon_high (`float`, *optional*):
            Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound
            specified in argument `epsilon`. Paper [DAPO](https://huggingface.co/papers/2503.14476) recommends `0.28`.

    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `10` instead of `500`.
    > - `gradient_checkpointing`: Defaults to `True` instead of `False`.
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

    # Parameters that control the model
    model_init_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `ZeroSyncGRPOTrainer` is provided as a string."
        },
    )

    # Parameters that control the data preprocessing
    remove_unused_columns: bool | None = field(
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
            "help": "Number of generations to sample. They form the group the advantages are computed over."
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
            "generation stops when the model generates a response turn with no tool calls or when the total response "
            "length reaches `max_completion_length`."
        },
    )
    fork_threshold_tokens: int = field(
        default=1024,
        metadata={
            "help": "A multi-turn conversation is reconciled into training rows by re-tokenizing the whole "
            "conversation every turn: a clean append stays one row, a rewrite forks a new row. A re-tokenization "
            "drift inside the last answer smaller than this many tokens (measured as the number of previously-trained "
            "tokens the realign would mask) is realigned as context; a larger drift forks a new row."
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
    continuous_batching_config: dict | str | None = field(
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
