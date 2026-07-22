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
class AsyncGRPOConfig(_BaseConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`AsyncGRPOTrainer`].

    This class includes only the parameters that are specific to asynchronous GRPO training. For a full list of
    training arguments, please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values
    in this class may differ from those in [`~transformers.TrainingArguments`].

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]` or `str`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when instantiating the
            model from a path.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to
            [`~transformers.AutoModelForCausalLM.from_pretrained`] and [`~transformers.AutoTokenizer.from_pretrained`].
        router_aux_loss_coef (`float`, *optional*, defaults to `0.001`):
            Coefficient of the load-balancing auxiliary loss. Only has an effect when training a Mixture-of-Experts
            (MoE) model; for other models it does nothing. The auxiliary loss is added to the training loss with this
            weight. Set to `0.0` to disable it.

        > Parameters that control generation

        num_generations (`int`, *optional*, defaults to `8`):
            Number of generations per prompt to sample.
        max_completion_length (`int`, *optional*, defaults to `2048`):
            Maximum length of the generated completion.
        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        chat_template_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the `apply_chat_template` function when generating completions.
        max_tool_calling_iterations (`int`, *optional*):
            Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and generation
            stops when the model generates a response turn with no tool calls or when the total response length reaches
            `max_completion_length`.
        fork_threshold_tokens (`int`, *optional*, defaults to `1024`):
            A multi-turn conversation is turned into training rows by re-tokenizing the whole conversation every turn
            and reconciling the result against the tokens held so far: a clean append stays one row, a rewrite (dropped
            reasoning, summarized history) forks a new row. When a turn's re-tokenized prompt drifts inside the last
            generated answer, the decision is made on the **drift size** — how many previously-trained tokens the
            realign would mask to context. A drift smaller than this many tokens is treated as a re-tokenization
            wobble (realigned as context); a larger drift — e.g. a long reasoning block dropped by the template —
            forks a new row so those trained tokens keep their training signal instead of being silently masked.

        > Parameters that control the vLLM server

        vllm_server_base_url (`str`, *optional*, defaults to `"http://localhost:8000"`):
            Base URL of the vLLM server used for generation (e.g., `"http://localhost:8000"`).
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Total timeout duration in seconds to wait for the vLLM server to be ready.
        request_timeout (`int`, *optional*, defaults to `600`):
            Timeout in seconds for individual HTTP requests to the vLLM server.

        > Parameters that control the training

        epsilon (`float`, *optional*, defaults to `0.2`):
            Epsilon value for clipping.
        epsilon_high (`float`, *optional*):
            Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound
            specified in argument `epsilon`. Paper [DAPO](https://huggingface.co/papers/2503.14476) recommends `0.28`.
        token_budget (`int`, *optional*):
            Maximum number of real tokens packed into a single row (one DP rank's forward) for dynamic
            token-budgeted micro-batching. When `> 0`, a `TokenBudgetBatcher` forms Σ Lᵢ²-balanced micro-batches
            whose rows each stay within this budget, bounding peak memory independently of the sample count (the
            number of samples per row becomes dynamic). If `None` (default), it is set to the vLLM server's
            `max_model_len` (queried at train start) — the cap on prompt + completion length — so no rollout sample
            can ever exceed the budget. A sample longer than `token_budget` fits in no row and is dropped with a
            warning. Set `<= 0` to disable token budgeting and instead pack a fixed `per_device_train_batch_size ×
            num_processes` samples per micro-batch, Σ Lᵢ²-balanced across the rows.

        > Parameters that control the async rollout pipeline

        max_inflight_tasks (`int`, *optional*, defaults to `-1`):
            Maximum number of concurrent generation tasks sent to the vLLM server. Defaults to `-1` (auto), which
            sets it to `max_staleness * per_device_train_batch_size * gradient_accumulation_steps * num_processes`.
            If using tool-use environments, you may want to set this manually based on how many parallel environments
            you can run.
        max_staleness (`int`, *optional*, defaults to `4`):
            Maximum number of weight update steps a rollout sample can lag behind the current model version before
            being discarded.
        queue_maxsize (`int`, *optional*, defaults to `1024`):
            Maximum number of rollout samples to buffer in the rollout queue.
        weight_sync_steps (`int`, *optional*, defaults to `1`):
            Number of training steps between weight synchronizations to the vLLM server.
        heartbeat_stale_after_s (`float`, *optional*, defaults to `300.0`):
            Seconds since the rollout worker's last heartbeat after which the trainer treats it as
            hung and aborts.

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
    > - `lr_scheduler_type`: Defaults to `constant` instead of `linear` (see below).

    > [!NOTE]
    > Training duration and learning rate under message-mode reconciliation:
    > A multi-turn conversation can fork into a variable number of training rows (a rewrite of the conversation
    > starts a new row), so the number of samples, and therefore the number of optimizer steps, per epoch is not
    > known up front. As a consequence:
    > - `num_train_epochs` bounds training by full passes over the *prompt* dataset, counted as the number of
    >   distinct prompts actually trained on. This is independent of how many rows the forks produce, so requesting
    >   N epochs always trains on N passes over the data. When `max_steps` is left unset, this is the stop condition
    >   and `max_steps` is only a safety ceiling.
    > - `max_steps`, if set explicitly (`> 0`), takes over as the stop condition (bounding by optimizer steps rather
    >   than by epochs) and disables the epoch-based stop.
    > - `lr_scheduler_type` defaults to `constant` because a decay horizon is measured in optimizer steps, which
    >   cannot be known up front when the step count depends on the fork rate. For a decaying learning rate, set a
    >   decaying schedule together with an explicit `max_steps`.
    """

    _VALID_DICT_FIELDS = _BaseConfig._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters that control the model
    model_init_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when instantiating "
            "the model from a path."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow loading models and tokenizers that ship custom Python code from the Hub. "
            "Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`."
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
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning-rate schedule. Defaults to `constant`: when training is bounded by `num_train_epochs`, "
            "message-mode forks make the total step count unknown up front, so `max_steps` is only a safety ceiling "
            "and a decay horizon can't be calibrated. Set a decaying schedule (e.g. `cosine`) together with an "
            "explicit `max_steps` if you want LR decay."
        },
    )

    # Parameters that control generation
    num_generations: int = field(
        default=8,
        metadata={"help": "Number of generations per prompt to sample."},
    )
    max_completion_length: int = field(
        default=2048,
        metadata={"help": "Maximum length of the generated completion."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    chat_template_kwargs: dict | None = field(
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

    # Parameters that control the vLLM server
    vllm_server_base_url: str = field(
        default="http://localhost:8000",
        metadata={"help": "Base URL of the vLLM server used for generation (e.g., 'http://localhost:8000')."},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={
            "help": "Total timeout duration in seconds to wait for the vLLM server to be ready. If the server is not "
            "up after the timeout, a `TimeoutError` is raised."
        },
    )
    request_timeout: int = field(
        default=600,
        metadata={"help": "Timeout in seconds for individual HTTP requests to the vLLM server."},
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
    token_budget: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of real tokens packed into a single row (one DP rank's forward) for dynamic "
            "token-budgeted micro-batching. When > 0, a `TokenBudgetBatcher` forms Σ Lᵢ²-balanced micro-batches "
            "whose rows each stay within this budget, bounding peak memory independently of the sample count. If "
            "None (default), it is set to the vLLM server's `max_model_len` (queried at train start), so no "
            "rollout sample can ever exceed the budget. A sample longer than `token_budget` fits in no row and is "
            "dropped with a warning. Set <= 0 to disable token budgeting and instead pack a fixed "
            "`per_device_train_batch_size × num_processes` samples per micro-batch, Σ Lᵢ²-balanced across the rows."
        },
    )

    # Parameters that control the async rollout pipeline
    max_inflight_tasks: int = field(
        default=-1,
        metadata={
            "help": "Maximum number of concurrent generation tasks sent to the vLLM server. Defaults to -1 (auto), "
            "which sets it to `max_staleness * per_device_train_batch_size * gradient_accumulation_steps * "
            "num_processes`. Generating more samples than this is wasteful since they will be discarded as stale "
            "before the trainer can consume them. If using tool-use environments, you may want to set this manually "
            "based on how many parallel environments you can run."
        },
    )
    max_staleness: int = field(
        default=4,
        metadata={
            "help": "Maximum number of weight update steps a rollout sample can lag behind the current model version "
            "before being discarded."
        },
    )
    queue_maxsize: int = field(
        default=1024,
        metadata={"help": "Maximum number of rollout samples to buffer in the rollout queue."},
    )
    weight_sync_steps: int = field(
        default=1,
        metadata={"help": "Number of training steps between weight synchronizations to the vLLM server."},
    )
    heartbeat_stale_after_s: float = field(
        default=300.0,
        metadata={
            "help": "Seconds since the rollout worker's last heartbeat after which the trainer treats it as hung "
            "and aborts."
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

        # Accelerator config: required for the async IterableDataset-backed dataloader to work correctly.
        # split_batches=True and dispatch_batches=True ensure that the main process drives the dataloader
        # and batches are broadcast to other processes rather than each process pulling independently.
        if not hasattr(self, "accelerator_config") or self.accelerator_config is None:
            self.accelerator_config = {"split_batches": True, "dispatch_batches": True}
        elif isinstance(self.accelerator_config, dict):
            self.accelerator_config["split_batches"] = True
            self.accelerator_config["dispatch_batches"] = True
        else:
            self.accelerator_config.split_batches = True
            self.accelerator_config.dispatch_batches = True
