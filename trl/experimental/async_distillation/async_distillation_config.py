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
class AsyncDistillationConfig(_BaseConfig):
    r"""
    Configuration class for the [`AsyncDistillationTrainer`].

    This class includes only the parameters that are specific to asynchronous on-policy distillation. For a full list
    of training arguments, please refer to the [`~transformers.TrainingArguments`] documentation. Note that default
    values in this class may differ from those in [`~transformers.TrainingArguments`]. Its structure mirrors
    [`~trl.experimental.async_grpo.AsyncGRPOConfig`] (async pipeline, vLLM server, logging fields are the same), with
    GRPO's clipping/group fields replaced by the teacher-distillation loss fields below.

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]` or `str`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when instantiating the
            student model from a path.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to
            [`~transformers.AutoModelForCausalLM.from_pretrained`] and [`~transformers.AutoTokenizer.from_pretrained`].

        > Parameters that control generation

        max_completion_length (`int`, *optional*, defaults to `2048`):
            Maximum number of tokens to generate per completion.
        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for sampling the student's on-policy completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Top-p (nucleus) sampling parameter for on-policy generation.
        top_k (`int`, *optional*, defaults to `0`):
            Top-k sampling parameter for on-policy generation. `0` disables top-k filtering.
        min_p (`float`, *optional*):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat
            tokens.
        chat_template_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the `apply_chat_template` function when generating completions.

        > Parameters that control the vLLM servers

        vllm_server_base_url (`str`, *optional*, defaults to `"http://localhost:8000"`):
            Base URL of the student's vLLM server, used both for generation and for streaming weight updates.
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Total timeout duration in seconds to wait for the student's vLLM server to be ready.
        teacher_server_urls (`dict[str, str]`, *optional*, defaults to `{"default": "http://localhost:8001"}`):
            Teacher vLLM server(s), each started with `vllm serve <teacher-model> --logprobs-mode processed_logprobs
            --max-logprobs -1`. Every teacher is static: this trainer never streams weight updates to any of them, so a
            teacher needs neither `VLLM_SERVER_DEV_MODE` nor a weight-transfer backend, only those two flags, which
            make its scoring exact (see `teacher_temperature` and `teacher_top_k`). Every teacher must share the
            student's tokenizer: completions are sent as raw token ids, and the ids a teacher reports back are used
            directly to index the student's own vocabulary in `compute_loss`. A teacher with a different vocabulary
            trains against the wrong tokens, silently if its vocabulary is no larger than the student's. Scoring is a
            teacher-forced request against the student's own completion tokens (`max_tokens=1`,
            `prompt_logprobs=teacher_top_k`, `temperature=teacher_temperature`), the same request
            [`~trl.generation.vllm_client.VLLMClient.get_sequence_logprobs`] issues for the synchronous server-teacher
            trainers. A single entry scores every sample (plain single-teacher on-policy distillation, the default).
            Multiple entries enable MOPD (multi-teacher on-policy distillation, see
            [MOPD](https://huggingface.co/papers/2606.30406)): each training row's `teacher_id` column selects which
            entry scores it, e.g. `{"math": "http://localhost:8002", "code": "http://localhost:8003"}` with a
            `teacher_id` of `"math"` or `"code"` per row.
        request_timeout (`int`, *optional*, defaults to `600`):
            Timeout in seconds for individual HTTP requests to any vLLM server.
        weight_sync_timeout (`int`, *optional*, defaults to `1800`):
            Timeout in seconds for a weight transfer to the student's vLLM server. A transfer that does not complete
            within this time raises instead of hanging the run.

        > Parameters that control the distillation loss

        beta (`float`, *optional*, defaults to `0.0`):
            Interpolation coefficient for the generalized Jensen-Shannon Divergence. `0.0` is forward KL
            (mean-seeking), `1.0` is reverse KL (mode-seeking), and values in between interpolate, following the same
            generalized-JSD formulation [`~trl.experimental.distillation.DistillationTrainer`] computes internally. The
            support the divergence is computed over differs by regime, mirroring
            [`~trl.experimental.server_distillation.ServerDistillationTrainer`]: at `beta=0.0`, the full
            `teacher_top_k`-wide teacher-reported support (plus tail bucket) is used, since forward KL's weighting is
            exactly what that support provides. At `beta != 0.0`, the support is narrowed to just two candidates — the
            teacher's own top-1 token and the completion's actual/realized token — since those are the only two token
            identities the wire protocol guarantees a teacher logprob for without transmitting a wider (or full)
            vocabulary; anything wider would only be a probabilistic approximation of covering the student's own likely
            tokens, not a guarantee.
        teacher_temperature (`float`, *optional*, defaults to `1.0`):
            Softmax temperature of the divergence, applied to *both* sides: sent to the teacher so vLLM computes its
            logprobs at this temperature server-side (exact, not a client-side rescaling), and applied to the student's
            own logits in `compute_loss`, mirroring
            [`~trl.experimental.server_distillation.ServerDistillationTrainer`]'s single `temperature`. Unrelated to
            `temperature`, which only controls how the student samples its completions. The teacher's server must run
            with `--logprobs-mode processed_logprobs` for this to reach its returned logprobs at all; without it the
            teacher silently reports raw logprobs and this setting only affects the student's side.
        teacher_top_k (`int`, *optional*, defaults to `8`):
            Number of per-position candidate tokens requested from the teacher via `prompt_logprobs`. Only these
            candidates (plus the realized token, which vLLM always reports even when it falls outside the top-k, plus a
            tail bucket capturing the remaining probability mass, see `add_tail_bucket`) are used to approximate the
            teacher's distribution, the full vocabulary is never transmitted over HTTP. The student side of the
            divergence is exact (computed locally, not approximated), since the student is the model being trained and
            its full logits are already available in `compute_loss`. `8` is a light default for smoke testing;
            production on-policy-distillation setups in adjacent RL frameworks use a sparse teacher support in roughly
            the same range (miles defaults to `16`, EasyOPD to `64`), so raising this towards `16`–`64` is reasonable
            once training for real. Anything above `20` requires the teacher's server to have been started with
            `--max-logprobs -1`, which lifts vLLM's default per-token logprob cap.
        add_tail_bucket (`bool`, *optional*, defaults to `True`):
            Whether to append a tail bucket representing the remaining probability mass outside `teacher_top_k`, to
            avoid a trivially small divergence when `teacher_top_k` is small.
        token_budget (`int`, *optional*):
            Maximum number of real tokens packed into a single row (one DP rank's forward) for dynamic token-budgeted
            micro-batching. When `> 0`, a `TokenBudgetBatcher` forms Σ Lᵢ²-balanced micro-batches whose rows each stay
            within this budget, bounding peak memory independently of the sample count (the number of samples per row
            becomes dynamic). If `None` (default), it is set to the student vLLM server's `max_model_len` (queried at
            train start) — the cap on prompt + completion length — so no rollout sample can ever exceed the budget. A
            sample longer than `token_budget` fits in no row and is dropped with a warning, counted as
            `batch/dropped_oversize_total`. Set `<= 0` to disable token budgeting and instead pack a fixed
            `per_device_train_batch_size × num_processes` samples per micro-batch, Σ Lᵢ²-balanced across the rows.

        > Parameters that control the async rollout pipeline

        max_inflight_tasks (`int`, *optional*, defaults to `-1`):
            Maximum number of concurrent generation+scoring tasks in flight against the two vLLM servers. Defaults to
            `-1` (auto), which sets it to `max_staleness * per_device_train_batch_size * gradient_accumulation_steps *
            num_processes`.
        max_staleness (`int`, *optional*, defaults to `4`):
            Maximum number of weight update steps a rollout sample can lag behind the current model version before
            being discarded.
        queue_maxsize (`int`, *optional*, defaults to `1024`):
            Maximum number of rollout samples to buffer in the rollout queue.
        weight_sync_steps (`int`, *optional*, defaults to `1`):
            Number of training steps between weight synchronizations to the student's vLLM server.
        heartbeat_stale_after_s (`float`, *optional*, defaults to `300.0`):
            Seconds since the rollout worker's last heartbeat after which the trainer treats it as hung and aborts.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `log_completions_steps` samples scored.
        log_completions_steps (`int`, *optional*, defaults to `100`):
            Number of scored samples between logging completions. Only used if `log_completions` is `True`. Counted in
            samples the rollout worker has scored, not optimizer steps: the worker runs in a separate process from the
            trainer and has no visibility into `self.state.global_step`.
        num_completions_to_print (`int`, *optional*):
            Number of completions to print with `rich`. If `None`, all completions are logged.

    > [!NOTE] > These parameters have default values different from [`~transformers.TrainingArguments`]: > -
    `logging_steps`: Defaults to `1` instead of `500`. > - `gradient_checkpointing`: Defaults to `True` instead of
    `False`. > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`. > - `learning_rate`: Defaults to
    `1e-6` instead of `5e-5`.
    """

    _VALID_DICT_FIELDS = _BaseConfig._VALID_DICT_FIELDS + ["model_init_kwargs", "teacher_server_urls"]

    # Parameters that control the model
    model_init_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when "
            "instantiating the student model from a path."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow loading models and tokenizers that ship custom Python code from the Hub. "
            "Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`."
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

    # Parameters that control generation
    max_completion_length: int = field(
        default=2048,
        metadata={"help": "Maximum number of tokens to generate per completion."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sampling the student's on-policy completions."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p (nucleus) sampling parameter for on-policy generation."},
    )
    top_k: int = field(
        default=0,
        metadata={"help": "Top-k sampling parameter for on-policy generation. 0 disables top-k filtering."},
    )
    min_p: float | None = field(
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
    chat_template_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to the `apply_chat_template` function when generating "
            "completions."
        },
    )

    # Parameters that control the vLLM servers
    vllm_server_base_url: str = field(
        default="http://localhost:8000",
        metadata={"help": "Base URL of the student's vLLM server (e.g., 'http://localhost:8000')."},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={"help": "Total timeout duration in seconds to wait for the student's vLLM server to be ready."},
    )
    teacher_server_urls: dict[str, str] | str | None = field(
        default=None,
        metadata={
            "help": "Teacher vLLM server(s), each started with `vllm serve <model> --logprobs-mode "
            "processed_logprobs --max-logprobs -1`; every teacher is static, no weights are ever streamed to any of "
            "them. Defaults to a single teacher at 'http://localhost:8001'. A single entry scores every sample "
            "(plain single-teacher on-policy distillation). Multiple entries enable MOPD: each row's `teacher_id` "
            "column selects which entry scores it."
        },
    )
    request_timeout: int = field(
        default=600,
        metadata={"help": "Timeout in seconds for individual HTTP requests to any vLLM server."},
    )
    weight_sync_timeout: int = field(
        default=1800,
        metadata={
            "help": "Timeout in seconds for a weight transfer to the student's vLLM server. A transfer that does not "
            "complete within this time raises instead of hanging the run."
        },
    )

    # Parameters that control the distillation loss
    beta: float = field(
        default=0.0,
        metadata={
            "help": "Interpolation coefficient for the generalized JSD. At beta=0.0 (forward KL), the divergence is "
            "computed over the full teacher_top_k-wide teacher-reported support (plus tail bucket). At beta != 0.0, "
            "the support narrows to two candidates only (the teacher's own top-1 token and the completion's "
            "actual/realized token), mirroring ServerDistillationTrainer's own beta>0 path, since those are the "
            "only two token identities the wire protocol guarantees a teacher logprob for."
        },
    )
    teacher_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Softmax temperature of the divergence, applied to both the teacher's returned logprobs "
            "(server-side) and the student's own logits. Unrelated to `temperature`, which controls sampling."
        },
    )
    teacher_top_k: int = field(
        default=8,
        metadata={
            "help": "Number of per-position candidate tokens requested from the teacher's `prompt_logprobs`, used "
            "to approximate its distribution without transmitting the full vocabulary over HTTP. Above 20, the "
            "teacher's server must have been started with `--max-logprobs -1`."
        },
    )
    add_tail_bucket: bool = field(
        default=True,
        metadata={
            "help": "Whether to append a tail bucket representing the remaining probability mass outside "
            "`teacher_top_k`, to avoid a trivially small divergence when `teacher_top_k` is small."
        },
    )
    token_budget: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of real tokens packed into a single row (one DP rank's forward) for dynamic "
            "token-budgeted micro-batching. When > 0, a `TokenBudgetBatcher` forms Σ Lᵢ²-balanced micro-batches "
            "whose rows each stay within this budget. If None (default), it is set to the student vLLM server's "
            "`max_model_len`. A sample longer than token_budget fits in no row and is dropped with a warning. Set "
            "<= 0 to pack a fixed `per_device_train_batch_size × num_processes` samples per micro-batch instead, Σ "
            "Lᵢ²-balanced across the rows."
        },
    )

    # Parameters that control the async rollout pipeline
    max_inflight_tasks: int = field(
        default=-1,
        metadata={
            "help": "Maximum number of concurrent generation+scoring tasks in flight. Defaults to -1 (auto), which "
            "sets it to `max_staleness * per_device_train_batch_size * gradient_accumulation_steps * num_processes`."
        },
    )
    max_staleness: int = field(
        default=4,
        metadata={
            "help": "Maximum number of weight update steps a rollout sample can lag behind the current model "
            "version before being discarded."
        },
    )
    queue_maxsize: int = field(
        default=1024,
        metadata={"help": "Maximum number of rollout samples to buffer in the rollout queue."},
    )
    weight_sync_steps: int = field(
        default=1,
        metadata={"help": "Number of training steps between weight synchronizations to the student's vLLM server."},
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
            "help": "Whether to log a sample of (prompt, completion) pairs every `log_completions_steps` samples scored."
        },
    )
    log_completions_steps: int = field(
        default=100,
        metadata={"help": "Number of scored samples between logging completions."},
    )
    num_completions_to_print: int | None = field(
        default=None,
        metadata={"help": "Number of completions to print with `rich`. If `None`, all completions are logged."},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.beta < 0.0 or self.beta > 1.0:
            raise ValueError(f"beta must be in [0.0, 1.0], got {self.beta}.")

        if self.parallelism_config is not None and (
            self.parallelism_config.cp_enabled or self.parallelism_config.sp_enabled
        ):
            raise ValueError(
                "AsyncDistillationTrainer does not support sequence-dim parallelism (`parallelism_config.cp_size > 1` "
                "or `parallelism_config.sp_size > 1`) yet. Distillation builds model inputs after generation "
                "inside the trainer, so Transformers' context-parallel / Ulysses sequence-parallel input "
                "sharding cannot be applied to the raw generation batch. Set both `cp_size=1` and `sp_size=1`, "
                "or disable `parallelism_config`."
            )

        if self.teacher_server_urls is None:
            self.teacher_server_urls = {"default": "http://localhost:8001"}
        if not self.teacher_server_urls:
            raise ValueError("teacher_server_urls must have at least one entry.")

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
