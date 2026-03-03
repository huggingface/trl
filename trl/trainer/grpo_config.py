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

from transformers import TrainingArguments

from .base_config import _BaseConfig


@dataclass
class GRPOConfig(_BaseConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`GRPOTrainer`].

    This class includes only the parameters that are specific to GRPO training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`str`, `dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`GRPOTrainer`] is provided as a string.
        disable_dropout (`bool`, *optional*, defaults to `False`):
            Whether to disable dropout in the model. This is useful for training with a reference model, as it prevents
            the model from generating different logprobs for the same input.
        cast_lm_head_to_fp32 (`bool`, *optional*, defaults to `False`):
            Whether to cast the language modeling head of the policy and reference models to float32. As recommended by
            the [ScaleRL](https://huggingface.co/papers/2510.13786) recipe. This flag is only supported when the model
            has untied word embedding and language modeling head layers i.e. `tie_word_embeddings` in the model config
            is False.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
        num_generations (`int`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. The effective batch size (num_processes * per_device_batch_size
            * gradient_accumulation_steps) must be evenly divisible by this value.
        num_generations_eval (`int` or `None`, *optional*):
            Number of generations to sample during evaluation. This allows using fewer generations during evaluation to
            save computation. If `None`, uses the value of `num_generations`.
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

        generation_batch_size: (`int`, *optional*):
            Batch size to use for generation. If `None`, it defaults to the effective training batch size:
            `per_device_train_batch_size * num_processes * steps_per_generation`. In other words, there is one
            generation batch processed per optimization step. Mutually exclusive with `steps_per_generation`.
        steps_per_generation: (`int`, *optional*):
            Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`. Mutually exclusive
            with `generation_batch_size`.
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
        use_transformers_paged (`bool`, *optional*, defaults to `False`):
            Whether to use the `transformers` paged implementation for generation. If set to `True`, the `transformers`
            paged implementation will be used for generation instead of the default padded implementation. This
            parameter is only effective when `use_vllm` is set to `False`.
        cache_implementation (`str`, *optional*):
            Implementation of the cache method for faster generation when `use_vllm` is set to `False`.

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

        > Parameters that control the training

        beta (`float`, *optional*, defaults to `0.0`):
            KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and improving
            training speed. [DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement
            learning](https://huggingface.co/papers/2501.12948) use a value of `0.001`.
        num_iterations (`int`, *optional*, defaults to `1`):
            Number of iterations per batch (denoted as μ in the algorithm).
        epsilon (`float`, *optional*, defaults to `0.2`):
            Epsilon value for clipping.
        delta (`float`, *optional*):
            Enables the upper clipping bound in two-sided GRPO loss when set to a float. If `None` (default), standard
            GRPO clipping is used. Recommended to be greater than `1 + ε` when enabled. This method is introduced in
            the [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291).
        epsilon_high (`float`, *optional*):
            Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound
            specified in argument `epsilon`. Paper [DAPO](https://huggingface.co/papers/2503.14476) recommends `0.28`.
            When used with `loss_type='cispo'`, this corresponds to the ε_max param specified in the [ScaleRL
            paper](https://huggingface.co/papers/2510.13786) and the recommended value is `5.0`.
        sapo_temperature_neg (`float`, *optional*, defaults to `1.05`):
            Temperature for tokens with non-positive advantage scores used in the `sapo` loss function. This parameter
            is introduced in the [Soft Adaptive Policy Optimization paper](https://huggingface.co/papers/2511.20347).
        sapo_temperature_pos (`float`, *optional*, defaults to `1.0`):
            Temperature for tokens with positive advantage scores used in the `sapo` loss function. This parameter is
            introduced in the [Soft Adaptive Policy Optimization paper](https://huggingface.co/papers/2511.20347).
        importance_sampling_level (`str`, *optional*, defaults to `"token"`):
            Controls whether importance sampling ratios are computed at the `"token"` or `"sequence"` level. `"token"`
            keeps the raw per-token log-probability ratios (one weight per token). `"sequence"` averages the
            log-probability ratios across valid tokens to produce a single ratio per sequence. The [GSPO
            paper](https://huggingface.co/papers/2507.18071) shows that sequence-level sampling often yields more
            stable training and better alignment with sequence-level rewards.
        reward_weights (`list[float]`, *optional*):
            Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are
            weighted equally with weight `1.0`.
        multi_objective_aggregation (`str`, *optional*, defaults to `"sum_then_normalize"`):
            Method to aggregate multiple reward functions. Supported values are:

            - `"sum_then_normalize"` (default): First sums the weighted rewards from each reward function, then applies
              reward scaling/normalization as specified by `scale_rewards` (see `scale_rewards` for details).
            - `"normalize_then_sum"`: First normalizes/scales each reward function across generations (within each
              group), then sums the normalized rewards using the specified weights. The aggregated reward is then
              normalized at the batch level when forming advantages. This is the suggested approach from the paper
              [GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL
              Optimization](https://huggingface.co/papers/2601.05242).
        scale_rewards (`str` or `bool`, *optional*, defaults to `"group"`):
            Specifies the scaling strategy for rewards. Supported values are:

            - `True` or `"group"` (default): rewards are scaled by the standard deviation within each group, ensuring
              unit variance within a group.
            - `"batch"`: rewards are scaled by the standard deviation across the entire batch, as recommended in the
              [PPO Lite paper](https://huggingface.co/papers/2508.08221).
            - `False` or `"none"`: no scaling is applied. The [Dr. GRPO
              paper](https://huggingface.co/papers/2503.20783) recommends not scaling rewards, as scaling by the
              standard deviation introduces a question-level difficulty bias.
        loss_type (`str`, *optional*, defaults to `"dapo"`):
            Specifies the loss formulation to use. Supported values are:

            - `"grpo"`: Aggregates token-level losses by normalizing over sequence length. Not recommended due to
              length bias—this approach tends to prefer shorter completions with positive advantages and longer ones
              with negative advantages.
            - `"dr_grpo"`: Aggregates token-level losses by normalizing with a global constant. This method was
              introduced in the [Dr. GRPO paper](https://huggingface.co/papers/2503.20783) to eliminate length bias.
              The value of the constant corresponds to `max_completion_length`.
            - `"dapo"` (default): Aggregates token-level losses by normalizing with the number of active token in the
              global accumulated batch. This method was introduced in the [DAPO
              paper](https://huggingface.co/papers/2503.14476) to eliminate length bias.
            - `"bnpo"`: Aggregates token-level losses by normalizing with the number of active token in the local
              batch. Note that normalization is performed over the local batch only, so results may slightly vary
              depending on the local batch size, despite a constant effective batch size. When using
              `per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss.
            - `"cispo"`: Clips the importance sampling weights instead of the advantage scaled importance weights. The
              clipped weights are then multiplied with the advantages and policy model's log probs. Individual token
              losses are aggregated by normalizing with the number of active tokens in the global accumulated batch.
              This method was introduced in the [MiniMax-M1 paper](https://huggingface.co/papers/2506.13585).
            - `"sapo"`: Soft Adaptive Policy Optimization loss, as introduced in the [Soft Adaptive Policy Optimization
              paper](https://huggingface.co/papers/2511.20347). Replaces hard clipping with a smooth,
              temperature-controlled gate that adaptively attenuates off-policy updates while preserving useful
              learning signals.
            - `"luspo"`: Length-Unbiased Sequence Policy Optimization loss. A sequence-level loss that scales each
              sequence's loss by its length. This is a modification of GSPO and requires
              `importance_sampling_level="sequence"`. Introduced in the [LUSPO
              paper](https://huggingface.co/papers/2602.05261).
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
        top_entropy_quantile (`float`, *optional*, defaults to `1.0`):
            ρ parameter from [Beyond the 80/20 Rule](https://huggingface.co/papers/2506.01939). Keeps in the policy
            loss term only the top-ρ quantile of tokens by entropy of the probability distribution at each sequence
            position, improving results. Range: `[0.0-1.0]`. A value of `0.0` masks all but the highest entropy token;
            `1.0` keeps all tokens. The paper recommends a value of `0.2`. If used with
            `mask_truncated_completions=True`, only tokens from non-truncated completions are considered.
        max_tool_calling_iterations (`int`, *optional*):
            Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and generation
            stops when the model generates a response turn with no tool calls or when the total response length reaches
            `max_model_length`.
        vllm_importance_sampling_correction (`bool`, *optional*, defaults to `True`):
            Whether to apply Importance Sampling (IS) to correct for the mismatch between vLLM completion logprobs and
            recomputed training logprobs. If set to `False`, no IS is applied regardless of
            `vllm_importance_sampling_mode`. When `True`, the selected mode determines how the IS ratios are computed
            and constrained.
        vllm_importance_sampling_mode (`str`, *optional*, defaults to `"sequence_mask"`):
            Specifies how Importance Sampling is performed when `vllm_importance_sampling_correction=True`. Possible
            values are:

                - `"token_truncate"`: Token-level truncated IS (default). Per-token ratios are clipped from above at C.
                - `"token_mask"`: Token-level masked IS. Per-token ratios above C are set to zero.
                - `"sequence_truncate"`: Sequence-level truncated IS. A single sequence ratio is clipped from above at
                  C and applied to all tokens in the sequence.
                - `"sequence_mask"`: Sequence-level masked IS. Sequences with ratios above C are masked out.
        vllm_importance_sampling_cap (`float`, *optional*, defaults to `3.0`):
            Importance sampling cap C used by `vllm_importance_sampling_mode`. For `*_truncate` modes, importance
            ratios are clipped from above at C. For `*_mask` modes, ratios larger than C are set to zero.
        off_policy_mask_threshold (`float`, *optional*):
            Threshold for off-policy sequence masking. If `None`, off-policy sequence masking is disabled. When set,
            sequences with negative advantages and high KL divergence are masked out to stabilize training. This
            parameter corresponds to the `delta` threshold in Equation 9 of the [DeepSeek-V3.2
            paper](https://huggingface.co/papers/2512.02556). It expects a positive value (e.g., 0.5).
        use_bias_correction_kl (`bool`, *optional*, defaults to `False`):
            Whether to use the unbiased KL divergence estimator with importance sampling correction. This corrects the
            KL divergence estimate by multiplying it with the importance sampling ratio. This is described in the
            [DeepSeek-V3.2 paper](https://huggingface.co/papers/2512.02556).

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is installed,
            it prints the sample. If `wandb` and/or `trackio` logging is enabled, it logs it to `wandb` and/or
            `trackio`.
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

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )

    # Parameters that control the model and reference model
    model_init_kwargs: dict | str | None = field(
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
    cast_lm_head_to_fp32: bool = field(
        default=False,
        metadata={
            "help": "Whether to cast the language modeling head of the policy and reference, models to float32."
            "As recommended by the [ScaleRL](https://huggingface.co/papers/2510.13786) recipe. This flag is only "
            "supported when the model has untied word embedding and language modeling head layers i.e. "
            "`tie_word_embeddings` in the model config is False."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: bool | None = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    num_generations: int | None = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The effective batch size (num_processes * per_device_batch_size "
            "* gradient_accumulation_steps) must be evenly divisible by this value."
        },
    )
    num_generations_eval: int | None = field(
        default=None,
        metadata={
            "help": "Number of generations to sample during evaluation. This allows using fewer generations during "
            "evaluation to save computation. If `None`, uses the value of `num_generations`."
        },
    )
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
    generation_batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Batch size to use for generation. If `None`, it defaults to the effective training batch size: "
            "`per_device_train_batch_size * num_processes * steps_per_generation`."
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
    use_transformers_paged: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the `transformers` paged implementation for generation. If set to `True`, the "
            "`transformers` paged implementation will be used for generation instead of the default padded "
            "implementation. This parameter is only effective when `use_vllm` is set to `False`."
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
        default=0.0,
        metadata={
            "help": "KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and "
            "improving training speed. [DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement "
            "learning](https://huggingface.co/papers/2501.12948) use a value of `0.001`."
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
    delta: float | None = field(
        default=None,
        metadata={
            "help": "Enables the upper clipping bound in two-sided GRPO loss when set to a float. If `None` "
            "(default), standard GRPO clipping is used. Recommended to be greater than `1 + ε` when enabled. This "
            "method is introduced in the [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291)."
        },
    )
    epsilon_high: float | None = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`. "
            "When used with `loss_type='cispo'`, this corresponds to the ε_max param specified in the"
            "[ScaleRL paper]https://huggingface.co/papers/2510.13786) and the recommended value is `5.0`."
        },
    )
    sapo_temperature_neg: float = field(
        default=1.05,
        metadata={
            "help": "Temperature for tokens with non-positive advantage scores used in the `sapo` loss function. "
            "This parameter is introduced in the [Soft Adaptive Policy Optimization "
            "paper](https://huggingface.co/papers/2511.20347)."
        },
    )
    sapo_temperature_pos: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for tokens with positive advantage scores used in the `sapo` loss function. "
            "This parameter is introduced in the [Soft Adaptive Policy Optimization "
            "paper](https://huggingface.co/papers/2511.20347)."
        },
    )
    importance_sampling_level: str = field(
        default="token",
        metadata={
            "help": "Controls whether importance sampling ratios are computed at the `'token'` or `'sequence'` level. "
            "`'token'` keeps the raw per-token log-probability ratios (one weight per token).  `'sequence'` averages "
            "the log-probability ratios across valid tokens to produce a single ratio per sequence. The GSPO paper "
            "shows that sequence-level sampling often yields more stable training and better alignment with "
            "sequence-level rewards."
        },
    )
    reward_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
    multi_objective_aggregation: str = field(
        default="sum_then_normalize",
        metadata={
            "help": "Method to aggregate multiple reward functions. Supported values are: "
            "`'sum_then_normalize'` (default): First sums the weighted rewards from each reward function, then "
            "applies reward scaling/normalization as specified by `scale_rewards` (see `scale_rewards` for details). "
            "`'normalize_then_sum'`: First normalizes/scales each reward function across generations (within each "
            "group), then sums the normalized rewards using the specified weights. The aggregated reward is then "
            "normalized at the batch level when forming advantages. This is the suggested approach from the paper "
            "GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization."
        },
    )
    scale_rewards: str = field(
        default="group",
        metadata={
            "help": "Specifies the scaling strategy for rewards. Supported values are: "
            "`True` or `group'` (default): rewards are scaled by the standard deviation within each group, ensuring "
            "unit variance within a group. "
            "`'batch'`: rewards are scaled by the standard deviation across the entire batch, as recommended in the "
            "PPO Lite paper. "
            "`False` or `'none'`: no scaling is applied. The Dr. GRPO paper recommends not scaling rewards, as "
            "scaling by the standard deviation introduces a question-level difficulty bias."
        },
    )
    loss_type: str = field(
        default="dapo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are 'grpo', 'dapo', 'bnpo', and "
            "'dr_grpo'. "
            "'grpo': Aggregates token-level losses by normalizing over sequence length. Not recommended due to length "
            "bias—this approach tends to prefer shorter completions with positive advantages and longer ones with "
            "negative advantages. "
            "'dapo' (default): Aggregates token-level losses by normalizing with the number of active token in the "
            "global accumulated batch. This method was introduced in the DAPO paper to eliminate length bias. "
            "'dr_grpo': Aggregates token-level losses by normalizing with a global constant. This method was "
            "introduced in the Dr. GRPO paper to eliminate length bias. The value of the constant corresponds to "
            "`max_completion_length`. "
            "'bnpo': Aggregates token-level losses by normalizing with the number of active token in the local batch. "
            "Note that normalization is performed over the local batch only, so results may slightly vary depending "
            "on the local batch size, despite a constant effective batch size. When using "
            "`per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss."
            "'cispo': Clips the importance sampling weights instead of the advantage scaled importance weights. "
            "The clipped weights are then multiplied with the advantages and policy model's log probs. "
            "Individual token losses are aggregated by normalizing with the number of active tokens in "
            "the global accumulated batch. This method was introduced in the "
            "[MiniMax-M1 paper](https://huggingface.co/papers/2506.13585). "
            "'sapo': Soft Adaptive Policy Optimization loss, as introduced in the "
            "[Soft Adaptive Policy Optimization paper](https://huggingface.co/papers/2511.20347). "
            "Replaces hard clipping with a smooth, temperature-controlled gate that adaptively attenuates "
            "off-policy updates while preserving useful learning signals."
            "'luspo': Length-Unbiased Sequence Policy Optimization loss. A sequence-level loss that scales each "
            "sequence's loss by its length. This is a modification of GSPO and requires "
            "`importance_sampling_level='sequence'`. Introduced in the [LUSPO "
            "paper](https://huggingface.co/papers/2602.05261)."
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
    top_entropy_quantile: float = field(
        default=1.0,
        metadata={
            "help": "ρ parameter from Beyond the 80/20 Rule. Keeps in the policy loss term only the top-ρ quantile of "
            "tokens by entropy of the probability distribution at each sequence position, improving results. Range: "
            "[0.0-1.0]. A value of `0.0` masks all but the highest entropy token; `1.0` keeps all tokens. The paper "
            "recommends a value of `0.2`. If used with `mask_truncated_completions=True`, only tokens from "
            "non-truncated completions are considered."
        },
    )
    max_tool_calling_iterations: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and "
            "generation stops when the model generates a response turn with no tool calls or when the total "
            "response length reaches `max_model_length`."
        },
    )
    vllm_importance_sampling_correction: bool = field(
        default=True,
        metadata={
            "help": "Whether to apply Importance Sampling (IS) to correct for the mismatch between vLLM "
            "completion logprobs and recomputed training logprobs. If set to `False`, no IS is applied "
            "regardless of `vllm_importance_sampling_mode`. When `True`, the selected mode determines how "
            "IS ratios are computed and constrained."
        },
    )
    vllm_importance_sampling_mode: str = field(
        default="sequence_mask",
        metadata={
            "help": "Specifies how Importance Sampling (IS) is performed when "
            "vllm_importance_sampling_correction=True. Modes are defined along two orthogonal "
            "dimensions: (1) constraint, which determines how to handle ratios above "
            "vllm_importance_sampling_cap (C)—either truncation (clip from above, ρ ← min(ρ, C)) or "
            "masking (set ratios above C to zero); and (2) granularity, which determines whether "
            "ratios are computed per token or as a single sequence-level ratio applied to all tokens. "
            "Supported options are: 'token_truncate', 'token_mask', 'sequence_truncate', and "
            "'sequence_mask'."
        },
    )
    vllm_importance_sampling_cap: float = field(
        default=3.0,
        metadata={
            "help": "Importance sampling cap C used by `vllm_importance_sampling_mode`. For '*_truncate' modes, "
            "ratios are clipped from above at C. For '*_mask' modes, ratios larger than C are set to zero."
        },
    )
    off_policy_mask_threshold: float | None = field(
        default=None,
        metadata={
            "help": "Threshold for off-policy sequence masking. If `None`, off-policy sequence masking is disabled. "
            "When set, sequences with negative advantages and high KL divergence are masked out to stabilize "
            "training. This parameter corresponds to the `delta` threshold in Equation 9 of the [DeepSeek-V3.2 "
            "paper](https://huggingface.co/papers/2512.02556). It expects a positive value (e.g., 0.5)."
        },
    )
    use_bias_correction_kl: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the unbiased KL divergence estimator with importance sampling correction. This "
            "corrects the KL divergence estimate by multiplying it with the importance sampling ratio. "
            "This is described in the [DeepSeek-V3.2 paper](https://huggingface.co/papers/2512.02556)."
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

    def __post_init__(self):
        super().__post_init__()

        self.scale_rewards = {True: "group", False: "none"}.get(self.scale_rewards, self.scale_rewards)

        if self.log_completions_hub_repo is not None and not self.log_completions:
            raise ValueError(
                "log_completions_hub_repo is set, but log_completions is False. Enable log_completions to upload "
                "completions to the Hub, or unset log_completions_hub_repo."
            )

        num_processes = self.world_size
        # The current default effective batch size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            # Just ensure the value is divisible by the global batch size
            if self.generation_batch_size % (self.per_device_train_batch_size * num_processes) != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                    f"({self.per_device_train_batch_size * num_processes})."
                )
            self.steps_per_generation = self.generation_batch_size // (
                self.per_device_train_batch_size * num_processes
            )
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        else:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        if self.do_eval and self.eval_strategy != "no":
            # Determine the number of generations to use for evaluation
            num_generations = self.num_generations_eval or self.num_generations

            # Just ensure the value is divisible by the global batch size
            if (self.per_device_eval_batch_size * num_processes) % num_generations != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by the number of generations used for evaluation ({num_generations})."
                )

        # The generation batch must contain full prompt groups (no partials), so it must be divisible by
        # num_generations.
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )

        if self.num_generations < 2:
            raise ValueError(
                "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
            )

        if self.delta is not None and self.use_liger_kernel:
            raise ValueError("Liger kernel does not support two-sided GRPO loss yet.")
