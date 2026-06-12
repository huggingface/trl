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
class SDPOConfig(_BaseConfig):
    r"""
    Configuration class for the [`SDPOTrainer`].

    Parameters:
        > Parameters that control the online policy objective

        beta (`float`, *optional*, defaults to `0.0`):
            KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and improving
            training speed. [DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement
            learning](https://huggingface.co/papers/2501.12948) use a value of `0.001`.
        epsilon (`float`, *optional*, defaults to `0.2`):
            Epsilon value for clipping.
        epsilon_high (`float`, *optional*):
            Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound
            specified in argument `epsilon`. Paper DAPO recommends `0.28`.
        importance_sampling_level (`str`, *optional*, defaults to `"token"`):
            Controls whether importance sampling ratios are computed at the `'token'` or `'sequence'` level. `'token'`
            keeps the raw per-token log-probability ratios (one weight per token). `'sequence'` averages the
            log-probability ratios across valid tokens to produce a single ratio per sequence. The GSPO paper shows
            that sequence-level sampling often yields more stable training and better alignment with sequence-level
            rewards.
        reward_weights (`list[float]`, *optional*):
            Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are
            weighted equally with weight `1.0`.
        scale_rewards (`str` or `bool`, *optional*, defaults to `"group"`):
            Specifies the scaling strategy for rewards. Supported values are: `True` or `'group'` (default): rewards
            are scaled by the standard deviation within each group, ensuring unit variance within a group. `'batch'`:
            rewards are scaled by the standard deviation across the entire batch, as recommended in the PPO Lite paper.
            `False` or `'none'`: no scaling is applied. The Dr. GRPO paper recommends not scaling rewards, as scaling
            by the standard deviation introduces a question-level difficulty bias.

        > Parameters that control the SDPO loss

        distillation_weight (`float`, *optional*, defaults to `1.0`):
            Convex combination weight between the policy and self-distillation objectives. The loss is `(1 -
            distillation_weight) * policy_loss + distillation_weight * distillation_loss`. Must be in `[0, 1]`. `1.0`
            (default) trains purely on self-distillation, `0.0` falls back to the standard GRPO-style policy gradient,
            and intermediate values blend both.
        distillation_alpha (`float`, *optional*, defaults to `1.0`):
            Divergence interpolation coefficient. Sampled-token SDPO requires the official reverse-KL setting
            `distillation_alpha=1.0`.
        distillation_mode (`Literal["sampled_token", "full_logits", "topk_logits"]`, *optional*, defaults to `"sampled_token"`):
            Distillation objective mode. `sampled_token` is the default SDPO mode and requires
            `distillation_alpha=1.0`.
        distillation_topk (`int`, *optional*):
            Top-k approximation for logit-level SDPO. Must be set when `distillation_mode=topk_logits` and left unset
            otherwise.
        distillation_is_clip (`float`, *optional*, defaults to `2.0`):
            Clipping coefficient for importance sampling in self-distillation. `None` disables clipping.
        distillation_add_tail (`bool`, *optional*, defaults to `False`):
            Whether to add a tail bucket for non-top-k probability mass.

        > Parameters that control the teacher

        teacher_model_kind (`str`, *optional*, defaults to `"ema"`):
            Semantic teacher choice. `base` uses the initial student, `live` uses the current student, and `ema` uses
            an exponentially averaged teacher.
        teacher_update_rate (`float`, *optional*, defaults to `0.05`):
            Teacher update rate used for EMA teacher synchronization.
        teacher_sync_steps (`int`, *optional*, defaults to `1`):
            How often to synchronize the EMA teacher model.

        > Parameters that control reprompting

        use_successful_as_teacher (`bool`, *optional*, defaults to `True`):
            Use successful rollouts as implicit feedback for self-distillation.
        success_reward_threshold (`float`, *optional*, defaults to `1.0`):
            Minimum reward for a rollout to be considered a successful demonstration.
        dont_reprompt_on_self_success (`bool`, *optional*, defaults to `True`):
            Skip reprompting when model generates correct response.
        max_reprompt_len (`int`, *optional*, defaults to `10240`):
            Maximum length for reprompting in self-distillation.
        reprompt_template (`str`, *optional*, defaults to `"{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n"`):
            Template for reprompting the teacher with a successful demonstration.
        solution_template (`str`, *optional*, defaults to `"\nCorrect solution:
            \n\n{successful_previous_attempt}\n\n"`): Template for formatting the successful demonstration text.
        feedback_template (`str`, *optional*, defaults to `"\nThe following is feedback from your unsuccessful earlier attempt:
            \n\n{feedback_raw}\n\n"`): Template for formatting environment feedback for reprompting.
        include_environment_feedback (`bool`, *optional*, defaults to `False`):
            Whether to include environment feedback in teacher reprompts when available.
        environment_feedback_only_without_solution (`bool`, *optional*, defaults to `False`):
            Whether to use feedback only when no successful solution is available.
        remove_thinking_from_demonstration (`bool`, *optional*, defaults to `False`):
            Whether to remove <think>...</think> blocks from the demonstration text.

        > Parameters that control diagnostics

        diagnostics_warning_interval (`int`, *optional*, defaults to `10`):
            Emit repeated trainer diagnostics every N consecutive degenerate steps. Set to 0 to disable.
        diagnostics_flat_tolerance (`float`, *optional*, defaults to `1e-8`):
            Tolerance used to decide whether reward variance or reprompt activity is effectively zero.

        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` argument
            of the `SDPOTrainer` is provided as a string.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model. This is useful for training with a reference model, as it prevents
            the model from generating different logprobs for the same input.

        > Parameters that control data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function that requires
            any column other than 'prompts' and 'completions', you should keep this to `False`.
        max_prompt_length (`int`, *optional*, defaults to `512`):
            Maximum prompt length. Longer prompts are truncated from the left.
        shuffle_dataset (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training dataset.

        > Parameters that control generation

        num_generations (`int`, *optional*, defaults to `8`):
            Number of generations to sample. The effective batch size (num_processes * per_device_batch_size *
            gradient_accumulation_steps) must be evenly divisible by this value.
        num_generations_eval (`int`, *optional*):
            Number of generations to sample during evaluation. This allows using fewer generations during evaluation to
            save computation. If `None`, uses the value of `num_generations`.
        max_completion_length (`int`, *optional*, defaults to `256`):
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
        use_teacher_server (`bool`, *optional*, defaults to `False`):
            Compute teacher logprobs from the running vLLM generation server instead of a local teacher forward. Only
            supported for `teacher_model_kind='live'` with `use_vllm=True`, `vllm_mode='server'`,
            `distillation_weight=1.0` (pure distillation), and `distillation_mode` in {'sampled_token', 'topk_logits'}
            (the server returns the teacher's top-k logprobs, not the full vocabulary; `topk_logits` distills over the
            teacher's own top-k support).
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

        loss_type (`str`, *optional*, defaults to `"dapo"`):
            Specifies the loss formulation to use. Supported values are 'grpo', 'bnpo', 'dr_grpo', and 'dapo'. 'grpo':
            Aggregates token-level losses by normalizing over sequence length. Not recommended due to length bias—this
            approach tends to prefer shorter completions with positive advantages and longer ones with negative
            advantages. 'dapo' (default): Aggregates token-level losses by normalizing with the number of active tokens
            in the global accumulated batch. This method was introduced in the DAPO paper to eliminate length bias.
            'dr_grpo': Aggregates token-level losses by normalizing with a global constant. This method was introduced
            in the Dr. GRPO paper to eliminate length bias. The value of the constant corresponds to
            `max_completion_length`. 'bnpo': Aggregates token-level losses by normalizing with the number of active
            tokens in the local batch. Note that normalization is performed over the local batch only, so results may
            slightly vary depending on the local batch size, despite a constant effective batch size. When using
            `per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss.
        num_iterations (`int`, *optional*, defaults to `1`):
            Number of iterations per batch (denoted as μ in the algorithm).
        generation_batch_size (`int`, *optional*):
            Batch size to use for generation. If `None`, it defaults to the effective training batch size:
            `per_device_train_batch_size * num_processes * steps_per_generation`.
        steps_per_generation (`int`, *optional*):
            Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`.
        mask_truncated_completions (`bool`, *optional*, defaults to `False`):
            When enabled, truncated completions are excluded from the loss calculation, preventing them from being
            incorrectly penalized and introducing noise during training. According to the DAPO paper, this is a good
            practice for training stability.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the `SDPOTrainer` is provided as a string."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={
            "help": "Whether to disable dropout in the model. This is useful for training with a reference model, as it prevents the model from generating different logprobs for the same input."
        },
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: int | None = field(
        default=512,
        metadata={"help": "Maximum prompt length. Longer prompts are truncated from the left."},
    )
    num_generations: int = field(
        default=8,
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
        default=256,
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
            "forward. Only supported for `teacher_model_kind='live'` with `use_vllm=True`, `vllm_mode='server'`, "
            "`distillation_weight=1.0` (pure distillation), and `distillation_mode` in {'sampled_token', "
            "'topk_logits'} (the server returns the teacher's top-k logprobs, not the full vocabulary; `topk_logits` "
            "distills over the teacher's own top-k support)."
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
    loss_type: str = field(
        default="dapo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are 'grpo', 'bnpo', 'dr_grpo', and 'dapo'. 'grpo': Aggregates token-level losses by normalizing over sequence length. Not recommended due to length bias—this approach tends to prefer shorter completions with positive advantages and longer ones with negative advantages. 'dapo' (default): Aggregates token-level losses by normalizing with the number of active tokens in the global accumulated batch. This method was introduced in the DAPO paper to eliminate length bias. 'dr_grpo': Aggregates token-level losses by normalizing with a global constant. This method was introduced in the Dr. GRPO paper to eliminate length bias. The value of the constant corresponds to `max_completion_length`. 'bnpo': Aggregates token-level losses by normalizing with the number of active tokens in the local batch. Note that normalization is performed over the local batch only, so results may slightly vary depending on the local batch size, despite a constant effective batch size. When using `per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss."
        },
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={
            "help": "When enabled, truncated completions are excluded from the loss calculation, preventing them from being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is a good practice for training stability."
        },
    )
    dont_reprompt_on_self_success: bool = field(
        default=True,
        metadata={"help": "Skip reprompting when model generates correct response."},
    )
    beta: float = field(
        default=0.0,
        metadata={
            "help": "KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and improving training speed. [DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning](https://huggingface.co/papers/2501.12948) use a value of `0.001`."
        },
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    epsilon_high: float | None = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`."
        },
    )
    importance_sampling_level: str = field(
        default="token",
        metadata={
            "help": "Controls whether importance sampling ratios are computed at the `'token'` or `'sequence'` level. `'token'` keeps the raw per-token log-probability ratios (one weight per token). `'sequence'` averages the log-probability ratios across valid tokens to produce a single ratio per sequence. The GSPO paper shows that sequence-level sampling often yields more stable training and better alignment with sequence-level rewards."
        },
    )
    reward_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are weighted equally with weight `1.0`."
        },
    )
    scale_rewards: str | bool = field(
        default="group",
        metadata={
            "help": "Specifies the scaling strategy for rewards. Supported values are: `True` or `'group'` (default): rewards are scaled by the standard deviation within each group, ensuring unit variance within a group. `'batch'`: rewards are scaled by the standard deviation across the entire batch, as recommended in the PPO Lite paper. `False` or `'none'`: no scaling is applied. The Dr. GRPO paper recommends not scaling rewards, as scaling by the standard deviation introduces a question-level difficulty bias."
        },
    )
    distillation_alpha: float = field(
        default=1.0,
        metadata={
            "help": "Divergence interpolation coefficient. Sampled-token SDPO requires the official reverse-KL setting "
            "`distillation_alpha=1.0`."
        },
    )
    distillation_mode: Literal["sampled_token", "full_logits", "topk_logits"] = field(
        default="sampled_token",
        metadata={
            "help": "Distillation objective mode. `sampled_token` is the default SDPO mode and requires "
            "`distillation_alpha=1.0`."
        },
    )
    distillation_topk: int | None = field(
        default=None,
        metadata={
            "help": "Top-k approximation for logit-level SDPO. Must be set when `distillation_mode=topk_logits` and left "
            "unset otherwise."
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
    distillation_weight: float = field(
        default=1.0,
        metadata={
            "help": "Convex combination weight between the policy and self-distillation objectives. The loss is "
            "`(1 - distillation_weight) * policy_loss + distillation_weight * distillation_loss`. Must be in `[0, 1]`. "
            "`1.0` (default) trains purely on self-distillation, `0.0` falls back to the standard GRPO-style policy "
            "gradient, and intermediate values blend both."
        },
    )
    teacher_model_kind: str = field(
        default="ema",
        metadata={
            "help": "Semantic teacher choice. `base` uses the initial student, `live` uses the current student, "
            "and `ema` uses an exponentially averaged teacher."
        },
    )
    teacher_update_rate: float = field(
        default=0.05,
        metadata={"help": "Teacher update rate used for EMA teacher synchronization."},
    )
    teacher_sync_steps: int = field(
        default=1,
        metadata={"help": "How often to synchronize the EMA teacher model."},
    )
    max_reprompt_len: int = field(
        default=10240,
        metadata={"help": "Maximum length for reprompting in self-distillation."},
    )
    use_successful_as_teacher: bool = field(
        default=True,
        metadata={"help": "Use successful rollouts as implicit feedback for self-distillation."},
    )
    success_reward_threshold: float = field(
        default=1.0,
        metadata={"help": "Minimum reward for a rollout to be considered a successful demonstration."},
    )
    reprompt_template: str = field(
        default="{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n",
        metadata={"help": "Template for reprompting the teacher with a successful demonstration."},
    )
    solution_template: str = field(
        default="\nCorrect solution:\n\n{successful_previous_attempt}\n\n",
        metadata={"help": "Template for formatting the successful demonstration text."},
    )
    feedback_template: str = field(
        default="\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n",
        metadata={"help": "Template for formatting environment feedback for reprompting."},
    )
    include_environment_feedback: bool = field(
        default=False,
        metadata={"help": "Whether to include environment feedback in teacher reprompts when available."},
    )
    environment_feedback_only_without_solution: bool = field(
        default=False,
        metadata={"help": "Whether to use feedback only when no successful solution is available."},
    )
    remove_thinking_from_demonstration: bool = field(
        default=False,
        metadata={"help": "Whether to remove <think>...</think> blocks from the demonstration text."},
    )
    diagnostics_warning_interval: int = field(
        default=10,
        metadata={
            "help": "Emit repeated trainer diagnostics every N consecutive degenerate steps. Set to 0 to disable."
        },
    )
    diagnostics_flat_tolerance: float = field(
        default=1e-8,
        metadata={
            "help": "Tolerance used to decide whether reward variance or reprompt activity is effectively zero."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.distillation_weight <= 1.0:
            raise ValueError(f"`distillation_weight` must be in [0, 1], got {self.distillation_weight}.")
        if self.distillation_mode == "sampled_token" and self.distillation_alpha != 1.0:
            raise ValueError(
                "`distillation_mode='sampled_token'` only supports reverse KL, so it requires "
                f"`distillation_alpha=1.0`, got {self.distillation_alpha}."
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

        self.scale_rewards = {True: "group", False: "none"}.get(self.scale_rewards, self.scale_rewards)

        if self.epsilon_high is None:
            self.epsilon_high = self.epsilon
