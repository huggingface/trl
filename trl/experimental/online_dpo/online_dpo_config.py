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

import warnings
from dataclasses import dataclass, field
from typing import Any

from ...trainer.base_config import BaseConfig


@dataclass
class OnlineDPOConfig(BaseConfig):
    r"""
    Configuration class for the [`experimental.online_dpo.OnlineDPOTrainer`].

    This class includes only the parameters that are specific to Online DPO training. For a full list of training
    arguments, please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this
    class may differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        reward_model_path (`str`, *optional*):
            Path to the reward model. Either `judge` or `reward_model_path` must be set, but not both.
        judge (`str`, *optional*):
            Name of the judge to use. Either `judge` or `reward_model_path` must be set, but not both.
        max_new_tokens (`int`, *optional*, defaults to `64`):
            Maximum number of tokens to generate per completion.
        max_length (`int`, *optional*, defaults to `256`):
            Maximum total length of the sequence (prompt + completion) used to compute log probabilities. If the
            sequence exceeds this limit, the leftmost tokens will be truncated to preserve as much of the completion as
            possible.
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        missing_eos_penalty (`float`, *optional*):
            Penalty applied to the score when the model fails to generate an EOS token. This is useful to encourage to
            generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be a positive
            value. This parameter only works when using `reward_funcs` and not when using `judge`.
        beta (`float` or `list[float]`, *optional*, defaults to `0.1`):
            Parameter controlling the deviation from the reference model. Higher β means less deviation from the
            reference model. For the IPO loss (`loss_type="ipo"`), β is the regularization parameter denoted by τ in
            the [paper](https://huggingface.co/papers/2310.12036). If a list of floats is provided then the β is
            selected for each new epoch and the last β is used for the rest of the epochs.
        loss_type (`str`, *optional*, defaults to `"sigmoid"`):
            Type of loss to use. Possible values are:

                - `"sigmoid"`: sigmoid loss from the original [DPO](https://huggingface.co/papers/2305.18290) paper.
                - `"ipo"`: IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model and reference model.

        > Parameters that control generation

        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int`, *optional*, defaults to `0`):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, top-k-filtering is
            disabled and all tokens are considered.
        min_p (`float`, *optional*):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
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
        generation_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to [`~transformers.GenerationConfig`] (if using transformers) or
            `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the
            generation behavior, such as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that conflict
            with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for generation
            instead of the default model.generate(). Requires `vllm` to be installed.
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation to use for vLLM. Must be one of `"transformers"` or `"vllm"`. `"transformers"`: Use
            the `transformers` backend for model implementation. `"vllm"`: Use the `vllm` library for model
            implementation.
        vllm_mode (`str`, *optional*, defaults to `"server"`):
            Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `"server"` or
            `"colocate"`.

            - `"server"`: The trainer will send generation requests to a separate vLLM server. Make sure a TRL vLLM
              server is running (start with `trl vllm-serve`).
            - `"colocate"`: vLLM will run in the same process and share the training GPUs. This avoids the need for a
              separate server but may cause resource contention with training.
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

        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.55`):
            Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to
            `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_gpu_memory_utilization` flag.
        vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to
            `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_tensor_parallel_size` flag.
        vllm_enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory usage low, but
            waking the engine adds host–device transfer latency.

        > Other parameters

        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.
        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model from a
            string.
    """

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=5e-7,
        metadata={"help": "The initial learning rate for AdamW."},
    )

    reward_model_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to the reward model. Either `judge` or `reward_model_path` must be set, but not both."
        },
    )
    judge: str | None = field(
        default=None,
        metadata={
            "help": "Name of the judge to use. Either `judge` or `reward_model_path` must be set, but not both."
        },
    )
    max_new_tokens: int = field(
        default=64,
        metadata={"help": "Maximum number of tokens to generate per completion."},
    )
    max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum total length of the sequence (prompt + completion) used to compute log probabilities. If "
            "the sequence exceeds this limit, the leftmost tokens will be truncated to preserve as much of the "
            "completion as possible."
        },
    )
    temperature: float = field(
        default=0.9,
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
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
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
    missing_eos_penalty: float | None = field(
        default=None,
        metadata={
            "help": "Penalty applied to the score when the model fails to generate an EOS token. This is useful to "
            "encourage to generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be "
            "a positive value."
        },
    )
    beta: list[float] = field(
        default_factory=lambda: [0.1],
        metadata={
            "help": "Parameter controlling the deviation from the reference model. Higher β means less deviation from "
            "the reference model. For the IPO loss (`loss_type='ipo'`), β is the regularization parameter denoted by "
            "τ in the [paper](https://huggingface.co/papers/2310.12036). If a list of floats is provided then the β "
            "is selected for each new epoch and the last β is used for the rest of the epochs."
        },
    )
    loss_type: str = field(
        default="sigmoid",
        metadata={
            "help": "Type of loss to use.",
            "choices": ["sigmoid", "ipo"],
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model."},
    )
    use_vllm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. Requires vLLM to be installed "
            "(`pip install trl[vllm]`)."
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
    vllm_structured_outputs_regex: str | None = field(
        default=None,
        metadata={"help": "Regex for vLLM structured outputs. If `None` (default), structured outputs is disabled."},
    )
    vllm_gpu_memory_utilization: float | None = field(
        default=0.55,
        metadata={
            "help": "Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_gpu_memory_utilization` flag.",
        },
    )
    vllm_mode: str = field(
        default="server",
        metadata={
            "help": "Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `'server'` or "
            "`'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure "
            "a TRL vLLM server is running (start with `trl vllm-serve`). `'colocate'`: vLLM will run in the same "
            "process and share the training GPUs. This avoids the need for a separate server but may cause resource "
            "contention with training.",
        },
    )
    vllm_server_base_url: str | None = field(
        default=None,
        metadata={
            "help": "Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` "
            "and `vllm_server_port` are ignored.",
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
            "after the timeout, a `ConnectionError` is raised.",
        },
    )
    vllm_group_port: int = field(
        default=51216,
        metadata={
            "help": "Port number for the weight update group. This is used to communicate with the vLLM server. "
            "Unless the port is occupied, there is no need to change it.",
        },
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_tensor_parallel_size` flag.",
        },
    )
    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={
            "help": "Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory "
            "usage low, but waking the engine adds host–device transfer latency."
        },
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
    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the model "
            "from a string."
        },
    )
    reward_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Weights for combining multiple reward functions. Must match the number of reward functions. "
            "If None, all reward functions are equally weighted."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        if hasattr(self.beta, "__len__") and len(self.beta) == 1:
            self.beta = self.beta[0]

        if self.max_new_tokens >= self.max_length:
            warnings.warn(
                f"The configuration has `max_new_tokens` ({self.max_new_tokens}) >= `max_length` ({self.max_length}). "
                "This will cause prompts to be truncated or completely removed in the forward pass. "
                "To preserve prompts, ensure  e.g. `max_length > max_new_tokens + 512`. ",
                stacklevel=3,
            )
