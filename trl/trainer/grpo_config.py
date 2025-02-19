# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from typing import Optional

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

        model_init_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`GRPOTrainer`] is provided as a string.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
        num_generations (`int` or `None`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. The global batch size (num_processes * per_device_batch_size)
            must be divisible by this value.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.

        > Parameters that control generation

        temperature (`float`, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int` or `None`, *optional*, defaults to `50`):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, top-k-filtering is
            disabled.
        min_p (`float` or `None`, *optional*, defaults to `None`):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat
            tokens.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept unused for
            training, as vLLM will require one for generation. vLLM must be installed (`pip install vllm`).
        vllm_device (`str`, *optional*, defaults to `"auto"`):
            Device where vLLM generation will run, e.g. `"cuda:1"`. If set to `"auto"` (default), the system will
            automatically select the next available GPU after the last one used for training.
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio of GPU memory to reserve for model weights, activations, and KV cache on the generation device.
        vllm_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation.
        vllm_max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM.
        vllm_guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
            Regex for vLLM guided decoding. If `None`, guided decoding is disabled.

        > Parameters that control generation acceleration powered by SGLang

        use_sglang (`bool`, *optional*, defaults to `False`):
            Whether to use SGLang for generating completions. If set to `True`, a SGLang server must be running.
        sglang_server_url (`str` or `None`, *optional*, defaults to `None`):
            The URL of the SGLang server (e.g. "http://localhost:30033"). Required if `use_sglang` is `True`.
        sglang_device (`str`, *optional*, defaults to `"cuda:1"`):
            GPU device to be used for SGLang generation if launching from this code. This is optional if the server is
            managed externally.
            Regex for vLLM guided decoding. If `None`, guided decoding is disabled.

        > Parameters that control generation acceleration powered by SGLang

        use_sglang (`bool`, *optional*, defaults to `False`):
            Whether to use SGLang for rollout. If set to `True`, a SGLang engine will be used.
        sglang_base_gpu_id (`int`, *optional*, defaults to `7`):
            Base GPU ID for SGLang engine initialization. If set to `None`, the last available GPU is used.
        sglang_mem_fraction_static (`float`, *optional*, defaults to `0.9`):
            Fraction of GPU memory reserved for static memory in SGLang.
        checkpoint_path (`str`, *optional*, defaults to `None`):
            Path to the checkpoint for updating or initializing SGLang weights.

        > Parameters that control the training

        learning_rate (`float`, *optional*, defaults to `1e-6`):
            Initial learning rate.
        beta (`float`, *optional*, defaults to `0.04`):
            KL coefficient. If `0.0`, the reference model is not loaded.
        reward_weights (`list[float]` or `None`, *optional*, defaults to `None`):
            Weights for each reward function.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to synchronize the reference model with the active model.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.9`):
            α parameter controlling the mix between the current and previous reference policy.
        ref_model_sync_steps (`int`, *optional*, defaults to `64`):
            Frequency (in steps) at which the current policy is synchronized with the reference policy.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is
            installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`.
    """

    # Parameters that control the model and reference model
    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`."
        },
    )

    # Parameters that control the data preprocessing
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to only keep the column 'prompt' in the dataset."},
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum length of the prompt."},
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={"help": "Number of generations per prompt to sample."},
    )
    temperature: Optional[float] = field(
        default=0.9,
        metadata={"help": "Temperature for sampling completions."},
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "If enabled, the policy model weights are gathered for generation (DeepSpeed ZeRO-3)."
        },
    )

    # Parameters for generation acceleration powered by vLLM
    use_vllm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. Requires `pip install vllm`."
        },
    )
    vllm_device: Optional[str] = field(
        default="auto",
        metadata={"help": "Device for vLLM generation (e.g., 'cuda:1')."},
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={"help": "Ratio of GPU memory reserved for vLLM generation."},
    )
    vllm_dtype: Optional[str] = field(
        default="auto",
        metadata={"help": "Data type to use for vLLM generation."},
    )
    sglang_server_url: Optional[str] = field(
        default=None,
        metadata={"help": "Optional max_model_len for vLLM."},
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={"help": "Regex for vLLM guided decoding (if enabled)."},
    )

    # When running the trainer, set the following command-line arguments (or JSON configuration) so that SGLang is used:
    # •	--use_sglang True
    # •	--sglang_server_url "http://localhost:30033"
    # •	Optionally, --sglang_device "cuda:1" if you wish to assign a specific GPU.
    # Parameters for generation acceleration powered by SGLang
    use_sglang: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use SGLang for generating completions. If True, a SGLang server must be running."
        },
    )
    sglang_server_url: Optional[str] = field(
        default=None,
        metadata={
            "help": "The URL of the SGLang server (e.g., 'http://localhost:30033'). Required if use_sglang is True."
        },
    )
    sglang_device: Optional[str] = field(
        default="cuda:1",
        metadata={
            "help": "The GPU device to be used for SGLang generation if launching internally. Optional if the server is managed externally."
        },
    )

    # Parameters that control generation acceleration powered by SGLang
    use_sglang: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use SGLang for generating completions."},
    )
    sglang_base_gpu_id: Optional[int] = field(
        default=7,
        metadata={"help": "Base GPU ID for SGLang engine initialization. If None, uses the last available GPU."},
    )
    sglang_mem_fraction_static: float = field(
        default=0.9,
        metadata={"help": "Fraction of GPU memory reserved for static memory in SGLang."},
    )
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the checkpoint for SGLang weight update."},
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Initial learning rate for the optimizer."},
    )
    beta: float = field(
        default=0.04,
        metadata={"help": "KL coefficient; if 0.0, the reference model is not loaded."},
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. If None, all rewards are weighted equally."
        },
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.9,
        metadata={"help": "Mixing coefficient for updating the reference model."},
    )
    ref_model_sync_steps: int = field(
        default=512,
        metadata={
            "help": "Frequency (in steps) for synchronizing the reference model."
        },
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={"help": "Whether to log completions during training."},
    )

    def __getstate__(self):
        """Custom state management for pickling support."""
        state = self.__dict__.copy()
        # Remove unpicklable attributes (SGLang engine and ZMQ-related objects)
        if "_sglang_engine" in state:
            del state["_sglang_engine"]
        return state
