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
from typing import Literal

from transformers import TrainingArguments


@dataclass
class PPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`experimental.ppo.PPOTrainer`].

    This class includes only the parameters that are specific to PPO training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        num_mini_batches (`int`, *optional*, defaults to `1`):
            Number of minibatches to split a batch into.
        total_episodes (`int`, *optional*):
            Total number of episodes in the dataset.
        local_rollout_forward_batch_size (`int`, *optional*, defaults to `64`):
            Per rank no grad forward pass in the rollout phase.
        num_sample_generations (`int`, *optional*, defaults to `10`):
            Number of debugging samples generations (i.e., `generate_completions` calls) throughout training.
        response_length (`int`, *optional*, defaults to `53`):
            Length of the response.
        stop_token (`str`, *optional*):
            Specifies the stop token to use for text generation. This parameter is mutually exclusive with
            `stop_token_id`.

            - `None`: No stop token is applied, unless `stop_token_id` is specified.
            - `'eos'`: Uses the tokenizer's `eos_token`.

        stop_token_id (`int`, *optional*):
            Specifies the ID of the stop token to use for text generation. If `None`, no stop token ID is applied,
            unless `stop_token` is specified. This parameter is mutually exclusive with `stop_token`.
        temperature (`float`, *optional*, defaults to `0.7`):
            Sampling temperature.
        missing_eos_penalty (`float`, *optional*):
            Penalty applied to the score when the model fails to generate an EOS token. This is useful to encourage to
            generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be a positive
            value.
        sft_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
            Path to the SFT model.
        world_size (`int`, *optional*):
            Number of processes (GPUs) to use for the training.
        num_total_batches (`int`, *optional*):
            Number of total batches to train.
        micro_batch_size (`int`, *optional*):
            Micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`).
        local_batch_size (`int`, *optional*):
            Batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`).
        batch_size (`int`, *optional*):
            Batch size across devices (HF's `per_device_train_batch_size` * `world_size` *
            `gradient_accumulation_steps`).
        local_mini_batch_size (`int`, *optional*):
            Mini batch size per GPU.
        mini_batch_size (`int`, *optional*):
            Mini batch size across GPUs.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the model to the Hub after training.
        reward_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
            Path to the reward model.
        model_adapter_name (`str`, *optional*):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`str`, *optional*):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
        num_ppo_epochs (`int`, *optional*, defaults to `4`):
            Number of epochs to train.
        whiten_rewards (`bool`, *optional*, defaults to `False`):
            Whether to whiten the rewards.
        kl_coef (`float`, *optional*, defaults to `0.05`):
            KL coefficient.
        kl_estimator (`Literal["k1", "k3"]`, *optional*, defaults to `"k1"`):
            Which estimator for KL-Divergence to use from [Approximating KL
            Divergence](http://joschu.net/blog/kl-approx.html). Defaults to "k1", a straightforward, unbiased
            estimator. Can be set to "k3", an unbiased estimator with lower variance which "appears to be a strictly
            better estimator". Cannot be set to "k2", as it is used for logging purposes.
        cliprange (`float`, *optional*, defaults to `0.2`):
            Clip range.
        vf_coef (`float`, *optional*, defaults to `0.1`):
            Value function coefficient.
        cliprange_value (`float`, *optional*, defaults to `0.2`):
            Clip range for the value function.
        gamma (`float`, *optional*, defaults to `1.0`):
            Discount factor.
        lam (`float`, *optional*, defaults to `0.95`):
            Lambda value for GAE.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation.
    """

    # Parameters whose default values are overridden from TrainingArguments
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
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
    # Transformers 4.57.0 introduced a bug that caused the dtype of `lr_scheduler_kwargs` to be unparsable. This issue
    # was fixed in https://github.com/huggingface/transformers/pull/41322 and released in 4.57.5. We add a temporary
    # workaround here, which can be removed once we drop support for versions older than 4.57.5.
    lr_scheduler_kwargs: dict | str | None = field(
        default=None,
        metadata={
            "help": "Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard "
            "restarts."
        },
    )

    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    num_mini_batches: int = field(
        default=1,
        metadata={"help": "Number of minibatches to split a batch into."},
    )
    total_episodes: int | None = field(
        default=None,
        metadata={"help": "Total number of episodes in the dataset."},
    )
    local_rollout_forward_batch_size: int = field(
        default=64,
        metadata={"help": "Per rank no grad forward pass in the rollout phase."},
    )
    num_sample_generations: int = field(
        default=10,
        metadata={
            "help": "Number of debugging samples generations (i.e., `generate_completions` calls) throughout training."
        },
    )
    response_length: int = field(
        default=53,
        metadata={"help": "Length of the response."},
    )
    stop_token: Literal["eos"] | None = field(
        default=None,
        metadata={
            "help": "Specifies the stop token to use for text generation. This parameter is mutually exclusive with "
            "`stop_token_id`."
        },
    )
    stop_token_id: int | None = field(
        default=None,
        metadata={
            "help": "Specifies the ID of the stop token to use for text generation. If `None`, no stop token ID is "
            "applied, unless `stop_token` is specified. This parameter is mutually exclusive with `stop_token`."
        },
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature."},
    )
    missing_eos_penalty: float | None = field(
        default=None,
        metadata={
            "help": "Penalty applied to the score when the model fails to generate an EOS token. This is useful to "
            "encourage to generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be "
            "a positive value."
        },
    )
    sft_model_path: str = field(
        default="EleutherAI/pythia-160m",
        metadata={"help": "Path to the SFT model."},
    )
    world_size: int | None = field(
        default=None,
        metadata={"help": "Number of processes (GPUs) to use for the training."},
    )
    num_total_batches: int | None = field(
        default=None,
        metadata={"help": "Number of total batches to train."},
    )
    micro_batch_size: int | None = field(
        default=None,
        metadata={"help": "Micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)."},
    )
    local_batch_size: int | None = field(
        default=None,
        metadata={"help": "Batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)."},
    )
    batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Batch size across devices (HF's `per_device_train_batch_size` * `world_size` * "
            "`gradient_accumulation_steps`)."
        },
    )
    local_mini_batch_size: int | None = field(
        default=None,
        metadata={"help": "Mini batch size per GPU."},
    )
    mini_batch_size: int | None = field(
        default=None,
        metadata={"help": "Mini batch size across GPUs."},
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the model to the Hub after training."},
    )
    reward_model_path: str = field(
        default="EleutherAI/pythia-160m",
        metadata={"help": "Path to the reward model."},
    )
    model_adapter_name: str | None = field(
        default=None,
        metadata={"help": "Name of the train target PEFT adapter, when using LoRA with multiple adapters."},
    )
    ref_adapter_name: str | None = field(
        default=None,
        metadata={"help": "Name of the reference PEFT adapter, when using LoRA with multiple adapters."},
    )
    num_ppo_epochs: int = field(
        default=4,
        metadata={"help": "Number of epochs to train."},
    )
    whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to whiten the rewards."},
    )
    kl_coef: float = field(
        default=0.05,
        metadata={"help": "KL coefficient."},
    )
    kl_estimator: Literal["k1", "k3"] = field(
        default="k1",
        metadata={
            "help": "Which estimator for KL-Divergence to use from Approximating KL Divergence "
            "(http://joschu.net/blog/kl-approx.html). Defaults to 'k1', a straightforward, unbiased estimator. Can be "
            "set to 'k3', an unbiased estimator with lower variance which 'appears to be a strictly better "
            "estimator'. Cannot be set to 'k2', as it is used for logging purposes."
        },
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "Clip range."},
    )
    vf_coef: float = field(
        default=0.1,
        metadata={"help": "Value function coefficient."},
    )
    cliprange_value: float = field(
        default=0.2,
        metadata={"help": "Clip range for the value function."},
    )
    gamma: float = field(
        default=1.0,
        metadata={"help": "Discount factor."},
    )
    lam: float = field(
        default=0.95,
        metadata={"help": "Lambda value for GAE."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation."
        },
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        super().__post_init__()
