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

from trl.trainer.base_config import _BaseConfig


@dataclass
class A2POConfig(_BaseConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`A2POTrainer`].

    This class includes only the parameters that are specific to A2PO training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`A2POTrainer`] is provided as a string.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to
            [`~transformers.AutoModelForCausalLM.from_pretrained`] and [`~transformers.AutoTokenizer.from_pretrained`].

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.

        > Parameters that control generation

        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. If the prompt is longer than this, it is left-truncated.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        temperature (`float`, *optional*, defaults to `1.0`):
            Sampling temperature, used in both Stage 1 and Stage 2 generation.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int` or `None`, *optional*):
            Number of highest-probability vocabulary tokens to keep. If `None`, top-k filtering is disabled.

        > Parameters that control Stage 1 (offline optimal value estimation)

        num_value_samples (`int`, *optional*, defaults to `8`):
            Number of samples drawn from the reference policy per prompt to estimate `V*`.
        beta1 (`float`, *optional*, defaults to `0.5`):
            KL temperature used to estimate `V*` in Stage 1.
        filter_all_incorrect (`bool`, *optional*, defaults to `True`):
            Whether to drop prompts for which all reference samples are incorrect.

        > Parameters that control Stage 2 (on-policy regression)

        beta2 (`float`, *optional*, defaults to `1e-3`):
            KL temperature used in the Stage 2 regression target.
        reward_weights (`list[float]`, *optional*):
            Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are
            weighted equally with weight `1.0`.
    """

    _VALID_DICT_FIELDS = _BaseConfig._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters that control the model and reference model
    model_init_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `A2POTrainer` is provided as a string."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow loading models and tokenizers that ship custom Python code from the Hub. "
            "Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`."
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
    max_prompt_length: int | None = field(
        default=512,
        metadata={"help": "Maximum length of the prompt. If the prompt is longer than this, it is left-truncated."},
    )
    max_completion_length: int | None = field(
        default=256, metadata={"help": "Maximum length of the generated completion."}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature, used in both Stage 1 and Stage 2 generation."},
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to `1.0` to consider all tokens."
        },
    )
    top_k: int | None = field(
        default=None,
        metadata={
            "help": "Number of highest-probability vocabulary tokens to keep. If `None`, top-k filtering is disabled."
        },
    )

    # Parameters that control Stage 1 (offline optimal value estimation)
    num_value_samples: int = field(
        default=8,
        metadata={"help": "Number of samples drawn from the reference policy per prompt to estimate `V*`."},
    )
    beta1: float = field(
        default=0.5,
        metadata={"help": "KL temperature used to estimate `V*` in Stage 1."},
    )
    filter_all_incorrect: bool = field(
        default=True,
        metadata={"help": "Whether to drop prompts for which all reference samples are incorrect."},
    )

    # Parameters that control Stage 2 (on-policy regression)
    beta2: float = field(
        default=1e-3,
        metadata={"help": "KL temperature used in the Stage 2 regression target."},
    )
    reward_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
