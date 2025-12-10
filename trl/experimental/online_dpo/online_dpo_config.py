# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

from ...trainer.grpo_config import GRPOConfig


@dataclass
class OnlineDPOConfig(GRPOConfig):
    r"""
    Configuration class for the [`experimental.online_dpo.OnlineDPOTrainer`].

    This class includes only the parameters that are specific to Online DPO training. For a full list of training
    arguments, please refer to the [`~transformers.TrainingArguments`] and [`GRPOConfig`] documentation. Note that
    default values in this class may differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
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
    """

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
    # Set default values for num_generations and num_generations_eval to 2 for Online DPO
    num_generations: int = field(
        default=2,
        metadata={
            "help": "Number of completions to generate per prompt during training. Must be set to 2 for Online DPO."
        },
    )
    num_generations_eval: int = field(
        default=2,
        metadata={
            "help": "Number of completions to generate per prompt during evaluation. Must be set to 2 for Online DPO."
        },
    )

    def __post_init__(self):
        # Instead of `missing_eos_penalty` it is possible to use `get_soft_overlong_punishment` reward function from
        # `trl.rewards.other_rewards` to penalize overlong completions.

        # Important: Online DPO requires exactly 2 generations per prompt
        if self.num_generations != 2 or self.num_generations_eval != 2:
            warnings.warn(
                "`num_generations` and `num_generations_eval` are forced to 2 in Online DPO Trainer. Overriding your "
                "values."
            )
        self.num_generations = 2
        self.num_generations_eval = 2

        super().__post_init__()
        # Important note: actual "generation batch size" used for generations is multiplied by two inside of
        # OnlineDPOTrainer to account for two completions per prompt and subsequent merging of those completions.
