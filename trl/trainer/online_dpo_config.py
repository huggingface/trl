from dataclasses import dataclass
from typing import Literal, Optional

from transformers import TrainingArguments


@dataclass
class OnlineDPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`OnlineDPOTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        reward_model_path (`Optional[str]`, *optional*, defaults to `None`):
            Path to the reward model.
        max_new_tokens (`int`, *optional*, defaults to `64`):
            Maximum number of tokens to generate per completion.
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        missing_eos_penalty (`Optional[float]`, *optional*, defaults to `None`):
            Penalty applied to the score when the model fails to generate an EOS token. This is useful to encourage
            to generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be a positive
            value.
        beta (`float`, *optional*, defaults to `0.1`):
            Parameter controlling the deviation from the reference model. Higher β means less deviation from the
            reference model. For the IPO loss (`loss_type="ipo"`), β is the regularization parameter denoted by τ in
            the [paper](https://huggingface.co/papers/2310.12036).
        loss_type (`str`, *optional*, defaults to `"sigmoid"`):
            Type of loss to use. Possible values are:

                - `"sigmoid"`: sigmoid loss from the original [DPO](https://huggingface.co/papers/2305.18290) paper.
                - `"ipo"`: IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.

        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
    """

    reward_model_path: Optional[str] = None
    max_new_tokens: int = 64
    temperature: float = 0.9
    missing_eos_penalty: Optional[float] = None
    beta: float = 0.1
    loss_type: Literal["sigmoid", "ipo"] = "sigmoid"
    dataset_num_proc: Optional[int] = None
