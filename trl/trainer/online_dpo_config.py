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

    Args:
        completion_length (`int`, *optional*, defaults to `53`):
            Length of the completions to generate.
        temperature (`float`, *optional*, defaults to `0.7`):
            Temperature for sampling.
        missing_eos_penalty (`Optional[float]`, *optional*, defaults to `None`):
            Penalty when the model fails to generate an EOS token.
        beta (`float`, *optional*, defaults to `0.05`):
            Beta parameter for the DPO loss.
        loss_type (`str`, *optional*, defaults to `"sigmoid"`):
            Type of DPO loss to use. Possible values are:

                - `"sigmoid"`: sigmoid loss from the original [DPO](https://huggingface.co/papers/2305.18290) paper.
                - `"ipo"`: IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.

        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of workers to use to process the data.
    """

    completion_length: int = 53
    temperature: float = 0.7
    missing_eos_penalty: Optional[float] = None
    beta: float = 0.05
    loss_type: Literal["sigmoid", "ipo"] = "sigmoid"
    dataset_num_proc: Optional[int] = None
