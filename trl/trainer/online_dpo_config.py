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
        reward_model_path (`Optional[str]`, *optional*, defaults to `None`):
            Path to the reward model.
        max_new_tokens (`int`, *optional*, defaults to `64`):
            The maximum number of tokens to generate per completion.
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        missing_eos_penalty (`Optional[float]`, *optional*, defaults to `None`):
            Penalty when the model fails to generate an EOS token.
        beta (`float`, *optional*, defaults to `0.1`):
            Beta parameter for the DPO loss.
        loss_type (`str`, *optional*, defaults to `"sigmoid"`):
            Type of DPO loss to use. Possible values are:

                - `"sigmoid"`: sigmoid loss from the original [DPO](https://huggingface.co/papers/2305.18290) paper.
                - `"ipo"`: IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.

        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of workers to use to process the data.
    """

    reward_model_path: Optional[str] = None
    max_new_tokens: int = 53
    temperature: float = 0.9
    missing_eos_penalty: Optional[float] = None
    beta: float = 0.1
    loss_type: Literal["sigmoid", "ipo"] = "sigmoid"
    dataset_num_proc: Optional[int] = None
