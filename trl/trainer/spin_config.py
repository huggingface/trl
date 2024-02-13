from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class SPINConfig(transformers.TrainingArguments):

    beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta factor in SPIN loss. Higher beta means less divergence from the initial policy."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={"help": ("The maximum length of the prompt to use for conditioning the model.")},
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": ("Used by TRL for reward model training, which tries to read this parameter in init.")},
    )
    optim: Optional[str] = field(default="rmsprop")
    remove_unused_columns: bool = field(default=False)
