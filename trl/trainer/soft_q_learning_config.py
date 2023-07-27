from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SoftQLearningConfig(object):
    """
    Configuration class for Soft Q-Learning Trainer
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of task to use - used only for tracking purposes"},
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of model to use - used only for tracking purposes"},
    )
    steps: Optional[int] = field(default=200, metadata={"help": "Number of training steps"})
    learning_rate: Optional[float] = field(default=1e-3, metadata={"help": "Adam learning rate"})
    target_learning_rate: Optional[float] = field(
        default=1e-3, metadata={"help": "Adam learning rate for the target model"}
    )
    target_sync_method: Optional[str] = field(
        default="polyak",
        metadata={"help": "Method for syncing target network weights with online network weights"},
    )

    loss_implementation: Optional[str] = field(
        default="v2_v2r_v3_v3r",
        metadata={
            "help": "Implementation of loss function. Options are 'v0' | 'v1' | 'v2' | 'v3' | 'v2_v2r' | 'v3_v3r' | 'v2_v2r_v3_v3r' "
        },
    )

    mix_strategy: Optional[str] = field(
        default="mix", metadata={"help": "Strategy for mixing on-policy/off-policy. Options are 'mix' | 'alternate'"}
    )

    loss_coefficient: Optional[float] = field(
        default=None, metadata={"help": "Coefficient to multiply with raw losses"}
    )

    loss_margin_constant: Optional[float] = field(
        default=None, metadata={"help": "Constant used in  large margin classification loss calculation"}
    )

    loss_margin_coefficient: Optional[float] = field(
        default=None, metadata={"help": "Coefficient to multiply with large margin classification loss"}
    )

    top_k: Optional[int] = field(default=None, metadata={"help": "Top k value for top k sampling"})
    top_p: Optional[float] = field(default=None, metadata={"help": "Top p value for top p sampling"})

    # adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    # init_kl_coef: Optional[float] = field(
    #    default=0.2,
    #    metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    # )
    # target: Optional[float] = field(default=6, metadata={"help": "Target KL value for adaptive KL control"})
    # horizon: Optional[float] = field(default=10000, metadata={"help": "Horizon for adaptive KL control"})
    # gamma: Optional[float] = field(default=1, metadata={"help": "Gamma parameter for advantage calculation"})
    # lam: Optional[float] = field(default=0.95, metadata={"help": "Lambda parameter for advantage calculation"})
    # cliprange: Optional[float] = field(
    #    default=0.2, metadata={"help": "Range for clipping in PPO policy gradient loss"}
    # )
    # cliprange_value: Optional[float] = field(
    #    default=0.2, metadata={"help": "Range for clipping values in loss calculation"}
    # )
    # vf_coef: Optional[float] = field(default=0.1, metadata={"help": "Scaling factor for value loss"})
    batch_size: Optional[int] = field(default=256, metadata={"help": "Number of samples per optimisation step"})
    forward_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples forward passed through model at a time"},
    )
    mini_batch_size: Optional[int] = field(
        default=1, metadata={"help": "Number of samples optimized inside PPO together"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "The number of gradient accumulation steps"}
    )
    # ppo_epochs: Optional[int] = field(
    #    default=4,
    #    metadata={"help": "Number of optimisation epochs per batch of samples"},
    # )
    # remove_unused_columns: Optional[bool] = field(
    #    default=True,
    #    metadata={"help": "Remove unused columns from the dataset if `datasets.Dataset` is used"},
    # )
    log_with: Optional[str] = field(
        default=None,
        metadata={
            "help": "Log with either 'wandb' or 'tensorboard', check  https://huggingface.co/docs/accelerate/usage_guides/tracking for more details"
        },
    )
    tracker_kwargs: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments for the tracker (e.g. wandb_project)"},
    )
    accelerator_kwargs: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments for the accelerator"},
    )
    project_kwargs: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments for the accelerator project config (e.g. `logging_dir`)"},
    )
    tracker_project_name: Optional[str] = field(
        default="trl", metadata={"help": "Name of project to use for tracking"}
    )
    max_grad_norm: Optional[float] = field(
        default=None, metadata={"help": "Maximum gradient norm for gradient clipping"}
    )
    seed: Optional[int] = field(default=0, metadata={"help": "Seed value for random generations"})
    optimize_cuda_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Optimize CUDA cache for slightly more memory-efficient training"},
    )
    # early_stopping: Optional[bool] = field(
    #    default=False, metadata={"help": "Whether to stop the PPO optimization loop early is the KL too high"}
    # )
    # target_kl: Optional[float] = field(
    #    default=0.1, metadata={"help": "Stop early if we exceed this value by over 50%"}
    # )
    # push_to_hub_if_best_kwargs: Optional[dict] = field(
    #    default_factory=dict,
    #    metadata={"help": "Keyword arguments for pushing model to the hub during training (e.g. repo_id)"},
    # )
    # compare_steps: Optional[int] = field(
    #    default=1,
    #    metadata={"help": "Number of steps between comparison of the current reward with the best seen so far"},
    # )

    def __post_init__(self):
        if self.loss_implementation not in ["v0", "v1", "v2", "v3", "v2_v2r", "v3_v3r", "v2_v2r_v3_v3r"]:
            raise ValueError(
                "loss_implementation must be one of 'v0' | 'v1' | 'v2' | 'v3' | 'v2_v2r' | 'v3_v3r' | 'v2_v2r_v3_v3r' "
            )

        if self.mix_strategy not in ["mix", "alternate"]:
            raise ValueError("mix_strategy must be one of 'mix' or 'alternate'")

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return output_dict
