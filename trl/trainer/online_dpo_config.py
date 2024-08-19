from dataclasses import dataclass
from typing import Literal, Optional

from transformers import TrainingArguments


@dataclass
class OnlineDPOConfig(TrainingArguments):
    # batch size related config
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    total_episodes: Optional[int] = None
    """The total number of episodes in the dataset"""
    local_rollout_forward_batch_size: int = 64
    """per rank no grad forward pass in the rollout phase"""
    num_sample_generations: int = 10
    """the number of debugging samples generations (i.e., `generate_completions` calls) throughout training"""

    # other config
    response_length: int = 53
    """the length of the response"""
    stop_token: Optional[Literal["eos"]] = None
    """the stop token"""
    stop_token_id: Optional[int] = None
    """the truncation token id"""
    temperature: float = 0.7
    """the sampling temperature"""
    penalty_reward_value: int = -1
    """the reward value for responses that do not contain `stop_token_id`"""
    non_eos_penalty: bool = False
    """whether to penalize responses that do not contain `stop_token_id`"""
    sft_model_path: str = "EleutherAI/pythia-160m"
    """the path to the sft model"""

    # various batch sizes
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    num_total_batches: Optional[int] = None
    """The number of total batches to train"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    local_mini_batch_size: Optional[int] = None
    """the mini batch size per GPU"""
    mini_batch_size: Optional[int] = None
    """the mini batch size across GPUs"""

    reward_model_path: Optional[str] = None
    """the path to the reward model"""
    judge: Optional[str] = None

    num_epochs: int = 4
    """the number of epochs to train"""

    beta: float = 0.05
    """the entropy regularization coefficient of DPO"""
    loss_type: Literal["sigmoid", "ipo"] = "sigmoid"
    """the type of loss to use for online DPO"""
    disable_dropout: bool = True
    """whether to disable dropout of the model during training"""
    dataset_num_proc: Optional[int] = None


    sanity_check: bool = False
    """wether to run in debug mode"""