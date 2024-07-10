import os
from dataclasses import dataclass
from typing import Literal, Optional

from transformers import (
    TrainingArguments,
)

from ..trainer.utils import (
    OnpolicyRuntimeConfig,
)


@dataclass
class PPOv2Config(OnpolicyRuntimeConfig, TrainingArguments):
    # common config
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    run_name: Optional[str] = None
    """a unique name of this run"""
    sanity_check: bool = False
    """wether to run in debug mode"""

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
    base_model: str = "EleutherAI/pythia-160m"
    """the name of the pretrained model to use"""
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
    reward_model_path: str = "EleutherAI/pythia-160m"
    """the path to the reward model"""
    reward_model_revision: str = "main"
    """the revision of the reward model"""
    sft_model_path: str = "EleutherAI/pythia-160m"
    """the path to the sft model"""
    sft_model_revision: str = "main"
    """the revision of the sft model"""

    # ppo config
    num_ppo_epochs: int = 4
    """the number of epochs to train"""
    vf_coef: float = 0.1
    """the value function coefficient"""
    cliprange: float = 0.2
    """the clip range"""
    cliprange_value: float = 0.2
    """the clip range for the value function"""
    gamma: float = 1
    """the discount factor"""
    lam: float = 0.95
    """the lambda value for GAE"""
    whiten_rewards: bool = False
    """whether to whiten the rewards"""
    kl_coef: float = 0.05
    """the KL coefficient"""

    # vllm deivce: only applies if we use the vllm mode
    # We actually needed to do precise model placement for the vllm model.
    # For this reason we created our own fork https://github.com/vwxyzjn/vllm/pull/1
    # you have to install via `pip install vllm-online`
    generation_backend: Literal["hf", "vllm"] = "hf"
    """the backend to use for the online generation"""
    vllm_device: str = "cuda:1"
    """the device placement of the vllm model; typically we place the vllm model on a decicated GPU"""
    vllm_gpu_memory_utilization: float = 0.8
    """the GPU memory utilization of the vllm model; passed to `gpu_memory_utilization` to the `vLLM` instance"""
