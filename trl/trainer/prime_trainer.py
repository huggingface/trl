import torch
from typing import Optional, Union, Callable
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from .grpo_trainer import GRPOTrainer
from .prime_config import PrimeConfig

class PrimeTrainer(GRPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args: PrimeConfig = None,
        reward_function: Optional[Callable] = None,
        reward_model: Optional[PreTrainedModel] = None,
        **kwargs
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = PrimeConfig(f"{model_name}-PRIME")
            
        self.reward_function = reward_function
        super().__init__(model=model, args=args, reward_model=reward_model, **kwargs)
        
        # Initialize metrics specific to PRIME
        self._metrics.update({
            "verifier_reward": [],
            "rm_reward": [],
            "process_reward": [],
            "correct_ratio": []
        })

    def filter_batch(self, rewards, batch):
        """Filter batch based on correct response ratio thresholds"""
        batch_size = len(batch) // self.args.num_generations
        rewards = rewards.view(batch_size, self.args.num_generations)
        
        # Calculate correct response ratio per prompt
        correct_responses = (rewards > 0.5).float().sum(dim=1) / self.args.num_generations
        
        # Apply thresholds
        min_thresh, max_thresh = self.args.correct_ratio_threshold
        valid_mask = (correct_responses > min_thresh) & (correct_responses < max_thresh)
        
        # Expand mask for all generations
        final_mask = valid_mask.repeat_interleave(self.args.num_generations)
        
        filtered_batch = {k: v[final_mask] for k, v in batch.items() if isinstance(v, torch.Tensor)}
        return filtered_batch, valid_mask

    def compute_loss(self, model, inputs, return_outputs=False):
        # Get completions and compute base rewards similar to GRPO
        loss, rewards, completions = super().compute_base_rewards(model, inputs)
        
        # Compute verifier rewards using reward function
        if self.reward_function is not None:
            verifier_rewards = self.reward_function(completions)
            rewards += self.args.verifier_reward_coef * verifier_rewards
            self._metrics["verifier_reward"].append(verifier_rewards.mean().item())

        # Filter batch based on correct ratio
        filtered_batch, valid_mask = self.filter_batch(rewards, inputs)
        if len(filtered_batch) == 0:
            return loss
            
        # Compute process rewards using implicit PRM
        process_rewards = self.compute_process_rewards(
            model, 
            filtered_batch,
            granularity=self.args.prime_granularity,
            norm_type=self.args.prime_norm
        )
        
        # Update metrics
        self._metrics["correct_ratio"].append(valid_mask.float().mean().item())
        self._metrics["process_reward"].append(process_rewards.mean().item())
        
        # Combine rewards and compute PPO loss
        total_rewards = rewards + process_rewards
        ppo_loss = self.compute_ppo_loss(model, filtered_batch, total_rewards)
        
        return ppo_loss

    def compute_process_rewards(self, model, batch, granularity="token", norm_type="batch_norm"):
        """Compute process rewards using the implicit PRM"""
        with torch.no_grad():
            # Get logits from current policy and reference model
            policy_logits = model(**batch).logits
            ref_logits = self.ref_model(**batch).logits
            
            # Compute KL divergence
            kl_div = torch.nn.functional.kl_div(
                policy_logits.log_softmax(-1),
                ref_logits.softmax(-1),
                reduction='none'
            ).sum(-1)
            
            # Apply normalization
            if norm_type == "batch_norm":
                kl_div = (kl_div - kl_div.mean()) / (kl_div.std() + 1e-8)
                
            # Convert to process rewards
            process_rewards = -self.args.beta_train * kl_div
            
            if granularity == "whole":
                process_rewards = process_rewards.mean(dim=1, keepdim=True).expand(-1, process_rewards.size(1))
                
        return process_rewards