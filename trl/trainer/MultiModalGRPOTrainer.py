from typing import Any, Optional, Union
import torch
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl.trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig

class MultimodalGRPOTrainer(GRPOTrainer):
    """
    Custom GRPO Trainer that handles multimodal inputs with images and text.
    """
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[str, PreTrainedModel, list],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list]] = None,
        data_collator=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            data_collator=data_collator,
            **kwargs
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The MultimodalGRPOTrainer does not support returning outputs")

        # Extract image and text inputs
        prompt_inputs = {
            k: v for k, v in inputs.items() 
            if k in ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        }
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs, 
                generation_config=self.generation_config
            )
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Get the per-token log probabilities for the completions
        def get_per_token_logps(model, input_ids, prompt_inputs):
            # Include image inputs when computing logits
            model_inputs = {
                "input_ids": input_ids,
                "pixel_values": prompt_inputs["pixel_values"],
                "image_grid_thw": prompt_inputs["image_grid_thw"],
            }
            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]
            input_ids = input_ids[:, 1:]
            
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        per_token_logps = get_per_token_logps(model, prompt_completion_ids, prompt_inputs)
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids, prompt_inputs)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids, prompt_inputs)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Rest of the GRPO logic remains the same
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Decode completions and compute rewards using parent class method
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Let the parent class handle reward computation - it will properly pass all dataset columns
        rewards_per_func = super()._compute_rewards(inputs, completions)

        # Sum rewards and compute loss
        rewards = rewards_per_func.sum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Update metrics
        self._update_metrics(completion_mask, rewards_per_func, rewards, std_grouped_rewards, per_token_kl)

        return loss

    def _update_metrics(self, completion_mask, rewards_per_func, rewards, std_grouped_rewards, per_token_kl):
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item()) 