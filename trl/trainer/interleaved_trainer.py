import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedModel, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction

from .grpo_trainer import GRPOTrainer
from .sft_trainer import SFTTrainer
from .interleaved_config import InterleaveConfig


class InterleaveTrainer:
    """
    A trainer that alternates between SFT and GRPO training every epoch.
    
    This trainer manages two separate trainers (SFT and GRPO) and switches between them
    based on the current epoch. It allows for interleaved training where the model
    alternates between supervised fine-tuning and reward-guided policy optimization.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, str],
        args: InterleaveConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        reward_function: Optional[Callable] = None,
        tokenizer=None,
        sft_kwargs: Optional[Dict[str, Any]] = None,
        grpo_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize the interleaved trainer."""
        self.args = args
        
        # Initialize SFT trainer with its specific parameters
        sft_kwargs = sft_kwargs or {}
        base_sft_kwargs = {
            "model": model,
            "args": args.sft_config,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "callbacks": callbacks,
            "compute_metrics": compute_metrics,
            "preprocess_logits_for_metrics": preprocess_logits_for_metrics,
            "processing_class": tokenizer,  # Pass tokenizer as processing_class
        }
        base_sft_kwargs.update(sft_kwargs)  # Add any SFT-specific kwargs
        base_sft_kwargs.update(kwargs)  # Add any remaining kwargs
        self.sft_trainer = SFTTrainer(**base_sft_kwargs)
        
        # Initialize GRPO trainer with its specific parameters
        grpo_kwargs = grpo_kwargs or {}
        base_grpo_kwargs = {
            "model": model,
            "args": args.grpo_config,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "callbacks": callbacks,
            "reward_funcs": reward_function,  # GRPO specific parameter
            "processing_class": tokenizer,  # Pass tokenizer as processing_class
        }
        base_grpo_kwargs.update(grpo_kwargs)  # Add any GRPO-specific kwargs
        # Add any remaining kwargs except those specific to SFT
        base_grpo_kwargs.update({k: v for k, v in kwargs.items() 
                               if k not in ["compute_metrics", "preprocess_logits_for_metrics"]})
        self.grpo_trainer = GRPOTrainer(**base_grpo_kwargs)
        
        # Track current epoch and training phase
        self.current_epoch = 0
        self.is_sft_phase = args.start_with_sft
        
        # Store the active trainer
        self._active_trainer = self.sft_trainer if self.is_sft_phase else self.grpo_trainer

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.
        
        This method manages the alternating training process between SFT and GRPO.
        At the end of each epoch, it switches between the two training approaches.
        """
        
        # Calculate total number of epochs
        total_epochs = self.args.num_train_epochs
        
        for epoch in range(self.current_epoch, int(total_epochs)):
            self.current_epoch = epoch
            
            # Determine which trainer to use for this epoch
            self.is_sft_phase = (epoch % 2 == 0) if self.args.start_with_sft else (epoch % 2 == 1)
            self._active_trainer = self.sft_trainer if self.is_sft_phase else self.grpo_trainer
            
            # Log the current training phase
            phase = "SFT" if self.is_sft_phase else "GRPO"
            print(f"\nStarting epoch {epoch + 1}/{total_epochs} with {phase} training")
            
            # Run one epoch of training
            self._active_trainer.train(
                resume_from_checkpoint=resume_from_checkpoint if epoch == 0 else None,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
                **kwargs
            )
            
            # Sync model weights between trainers
            self._sync_model_weights()
    
    def _sync_model_weights(self):
        """Synchronize model weights between SFT and GRPO trainers."""
        if self.is_sft_phase:
            # Copy weights from SFT to GRPO
            self.grpo_trainer.model.load_state_dict(self.sft_trainer.model.state_dict())
        else:
            # Copy weights from GRPO to SFT
            self.sft_trainer.model.load_state_dict(self.grpo_trainer.model.state_dict())
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation using both SFT and GRPO evaluation methods.
        
        Returns combined metrics from both evaluation approaches.
        """
        # Run both evaluations
        sft_metrics = self.sft_trainer.evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=f"sft_{metric_key_prefix}"
        )
        
        grpo_metrics = self.grpo_trainer.evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=f"grpo_{metric_key_prefix}"
        )
        
        # Combine metrics
        combined_metrics = {**sft_metrics, **grpo_metrics}
        
        # Add weighted average of key metrics if they exist in both
        for key in sft_metrics:
            if key.startswith(f"sft_{metric_key_prefix}"):
                grpo_key = f"grpo_{key[4:]}"  # Replace "sft_" with "grpo_"
                if grpo_key in grpo_metrics:
                    # Calculate weighted average
                    weighted_avg = (
                        self.args.sft_weight * sft_metrics[key] +
                        (1 - self.args.sft_weight) * grpo_metrics[grpo_key]
                    )
                    # Store with "combined_" prefix
                    combined_key = f"combined_{key[4:]}"  # Remove "sft_" prefix
                    combined_metrics[combined_key] = weighted_avg
        
        return combined_metrics
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save the current model state."""
        # Save using the active trainer
        self._active_trainer.save_model(output_dir, _internal_call)
    
    def save_state(self):
        """Save trainer state."""
        # Save state for both trainers
        self.sft_trainer.save_state()
        self.grpo_trainer.save_state()
    
    @property
    def model(self) -> PreTrainedModel:
        """Get the current model."""
        return self._active_trainer.model
    
    @property
    def tokenizer(self):
        """Get the tokenizer."""
        return self._active_trainer.tokenizer
    
    @property
    def data_collator(self):
        """Get the current data collator."""
        return self._active_trainer.data_collator 