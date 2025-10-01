#!/usr/bin/env python3

# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Enhanced Supervised Fine-Tuning with Multi-Token Prediction (MTP) Demo Script

This script demonstrates how to fine-tune mainstream language models with enhanced MTP features:
1. Identical head structure copying from LM head
2. Multi-layer MTP heads for increased capacity  
3. Flexible parameter initialization strategies
4. Various head architectures (linear, ffn, mha_ffn, cnn, identical)

Supported models include Llama, Qwen, DeepSeek, Mistral, and other popular architectures.

Example usage:
    # Complete MTP training with Qwen2.5 and evaluation
    CUDA_VISIBLE_DEVICES=0 python examples/scripts/sft_with_mtp.py \
        --model_name_or_path Qwen/Qwen2.5-0.5B \
        --dataset_name trl-lib/Capybara \
        --train_split train \
        --eval_split test \
        --output_dir /root/autodl-tmp/qwen-identical-mtp \
        --mtp_enabled \
        --mtp_head_type identical \
        --mtp_init_strategy copy_lm_head \
        --mtp_num_layers 1 \
        --mtp_num_predictions 2 \
        --mtp_loss_weight 0.5 \
        --learning_rate 2e-5 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 3 \
        --warmup_steps 100 \
        --weight_decay 0.01 \
        --lr_scheduler_type cosine \
        --max_length 512 \
        --save_steps 100 \
        --logging_steps 10 \
        --eval_steps 100

    # MTP training with UltraChat dataset
    python examples/scripts/sft_with_mtp.py \
        --model_name_or_path Qwen/Qwen2.5-0.5B \
        --dataset_name HuggingFaceH4/ultrachat_200k \
        --train_split train_sft \
        --eval_split test_sft \
        --mtp_enabled \
        --mtp_head_type identical \
        --mtp_num_predictions 2 \
        --output_dir /root/autodl-tmp/qwen-mtp-ultrachat

    # MTP training with math problems dataset  
    python examples/scripts/sft_with_mtp.py \
        --model_name_or_path Qwen/Qwen2.5-0.5B \
        --dataset_name microsoft/orca-math-word-problems-200k \
        --train_split train \
        --mtp_enabled \
        --mtp_head_type ffn \
        --mtp_num_predictions 3 \
        --output_dir /root/autodl-tmp/qwen-mtp-math

    # --dataset_name trl-lib/Capybara \
    # --train_split train \
    # --eval_split test \
    # --output_dir /root/autodl-tmp/Qwen2.5-0.5B-MTP-Identical-Capybara \
        
    # Multi GPU training using accelerate config with evaluation
    nohup accelerate launch \
        --config_file examples/accelerate_configs/multi_gpu.yaml \
        examples/scripts/sft_with_mtp.py \
        --model_name_or_path Qwen/Qwen2.5-0.5B \
        --dataset_name trl-lib/Capybara \
        --train_split train \
        --eval_split test \
        --output_dir /root/autodl-tmp/Qwen2.5-0.5B-MTP-Identical-Capybara \
        --mtp_enabled \
        --mtp_num_predictions 2 \
        --mtp_head_type identical \
        --mtp_init_strategy copy_lm_head \
        --mtp_num_layers 2 \
        --mtp_loss_weight 0.5 \
        --learning_rate 2e-5 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 1 \
        --warmup_steps 50 \
        --max_length 1024 \
        --save_steps 250 \
        --logging_steps 10 \
        --eval_strategy epoch \
        --bf16 true

    # Identical head structure with parameter copying
    python sft_with_mtp.py \
        --model_name_or_path Qwen/Qwen2.5-0.5B \
        --dataset_name trl-lib/Capybara \
        --mtp_enabled \
        --mtp_head_type identical \
        --mtp_init_strategy copy_lm_head \
        --mtp_num_layers 1 \
        --output_dir /root/autodl-tmp/qwen-identical-mtp

    # Multi-layer FFN heads with advanced initialization
    python sft_with_mtp.py \
        --model_name_or_path meta-llama/Llama-3.2-1B \
        --dataset_name HuggingFaceH4/ultrachat_200k \
        --mtp_enabled \
        --mtp_head_type ffn \
        --mtp_num_layers 3 \
        --mtp_init_strategy kaiming_uniform \
        --mtp_weight_decay_strategy harmonic \
        --output_dir /root/autodl-tmp/llama-multilayer-mtp

    # Deep MHA+FFN heads for complex modeling (NEW!)
    python sft_with_mtp.py \
        --model_name_or_path deepseek-ai/deepseek-llm-7b-base \
        --dataset_name trl-lib/Capybara \
        --mtp_enabled \
        --mtp_head_type mha_ffn \
        --mtp_num_layers 4 \
        --mtp_num_predictions 4 \
        --mtp_init_strategy xavier_normal \
        --output_dir /root/autodl-tmp/deepseek-deep-mtp

    # Parameter-efficient setup with weight tying
    python sft_with_mtp.py \
        --model_name_or_path mistralai/Mistral-7B-v0.1 \
        --dataset_name trl-lib/Capybara \
        --mtp_enabled \
        --mtp_head_type linear \

        --mtp_num_layers 1 \
        --mtp_dropout_prob 0.0 \
        --output_dir /root/autodl-tmp/mistral-efficient-mtp
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser

from trl import SFTConfig, SFTTrainer


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""
    
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split: str = field(
        default="train",
        metadata={"help": "The name of the train split in the dataset."}
    )
    eval_split: Optional[str] = field(
        default="test",
        metadata={"help": "The name of the evaluation split in the dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples."},
    )


def analyze_mtp_configuration(trainer, training_args):
    """Analyze and log detailed MTP configuration and model structure."""
    if not training_args.mtp_enabled or not hasattr(trainer.model, 'mtp_heads'):
        return
    
    mtp_heads = trainer.model.mtp_heads
    logger.info("=" * 60)
    logger.info("ENHANCED MTP CONFIGURATION ANALYSIS")
    logger.info("=" * 60)
    
    # Basic configuration
    logger.info("MTP Configuration:")
    logger.info(f"  - Head type: {training_args.mtp_head_type}")
    logger.info(f"  - Number of predictions: {training_args.mtp_num_predictions}")
    logger.info(f"  - Number of layers per head: {training_args.mtp_num_layers}")
    logger.info(f"  - Initialization strategy: {training_args.mtp_init_strategy}")
    logger.info(f"  - Loss weight: {training_args.mtp_loss_weight}")
    logger.info(f"  - Weight decay strategy: {training_args.mtp_weight_decay_strategy}")
    logger.info(f"  - Dropout probability: {training_args.mtp_dropout_prob}")
    
    # Parameter analysis
    total_model_params = sum(p.numel() for p in trainer.model.parameters())
    mtp_params = sum(p.numel() for p in mtp_heads.parameters())
    trainable_mtp_params = sum(p.numel() for p in mtp_heads.parameters() if p.requires_grad)
    
    logger.info(f"\nParameter Analysis:")
    logger.info(f"  - Total model parameters: {total_model_params:,}")
    logger.info(f"  - MTP head parameters: {mtp_params:,}")
    logger.info(f"  - Trainable MTP parameters: {trainable_mtp_params:,}")
    logger.info(f"  - MTP overhead: {(mtp_params / total_model_params * 100):.2f}%")
    
    # Compare with LM head
    if hasattr(trainer.model, 'lm_head'):
        lm_head_params = sum(p.numel() for p in trainer.model.lm_head.parameters())
        ratio = mtp_params / lm_head_params if lm_head_params > 0 else 0
        logger.info(f"  - LM head parameters: {lm_head_params:,}")
        logger.info(f"  - MTP/LM head ratio: {ratio:.2f}x")
    
    # Head structure analysis
    logger.info(f"\nHead Structure Analysis:")
    for i, head in enumerate(mtp_heads.heads):
        logger.info(f"  Head {i+1} ({training_args.mtp_head_type}):")
        if isinstance(head, torch.nn.Linear):
            logger.info(f"    - Linear: {head.in_features} -> {head.out_features}")
            logger.info(f"    - Parameters: {sum(p.numel() for p in head.parameters()):,}")
        elif isinstance(head, torch.nn.Sequential):
            logger.info(f"    - Sequential with {len(head)} layers")
            total_head_params = sum(p.numel() for p in head.parameters())
            logger.info(f"    - Parameters: {total_head_params:,}")
            for j, layer in enumerate(head):
                if isinstance(layer, torch.nn.Linear):
                    logger.info(f"      Layer {j+1}: Linear {layer.in_features} -> {layer.out_features}")
                else:
                    logger.info(f"      Layer {j+1}: {type(layer).__name__}")
        elif hasattr(head, 'layers'):  # MHA+FFN or CNN heads
            total_head_params = sum(p.numel() for p in head.parameters())
            logger.info(f"    - {type(head).__name__} with {len(head.layers)} transformer layers")
            logger.info(f"    - Parameters: {total_head_params:,}")
    
    logger.info("=" * 60)


def test_mtp_predictions(trainer, tokenizer, training_args):
    """Test MTP predictions and show example outputs."""
    if not training_args.mtp_enabled or not hasattr(trainer.model, 'mtp_heads'):
        return
    
    logger.info("TESTING MTP PREDICTIONS")
    logger.info("=" * 40)
    
    # Create test input
    test_texts = [
        "The future of artificial intelligence is",
        "Multi-token prediction helps models to",
        "In the context of language modeling,",
    ]
    
    trainer.model.eval()
    
    for test_text in test_texts:
        logger.info(f"\nTest input: '{test_text}'")
        inputs = tokenizer(test_text, return_tensors="pt", padding=True)
        
        # Move to model device
        inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}
        
        # Set to training mode to enable MTP
        trainer.model.train()
        
        with torch.no_grad():
            outputs = trainer.model(**inputs)
        
        if hasattr(outputs, 'mtp_logits') and outputs.mtp_logits is not None:
            input_ids = inputs['input_ids'][0]
            
            # Show predictions for the last input token
            last_pos = input_ids.shape[0] - 1
            current_token = tokenizer.decode([input_ids[last_pos].item()])
            
            logger.info(f"  Current token: '{current_token}'")
            logger.info(f"  MTP predictions:")
            
            for i, mtp_logit in enumerate(outputs.mtp_logits):
                # Get top 3 predictions
                top_k_probs, top_k_ids = torch.topk(torch.softmax(mtp_logit[0, last_pos, :], dim=-1), k=3)
                top_tokens = [tokenizer.decode([token_id.item()]) for token_id in top_k_ids]
                
                logger.info(f"    Step t+{i+1}: {top_tokens[0]} ({top_k_probs[0].item():.3f}), "
                           f"{top_tokens[1]} ({top_k_probs[1].item():.3f}), "
                           f"{top_tokens[2]} ({top_k_probs[2].item():.3f})")
    
    logger.info("=" * 40)


def evaluate_model_on_test_set(trainer, eval_dataset, training_args):
    """Evaluate model performance on test set and output detailed results."""
    logger.info("=" * 60)
    logger.info("EVALUATING MODEL ON TEST SET")
    logger.info("=" * 60)
    
    if eval_dataset is None:
        logger.warning("No evaluation dataset provided, skipping evaluation.")
        return None
    
    logger.info(f"Evaluating on {len(eval_dataset)} test samples...")
    
    # Use trainer's evaluate method without passing eval_dataset parameter
    # This will use the eval_dataset that was already processed during trainer initialization
    eval_results = trainer.evaluate()
    
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    
    # Separate standard metrics and MTP metrics
    standard_metrics = {}
    mtp_metrics = {}
    
    for key, value in eval_results.items():
        if isinstance(value, (int, float)):
            if "mtp" in key.lower() or "ntp" in key.lower():
                mtp_metrics[key] = value
            else:
                standard_metrics[key] = value
    
    # Output standard evaluation metrics
    logger.info("Standard Evaluation Metrics:")
    for key, value in standard_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Output MTP specific metrics
    if mtp_metrics:
        logger.info("\nMTP-Specific Evaluation Metrics:")
        for key, value in mtp_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.6f}")
            else:
                logger.info(f"  {key}: {value}")
    
    # Calculate improvement metrics (if MTP is enabled)
    if training_args.mtp_enabled and mtp_metrics:
        logger.info("\nMTP PERFORMANCE ANALYSIS:")
        logger.info("-" * 30)
        
        # Compare standard loss and MTP loss
        if 'eval_loss' in standard_metrics:
            base_loss = standard_metrics['eval_loss']
            logger.info(f"Standard Loss: {base_loss:.6f}")
            
            # Find MTP loss
            mtp_loss_keys = [k for k in mtp_metrics.keys() if 'loss' in k.lower()]
            if mtp_loss_keys:
                for mtp_key in mtp_loss_keys:
                    mtp_loss = mtp_metrics[mtp_key]
                    logger.info(f"{mtp_key}: {mtp_loss:.6f}")
                    
                    # Calculate relative improvement
                    if base_loss > 0:
                        improvement = ((base_loss - mtp_loss) / base_loss) * 100
                        logger.info(f"Relative improvement: {improvement:.2f}%")
        
        # Output MTP head configuration summary
        logger.info(f"\nMTP Configuration Summary:")
        logger.info(f"  Head Type: {training_args.mtp_head_type}")
        logger.info(f"  Predictions: {training_args.mtp_num_predictions}")
        logger.info(f"  Layers: {training_args.mtp_num_layers}")
        logger.info(f"  Loss Weight: {training_args.mtp_loss_weight}")
    
    logger.info("=" * 60)
    return eval_results


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Print GPU information (like test.py)
    logger.info(f"GPU Number: {torch.cuda.device_count()}")
    logger.info(f"Current Device: {torch.cuda.current_device()}")
    
    # Force single GPU if CUDA_VISIBLE_DEVICES is set
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Print training configuration (like test.py)
    logger.info(f"Training Configuration:")
    logger.info(f"  per_device_train_batch_size: {training_args.per_device_train_batch_size}")
    logger.info(f"  gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"  gradient_checkpointing: {training_args.gradient_checkpointing}")
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Log enhanced MTP configuration if enabled
    if training_args.mtp_enabled:
        logger.info("=" * 60)
        logger.info("ENHANCED MULTI-TOKEN PREDICTION (MTP) ENABLED")
        logger.info("=" * 60)
        logger.info("New Features:")
        logger.info("  Identical head structure copying")
        logger.info("  Multi-layer MTP heads")
        logger.info("  Flexible parameter initialization")
        logger.info("  Advanced head architectures like MHA+FFN, CNN, etc.")
        logger.info("")
        logger.info("Configuration:")
        logger.info(f"  - Head type: {training_args.mtp_head_type}")
        logger.info(f"  - Number of predictions: {training_args.mtp_num_predictions}")
        logger.info(f"  - Number of layers: {training_args.mtp_num_layers}")
        logger.info(f"  - Initialization: {training_args.mtp_init_strategy}")
        logger.info(f"  - Loss weight: {training_args.mtp_loss_weight}")
        logger.info(f"  - Weight decay: {training_args.mtp_weight_decay_strategy}")

        logger.info(f"  - Dropout: {training_args.mtp_dropout_prob}")
        logger.info("=" * 60)
    
    # Load dataset
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    if data_args.dataset_name is not None:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.train_split,
        )
        if data_args.max_train_samples is not None:
            dataset = dataset.select(range(data_args.max_train_samples))
        
        eval_dataset = None
        if data_args.eval_split is not None:
            try:
                eval_dataset = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=data_args.eval_split,
                )
                if data_args.max_eval_samples is not None:
                    eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
                logger.info(f"Successfully loaded evaluation dataset: {data_args.eval_split}")
            except Exception as e:
                logger.warning(f"Failed to load evaluation split '{data_args.eval_split}': {e}")
                logger.info("Available splits will be checked...")
                try:
                    dataset_info = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
                    available_splits = list(dataset_info.keys())
                    logger.info(f"Available splits: {available_splits}")
                    if 'test' in available_splits:
                        eval_dataset = dataset_info['test']
                        logger.info("Using 'test' split for evaluation")
                    elif 'validation' in available_splits:
                        eval_dataset = dataset_info['validation']  
                        logger.info("Using 'validation' split for evaluation")
                except Exception as e2:
                    logger.warning(f"Could not determine available splits: {e2}")
                    eval_dataset = None
    else:
        raise ValueError("You must specify a dataset_name")
    
    logger.info(f"Training samples: {len(dataset)}")
    if eval_dataset is not None:
        logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    # Set model initialization kwargs
    model_init_kwargs = {}
    if model_args.torch_dtype is not None:
        model_init_kwargs["torch_dtype"] = model_args.torch_dtype
    if model_args.trust_remote_code:
        model_init_kwargs["trust_remote_code"] = model_args.trust_remote_code
    
    training_args.model_init_kwargs = model_init_kwargs
    
    if eval_dataset is not None and training_args.eval_strategy == "no":
        training_args.eval_strategy = "epoch"
        logger.info("Evaluation dataset found, setting eval_strategy to 'epoch'")
    
    # Preprocess datasets to handle format conflicts
    # Some datasets (like UltraChat) have both 'prompt' and 'messages' fields
    # SFTTrainer prioritizes 'prompt' field and expects 'completion', but we want conversational format
    def preprocess_dataset_format(dataset):
        """Handle dataset format conflicts intelligently."""
        if dataset is None:
            return None
        
        # Check the first example to understand the format
        first_example = dataset[0] if len(dataset) > 0 else {}
        has_prompt = 'prompt' in first_example
        has_completion = 'completion' in first_example
        has_messages = 'messages' in first_example
        
        logger.info(f"Dataset format analysis: prompt={has_prompt}, completion={has_completion}, messages={has_messages}")
        
        # Case 1: Standard prompt+completion format - don't touch it
        if has_prompt and has_completion and not has_messages:
            logger.info("Detected standard prompt+completion format. No preprocessing needed.")
            return dataset
        
        # Case 2: Pure conversational format - don't touch it  
        if has_messages and not has_prompt:
            logger.info("Detected pure conversational format. No preprocessing needed.")
            return dataset
        
        # Case 3: Conflicting format (prompt + messages, but no completion)
        # This is the UltraChat case - we prefer conversational format
        if has_prompt and has_messages and not has_completion:
            logger.info("Detected conflicting format (prompt+messages without completion).")
            logger.info("Removing 'prompt' field to use conversational format.")
            # Remove 'prompt' and other non-essential fields to force conversational processing
            columns_to_remove = [col for col in dataset.column_names if col not in ['messages']]
            dataset = dataset.remove_columns(columns_to_remove)
            return dataset
        
        # Case 4: Other formats - let SFTTrainer handle it
        logger.info("Unknown dataset format. Letting SFTTrainer handle it as-is.")
        return dataset
    
    # Apply preprocessing
    dataset = preprocess_dataset_format(dataset)
    eval_dataset = preprocess_dataset_format(eval_dataset)
    
    # Initialize trainer
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Analyze MTP configuration and model structure
    analyze_mtp_configuration(trainer, training_args)
    
    # Test MTP predictions before training
    if training_args.mtp_enabled:
        test_mtp_predictions(trainer, tokenizer, training_args)
    
    # Log basic model information
    logger.info("MODEL INFORMATION")
    logger.info("=" * 30)
    logger.info(f"Model class: {trainer.model.__class__.__name__}")
    logger.info(f"Total parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    # Print a sample from the dataset
    logger.info("\nDATASET SAMPLE")
    logger.info("=" * 20)
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 100:
            logger.info(f"  {key}: {value[:100]}...")
        else:
            logger.info(f"  {key}: {value}")
    logger.info("=" * 20)
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("Training completed successfully!")
    
    if eval_dataset is not None:
        logger.info("\nStarting post-training evaluation...")
        final_eval_results = evaluate_model_on_test_set(trainer, eval_dataset, training_args)
    else:
        logger.info("No evaluation dataset available, skipping post-training evaluation.")
    
    # Print final metrics with MTP-specific information
    if trainer.state.log_history:
        final_metrics = trainer.state.log_history[-1]
        logger.info("=" * 50)
        logger.info("FINAL TRAINING METRICS")
        logger.info("=" * 50)
        
        # Separate MTP and standard metrics
        mtp_metrics = {}
        standard_metrics = {}
        
        for key, value in final_metrics.items():
            if isinstance(value, float):
                if "mtp" in key.lower() or "ntp" in key.lower():
                    mtp_metrics[key] = value
                else:
                    standard_metrics[key] = value
        
        # Log standard metrics
        logger.info("Standard Metrics:")
        for key, value in standard_metrics.items():
            logger.info(f"  {key}: {value:.6f}")
        
        # Log MTP-specific metrics if available
        if mtp_metrics:
            logger.info("\nMTP-Specific Metrics:")
            for key, value in mtp_metrics.items():
                logger.info(f"  {key}: {value:.6f}")
        
        logger.info("=" * 50)
    
    # Final MTP analysis after training
    if training_args.mtp_enabled:
        logger.info("\nFINAL MTP ANALYSIS")
        logger.info("=" * 30)
        logger.info("Enhanced MTP training completed with:")
        logger.info(f"  - {training_args.mtp_head_type} heads ({training_args.mtp_num_layers} layers)")
        logger.info(f"  - {training_args.mtp_init_strategy} initialization")
        logger.info(f"  - {training_args.mtp_num_predictions} future token predictions")
        logger.info("=" * 30)


if __name__ == "__main__":
    main()