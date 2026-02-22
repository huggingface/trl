#!/usr/bin/env python3
"""
Test script for GRPO with HA-DW on MPS.

This script trains a small model on a synthetic math dataset to verify
that the HA-DW implementation works correctly.

Note: BitsAndBytes quantization is not well-supported on MPS, so this
script uses the full precision model instead.
"""

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


def create_synthetic_math_dataset(num_samples=50):
    """Create a small synthetic math dataset for testing."""
    prompts = []

    # Simple addition problems
    for i in range(num_samples):
        a = i % 10
        b = (i + 3) % 10
        prompts.append({
            "prompt": f"What is {a} + {b}? Answer with just the number.",
            "answer": str(a + b)
        })

    return Dataset.from_list(prompts)


def accuracy_reward(prompts, completions, answer, **kwargs):
    """
    Simple reward function that checks if the completion contains the correct answer.

    Returns:
        List of rewards (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []
    for completion, correct_answer in zip(completions, answer):
        # Extract first number from completion
        completion_clean = completion.strip()
        # Simple check: does the completion contain the correct answer?
        if correct_answer in completion_clean.split():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def main(use_hadw=True):
    print("=" * 80)
    if use_hadw:
        print("Testing GRPO with HA-DW on MPS")
    else:
        print("Testing GRPO (baseline) on MPS")
    print("=" * 80)

    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("⚠️  MPS not available! Falling back to CPU.")
        device = "cpu"
    else:
        print("✓ MPS is available")
        device = "mps"

    # Model configuration
    # Using a small model that works well on Apple Silicon
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n📦 Loading model: {model_name}")
    print(f"   Device: {device}")

    # Model initialization kwargs for MPS
    # Note: Using fp32 for better numerical stability on MPS
    # The paper used bf16/fp32, but MPS doesn't support bf16 well
    model_init_kwargs = {
        "torch_dtype": torch.float32,
        "device_map": None,  # Let the trainer handle device placement
    }
    print(f"   Using dtype: {model_init_kwargs['torch_dtype']}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    print("\n📊 Creating synthetic dataset...")
    dataset = create_synthetic_math_dataset(num_samples=32)  # Small dataset for quick testing
    print(f"   Dataset size: {len(dataset)} samples")
    print(f"   Example prompt: {dataset[0]['prompt']}")
    print(f"   Example answer: {dataset[0]['answer']}")

    # Configure GRPO with HA-DW
    print(f"\n⚙️  Configuring GRPO{'with HA-DW' if use_hadw else ' (baseline)'}...")
    config = GRPOConfig(
        output_dir="./test_hadw_output",
        # Model initialization
        model_init_kwargs=model_init_kwargs,
        # HA-DW parameters
        use_hadw=use_hadw,
        hadw_eta=0.1,
        hadw_lambda_scale=1.0,
        hadw_history_window=5,  # Smaller window for small dataset
        # GRPO parameters
        num_generations=2,  # 2 generations per prompt (must divide batch size)
        max_completion_length=32,
        temperature=0.7,
        # Training parameters
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_steps=100,
        learning_rate=1e-6,
        # Optimization - Using fp32 for numerical stability
        # (MPS doesn't handle bf16 well, and fp16 can cause overflow in HA-DW)
        fp16=False,
        bf16=False,
        remove_unused_columns=False,
        # Disable features not needed for testing
        report_to=[],
        save_strategy="no",
    )

    print(f"   ✓ HA-DW enabled: {config.use_hadw}")
    print(f"   ✓ Eta: {config.hadw_eta}")
    print(f"   ✓ Lambda scale: {config.hadw_lambda_scale}")
    print(f"   ✓ History window: {config.hadw_history_window}")
    print(f"   ✓ Num generations: {config.num_generations}")

    # Initialize trainer
    print("\n🚀 Initializing GRPO Trainer...")
    try:
        trainer = GRPOTrainer(
            model=model_name,
            reward_funcs=accuracy_reward,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        print("   ✓ Trainer initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize trainer: {e}")
        raise

    # Run training
    print("\n🏋️  Starting training...")
    print("-" * 80)
    try:
        trainer.train()
        print("-" * 80)
        print("   ✓ Training completed successfully!")
    except Exception as e:
        print(f"   ✗ Training failed: {e}")
        raise

    # Print HA-DW metrics if available
    print("\n📈 HA-DW Metrics:")
    if hasattr(trainer, '_metrics') and 'train' in trainer._metrics:
        metrics = trainer._metrics['train']
        hadw_keys = [k for k in metrics.keys() if k.startswith('hadw/')]
        if hadw_keys:
            for key in hadw_keys:
                if metrics[key]:
                    values = metrics[key]
                    print(f"   {key}:")
                    print(f"     - First: {values[0]:.4f}")
                    print(f"     - Last:  {values[-1]:.4f}")
                    if len(values) > 1:
                        print(f"     - Mean:  {sum(values)/len(values):.4f}")
        else:
            print("   No HA-DW metrics found (this is unexpected)")

    print("\n" + "=" * 80)
    print("✅ Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GRPO with HA-DW")
    parser.add_argument(
        "--no-hadw",
        action="store_true",
        help="Disable HA-DW (run baseline GRPO for comparison)"
    )
    args = parser.parse_args()

    main(use_hadw=not args.no_hadw)
