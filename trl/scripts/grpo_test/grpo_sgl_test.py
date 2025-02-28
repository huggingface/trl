import os
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import tempfile
import torch
import sys
import traceback


def main():
    try:
        # Display environment info for debugging
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(
                f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
            )

        # Use checkpoint directory within project path
        checkpoint_dir = os.path.join("/home/misc/jinpan/trl-jin", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Load dataset
        dataset = load_dataset(
            "trl-internal-testing/zen", "standard_prompt_only", split="train"
        )
        print(f"Loaded dataset with {len(dataset)} examples")

        # Configure training with explicit GPU assignment for SGLang

        training_args = GRPOConfig(
            output_dir=checkpoint_dir,
            learning_rate=1.0e-03,
            per_device_train_batch_size=3,
            num_generations=9,
            max_completion_length=32,
            report_to="none",
            use_sglang=True,
            sglang_base_gpu_id=3,  # Use the correct parameter name
            sglang_mem_fraction_static=0.9,
            checkpoint_path=checkpoint_dir,
        )

        # Initialize trainer with detailed error handling
        try:
            trainer = GRPOTrainer(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )
            print("Trainer successfully initialized")
        except Exception as e:
            print(f"Failed to initialize trainer: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Save a checkpoint for SGLang weight updates
        trainer.model.save_pretrained(checkpoint_dir)
        print(f"Initial model checkpoint saved to {checkpoint_dir}")

        # Clone parameters for verification after training
        previous_trainable_params = {
            n: param.clone()
            for n, param in trainer.model.named_parameters()
            if param.requires_grad
        }
        print(f"Captured {len(previous_trainable_params)} trainable parameters")

        # Start training with error handling
        try:
            print(
                "[DEBUG] Starting training; expecting generation requests from SGLang engine..."
            )
            trainer.train()
            print("[DEBUG] Training finished.")
        except Exception as e:
            print(f"Training failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Verify that parameters have changed
        changed_params = 0
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if not torch.equal(param, new_param):
                changed_params += 1

        print(
            f"{changed_params}/{len(previous_trainable_params)} parameters changed during training"
        )
        print("Test completed successfully")

    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
