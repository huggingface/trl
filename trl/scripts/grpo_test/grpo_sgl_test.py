from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import os
import torch
import time


def print_gpu_memory():
    """Print memory usage for all available GPUs"""
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(
            f"GPU {i}: {allocated:.2f}GB/{total:.2f}GB allocated, {reserved:.2f}GB reserved"
        )


def main():
    # Print diagnostic information
    print("\n=== System Configuration ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    print("\n=== Initial GPU Memory Usage ===")
    print_gpu_memory()

    # Create checkpoint directory
    checkpoint_dir = os.path.join("/home/misc/jinpan/trl-jin", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Clear any existing model files to ensure clean state
    import shutil

    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            item_path = os.path.join(checkpoint_dir, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    # Prepare model to have a valid checkpoint from the start
    print("\n=== Preparing Initial Checkpoint ===")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    start_time = time.time()
    print(f"Downloading and saving model: {model_name}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Model saved to {checkpoint_dir} in {time.time() - start_time:.1f}s")

    # Load dataset
    print("\n=== Loading Dataset ===")
    dataset = load_dataset(
        "trl-internal-testing/zen", "standard_prompt_only", split="train"
    )
    print(f"Loaded dataset with {len(dataset)} examples")

    # Configure training arguments
    print("\n=== Configuring Training ===")
    training_args = GRPOConfig(
        output_dir=checkpoint_dir,
        learning_rate=1.0e-03,
        per_device_train_batch_size=3,
        num_generations=9,
        max_completion_length=32,
        report_to="none",
        # SGLang configuration - now using proper options
        use_sglang=True,
        sglang_device="auto",  # Let it select GPU automatically
        sglang_gpu_memory_utilization=0.7,
        sglang_fallback_to_transformers=True,
        # Set checkpoint path from the beginning
        checkpoint_path=checkpoint_dir,
        # Add more verbose output
        logging_steps=1,
        logging_first_step=True,
    )

    # Initialize trainer
    print("\n=== Initializing Trainer ===")
    start_time = time.time()

    trainer = GRPOTrainer(
        model=model_name,  # Use model name here, not the loaded model
        reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
        args=training_args,
        train_dataset=dataset,
    )

    print(f"Trainer initialized in {time.time() - start_time:.1f}s")

    # Memory state after initialization
    print("\n=== GPU Memory After Initialization ===")
    print_gpu_memory()

    # Store initial model parameters to verify training
    print("\n=== Capturing Initial Model State ===")
    previous_trainable_params = {
        n: param.clone()
        for n, param in trainer.model.named_parameters()
        if param.requires_grad
    }
    print(f"Captured {len(previous_trainable_params)} trainable parameters")

    # Run training
    print("\n=== Starting Training ===")
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start
    print(f"Training completed in {train_time:.1f}s")

    # Memory state after training
    print("\n=== GPU Memory After Training ===")
    print_gpu_memory()

    # Verify parameters changed
    print("\n=== Verifying Parameter Updates ===")
    unchanged_count = 0
    changed_count = 0

    for n, param in previous_trainable_params.items():
        new_param = trainer.model.get_parameter(n)
        if torch.allclose(param, new_param, rtol=1e-4, atol=1e-4):
            unchanged_count += 1
            print(f"WARNING: Parameter {n} did not change significantly")
        else:
            changed_count += 1

    print(f"Results: {changed_count} parameters changed, {unchanged_count} unchanged")

    if unchanged_count == 0:
        print("\n=== TEST SUCCESSFUL: All parameters were updated ===")
    else:
        print(f"\n=== TEST WARNING: {unchanged_count} parameters were not updated ===")

    print("\nTest completed successfully")


if __name__ == "__main__":
    main()
