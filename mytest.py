
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import tempfile
import torch

dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
with tempfile.TemporaryDirectory() as tmp_dir:
    training_args = GRPOConfig(
        output_dir=tmp_dir,
        learning_rate=0.1,  # increase the learning rate to speed up the test
        per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
        num_generations=3,  # reduce the number of generations to reduce memory usage
        max_completion_length=32,  # reduce the completion length to reduce memory usage
        report_to="none",
        use_sglang=True,
        sglang_device="auto",  # will raise a warning, but allows this test to work with only one GPU
        sglang_gpu_memory_utilization=0.5,  # reduce since because we use the same device for training and vllm
    )
    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
        args=training_args,
        train_dataset=dataset,
    )
    previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
    trainer.train()
    # Check that the params have changed
    for n, param in previous_trainable_params.items():
        new_param = trainer.model.get_parameter(n)
        assert not(torch.equal(param, new_param), f"Parameter {n} has not changed.")
    print("test over")