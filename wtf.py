from transformers import AutoProcessor, AutoModelForImageTextToText
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
import tempfile
import torch
from parameterized import parameterized


# dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

# with tempfile.TemporaryDirectory() as tmp_dir:
#     training_args = GRPOConfig(
#         output_dir=tmp_dir,
#         learning_rate=0.1,  # increase the learning rate to speed up the test
#         per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
#         num_generations=3,  # reduce the number of generations to reduce memory usage
#         max_completion_length=8,  # reduce the completion length to reduce memory usage
#         steps_per_generation=2,  # increase the steps per generation to trigger IS
#         report_to="none",
#     )
#     trainer = GRPOTrainer(
#         model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
#         # model="Qwen/Qwen2.5-VL-3B-Instruct",
#         reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
#         args=training_args,
#         train_dataset=dataset,
#     )

#     previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

#     trainer.train()


model_id = "trl-internal-testing/tiny-SmolVLMForConditionalGeneration"
# model_id = "trl-internal-testing/tiny-Qwen2VLForConditionalGeneration"
dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

with tempfile.TemporaryDirectory() as tmp_dir:
    training_args = GRPOConfig(
        output_dir=tmp_dir,
        learning_rate=0.1,  # increase the learning rate to speed up the test
        per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
        num_generations=3,  # reduce the number of generations to reduce memory usage
        max_completion_length=8,  # reduce the completion length to reduce memory usage
        max_prompt_length=None,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
        args=training_args,
        train_dataset=dataset,
    )

    previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

    trainer.train()
