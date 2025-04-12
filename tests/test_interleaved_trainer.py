import copy
import os
import tempfile
import unittest

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from trl import InterleaveConfig, InterleaveTrainer, SFTConfig, GRPOConfig
from trl.trainer import ConstantLengthDataset


def formatting_prompts_func(example):
    text = f"### Question: {example['question']}\n ### Answer: {example['answer']}"
    return text

def grpo_formatting_func(example):
    text = f"### Question: {example['question']}\n ### Answer: {example['answer']}"
    return {"prompt": text}

class InterleaveTrainerTester(unittest.TestCase):
    """Test cases for the InterleaveTrainer."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create base dataset
        self.dummy_dataset = Dataset.from_dict({
            "question": [
                "Does llamas know how to code?",
                "Does llamas know how to fly?",
                "Does llamas know how to talk?",
            ],
            "answer": [
                "Yes, llamas are very good at coding.",
                "No, llamas can't fly.",
                "Yes, llamas are very good at talking.",
            ],
        })

        # Create formatted text for both datasets
        formatted_texts = [
            f"### Question: {q}\n ### Answer: {a}"
            for q, a in zip(self.dummy_dataset["question"], self.dummy_dataset["answer"])
        ]

        # Tokenize the texts
        tokenized = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=16,
            return_tensors="pt"
        )

        # Create a unified dataset with all required fields for both SFT and GRPO
        self.sft_dataset = Dataset.from_dict({
            "text": formatted_texts,  # Required for SFT
            "prompt": formatted_texts,  # Required for GRPO
            "input_ids": tokenized["input_ids"].tolist(),
            "attention_mask": tokenized["attention_mask"].tolist(),
            "labels": tokenized["input_ids"].tolist(),  # For causal LM, labels = input_ids
            "completion": ["" for _ in formatted_texts],  # Required for GRPO
        })

        # Use the same dataset for both SFT and GRPO
        self.grpo_dataset = self.sft_dataset

        # Create a simple reward function for GRPO
        def dummy_reward_func(prompts, completions, **kwargs):  # Add **kwargs to accept any additional arguments
            return [1.0] * len(prompts)  # Return constant rewards
        
        self.reward_function = dummy_reward_func

    def test_interleave_trainer_initialization(self):
        """Test that the InterleaveTrainer initializes correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create SFT config with text field parameter
            sft_config = SFTConfig(
                output_dir=tmp_dir,
                dataset_text_field="text",
                per_device_train_batch_size=4,
            )

            # Create GRPO config with appropriate settings
            grpo_config = GRPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=4,
                num_generations=2,  # Must be compatible with batch size
                max_prompt_length=16,
                max_completion_length=16,
            )

            # Create configs with compatible batch size and num_generations
            training_args = InterleaveConfig(
                output_dir=tmp_dir,
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=4,
                report_to="none",
                start_with_sft=True,
                sft_weight=0.6,
                sft_config=sft_config,
                grpo_config=grpo_config,
            )
            
            # Initialize trainer with GRPO-specific kwargs
            trainer = InterleaveTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.sft_dataset,
                eval_dataset=self.grpo_dataset,
                reward_function=self.reward_function,
                tokenizer=self.tokenizer,  # Pass tokenizer directly
            )

            # Check trainer attributes
            self.assertIsNotNone(trainer.sft_trainer)
            self.assertIsNotNone(trainer.grpo_trainer)
            self.assertTrue(trainer.is_sft_phase)
            self.assertEqual(trainer.current_epoch, 0)

    def test_interleave_trainer_phase_switching(self):
        """Test that the trainer correctly switches between SFT and GRPO phases."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create SFT config with text field parameter
            sft_config = SFTConfig(
                output_dir=tmp_dir,
                dataset_text_field="text",
                per_device_train_batch_size=4,
            )

            # Create GRPO config with appropriate settings
            grpo_config = GRPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=4,
                num_generations=2,  # Must be compatible with batch size
                max_prompt_length=16,
                max_completion_length=16,
            )

            # Create configs with compatible batch size and num_generations
            training_args = InterleaveConfig(
                output_dir=tmp_dir,
                num_train_epochs=2,
                per_device_train_batch_size=4,
                report_to="none",
                start_with_sft=True,
                sft_config=sft_config,
                grpo_config=grpo_config,
            )

            # Initialize trainer with GRPO-specific kwargs
            trainer = InterleaveTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.sft_dataset,
                eval_dataset=self.grpo_dataset,
                reward_function=self.reward_function,
                tokenizer=self.tokenizer,  # Pass tokenizer directly
            )

            # Store initial weights
            initial_weights = copy.deepcopy(trainer.model.state_dict())

            # Train for one step
            trainer.train()

            # Check that weights were updated
            final_weights = trainer.model.state_dict()
            for key in initial_weights:
                self.assertFalse(
                    torch.allclose(initial_weights[key], final_weights[key]),
                    f"Weights for {key} were not updated during training",
                )

    def test_interleave_trainer_evaluation(self):
        """Test that evaluation works correctly and combines metrics from both trainers."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create SFT config with text field parameter
            sft_config = SFTConfig(
                output_dir=tmp_dir,
                dataset_text_field="text",
                per_device_train_batch_size=4,
            )

            # Create GRPO config with appropriate settings
            grpo_config = GRPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=4,
                num_generations=2,  # Must be compatible with batch size
                max_prompt_length=16,
                max_completion_length=16,
            )

            # Create configs with compatible batch size and num_generations
            training_args = InterleaveConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=4,
                report_to="none",
                start_with_sft=True,
                sft_weight=0.7,
                sft_config=sft_config,
                grpo_config=grpo_config,
            )

            # Initialize trainer with GRPO-specific kwargs
            trainer = InterleaveTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.sft_dataset,
                eval_dataset=self.grpo_dataset,
                reward_function=self.reward_function,
                tokenizer=self.tokenizer,  # Pass tokenizer directly
            )

            # Run evaluation
            metrics = trainer.evaluate()

            # Check that metrics contain both SFT and GRPO results
            self.assertTrue(any(key.startswith("sft_") for key in metrics))
            self.assertTrue(any(key.startswith("grpo_") for key in metrics))
            self.assertTrue(any(key.startswith("combined_") for key in metrics))

    def test_interleave_trainer_weight_sync(self):
        """Test that model weights are properly synchronized between trainers."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create SFT config with text field parameter
            sft_config = SFTConfig(
                output_dir=tmp_dir,
                dataset_text_field="text",
                per_device_train_batch_size=4,
            )

            # Create GRPO config with appropriate settings
            grpo_config = GRPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=4,
                num_generations=2,  # Must be compatible with batch size
                max_prompt_length=16,
                max_completion_length=16,
            )

            # Create configs with compatible batch size and num_generations
            training_args = InterleaveConfig(
                output_dir=tmp_dir,
                num_train_epochs=2,
                per_device_train_batch_size=4,
                report_to="none",
                start_with_sft=True,
                sft_config=sft_config,
                grpo_config=grpo_config,
            )

            # Initialize trainer with GRPO-specific kwargs
            trainer = InterleaveTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.sft_dataset,
                eval_dataset=self.grpo_dataset,
                reward_function=self.reward_function,
                tokenizer=self.tokenizer,  # Pass tokenizer directly
            )

            # Force a weight sync
            trainer._sync_model_weights()

            # Check that weights are identical between trainers
            sft_weights = trainer.sft_trainer.model.state_dict()
            grpo_weights = trainer.grpo_trainer.model.state_dict()

            for key in sft_weights:
                self.assertTrue(
                    torch.allclose(sft_weights[key], grpo_weights[key]),
                    f"Weights for {key} are not synchronized between trainers",
                )

    def test_interleave_trainer_save_load(self):
        """Test that the trainer can save and load model states correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create SFT config with text field parameter
            sft_config = SFTConfig(
                output_dir=tmp_dir,
                dataset_text_field="text",
                per_device_train_batch_size=4,
                max_steps=2,
            )

            # Create GRPO config with appropriate settings
            grpo_config = GRPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=4,
                num_generations=2,  # Must be compatible with batch size
                max_prompt_length=16,
                max_completion_length=16,
            )

            # Create configs with compatible batch size and num_generations
            training_args = InterleaveConfig(
                output_dir=tmp_dir,
                num_train_epochs=1,
                per_device_train_batch_size=4,
                report_to="none",
                start_with_sft=True,
                sft_config=sft_config,
                grpo_config=grpo_config,
            )

            # Initialize trainer with GRPO-specific kwargs
            trainer = InterleaveTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.sft_dataset,
                eval_dataset=self.grpo_dataset,
                reward_function=self.reward_function,
                tokenizer=self.tokenizer,  # Pass tokenizer directly
            )

            # Train for a few steps to ensure state is created
            trainer.train()

            # Save model and state
            trainer.save_model(tmp_dir)
            trainer.save_state()

            # Check that files were created
            self.assertTrue(any(f.endswith(".bin") for f in os.listdir(tmp_dir)))
            self.assertTrue("trainer_state.json" in os.listdir(tmp_dir))

    def test_interleave_trainer_with_custom_configs(self):
        """Test that the trainer works with custom SFT and GRPO configs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create custom configs with compatible batch size and num_generations
            sft_config = SFTConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=4,
                max_steps=2,
                dataset_text_field="text",  # Add text field parameter
            )
            
            grpo_config = GRPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=4,
                max_steps=2,
                num_generations=2,  # Must be compatible with batch size
                beta=0.1,  # Custom GRPO parameter
            )

            training_args = InterleaveConfig(
                output_dir=tmp_dir,
                num_train_epochs=2,
                per_device_train_batch_size=4,
                report_to="none",
                start_with_sft=True,
                sft_config=sft_config,
                grpo_config=grpo_config,
            )

            # Initialize trainer with GRPO-specific kwargs
            trainer = InterleaveTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.sft_dataset,
                eval_dataset=self.grpo_dataset,
                reward_function=self.reward_function,
                tokenizer=self.tokenizer,  # Pass tokenizer directly
            )

            # Check that configs were properly set
            self.assertEqual(trainer.sft_trainer.args.max_steps, 2)
            self.assertEqual(trainer.grpo_trainer.args.beta, 0.1)
            self.assertEqual(trainer.grpo_trainer.args.num_generations, 2)

    def test_interleave_trainer_error_handling(self):
        """Test that the trainer properly handles invalid configurations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Try to create trainer with invalid sft_weight
            with self.assertRaises(ValueError):
                training_args = InterleaveConfig(
                    output_dir=tmp_dir,
                    per_device_train_batch_size=4,
                    report_to="none",
                    sft_weight=1.5,  # Invalid weight > 1
                )

                InterleaveTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.sft_dataset,
                    eval_dataset=self.grpo_dataset,
                    reward_function=self.reward_function,
                ) 