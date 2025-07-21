# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import unittest
import tempfile
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset

from trl.trainer.repo_config import RePOConfig
from trl.trainer.repo_trainer import RePOTrainer
from trl.trainer.utils import ReplayBuffer


class ReplayBufferTests(unittest.TestCase):
    def test_replay_buffer_add_sample(self):
        capacity = 10
        buffer = ReplayBuffer(capacity)

        # Test add
        for i in range(15):
            experience = {
                "query_tensors": torch.tensor([i]),
                "response_tensors": torch.tensor([i * 2]),
                "logprobs": torch.tensor([0.1 * i]),
            }
            buffer.add(experience)

        self.assertEqual(len(buffer), capacity) # Should be at capacity
        # Check if oldest elements are discarded
        self.assertEqual(buffer.buffer[0]["query_tensors"].item(), 5)

        # Test sample
        sample_batch_size_small = 3
        sampled_small = buffer.sample(sample_batch_size_small)
        self.assertIsNotNone(sampled_small)
        self.assertEqual(len(sampled_small), sample_batch_size_small)
        for exp in sampled_small:
            self.assertIn("query_tensors", exp)
            self.assertTrue(exp["query_tensors"].item() >= 5) # Check if samples are from the correct range

        sample_batch_size_full = 10
        sampled_full = buffer.sample(sample_batch_size_full)
        self.assertIsNotNone(sampled_full)
        self.assertEqual(len(sampled_full), sample_batch_size_full)

        # Test sample when buffer is smaller than batch_size
        buffer_small = ReplayBuffer(5)
        for i in range(3):
             buffer_small.add({"dummy": torch.tensor([i])})
        self.assertIsNone(buffer_small.sample(5))

        # Test sampling with device
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            sampled_gpu = buffer.sample(sample_batch_size_small, device=device)
            self.assertIsNotNone(sampled_gpu)
            self.assertEqual(sampled_gpu[0]["query_tensors"].device.type, "cuda")


class RePOTrainerTests(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/tiny-GPT2-L1-H4-ppo-critics" # A very small model for quick testing
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token # Standard practice

        # Create dummy dataset
        dummy_data = [{"query": "Hello", "response": "World"}, {"query": "How are", "response": "you?"}]

        def tokenize(example):
            example["query_tokens"] = self.tokenizer.encode(example["query"])
            # For PPOTrainer, dataset usually contains tokenized queries
            # For test simplicity, we'll pass 'input_ids' directly as if they are queries
            return {"input_ids": example["query_tokens"], "query": example["query"]}

        self.dataset = Dataset.from_list(dummy_data).map(tokenize)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_repo_trainer_init(self):
        config = RePOConfig(output_dir=self.temp_dir, batch_size=1, ppo_epochs=1, mini_batch_size=1)
        model = AutoModelForCausalLM.from_pretrained(self.model_id)

        trainer = RePOTrainer(
            config=config,
            model=model,
            ref_model=None, # PPOTrainer can create this if None
            tokenizer=self.tokenizer,
            dataset=self.dataset,
        )
        self.assertIsInstance(trainer.replay_buffer, ReplayBuffer)
        self.assertEqual(trainer.replay_buffer.capacity, config.replay_buffer_size)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
    def test_repo_trainer_simple_run(self):
        # This test is to ensure a few training steps can run without crashing.
        # It will initially test the on-policy path and buffer filling.
        # Loss values are not strictly checked here, only that it runs.

        config = RePOConfig(
            output_dir=self.temp_dir,
            batch_size=2, # Must be >= number of samples if dataset is small
            mini_batch_size=1,
            ppo_epochs=1,
            gradient_accumulation_steps=1,
            max_steps=2, # Run for a few steps
            kl_penalty="kl", # Required by PPOTrainer
            adap_kl_ctrl=False, # Simpler for testing
            use_replay_buffer=True,
            replay_warmup_steps=1,
            replay_batch_size=1, # Sample 1 from buffer
            learning_rate=1e-4, # Added learning_rate
            # PPOTrainer specific arguments that might be needed
            log_with="tensorboard", # or wandb, or None
            tracker_project_name="test_repo_trainer",
            optimize_cuda_cache=True,
        )

        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        # ref_model = AutoModelForCausalLM.from_pretrained(self.model_id) # PPOTrainer can create this

        trainer = RePOTrainer(
            config=config,
            model=model,
            ref_model=None,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
        )

        try:
            trainer.train()
        except Exception as e:
            self.fail(f"RePOTrainer training failed with exception: {e}")

        # Check if replay buffer was filled (at least one on-policy step happened)
        self.assertTrue(len(trainer.replay_buffer) > 0, "Replay buffer should have been filled.")

        # More detailed checks for off-policy loss would require inspecting metrics,
        # which will be part of step 4 of the plan (Implement Off-Policy Logic).

if __name__ == "__main__":
    unittest.main()
