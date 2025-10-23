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

import tempfile
import unittest
from unittest.mock import Mock

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer
from trl.models.modeling_mtp_extension import MTPExtension, MTPHeads
from trl.trainer.mtp_data_collator import DataCollatorForMTPLanguageModeling


class TestMTPFunctionality(unittest.TestCase):
    """Test suite for Multi-Token Prediction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_name = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create a small test dataset
        self.test_data = [
            {"text": "Hello world, this is a test."},
            {"text": "Another test sentence for training."},
            {"text": "Multi-token prediction is awesome!"},
            {"text": "This is the fourth test sentence."},
        ]
        self.dataset = Dataset.from_list(self.test_data)
    
    def _get_lm_head(self, model):
        """Get LM head from model, supporting different architectures."""
        possible_lm_head_names = ['lm_head', 'head', 'output_layer', 'embed_out', 'score', 'classifier']
        for head_name in possible_lm_head_names:
            if hasattr(model, head_name):
                return getattr(model, head_name)
        return None
    
    def test_mtp_config_validation(self):
        """Test MTP configuration validation."""
        # Valid configuration
        config = SFTConfig(
            output_dir=tempfile.mkdtemp(),
            mtp_enabled=True,
            mtp_num_predictions=2,
            mtp_loss_weight=0.5,
            mtp_head_type="linear",
            num_train_epochs=1,
            per_device_train_batch_size=1,
        )
        self.assertTrue(config.mtp_enabled)
        self.assertEqual(config.mtp_num_predictions, 2)
        
        # Invalid configurations should raise ValueError
        with self.assertRaises(ValueError):
            SFTConfig(
                output_dir=tempfile.mkdtemp(),
                mtp_enabled=True,
                mtp_num_predictions=0,  # Invalid: must be >= 1
            )
        
        with self.assertRaises(ValueError):
            SFTConfig(
                output_dir=tempfile.mkdtemp(),
                mtp_enabled=True,
                mtp_head_type="invalid_type",  # Invalid head type
            )
    
    def test_mtp_heads_creation(self):
        """Test MTP heads creation and forward pass."""
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        config = model.config
        
        # Test different head types including new ones
        for head_type in ["linear", "ffn", "mha_ffn", "cnn", "identical"]:
            with self.subTest(head_type=head_type):
                mtp_heads = MTPHeads(
                    config=config,
                    num_predictions=2,
                    head_type=head_type,
                    dropout_prob=0.1,
                    num_layers=2,  # Test multi-layer
                    init_strategy="kaiming_uniform",
                    lm_head_module=self._get_lm_head(model),
                )
                
                # Test forward pass
                batch_size, seq_len = 2, 10
                hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
                
                mtp_logits = mtp_heads(hidden_states)
                
                # Check output shape
                self.assertEqual(len(mtp_logits), 2)  # num_predictions
                for logits in mtp_logits:
                    self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))
    
    def test_mtp_initialization_strategies(self):
        """Test different MTP initialization strategies."""
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        config = model.config
        
        # Test different initialization strategies
        init_strategies = ["default", "kaiming_uniform", "kaiming_normal", 
                          "xavier_uniform", "xavier_normal", "copy_lm_head"]
        
        for init_strategy in init_strategies:
            with self.subTest(init_strategy=init_strategy):
                mtp_heads = MTPHeads(
                    config=config,
                    num_predictions=2,
                    head_type="linear",
                    init_strategy=init_strategy,
                    lm_head_module=self._get_lm_head(model),
                )
                
                # Check that parameters are initialized (not all zeros)
                for head in mtp_heads.heads:
                    if isinstance(head, torch.nn.Linear):
                        self.assertFalse(torch.all(head.weight == 0))
                    elif isinstance(head, torch.nn.Sequential):
                        for layer in head:
                            if isinstance(layer, torch.nn.Linear):
                                self.assertFalse(torch.all(layer.weight == 0))
    
    def test_multi_layer_mtp_heads(self):
        """Test multi-layer MTP heads."""
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        config = model.config
        
        # Test different numbers of layers
        for num_layers in [1, 2, 3]:
            with self.subTest(num_layers=num_layers):
                mtp_heads = MTPHeads(
                    config=config,
                    num_predictions=2,
                    head_type="linear",
                    num_layers=num_layers,
                )
                
                # Test forward pass
                batch_size, seq_len = 2, 8
                hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
                
                mtp_logits = mtp_heads(hidden_states)
                
                # Check output shape is correct regardless of number of layers
                self.assertEqual(len(mtp_logits), 2)
                for logits in mtp_logits:
                    self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))
    
    def test_identical_head_structure(self):
        """Test identical head structure copying."""
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        config = model.config
        
        # Test identical head creation
        lm_head = self._get_lm_head(model)
        
        mtp_heads = MTPHeads(
            config=config,
            num_predictions=2,
            head_type="identical",
            lm_head_module=lm_head,
        )
        
        # Test forward pass
        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        mtp_logits = mtp_heads(hidden_states)
        
        # Check output shape
        self.assertEqual(len(mtp_logits), 2)
        for logits in mtp_logits:
            self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))
        
        # Check that heads have same structure as LM head
        for head in mtp_heads.heads:
            if isinstance(lm_head, torch.nn.Linear) and isinstance(head, torch.nn.Linear):
                self.assertEqual(head.in_features, lm_head.in_features)
                self.assertEqual(head.out_features, lm_head.out_features)
    
    def test_mtp_extension(self):
        """Test MTP extension functionality."""
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Test adding MTP to model
        self.assertFalse(hasattr(model, 'mtp_heads'))
        
        MTPExtension.add_mtp_to_model(
            model=model,
            num_predictions=2,
            head_type="linear",
            dropout_prob=0.1,
        )
        
        self.assertTrue(hasattr(model, 'mtp_heads'))
        self.assertEqual(len(model.mtp_heads.heads), 2)  # Should have 2 prediction heads
        
        # Test removing MTP from model
        MTPExtension.remove_mtp_from_model(model)
        self.assertFalse(hasattr(model, 'mtp_heads'))
    
    def test_mtp_data_collator(self):
        """Test MTP data collator."""
        collator = DataCollatorForMTPLanguageModeling(
            pad_token_id=self.tokenizer.pad_token_id,
            mtp_num_predictions=2,
            padding_free=False,
        )
        
        # Create sample tokenized data
        examples = [
            {"input_ids": [1, 2, 3, 4, 5], "labels": [1, 2, 3, 4, 5]},
            {"input_ids": [6, 7, 8], "labels": [6, 7, 8]},
        ]
        
        batch = collator(examples)
        
        # Check that MTP labels are generated
        self.assertIn("mtp_labels", batch)
        self.assertEqual(batch["mtp_labels"].shape[-1], 2)  # num_predictions
        
        # Check MTP label values
        # For position t, mtp_labels[t, 0] should be labels[t+1]
        # For position t, mtp_labels[t, 1] should be labels[t+2]
        labels = batch["labels"]
        mtp_labels = batch["mtp_labels"]
        
        # Check first example (longer sequence)
        for i in range(labels.shape[1] - 2):  # Avoid boundary issues
            if labels[0, i] != -100:  # Skip padding
                # Check t+1 prediction
                if i + 1 < labels.shape[1] and labels[0, i + 1] != -100:
                    self.assertEqual(mtp_labels[0, i, 0].item(), labels[0, i + 1].item())
                # Check t+2 prediction  
                if i + 2 < labels.shape[1] and labels[0, i + 2] != -100:
                    self.assertEqual(mtp_labels[0, i, 1].item(), labels[0, i + 2].item())
    
    def test_mtp_trainer_initialization(self):
        """Test SFTTrainer initialization with MTP."""
        config = SFTConfig(
            output_dir=tempfile.mkdtemp(),
            mtp_enabled=True,
            mtp_num_predictions=2,
            mtp_loss_weight=0.5,
            mtp_head_type="identical",
            mtp_num_layers=2,
            mtp_init_strategy="copy_lm_head",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            max_length=64,
        )
        
        trainer = SFTTrainer(
            model=self.model_name,
            args=config,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,
        )
        
        # Check that MTP is enabled
        self.assertTrue(trainer.mtp_enabled)
        
        # Initialize MTP (happens on first training step)
        trainer._initialize_mtp_after_prepare()
        
        # Now check that MTP heads are added
        actual_model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        self.assertTrue(hasattr(actual_model, 'mtp_heads'))
        
        # Check that MTP data collator is used
        self.assertIsInstance(trainer.data_collator, DataCollatorForMTPLanguageModeling)
    
    def test_mtp_loss_computation(self):
        """Test MTP loss computation."""
        config = SFTConfig(
            output_dir=tempfile.mkdtemp(),
            mtp_enabled=True,
            mtp_num_predictions=2,
            mtp_loss_weight=0.5,
            mtp_weight_decay_strategy="uniform",
            num_train_epochs=1,
            per_device_train_batch_size=1,
        )
        
        trainer = SFTTrainer(
            model=self.model_name,
            args=config,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,
        )
        
        # Create mock inputs
        batch_size, seq_len = 2, 10
        vocab_size = trainer.model.config.vocab_size
        
        # Mock MTP logits
        mtp_logits = [
            torch.randn(batch_size, seq_len, vocab_size),
            torch.randn(batch_size, seq_len, vocab_size),
        ]
        
        # Mock MTP labels
        mtp_labels = torch.randint(0, vocab_size, (batch_size, seq_len, 2))
        
        # Test MTP loss computation
        mtp_loss = trainer._compute_mtp_loss(mtp_logits, mtp_labels)
        
        self.assertIsInstance(mtp_loss, torch.Tensor)
        self.assertEqual(mtp_loss.dim(), 0)  # Scalar loss
        self.assertFalse(torch.isnan(mtp_loss))
        self.assertTrue(mtp_loss.item() >= 0)  # Loss should be non-negative
    
    def test_mtp_training_step(self):
        """Test a single training step with MTP."""
        config = SFTConfig(
            output_dir=tempfile.mkdtemp(),
            mtp_enabled=True,
            mtp_num_predictions=2,
            mtp_loss_weight=0.3,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            max_length=32,
            logging_steps=1,
        )
        
        trainer = SFTTrainer(
            model=self.model_name,
            args=config,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,
        )
        
        # Initialize MTP (happens on first training step)
        trainer._initialize_mtp_after_prepare()
        
        # Get a batch from the dataloader
        train_dataloader = trainer.get_train_dataloader()
        batch = next(iter(train_dataloader))
        
        # Move batch to device
        batch = {k: v.to(trainer.model.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Test compute_loss
        trainer.model.train()
        loss = trainer.compute_loss(trainer.model, batch)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertFalse(torch.isnan(loss))
        self.assertTrue(loss.item() >= 0)
        
        # Check that MTP metrics are logged
        self.assertIn("mtp_loss", trainer._metrics["train"])
        self.assertIn("ntp_loss", trainer._metrics["train"])


if __name__ == "__main__":
    unittest.main()
