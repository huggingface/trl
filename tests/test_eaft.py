
import pytest
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from unittest.mock import MagicMock

from trl import SFTConfig, SFTTrainer
from .testing_utils import TrlTestCase
from trl.trainer.sft_trainer import eaft_loss_func


class TestEAFTLoss(TrlTestCase):
    def test_eaft_loss(self):
        batch_size = 2
        seq_len = 3
        vocab_size = 25
        
        # Create random logits and labels
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Use a dict for outputs to behave like the model output dict
        outputs = {"logits": logits}
        
        # Calculate loss
        loss = eaft_loss_func(outputs, labels, alpha=1.0)
        
        # Simple assertions
        assert torch.is_tensor(loss)
        assert loss.dim() == 0

    def test_eaft_loss_zero_alpha(self):
        batch_size = 2
        seq_len = 3
        vocab_size = 25
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # ensure ignore_index handling matches
        labels[0, 0] = -100
        
        outputs = {"logits": logits}
        
        # EAFT with alpha=0
        eaft_loss = eaft_loss_func(outputs, labels, alpha=0.0)
        
        # manually replicate the padding and shifting logic from `eaft_loss_func`
        # in sft_trainer.py because eaft_loss_func performs this
        # internally before computing the loss. To verify alpha=0.0 matches standard CE,
        # we apply the same transformations to the labels here
        labels_padded = torch.nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels_padded[..., 1:].contiguous()
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        
        # standard CE loss check
        ce_loss = torch.nn.functional.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
        
        torch.testing.assert_close(eaft_loss, ce_loss)


class TestSFTTrainerEAFT(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train[:100]")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

    def test_train_eaft_loss(self):
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            loss_type="eaft",
            eaft_alpha=0.5,
            learning_rate=1e-3,
            report_to="none",
            max_steps=3,
        )
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
        )
        
        trainer.train()
        # check that loss is recorded
        assert trainer.state.log_history[-1]["train_loss"] is not None
        # check that it ran 3 steps
        assert trainer.state.global_step == 3

    def test_train_eaft_init_error(self):
        # should raise error if compute_loss_func is provided with loss_type="eaft"
        training_args = SFTConfig(
            output_dir=self.tmp_dir,
            loss_type="eaft",
            report_to="none",
        )
        
        with pytest.raises(ValueError, match="compute_loss_func"):
            SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                compute_loss_func=lambda x, y: 0.0
            )
