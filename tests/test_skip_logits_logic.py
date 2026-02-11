"""
Tests for the skip_logits fix in SFTTrainer.compute_loss.

Verifies that when use_liger_kernel=True, the skip_logits flag is correctly
set based on model.training, prediction_loss_only, and compute_metrics.

This does NOT require liger-kernel to be installed — it tests the skip_logits
decision logic by setting the flag after trainer construction and intercepting
the inputs dict before the parent compute_loss is called.

See: https://github.com/huggingface/trl/issues/4679
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM

from trl import SFTConfig, SFTTrainer


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture
def dummy_dataset():
    """Minimal tokenized dataset for SFTTrainer."""
    return Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5]] * 4,
            "labels": [[1, 2, 3, 4, 5]] * 4,
            "attention_mask": [[1, 1, 1, 1, 1]] * 4,
        }
    )


def _build_trainer(tmp_path, dummy_dataset, prediction_loss_only=False, compute_metrics=None):
    """
    Create an SFTTrainer and then flip use_liger_kernel=True after construction.

    We cannot pass use_liger_kernel=True to the constructor because the base
    Trainer.__init__ checks for the liger-kernel package and raises ImportError
    if it's not installed. Instead, we construct normally and set the flag after,
    which is sufficient to test the skip_logits logic in compute_loss.
    """
    args = SFTConfig(
        output_dir=str(tmp_path),
        use_liger_kernel=False,  # Avoid ImportError during __init__
        prediction_loss_only=prediction_loss_only,
        report_to="none",
        max_length=8,
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dummy_dataset,
        compute_metrics=compute_metrics,
    )
    # Now enable the flag so compute_loss enters the liger code path
    trainer.args.use_liger_kernel = True
    return trainer


def _capture_skip_logits(trainer, training_mode):
    """
    Call trainer.compute_loss, intercepting the inputs dict right before
    super().compute_loss is called, and return the captured skip_logits value.
    """
    captured = {}

    def mock_super_compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
        """Intercept inputs to capture skip_logits, return dummy outputs."""
        captured["skip_logits"] = inputs.get("skip_logits", "NOT_SET")
        captured["return_token_accuracy"] = inputs.get("return_token_accuracy", "NOT_SET")
        captured["use_token_scaling"] = inputs.get("use_token_scaling", "NOT_SET")

        # Return dummy (loss, outputs) to satisfy the rest of compute_loss
        dummy_loss = torch.tensor(1.0, requires_grad=True)
        dummy_outputs = MagicMock()
        dummy_outputs.token_accuracy = None
        dummy_outputs.logits = torch.randn(1, 5, trainer.model.config.vocab_size)
        return (dummy_loss, dummy_outputs)

    # Set model to the right mode
    if training_mode:
        trainer.model.train()
    else:
        trainer.model.eval()

    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "labels": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }

    # Patch the parent class compute_loss to intercept inputs
    with patch("trl.trainer.sft_trainer.BaseTrainer.compute_loss", side_effect=mock_super_compute_loss):
        try:
            trainer.compute_loss(trainer.model, inputs)
        except Exception:
            pass  # We only care about the captured values

    return captured


class TestSkipLogitsLogic:
    """Test that skip_logits is set correctly in compute_loss for all combinations."""

    def test_training_mode_skip_logits_true(self, tmp_path, dummy_dataset):
        """During training, skip_logits should always be True."""
        trainer = _build_trainer(tmp_path, dummy_dataset)
        result = _capture_skip_logits(trainer, training_mode=True)
        assert result["skip_logits"] is True, "skip_logits should be True during training"

    def test_eval_no_compute_metrics_skip_logits_true(self, tmp_path, dummy_dataset):
        """During eval with no compute_metrics, skip_logits should be True (only loss needed)."""
        trainer = _build_trainer(tmp_path, dummy_dataset, compute_metrics=None)
        result = _capture_skip_logits(trainer, training_mode=False)
        assert result["skip_logits"] is True, "skip_logits should be True during eval when compute_metrics is None"

    def test_eval_prediction_loss_only_skip_logits_true(self, tmp_path, dummy_dataset):
        """During eval with prediction_loss_only=True, skip_logits should be True."""
        trainer = _build_trainer(
            tmp_path,
            dummy_dataset,
            prediction_loss_only=True,
            compute_metrics=lambda x: {"accuracy": 0.5},
        )
        result = _capture_skip_logits(trainer, training_mode=False)
        assert result["skip_logits"] is True, "skip_logits should be True during eval when prediction_loss_only=True"

    def test_eval_with_compute_metrics_skip_logits_false(self, tmp_path, dummy_dataset):
        """
        During eval with compute_metrics set and prediction_loss_only=False,
        skip_logits should be False — logits MUST be materialized for metrics.
        This is the critical regression test.
        """

        def dummy_metrics(eval_pred):
            return {"accuracy": 0.5}

        trainer = _build_trainer(
            tmp_path,
            dummy_dataset,
            prediction_loss_only=False,
            compute_metrics=dummy_metrics,
        )
        result = _capture_skip_logits(trainer, training_mode=False)
        assert result["skip_logits"] is False, (
            "skip_logits should be False during eval when compute_metrics is set "
            "and prediction_loss_only is False — logits are needed for metrics!"
        )

    @pytest.mark.parametrize(
        "training, prediction_loss_only, has_compute_metrics, expected_skip",
        [
            # Training mode: always skip
            (True, False, False, True),
            (True, False, True, True),
            (True, True, True, True),
            # Eval mode: skip unless compute_metrics is set AND prediction_loss_only is False
            (False, False, False, True),  # no metrics -> skip
            (False, True, False, True),  # loss only -> skip
            (False, True, True, True),  # loss only trumps metrics -> skip
            (False, False, True, False),  # metrics needed -> DON'T skip
        ],
        ids=[
            "train",
            "train+metrics",
            "train+loss_only+metrics",
            "eval_no_metrics",
            "eval_loss_only",
            "eval_loss_only+metrics",
            "eval+metrics_NEEDS_LOGITS",
        ],
    )
    def test_skip_logits_truth_table(
        self,
        tmp_path,
        dummy_dataset,
        training,
        prediction_loss_only,
        has_compute_metrics,
        expected_skip,
    ):
        """Exhaustive truth table for all skip_logits conditions."""
        compute_metrics = (lambda x: {"acc": 0.5}) if has_compute_metrics else None
        trainer = _build_trainer(
            tmp_path,
            dummy_dataset,
            prediction_loss_only=prediction_loss_only,
            compute_metrics=compute_metrics,
        )
        result = _capture_skip_logits(trainer, training_mode=training)
        assert result["skip_logits"] is expected_skip, (
            f"Expected skip_logits={expected_skip} for "
            f"training={training}, prediction_loss_only={prediction_loss_only}, "
            f"has_compute_metrics={has_compute_metrics}"
        )
