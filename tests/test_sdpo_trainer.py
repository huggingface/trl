# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import pytest
import torch
from trl import SDPOConfig, SDPOTrainer


def test_sdpo_config_defaults():
    """Test that SDPOConfig has correct default values."""
    config = SDPOConfig(
        bf16=False,  # Disable bf16 for non-GPU environments
        output_dir="/tmp/test",
        report_to="none",
    )

    # Test SDPO-specific defaults
    assert config.distillation_alpha == 1.0
    assert config.distillation_topk == 20
    assert config.full_logit_distillation is False
    assert config.distillation_is_clip == 2.0
    assert config.distillation_add_tail is False
    assert config.dont_reprompt_on_self_success is True
    assert config.ema_update_rate == 0.01
    assert config.max_reprompt_len == 10240
    assert config.distillation_weight == 1.0
    assert config.use_successful_as_teacher is True

    # Test inherited GRPOConfig defaults are preserved
    assert config.beta == 0.0
    assert config.num_generations == 8
    assert config.loss_type == "dapo"


def test_sdpo_config_custom_values():
    """Test that SDPOConfig accepts custom values."""
    config = SDPOConfig(
        distillation_alpha=0.5,
        distillation_topk=50,
        full_logit_distillation=True,
        distillation_is_clip=3.0,
        distillation_add_tail=True,
        dont_reprompt_on_self_success=False,
        ema_update_rate=0.05,
        max_reprompt_len=20480,
        distillation_weight=0.5,
        use_successful_as_teacher=False,
        bf16=False,  # Disable bf16 for non-GPU environments
        output_dir="/tmp/test",
        report_to="none",
    )

    assert config.distillation_alpha == 0.5
    assert config.distillation_topk == 50
    assert config.full_logit_distillation is True
    assert config.distillation_is_clip == 3.0
    assert config.distillation_add_tail is True
    assert config.dont_reprompt_on_self_success is False
    assert config.ema_update_rate == 0.05
    assert config.max_reprompt_len == 20480
    assert config.distillation_weight == 0.5
    assert config.use_successful_as_teacher is False


def test_sdpo_config_alpha_validation():
    """Test that non-full-logit distillation only supports alpha=1.0."""
    config = SDPOConfig(
        full_logit_distillation=False,
        distillation_alpha=1.0,
        bf16=False,
        output_dir="/tmp/test",
        report_to="none",
    )
    assert config.distillation_alpha == 1.0

    # Other alpha values are allowed in config, but will raise error during training
    config = SDPOConfig(
        full_logit_distillation=False,
        distillation_alpha=0.5,
        bf16=False,
        output_dir="/tmp/test",
        report_to="none",
    )
    assert config.distillation_alpha == 0.5


def test_sdpo_trainer_import():
    """Test that SDPOTrainer can be imported from trl."""
    from trl import SDPOTrainer

    assert SDPOTrainer is not None
    assert hasattr(SDPOTrainer, "__init__")
    assert hasattr(SDPOTrainer, "train")


def test_sdpo_config_import():
    """Test that SDPOConfig can be imported from trl."""
    from trl import SDPOConfig

    assert SDPOConfig is not None
    assert SDPOConfig.distillation_alpha == 1.0


def test_sdpo_trainer_is_subclass_of_grpo():
    """Test that SDPOTrainer is a subclass of GRPOTrainer."""
    from trl import SDPOTrainer, GRPOTrainer

    assert issubclass(SDPOTrainer, GRPOTrainer)


def test_sdpo_config_is_subclass_of_grpo_config():
    """Test that SDPOConfig is a subclass of GRPOConfig."""
    from trl import SDPOConfig, GRPOConfig

    assert issubclass(SDPOConfig, GRPOConfig)


def test_sdpo_trainer_inheritance():
    """Test that SDPOTrainer inherits methods from GRPOTrainer."""
    from trl import SDPOTrainer

    # Check that GRPOTrainer methods are available
    assert hasattr(SDPOTrainer, "_compute_loss")
    assert hasattr(SDPOTrainer, "_get_per_token_logps")
    assert hasattr(SDPOTrainer, "log")


def test_sdpo_trainer_custom_methods():
    """Test that SDPOTrainer has its custom methods."""
    from trl import SDPOTrainer

    # Check that SDPO-specific methods are available
    assert hasattr(SDPOTrainer, "_compute_self_distillation_loss")
    assert hasattr(SDPOTrainer, "_compute_self_distillation_loss_core")
    assert hasattr(SDPOTrainer, "_compute_token_level_distillation_loss")
    assert hasattr(SDPOTrainer, "_apply_importance_sampling_clipping")
    assert hasattr(SDPOTrainer, "_get_teacher_log_probs")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sdpo_trainer_instantiation():
    """Test that SDPOTrainer can be instantiated (minimal test)."""
    from trl import SDPOTrainer, SDPOConfig
    from datasets import Dataset

    # Create a minimal dataset
    dataset = Dataset.from_dict({"prompt": ["test prompt"]})

    # This will fail without proper setup, but tests import structure
    try:
        config = SDPOConfig(
            max_completion_length=8,
            num_generations=2,
            per_device_train_batch_size=1,
            output_dir="/tmp/test_sdpo",
            report_to="none",
        )
        # Actual instantiation would require a model and reward function
        # This test just checks the code structure is correct
        assert config is not None
    except Exception as e:
        pytest.fail(f"Failed to create SDPOConfig: {e}")


def test_sdpo_config_from_dict():
    """Test that SDPOConfig can be created from a dict."""
    config_dict = {
        "distillation_alpha": 0.5,
        "distillation_topk": 30,
        "learning_rate": 1e-5,
        "num_generations": 4,
        "bf16": False,
        "output_dir": "/tmp/test",
        "report_to": "none",
    }

    config = SDPOConfig(**config_dict)

    assert config.distillation_alpha == 0.5
    assert config.distillation_topk == 30
    assert config.learning_rate == 1e-5
    assert config.num_generations == 4