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

from dataclasses import dataclass
from typing import Literal

from ...trainer.grpo_config import GRPOConfig


@dataclass
class PAPOConfig(GRPOConfig):
    """
    Configuration class for PAPOTrainer.

    PAPO (Perception-Aware Policy Optimization) extends GRPO/DAPO for multimodal reasoning by adding an implicit
    perception loss and double entropy regularization.

    Args:
        perception_loss_weight (`float`, *optional*, defaults to `0.1`):
            gamma Weight coefficient for the perception loss term. This encourages the model to be sensitive to visual
            changes.

        mask_ratio (`float`, *optional*, defaults to `0.3`):
            Ratio of the image to mask when computing perception loss.

        mask_type (`Literal["random", "patch", "grid"]`, *optional*, defaults to `"random"`):
            Type of masking strategy to use.

        der_loss_weight1 (`float`, *optional*, defaults to `0.03`):
            eta1 Weight coefficient for the Double Entropy Regularization (DER) term. This term encourages confident
            predictions with original images (low entropy) and uncertain predictions with masked images (high entropy).

        der_loss_weight2 (`float`, *optional*, defaults to `0.03`):
            eta2 Weight coefficient for the Double Entropy Regularization (DER) term. This term encourages confident
            predictions with original images (low entropy) and uncertain predictions with masked images (high entropy).

        loss_type (`Literal["grpo", "dapo"]`, inherited from GRPOConfig):
            Base loss type to use. Set to "grpo" for PAPO-G or "dapo" for PAPO-D.
    """

    perception_loss_weight: float = 0.1
    mask_ratio: float = 0.3
    mask_type: Literal["random", "patch", "grid"] = "random"

    # Added for Double Entropy Regularization
    der_loss_weight1: float = 0.03
    der_loss_weight2: float = 0.03

    def __post_init__(self):
        super().__post_init__()

        # Validation
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be between 0 and 1, got {self.mask_ratio}")

        if self.der_loss_weight1 < 0 or self.der_loss_weight2 < 0:
            raise ValueError(
                f"der_loss_weight1 and der_loss_weight2 must be non-negative, got {self.der_loss_weight1} and {self.der_loss_weight2}"
            )

        if self.mask_type not in ["random", "patch", "grid"]:
            raise ValueError(f"mask_type must be one of ['random', 'patch', 'grid'], got {self.mask_type}")
