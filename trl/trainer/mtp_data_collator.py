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

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch

from .sft_trainer import DataCollatorForLanguageModeling


@dataclass
class DataCollatorForMTPLanguageModeling(DataCollatorForLanguageModeling):
    """
    Data collator for Multi-Token Prediction language modeling.
    
    Extends the standard language modeling collator to generate MTP labels
    that predict multiple future tokens at each position.
    
    Args:
        mtp_num_predictions (`int`, defaults to 2):
            Number of future tokens to predict at each position.
        mtp_ignore_index (`int`, defaults to -100):
            Index to ignore in loss computation for MTP labels.
    """
    
    mtp_num_predictions: int = 2
    mtp_ignore_index: int = -100
    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Process a batch of examples and add MTP labels.
        
        Args:
            examples: List of examples to process.
            
        Returns:
            Dictionary containing input_ids, attention_mask, labels, and mtp_labels.
        """
        # Call parent method to get standard batch
        batch = super().torch_call(examples)
        
        # Generate MTP labels if we have standard labels
        if "labels" in batch and self.mtp_num_predictions > 0:
            batch["mtp_labels"] = self._generate_mtp_labels(batch["labels"])
        
        return batch
    
    def _generate_mtp_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate MTP labels from standard labels.
        
        For each position t, generates labels for positions t+1, t+2, ..., t+k
        where k is mtp_num_predictions.
        
        Args:
            labels: Standard labels tensor of shape (batch_size, seq_len).
            
        Returns:
            MTP labels tensor of shape (batch_size, seq_len, mtp_num_predictions).
        """
        batch_size, seq_len = labels.shape
        device = labels.device
        dtype = labels.dtype
        
        # Initialize MTP labels with ignore_index
        mtp_labels = torch.full(
            (batch_size, seq_len, self.mtp_num_predictions),
            self.mtp_ignore_index,
            dtype=dtype,
            device=device
        )
        
        # Fill MTP labels for each prediction step
        for k in range(self.mtp_num_predictions):
            shift = k + 1
            if shift < seq_len:
                # For position t, predict token at t+shift
                mtp_labels[:, :-shift, k] = labels[:, shift:]
        
        return mtp_labels
