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
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel


class RewardHead(nn.Module):
    """Head for reward prediction."""

    def __init__(self, config: PretrainedConfig, n_labels: int = 1):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # use same dropout as attention dropout
        self.dropout = nn.Dropout(config.attention_dropout if hasattr(config, "attention_dropout") else 0.1)
        self.out_proj = nn.Linear(config.hidden_size, n_labels)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output


class CLoudRewardModelConfig(PretrainedConfig):
    """Configuration class for CLoud Reward Model."""

    def __init__(self, feedback_method="vanilla", base_model_name_or_path=None, **kwargs):
        assert feedback_method in ["vanilla", "teacher"]
        self.feedback_method = feedback_method
        self.base_model_name_or_path = base_model_name_or_path
        super().__init__(**kwargs)


class CLoudRewardModel(PreTrainedModel):
    """CLoud Reward Model combining a base LLM with a reward head."""

    config_class = CLoudRewardModelConfig

    def __init__(self, config, pretrained_reward_base_model=None):
        super().__init__(config)
        self.feedback_method = config.feedback_method

        # Initialize base model
        if pretrained_reward_base_model is None:
            reward_base_model_cfg = AutoConfig.from_pretrained(config.base_model_name_or_path)
            self.reward_base_model = AutoModelForCausalLM.from_config(reward_base_model_cfg)
        else:
            self.reward_base_model = pretrained_reward_base_model

        # Add reward head
        self.reward_head = RewardHead(self.reward_base_model.config)
        self._no_split_modules = self.reward_base_model._no_split_modules

    def forward(self, input_ids, attention_mask, **kwargs):
        batch_size, _ = input_ids.shape

        outputs = self.reward_base_model(
            input_ids, attention_mask, output_hidden_states=True, return_dict=True, **kwargs
        )

        # Get hidden states and compute rewards
        hidden_states = outputs.hidden_states[-1]
        rewards = self.reward_head(hidden_states)
        sequence_lengths = torch.sum(attention_mask, dim=-1) - 1
        rewards = rewards[torch.arange(batch_size, device=rewards.device), sequence_lengths]

        return {
            "logits": rewards,
            "hidden_states": outputs.hidden_states,
            "lm_logits": outputs.logits if self.feedback_method == "teacher" else None,
        }
