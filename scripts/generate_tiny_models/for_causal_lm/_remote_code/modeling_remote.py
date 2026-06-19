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

import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel, LlamaPreTrainedModel

from .configuration_remote import RemoteConfig


class RemoteModel(LlamaModel):
    config_class = RemoteConfig


class RemoteForCausalLM(LlamaForCausalLM):
    config_class = RemoteConfig


class RemoteForSequenceClassification(LlamaForSequenceClassification):
    config_class = RemoteConfig

    def __init__(self, config):
        # The parent's MRO calls `AutoModel.from_config(config)`, which would re-trigger the
        # trust_remote_code prompt for `RemoteConfig`. Wire `RemoteModel` in directly instead.
        LlamaPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels
        self.model = RemoteModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()
