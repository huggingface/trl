# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from typing import Callable, List, Optional, Union

import torch
from transformers import (
    LogitsProcessor
)
from . import BaseInference


class VASInference(BaseInference):
    """
    VASInference is a class that is used to generate responses from a model trained with the VAS framework.
    """

    def __init__(self, config, model, value_model):
        super().__init__(config)
        self.model = model
        self.value_model = value_model

    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, **generation_kwargs) -> torch.LongTensor:
        """
        Generate a response using the VAS framework.

        input_ids: torch.LongTensor, the input ids to use for generation
        beta: float, the beta value to use for weighting the Value model
        topk: int, the number of topk to use for the Value model
        value_model_batch_size: int, the batch size of tokens to evaluate at once
        generation_kwargs: dict, additional generation kwargs to pass to the model
        """
        # get the logits processor
        beta = self.config.beta
        topk = self.config.topk
        value_model_batch_size = self.config.value_model_batch_size
        logits_processor = VASLogitsProcessor(self.value_model, beta=beta, topk=topk, value_model_batch_size=value_model_batch_size)

        # generate the response
        outputs = self.model.generate(input_ids, logits_processor=logits_processor, **generation_kwargs)

        return outputs


class VASLogitsProcessor(LogitsProcessor, torch.nn.Module):
    """
    A class to process logits to perform VAS decoding

    model: AutoModelForCausalLMWithValueHead, the Value model to use
    beta: float, the beta value to use for weighting the q model
    topk: int, the number of topk to use for the Value model
    topk_per_device_batch_size: int, the batch suze of tokens to evaluate at once
    """

    def __init__(self, value_model: torch.nn.Module, beta: float, topk: int = 20, value_model_batch_size: int = 1):
        super().__init__()
        self.value_model = value_model
        self.beta = beta
        self.topk = topk
        self.value_model_batch_size = value_model_batch_size

        assert self.topk > 0, 'topk must be larger than zero'

        self.last_input_ids = None
        self.past_key_values = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        augmented_outputs = torch.clone(scores)
        batch_size = input_ids.shape[0]

        orig_input_ids = input_ids

        if self.last_input_ids is not None and (input_ids[0, :-1].shape == self.last_input_ids.shape) and torch.all(input_ids[0, :-1] == self.last_input_ids):
            # if the last input ids are the same as the current input ids, we can reuse the past key values
            _, _, _, past_key_values = self.value_model(input_ids, past_key_values=self.past_key_values, return_past_key_values=True)
        else:
            _, _, _, past_key_values = self.value_model(input_ids, return_past_key_values=True)
        self.past_key_values = past_key_values
        self.last_input_ids = input_ids[0, :]

        values = torch.zeros_like(scores, device=scores.device)
        topk_ids = torch.topk(scores, self.topk, dim=-1).indices

        for i in range(0, topk_ids.shape[1], self.topk_per_device_batch_size):
            curr_topk_ids = topk_ids[:, i:i + self.topk_per_device_batch_size]
            curr_input_ids = orig_input_ids.unsqueeze(1).repeat(1, curr_topk_ids.shape[1], 1)
            curr_input_ids = torch.cat([curr_input_ids, curr_topk_ids.unsqueeze(-1)], dim=-1)
            curr_input_ids = curr_input_ids.reshape((batch_size*self.topk_per_device_batch_size, -1))

            _, _, value, _ = self.value_model(curr_input_ids, past_key_values=tuple((t1.repeat(curr_topk_ids.shape[1],1, 1, 1), t2.repeat(curr_topk_ids.shape[1],1, 1, 1)) for t1, t2 in self.past_key_values), return_past_key_values=True)
            value = value.reshape((batch_size, self.topk_per_device_batch_size, -1))[:,:,-1]
            values = values.scatter_(1, curr_topk_ids, value)

        values = values.scatter_(1, topk_ids, values.gather(1, topk_ids) - values.gather(1, topk_ids).mean(-1, keepdim=True))
        augmented_outputs += self.beta * values

        return augmented_outputs
