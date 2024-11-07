# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from typing import List, Optional, Union

import torch
from transformers import GenerationConfig, LogitsProcessor

from ..core import set_seed
from ..models import (
    SUPPORTED_ARCHITECTURES,
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
    PreTrainedModelWrapper,
)


class VASSampler:
    def __init__(
        self,
        model: PreTrainedModelWrapper,
        value_model: Union[AutoModelForSeq2SeqLMWithValueHead, AutoModelForCausalLMWithValueHead],
        beta: float = 1.0,
        top_k: int = 10,
        value_model_batch_size: int = 1,
        seed: Optional[int] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> None:
        """
        VASSampler is used to generate responses from a model trained with the VAS framework (see /trainers/VASTrainer.py).
        Args:
            model (`PreTrainedModelWrapper`):
                The pretrained model to use for generation
            value_model (`AutoModelForSeq2SeqLMWithValueHead` or `AutoModelForCausalLMWithValueHead`):
                The Pretrained Value model use to augment the sampling process
            beta (`float`):
                The value to use for weighting the Value outputs versus the logits of the LLM
            top_k (`int`):
                The number of top-k tokens that will be evaluated using the Value model
            value_model_batch_size (`int`):
                Batch size for the Value model, can be different from the batch size of the LLM
            seed (`int`, *optional*):
                Random seed used to control generation
            generation_config (`GenerationConfig`, *optional*):
                Generation config passed to the underlying model's `generate` method.
                See `GenerationConfig` (https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation#transformers.GenerationConfig) for more details

        """
        if seed is not None:
            set_seed(seed)

        if not isinstance(model, (SUPPORTED_ARCHITECTURES)):
            raise ValueError(
                f"model must be a PreTrainedModelWrapper, got {type(model)} - supported architectures are: {SUPPORTED_ARCHITECTURES}"
            )

        if not isinstance(value_model, (SUPPORTED_ARCHITECTURES)):
            raise ValueError(
                f"value_model must be a PreTrainedModelWrapper, got {type(model)} - supported architectures are: {SUPPORTED_ARCHITECTURES}"
            )

        self.model = model
        self.value_model = value_model
        self.beta = (beta,)
        self.top_k = top_k
        self.value_model_batch_size = value_model_batch_size
        self.gen_config = generation_config

        # Create a VAS logits processor
        self.logits_processor = VASLogitsProcessor(
            self.value_model, beta=self.beta, top_k=self.top_k, value_model_batch_size=self.value_model_batch_size
        )

    def generate(
        self,
        tokenized_query: Union[List[int], torch.Tensor, List[torch.Tensor], List[List[int]]],
        device: Optional[Union[str, torch.device]] = None,
        **generation_kwargs,
    ) -> List[List[str]]:
        """
        Generate a response using the VAS framework.

        Args:
            tokenized_query (`List[int]` or `torch.Tensor` or `List[torch.Tensor]` or `List[int]`):
                represents either a single tokenized query (a single tensor or a list of integers) or a batch of tokenized queries (a list of tensors or a list of lists of integers)
            device (`str` or `torch.device`, *optional*):
                The device on which the model will be loaded
            **generation_kwargs (`dict`, *optional*):
                Additional keyword arguments passed along to the underlying model's `generate` method.
                This is used to override generation config

        Returns:
            List[List[str]]: A list of lists of generated texts
        """

        # generate the response
        outputs = self.model.generate(
            tokenized_query.to(device),
            logits_processor=self.logits_processor,
            generation_config=self.gen_config,
            **generation_kwargs,
        )

        return outputs


class VASLogitsProcessor(LogitsProcessor, torch.nn.Module):
    """

    value_model: AutoModelForCausalLMWithValueHead, the Value model to use
    beta: float, the beta value to use for weighting the q model
    topk: int, the number of topk to use for the Value model
    topk_per_device_batch_size: int, the batch suze of tokens to evaluate at once
    """

    def __init__(
        self,
        value_model: Union[AutoModelForSeq2SeqLMWithValueHead, AutoModelForCausalLMWithValueHead],
        beta: float = 1.0,
        top_k: int = 10,
        value_model_batch_size: int = 1,
    ):
        """
        A logit processor that augment the output logits with Value model as per the VAS decoding scheme.

        Args:
            value_model (`AutoModelForSeq2SeqLMWithValueHead` or `AutoModelForCausalLMWithValueHead`):
                The Pretrained Value model use to augment the sampling process
            beta (`float`):
                The value to use for weighting the Value outputs versus the logits of the LLM
            top_k (`int`):
                The number of top-k tokens that will be evaluated using the Value model
            value_model_batch_size (`int`):
                Batch size for the Value model, can be different from the batch size of the LLM
        """
        super().__init__()
        self.value_model = value_model
        self.beta = beta
        self.top_k = top_k
        self.value_model_batch_size = value_model_batch_size

        assert self.topk > 0, "topk must be larger than zero"

        self.last_input_ids = None
        self.past_key_values = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        augmented_outputs = torch.clone(scores)
        batch_size = input_ids.shape[0]

        orig_input_ids = input_ids

        if (
            self.last_input_ids is not None
            and (input_ids[0, :-1].shape == self.last_input_ids.shape)
            and torch.all(input_ids[0, :-1] == self.last_input_ids)
        ):
            # if the last input ids are the same as the current input ids, we can reuse the past key values
            _, _, _, past_key_values = self.value_model(
                input_ids, past_key_values=self.past_key_values, return_past_key_values=True
            )
        else:
            _, _, _, past_key_values = self.value_model(input_ids, return_past_key_values=True)
        self.past_key_values = past_key_values
        self.last_input_ids = input_ids[0, :]

        values = torch.zeros_like(scores, device=scores.device)
        topk_ids = torch.topk(scores, self.topk, dim=-1).indices

        for i in range(0, topk_ids.shape[1], self.topk_per_device_batch_size):
            curr_topk_ids = topk_ids[:, i : i + self.topk_per_device_batch_size]
            curr_input_ids = orig_input_ids.unsqueeze(1).repeat(1, curr_topk_ids.shape[1], 1)
            curr_input_ids = torch.cat([curr_input_ids, curr_topk_ids.unsqueeze(-1)], dim=-1)
            curr_input_ids = curr_input_ids.reshape((batch_size * self.topk_per_device_batch_size, -1))

            _, _, value, _ = self.value_model(
                curr_input_ids,
                past_key_values=tuple(
                    (t1.repeat(curr_topk_ids.shape[1], 1, 1, 1), t2.repeat(curr_topk_ids.shape[1], 1, 1, 1))
                    for t1, t2 in self.past_key_values
                ),
                return_past_key_values=True,
            )
            value = value.reshape((batch_size, self.topk_per_device_batch_size, -1))[:, :, -1]
            values = values.scatter_(1, curr_topk_ids, value)

        values = values.scatter_(
            1, topk_ids, values.gather(1, topk_ids) - values.gather(1, topk_ids).mean(-1, keepdim=True)
        )
        augmented_outputs += self.beta * values

        return augmented_outputs
