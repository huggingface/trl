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

import warnings

import torch
from transformers import GenerationMixin

from ..import_utils import suppress_experimental_warning


with suppress_experimental_warning():
    from ..experimental.ppo.modeling_value_head import PreTrainedModelWrapper as _PreTrainedModelWrapper


LAYER_PATTERNS = [
    "transformer.h.{layer}",
    "model.decoder.layers.{layer}",
    "gpt_neox.layers.{layer}",
    "model.layers.{layer}",
]


class PreTrainedModelWrapper(_PreTrainedModelWrapper):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The `PreTrainedModelWrapper` is now located in `trl.experimental`. Please update your imports to "
            "`from trl.experimental.bco import PreTrainedModelWrapper`. The current import path will be removed and "
            "no longer supported in TRL 0.29. For more information, see "
            "https://github.com/huggingface/trl/issues/4223.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class GeometricMixtureWrapper(GenerationMixin):
    """
    Geometric Mixture generation wrapper that samples from the logits of two model's geometric mixture.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be wrapped.
        ref_model ([`~transformers.PreTrainedModel`]): The reference model.
        generation_config ([`~transformers.GenerationConfig`]): The generation config.
        mixture_coef (`float`, *optional* - default: 0.5): The mixture coefficient.
    """

    main_input_name = "input_ids"
    _supports_cache_class = False
    _supports_static_cache = False
    _is_stateful = False

    def __init__(self, model, ref_model, generation_config, mixture_coef=0.5, device=None):
        super().__init__()

        self.model = model
        self.config = model.config
        self.ref_model = ref_model
        self.generation_config = generation_config
        self.mixture_coef = mixture_coef
        self.device = device
        if hasattr(self.model, "_is_stateful"):
            self._is_stateful = self.model._is_stateful

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @torch.inference_mode()
    def forward(self, *args, **kwargs):
        model_outputs = self.model(*args, **kwargs)
        model_logits = model_outputs.logits
        ref_model_logits = self.ref_model(*args, **kwargs).logits

        model_outputs.logits = torch.nn.functional.log_softmax(
            self.mixture_coef * ref_model_logits + (1 - self.mixture_coef) * model_logits, dim=-1
        )

        return model_outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # turn off cache in the generation config
        kwargs["use_cache"] = False
        model_inputs = self.model.prepare_inputs_for_generation(*args, **kwargs)
        _ = self.ref_model.prepare_inputs_for_generation(*args, **kwargs)

        return model_inputs

    def _validate_model_class(self):
        self.model._validate_model_class()

    def _validate_model_kwargs(self, model_kwargs):
        return self.model._validate_model_kwargs(model_kwargs)
