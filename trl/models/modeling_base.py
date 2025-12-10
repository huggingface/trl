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

import logging
import warnings
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import GenerationMixin
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

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


def create_reference_model(
    model: nn.Module, num_shared_layers: int | None = None, pattern: str | None = None
) -> nn.Module:
    """
    Creates a static reference copy of a model. Note that model will be in `.eval()` mode.

    Args:
        model ([`PreTrainedModelWrapper`]): The model to be copied.
        num_shared_layers (`int`, *optional*):
            The number of initial layers that are shared between both models and kept frozen.
        pattern (`str`, *optional*): The shared layers are selected with a string pattern
            (e.g. "transformer.h.{layer}" for GPT2) and if a custom pattern is necessary it can be passed here.

    Returns:
        [`PreTrainedModelWrapper`]
    """
    if is_deepspeed_zero3_enabled():
        raise ValueError(
            "DeepSpeed ZeRO-3 is enabled and is not compatible with `create_reference_model()`. Please instantiate your reference model directly with `AutoModelForCausalLM.from_pretrained()`."
        )

    parameter_names = [n for n, _ in model.named_parameters()]
    ref_model = deepcopy(model)

    # if no layers are shared, return copy of model
    if num_shared_layers is None:
        for param_name in parameter_names:
            param = ref_model.get_parameter(param_name)
            param.requires_grad = False
        return ref_model.eval()

    # identify layer name pattern
    if pattern is not None:
        pattern = pattern.format(layer=num_shared_layers)
    else:
        for pattern_candidate in LAYER_PATTERNS:
            pattern_candidate = pattern_candidate.format(layer=num_shared_layers)
            if any(pattern_candidate in name for name in parameter_names):
                pattern = pattern_candidate
                break

    if pattern is None:
        raise ValueError("Layer pattern could not be matched.")

    # divide parameters in shared and unshared parameter lists
    shared_param_list = []
    unshared_param_list = []

    shared_parameter = True
    for name, _param in model.named_parameters():
        if pattern in name:
            shared_parameter = False
        if shared_parameter:
            shared_param_list.append(name)
        else:
            unshared_param_list.append(name)

    # create reference of the original parameter if they are shared
    for param_name in shared_param_list:
        param = model.get_parameter(param_name)
        param.requires_grad = False

        _ref_param = ref_model.get_parameter(param_name)

    # for all other parameters just make sure they don't use gradients
    for param_name in unshared_param_list:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False

    if pattern is not None and len(unshared_param_list) == 0:
        logging.warning("Pattern passed or found, but no layers matched in the model. Check for a typo.")

    return ref_model.eval()


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
