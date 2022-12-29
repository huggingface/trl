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
from copy import deepcopy
import torch.nn as nn

from transformers import PreTrainedModel

LAYER_PATTERNS = ["transformer.h.{layer}", "model.decoder.layers.{layer}", "gpt_neox.layers.{layer}"]


class PreTrainedModelWrapper(nn.Module):
    r"""
    A wrapper class around a (`transformers.PreTrainedModel`) to be compatible with the
    (`~transformers.PreTrained`) class in order to keep some attributes and methods of the
    (`~transformers.PreTrainedModel`) class.

    Attributes
    ----------
    pretrained_model: (`transformers.PreTrainedModel`)
        The model to be wrapped.
    parent_class: (`transformers.PreTrainedModel`)
        The parent class of the model to be wrapped.
    supported_args: (`list`)
        The list of arguments that are supported by the wrapper class.
    """
    transformers_parent_class = None
    supported_args = None

    def __init__(self, pretrained_model=None, **kwargs):
        super().__init__()
        self.pretrained_model = pretrained_model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Instantiates a new model from a pretrained model.

        Parameters
        ----------
        pretrained_model_name_or_path: (`str` or `transformers.PreTrainedModel`)
            The path to the pretrained model or its name.
        *model_args:
            Additional positional arguments passed along to the underlying model's
            `from_pretrained` method.
        **kwargs:
            Additional keyword arguments passed along to the underlying model's
            `from_pretrained` method. We also pre-process the kwargs to extract 
            the arguments that are specific to the `transformers.PreTrainedModel`
            class and the arguments that are specific to trl models.
        """
        if kwargs is not None:
            trl_model_args, pretrained_kwargs = cls._split_kwargs(kwargs)
        else:
            trl_model_args = {}
            pretrained_kwargs = {}

        # First, load the pre-trained model using the parent-class
        # either `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM`
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model = cls.transformers_parent_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, **pretrained_kwargs
            )
        elif isinstance(pretrained_model_name_or_path, PreTrainedModel):
            pretrained_model = pretrained_model_name_or_path
        else:
            raise ValueError(
                "pretrained_model_name_or_path should be a string or a PreTrainedModel, "
                f"but is {type(pretrained_model_name_or_path)}"
            )

        # Then, create the full model by instantiating the wrapper class
        model = cls(pretrained_model, **trl_model_args)

        return model

    @classmethod
    def _split_kwargs(cls, kwargs):
        """
        Separate the kwargs from the arguments that we support inside
        `supported_args` and the ones that we don't.
        """
        supported_kwargs = {}
        unsupported_kwargs = {}

        for key, value in kwargs.items():
            if key in cls.supported_args:
                supported_kwargs[key] = value
            else:
                unsupported_kwargs[key] = value

        return supported_kwargs, unsupported_kwargs

    def push_to_hub(self, *args, **kwargs):
        r"""
        Push the pretrained model to the hub.
        """
        return self.pretrained_model.push_to_hub(*args, **kwargs)
    
    def save_pretrained(self, *args, **kwargs):
        r"""
        Save the pretrained model to a directory.
        """
        return self.pretrained_model.save_pretrained(*args, **kwargs)


def create_reference_model(model: PreTrainedModelWrapper, share_layers: int = None, pattern: str = None):
    """
    Creates a static reference copy of a model. Note that model will be in `.eval()` mode.
    
    Args:
        model (`PreTrainedModelWrapper`): model to be copied
        share_layers (`int`, *optional*): first layers are shared between both models and kept frozen. 
        pattern (`str`, *optional*): The shared layers are selected with a string pattern 
            (e.g. "transformer.h.{layer}" for GPT2) and if a custom pattern is necessary it can be passed here.

    Returns
        `PreTrainedModelWrapper`
    """

    parameter_names = [n for n, _ in model.named_parameters()]
    ref_model = deepcopy(model)

    # if no layers are shared, return copy of model
    if share_layers is None:
        for param_name in parameter_names:
            param = ref_model.get_parameter(param_name)
            param.requires_grad = False
        return ref_model.eval()

    # identify layer name pattern
    if pattern is not None:
        pattern = pattern.format(layer=share_layers)
    else:
        for pattern_candidate in LAYER_PATTERNS:
            pattern_candidate = pattern_candidate.format(layer=share_layers)
            if any([pattern_candidate in name for name in parameter_names]):
                pattern = pattern_candidate
                break

    if pattern is None:
        raise ValueError("Layer pattern could not be matched.")

    # divide parameters in shared and unshared paramter lists
    shared_param_list = []
    unshared_param_list = []

    shared_parameter = True
    for name, param in model.named_parameters():
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

        ref_param = ref_model.get_parameter(param_name)
        ref_param = param

    # for all other parameters just make sure they don't use gradients
    for param_name in unshared_param_list:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False

    return ref_model.eval()
