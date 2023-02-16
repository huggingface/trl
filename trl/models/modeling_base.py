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
import json
import os
from copy import deepcopy

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import PreTrainedModel


LAYER_PATTERNS = ["transformer.h.{layer}", "model.decoder.layers.{layer}", "gpt_neox.layers.{layer}"]


class PreTrainedModelWrapper(nn.Module):
    r"""
    A wrapper class around a (`transformers.PreTrainedModel`) to be compatible with the
    (`~transformers.PreTrained`) class in order to keep some attributes and methods of the
    (`~transformers.PreTrainedModel`) class.

    Attributes:
        pretrained_model: (`transformers.PreTrainedModel`)
            The model to be wrapped.
        parent_class: (`transformers.PreTrainedModel`)
            The parent class of the model to be wrapped.
        supported_args: (`list`)
            The list of arguments that are supported by the wrapper class.
    """
    transformers_parent_class = None
    supported_args = None
    supported_modules = ("v_head",)

    def __init__(self, pretrained_model=None, **kwargs):
        super().__init__()
        self.pretrained_model = pretrained_model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Instantiates a new model from a pretrained model from `transformers`. The
        pretrained model is loaded using the `from_pretrained` method of the
        `transformers.PreTrainedModel` class. The arguments that are specific to the
        `transformers.PreTrainedModel` class are passed along this method and filtered
        out from the `kwargs` argument.


        Args:
            pretrained_model_name_or_path (`str` or `transformers.PreTrainedModel`):
                The path to the pretrained model or its name.
            *model_args (`list`, *optional*)):
                Additional positional arguments passed along to the underlying model's
                `from_pretrained` method.
            **kwargs (`dict`, *optional*):
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

        # if resume_training, load the state_dict again - this is ok since the
        # state_dict is removed from the model after loading it.
        if isinstance(pretrained_model_name_or_path, str):
            filename = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            sharded_index_filename = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin.index.json")
            is_shared = False

            if not os.path.exists(filename):
                try:
                    filename = hf_hub_download(pretrained_model_name_or_path, "pytorch_model.bin")
                # sharded
                except:  # noqa
                    if os.path.exists(sharded_index_filename):
                        index_file_name = sharded_index_filename
                    else:
                        index_file_name = hf_hub_download(
                            pretrained_model_name_or_path, "pytorch_model.bin.index.json"
                        )
                    # load json
                    with open(index_file_name, "r") as f:
                        index = json.load(f)
                    # check filename with `v_head` or any known extra module:
                    files_to_download = set()
                    for k, v in index["weight_map"].items():
                        if any([module in k for module in cls.supported_modules]):
                            files_to_download.add(v)
                    is_shared = True

            if is_shared:
                # download each file and add it to the state_dict
                state_dict = {}
                for shard_file in files_to_download:
                    filename = hf_hub_download(pretrained_model_name_or_path, shard_file)
                    state_dict.update(torch.load(filename, map_location="cpu"))
            else:
                state_dict = torch.load(filename, map_location="cpu")

        else:
            state_dict = pretrained_model_name_or_path.state_dict()

        model.post_init(state_dict=state_dict)

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
        Push the pretrained model to the hub. This method is a wrapper around
        `transformers.PreTrainedModel.push_to_hub`. Please refer to the documentation
        of `transformers.PreTrainedModel.push_to_hub` for more information.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed along to the underlying model's
                `push_to_hub` method.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed along to the underlying model's
                `push_to_hub` method.
        """
        raise NotImplementedError

    def save_pretrained(self, *args, **kwargs):
        r"""
        Save the pretrained model to a directory. This method is a wrapper around
        `transformers.PreTrainedModel.save_pretrained`. Please refer to the documentation
        of `transformers.PreTrainedModel.save_pretrained` for more information.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed along to the underlying model's
                `save_pretrained` method.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed along to the underlying model's
                `save_pretrained` method.
        """
        state_dict = kwargs.pop("state_dict", None)
        if state_dict is None:
            state_dict = self.state_dict()
            kwargs["state_dict"] = state_dict

        return self.pretrained_model.save_pretrained(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        r"""
        Return the state_dict of the pretrained model.
        """
        raise NotImplementedError

    def post_init(self, *args, **kwargs):
        r"""
        Post initialization method. This method is called after the model is
        instantiated and loaded from a checkpoint. It can be used to perform
        additional operations such as loading the state_dict.
        """
        raise NotImplementedError


def create_reference_model(
    model: PreTrainedModelWrapper, num_shared_layers: int = None, pattern: str = None
) -> PreTrainedModelWrapper:
    """
    Creates a static reference copy of a model. Note that model will be in `.eval()` mode.

    Args:
        model (`PreTrainedModelWrapper`): The model to be copied.
        num_shared_layers (`int`, *optional*): The number of initial layers that are shared between both models and kept frozen.
        pattern (`str`, *optional*): The shared layers are selected with a string pattern
            (e.g. "transformer.h.{layer}" for GPT2) and if a custom pattern is necessary it can be passed here.

    Returns
        `PreTrainedModelWrapper`
    """

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
            if any([pattern_candidate in name for name in parameter_names]):
                pattern = pattern_candidate
                break

    if pattern is None:
        raise ValueError("Layer pattern could not be matched.")

    # divide parameters in shared and unshared parameter lists
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

        ref_param = ref_model.get_parameter(param_name)  # noqa
        ref_param = param  # noqa

    # for all other parameters just make sure they don't use gradients
    for param_name in unshared_param_list:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False

    return ref_model.eval()
