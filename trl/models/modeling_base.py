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
import torch.nn as nn

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
    """
    pretrained_model = None
    transformers_parent_class = None
    

    def __init__(self, pretrained_model=None, **kwargs):
        super().__init__()
        self.pretrained_model = pretrained_model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiates a new model from a pretrained model.

        Parameters
        ----------
        pretrained_model_name_or_path: (`str`)
            The path to the pretrained model or its name.
        **kwargs:
            Additional keyword arguments passed along to the underlying model's
            `from_pretrained` method.
        """
        # First, load the pre-trained model using the parent-class
        # either `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM`
        pretrained_model = cls.transformers_parent_class.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        # Then, create the full model by instantiating the wrapper class
        model = cls(pretrained_model, **kwargs)

        return model
    
    def push_to_hub(self, *args, **kwargs):
        r"""
        Push the model to the hub.
        """
        return self.pretrained_model.push_to_hub(*args, **kwargs)
    
    def save_pretrained(self, *args, **kwargs):
        r"""
        Save the model to a directory.
        """
        return self.pretrained_model.save_pretrained(*args, **kwargs)
