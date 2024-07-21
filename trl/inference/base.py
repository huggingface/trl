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

from huggingface_hub import PyTorchModelHubMixin


class BaseInference():
    r"""
    Base class for all inference options - this base class implements the basic functions that we
    need for a inference.

    The trainer needs to have the following functions:
        - generate: takes in a batch of prompts and generate a response
    Each user is expected to implement their own inference class that inherits from this base
    if they want to use a new inference algorithm.
    """

    def __init__(self, config):
        self.config = config

    def generate(self, *args):
        raise NotImplementedError("Not implemented")
