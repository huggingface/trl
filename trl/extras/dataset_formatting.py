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


import datasets
from datasets import Value
from packaging.version import Version


if Version(datasets.__version__) >= Version("4.0.0"):
    from datasets import List

    FORMAT_MAPPING = {
        "chatml": List({"content": Value(dtype="string", id=None), "role": Value(dtype="string", id=None)}),
        "instruction": {"completion": Value(dtype="string", id=None), "prompt": Value(dtype="string", id=None)},
    }
else:
    FORMAT_MAPPING = {
        "chatml": [{"content": Value(dtype="string", id=None), "role": Value(dtype="string", id=None)}],
        "instruction": {"completion": Value(dtype="string", id=None), "prompt": Value(dtype="string", id=None)},
    }
