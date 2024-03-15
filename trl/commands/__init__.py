# flake8: noqa

# Copyright 2024 The HuggingFace Team. All rights reserved.
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
# flake8: noqa

from typing import TYPE_CHECKING
from ..import_utils import _LazyModule, OptionalDependencyNotAvailable


_import_structure = {
    "cli_utils": ["SftArgumentParser", "init_zero_verbose", "DpoScriptArguments", "TrlParser"],
    "config_parser": ["YamlConfigParser"],
}


if TYPE_CHECKING:
    from .cli_utils import SftScriptArguments, init_zero_verbose, DpoScriptArguments, TrlParser
    from .config_parser import YamlConfigParser
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
