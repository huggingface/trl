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

import pytest

from trl.cli.accelerate_config import resolve_accelerate_config_argument


class TestResolveAccelerateConfigArgument:
    def test_two_token_form(self):
        """`--accelerate_config <name>` is resolved into `--config_file <path>`."""
        result = resolve_accelerate_config_argument(["--accelerate_config", "single_gpu", "--num_processes", "1"])
        assert result[0] == "--config_file"
        assert result[1].endswith("single_gpu.yaml")
        assert result[2:] == ["--num_processes", "1"]

    def test_equals_form(self):
        """`--accelerate_config=<name>` (equals-sign form) is resolved the same way as the two-token form."""
        result = resolve_accelerate_config_argument(["--accelerate_config=single_gpu", "--num_processes", "1"])
        assert result[0] == "--config_file"
        assert result[1].endswith("single_gpu.yaml")
        assert result[2:] == ["--num_processes", "1"]

    def test_equals_form_matches_two_token_form(self):
        """Both syntaxes should resolve to the exact same launch arguments."""
        equals_form = resolve_accelerate_config_argument(["--accelerate_config=single_gpu", "--foo", "bar"])
        two_token_form = resolve_accelerate_config_argument(["--accelerate_config", "single_gpu", "--foo", "bar"])
        assert equals_form == two_token_form

    def test_no_accelerate_config_argument(self):
        """When `--accelerate_config` isn't present, the arguments are returned unchanged."""
        args = ["--foo", "bar"]
        assert resolve_accelerate_config_argument(args) == args

    def test_missing_value_raises(self):
        """`--accelerate_config` with no value after it raises a clear error."""
        with pytest.raises(ValueError, match="Expected a value after `--accelerate_config`"):
            resolve_accelerate_config_argument(["--accelerate_config"])

    def test_invalid_config_name_raises(self):
        """An unknown config name (and not a file) raises a clear error, for both syntaxes."""
        with pytest.raises(ValueError, match="is neither a file nor a valid config"):
            resolve_accelerate_config_argument(["--accelerate_config", "does_not_exist"])
        with pytest.raises(ValueError, match="is neither a file nor a valid config"):
            resolve_accelerate_config_argument(["--accelerate_config=does_not_exist"])
