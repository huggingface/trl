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

from pathlib import Path

import trl
from trl.cli.commands import get_commands


def test_skill_uses_the_installed_library_convention():
    skill = Path(trl.__file__).parent / ".agents" / "skills" / "trl" / "SKILL.md"

    assert skill.is_file()
    assert skill.read_text(encoding="utf-8").startswith("---\nname: trl\n")


def test_registered_commands_do_not_include_per_library_skill_installer():
    assert "skills" not in {command.name for command in get_commands()}
