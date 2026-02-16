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

from trl.skills import _get_skills_dir, list_skills


def test_get_skills_dir():
    """Test that get_skills_dir() returns a valid directory path."""
    skills_dir = _get_skills_dir()

    assert isinstance(skills_dir, Path)
    assert skills_dir.exists(), f"Skills directory does not exist: {skills_dir}"
    assert skills_dir.is_dir(), f"Skills path is not a directory: {skills_dir}"


def test_list_skills():
    """Test that list_skills() returns expected skills."""
    skills = list_skills()

    assert isinstance(skills, list)
    assert skills == ["trl-training"]
