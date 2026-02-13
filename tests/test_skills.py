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

import pytest

from trl.skills import get_skill_metadata, get_skill_path, get_skills_dir, list_skills, print_skill_info


def test_get_skills_dir():
    """Test that get_skills_dir() returns a valid directory path."""
    skills_dir = get_skills_dir()

    assert isinstance(skills_dir, Path)
    assert skills_dir.exists(), f"Skills directory does not exist: {skills_dir}"
    assert skills_dir.is_dir(), f"Skills path is not a directory: {skills_dir}"


def test_list_skills():
    """Test that list_skills() returns expected skills."""
    skills = list_skills()

    assert isinstance(skills, list)
    assert skills == []


def test_get_skill_path_not_found():
    """Test that get_skill_path() raises FileNotFoundError for non-existent skill."""
    with pytest.raises(FileNotFoundError) as exc_info:
        get_skill_path("non-existent-skill")

    msg = str(exc_info.value).lower()
    assert "non-existent-skill" in msg
    assert "not found" in msg


def test_print_skill_info_no_args(capsys):
    """Test print_skill_info() without arguments lists all skills."""
    print_skill_info()
    captured = capsys.readouterr()
    output = captured.out

    # Verify output contains expected content
    assert "No agent skills found" in output


def test_all_skills_have_valid_structure():
    """Test that all discovered skills have valid structure."""
    skills = list_skills()

    for skill_name in skills:
        # Can get path
        skill_path = get_skill_path(skill_name)
        assert skill_path.exists()

        # Can parse metadata
        metadata = get_skill_metadata(skill_name)
        assert "name" in metadata
        assert "description" in metadata

        # Name matches directory name
        assert metadata["name"] == skill_name
