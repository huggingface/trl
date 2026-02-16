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

"""
Agent Skills for TRL

This module provides utilities for discovering and accessing TRL skills that can be used by AI agents to learn how to
use the TRL CLI.

Agent Skills are folders of instructions, scripts, and resources that agents can discover and use to perform tasks more
accurately and efficiently. Learn more at https://agentskills.io
"""

import importlib.resources as resources
from pathlib import Path


def get_skills_dir() -> Path:
    """
    Get the path to the TRL skills directory.

    This function works in both development and installed package environments.

    Returns:
        Path to the skills directory.

    Example:
        ```python
        from trl.skills import get_skills_dir

        skills_dir = get_skills_dir()
        print(skills_dir)  # /path/to/site-packages/trl/skills
        ```
    """
    return Path(str(resources.files("trl.skills")))


def list_skills() -> list[str]:
    """
    List all available TRL skills.

    A skill is a directory containing a SKILL.md file.

    Returns:
        Sorted list of skill names (directory names containing SKILL.md).

    Example:
        ```python
        from trl.skills import list_skills

        skills = list_skills()
        print(skills)  # ['trl-training']
        ```
    """
    skills_dir = get_skills_dir()
    skills = []

    for item in skills_dir.iterdir():
        if item.is_dir() and (item / "SKILL.md").exists():
            skills.append(item.name)

    return sorted(skills)
