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
from typing import Any

import yaml


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


def get_skill_path(skill_name: str) -> Path:
    """
    Get the path to a specific skill's SKILL.md file.

    Args:
        skill_name: Name of the skill (e.g., "trl-training").

    Returns:
        Path to the SKILL.md file.

    Raises:
        FileNotFoundError: If the skill doesn't exist.

    Example:
        ```python
        from trl.skills import get_skill_path

        skill_path = get_skill_path("trl-training")
        with open(skill_path) as f:
            content = f.read()
        ```
    """
    skill_file = get_skills_dir() / skill_name / "SKILL.md"
    if not skill_file.exists():
        available = list_skills()
        raise FileNotFoundError(
            f"Skill '{skill_name}' not found. Available skills: {', '.join(available) if available else 'none'}"
        )
    return skill_file


def get_skill_metadata(skill_name: str) -> dict[str, Any]:
    """
    Parse and return metadata from a skill's YAML frontmatter.

    The YAML frontmatter is enclosed between --- delimiters at the start of SKILL.md.

    Args:
        skill_name: Name of the skill.

    Returns:
        Dictionary containing skill metadata (name, description, license, metadata, etc.).

    Raises:
        FileNotFoundError: If the skill doesn't exist. ValueError: If the YAML frontmatter is invalid.

    Example:
        ```python
        from trl.skills import get_skill_metadata

        metadata = get_skill_metadata("trl-training")
        print(metadata["name"])  # trl-training
        print(metadata["description"])  # Train and fine-tune transformer...
        ```
    """
    skill_file = get_skill_path(skill_name)

    with open(skill_file, encoding="utf-8") as f:
        content = f.read()

    # Extract YAML frontmatter
    if not content.startswith("---"):
        raise ValueError(f"Skill '{skill_name}' SKILL.md is missing YAML frontmatter")

    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"Skill '{skill_name}' SKILL.md has invalid YAML frontmatter structure")

    try:
        frontmatter = yaml.safe_load(parts[1])
    except yaml.YAMLError as e:
        raise ValueError(f"Skill '{skill_name}' SKILL.md has invalid YAML: {e}") from e

    if not isinstance(frontmatter, dict):
        raise ValueError(f"Skill '{skill_name}' SKILL.md frontmatter must be a YAML mapping")

    return frontmatter


def print_skill_info(skill_name: str | None = None) -> None:
    """
    Print information about available skills.

    If a specific skill name is provided, prints detailed information about that skill. Otherwise, prints a list of all
    available skills with their descriptions.

    Args:
        skill_name: Optional specific skill to show info for.

    Example:
        ```python
        from trl.skills import print_skill_info

        # List all skills
        print_skill_info()

        # Show details for specific skill
        print_skill_info("trl-training")
        ```
    """
    if skill_name:
        try:
            metadata = get_skill_metadata(skill_name)
            path = get_skill_path(skill_name)
            print(f"\nSkill: {skill_name}")
            print(f"Description: {metadata.get('description', 'N/A')}")
            print(f"License: {metadata.get('license', 'N/A')}")
            print(f"Path: {path}")

            # Print additional metadata if available
            if "metadata" in metadata:
                meta = metadata["metadata"]
                if "version" in meta:
                    print(f"Version: {meta['version']}")
                if "commands" in meta:
                    print("\nCommands:")
                    for cmd in meta["commands"]:
                        print(f"  - {cmd}")
                if "documentation" in meta:
                    print(f"\nDocumentation: {meta['documentation']}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
    else:
        skills = list_skills()
        if not skills:
            print("No agent skills found.")
            return

        print(f"\nTRL Agent Skills ({len(skills)}):\n")
        print("Agent Skills are instructions that AI agents can use to learn how to use TRL CLI commands.")
        print("Learn more at https://agentskills.io\n")

        for skill in skills:
            try:
                metadata = get_skill_metadata(skill)
                print(f"  {skill}")
                print(f"    {metadata.get('description', 'No description')}\n")
            except (FileNotFoundError, ValueError):
                print(f"  {skill}")
                print("    (Invalid skill metadata)\n")

        print("To see detailed information about a specific skill:")
        print(f"  trl skills {skills[0]}")
        print("\nTo use a skill with an AI agent, read the SKILL.md file:")
        print(f"  cat $(python -c \"from trl.skills import get_skill_path; print(get_skill_path('{skills[0]}'))\")")
