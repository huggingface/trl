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
Agent Skills.

This module:
- provides utilities for discovering and accessing TRL skills that can be used by AI agents to learn how to use the TRL
  CLI
- handles installation, uninstallation, and management of TRL skills
- defines where different AI agents and coding tools look for skills, enabling easy installation of TRL skills to the
  appropriate directories

Agent Skills are folders of instructions, scripts, and resources that agents can discover and use to perform tasks more
accurately and efficiently. Learn more at https://agentskills.io
"""

import importlib.resources as resources
import shutil
from pathlib import Path


AGENT_PATHS = {
    "claude": {
        "global": Path("~/.claude/skills"),
        "project": Path("./.claude/skills"),
    },
    "codex": {
        "global": Path("~/.codex/skills"),
        "project": Path("./.codex/skills"),
    },
    "opencode": {
        "global": Path("~/.config/opencode/skills"),
        "project": Path(".opencode/skills"),
    },
}


def _get_trl_skills_dir() -> Path:
    """
    Get the path to the TRL skills directory.

    This is the directory inside the TRL package containing skills that can be installed to AI agent directories.

    Returns:
        `Path`: TRL skills directory.

    Example:
        ```python
        from trl.skills import get_skills_dir

        skills_dir = get_skills_dir()
        print(skills_dir)  # /path/to/site-packages/trl/skills
        ```
    """
    return Path(str(resources.files("trl.skills")))


def list_skills(skills_dir: Path | None = None) -> list[str]:
    """
    List skills in skills directory.

    A skill is a directory containing a SKILL.md file.

    Args:
        skills_dir (`Path`, *optional*): Skills directory. If `None`, it defaults to TRL skills directory.

    Returns:
        `list[str]`: Skill names (directory names containing SKILL.md).

    Example:
        ```python
        from trl.skills import list_skills

        skills = list_skills()
        print(skills)  # ['trl-training']
        ```
    """
    skills_dir = skills_dir or _get_trl_skills_dir()
    if not skills_dir.exists():
        return []
    skills = []
    for item in skills_dir.iterdir():
        if item.is_dir() and (item / "SKILL.md").exists():
            skills.append(item.name)
    return sorted(skills)


def install_skill(
    skill_name: str,
    target_dir: Path,
    source_dir: Path | None = None,
    force: bool = False,
) -> bool:
    """
    Install a skill to target directory.

    Args:
        skill_name (`str`): Name of skill to install.
        target_dir (`Path`): Target installation directory.
        source_dir (`Path`, *optional*):
            Source directory containing skills. If `None`, it defaults to TRL skills directory.
        force (`bool`, defaults to `False`): Whether to overwrite if exists.

    Returns:
        `bool`: True if installed successfully.

    Raises:
        - `FileNotFoundError`: If skill doesn't exist in TRL.
        - `FileExistsError`: If skill already installed and not force.
        - `PermissionError`: If no permission to write to target.
    """
    source_dir = source_dir or _get_trl_skills_dir()
    source_skill = source_dir / skill_name

    # Check if source skill exists
    if not source_skill.exists():
        raise FileNotFoundError(f"Skill '{skill_name}' not found in TRL. Available skills: {', '.join(list_skills())}")

    if not source_skill.is_dir():
        raise ValueError(f"Skill '{skill_name}' is not a directory")

    target_skill = target_dir / skill_name

    # Check if already exists
    if target_skill.exists() and not force:
        raise FileExistsError(f"Skill '{skill_name}' already installed at {target_skill}. Use --force to overwrite.")

    # Create target directory
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Cannot create directory {target_dir}: {e}") from e

    # Remove existing if force
    if target_skill.exists() and force:
        if target_skill.is_symlink():
            target_skill.unlink()
        else:
            shutil.rmtree(target_skill)

    # Install
    try:
        shutil.copytree(source_skill, target_skill)
    except OSError as e:
        raise OSError(f"Failed to install skill: {e}") from e

    return True


def uninstall_skill(skill_name: str, target_dir: Path) -> bool:
    """
    Uninstall a skill from target directory.

    Args:
        skill_name (`str`): Name of skill to uninstall.
        target_dir (`Path`): Directory skill is installed in.

    Returns:
        `bool`: True if uninstalled successfully.

    Raises:
        `FileNotFoundError`: If skill not installed. PermissionError: If no permission to remove.
    """
    target_skill = target_dir / skill_name

    if not target_skill.exists():
        raise FileNotFoundError(f"Skill '{skill_name}' not installed at {target_dir}")

    # Remove symlink or directory
    try:
        shutil.rmtree(target_skill)
    except OSError as e:
        raise PermissionError(f"Cannot remove skill: {e}") from e

    return True
