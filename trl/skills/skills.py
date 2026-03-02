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


def list_agent_names() -> list[str]:
    """
    List available predefined agent names.

    Returns:
        `list[str]`: Sorted list of agent names (e.g., ['claude', 'codex', 'opencode']).
    """
    return sorted(AGENT_PATHS.keys())


def _get_trl_skills_dir() -> Path:
    """
    Get the path to the TRL skills directory.

    This is the directory inside the TRL package containing skills that can be installed to AI agent directories.

    Returns:
        `Path`: TRL skills directory.
    """
    return Path(str(resources.files("trl.skills")))


def resolve_target_path(target: str | Path, scope: str = "project") -> Path:
    """
    Resolve target to a concrete directory path.

    Converts semantic agent names (e.g., 'claude') with scope to actual filesystem paths, or normalizes provided paths.

    Args:
        target (`str | Path`): Agent name (e.g., 'claude', 'codex') or directory path.
        scope (`str`, defaults to `"project"`):
            Scope for agent names: 'global' (user-level like ~/.agent/skills/) or 'project' (./agent/skills/).

    Returns:
        `Path`: Resolved absolute path.

    Raises:
        `ValueError`: If `scope` is invalid for a predefined agent target.

    Example:
        ```python
        from trl.skills import resolve_target_path

        # Resolve agent name with scope
        path = resolve_target_path("claude", "global")
        print(path)  # /home/user/.claude/skills

        # Resolve custom path
        path = resolve_target_path("/custom/skills")
        print(path)  # /custom/skills
        ```
    """
    if isinstance(target, Path):
        return target.expanduser().resolve()

    # Check if it's a predefined agent
    if target in AGENT_PATHS:
        if scope not in AGENT_PATHS[target]:
            valid_scopes = ", ".join(sorted(AGENT_PATHS[target]))
            raise ValueError(f"Invalid scope '{scope}' for agent '{target}'. Expected one of: {valid_scopes}")
        agent_path = AGENT_PATHS[target][scope]
        return agent_path.expanduser().resolve()

    # Treat as custom path string
    return Path(target).expanduser().resolve()


def _list_skills_in_dir(skills_dir: Path) -> list[str]:
    """
    List skills in directory.

    A skill is a directory containing a SKILL.md file.

    Args:
        skills_dir (`Path`): Skills directory to scan.

    Returns:
        `list[str]`: Skill names (directory names containing SKILL.md).
    """
    if not skills_dir.exists():
        return []
    skills = []
    for item in skills_dir.iterdir():
        if item.is_dir() and (item / "SKILL.md").exists():
            skills.append(item.name)
    return sorted(skills)


def list_skills(target: str | Path | None = None, scope: str = "project") -> list[str]:
    """
    List skills.

    A skill is a directory containing a SKILL.md file.

    Args:
        target (`str | Path`, *optional*):
            Agent name (e.g., 'claude'), directory path, or `None` for TRL's built-in skills.
        scope (`str`, defaults to `"project"`):
            For agent names: 'global' (user-level) or 'project' (current directory).

    Returns:
        `list[str]`: Skill names (directory names containing SKILL.md).

    Example:
        ```python
        from trl.skills import list_skills

        # List TRL's built-in skills
        skills = list_skills()
        print(skills)  # ['trl-training']

        # List skills installed for Claude globally
        installed = list_skills(target="claude", scope="global")
        print(installed)  # ['trl-training', 'custom-skill']

        # List skills in custom directory
        custom = list_skills(target="/path/to/skills")
        print(custom)  # [...]
        ```
    """
    if target is None:
        # List TRL's built-in skills
        return _list_skills_in_dir(_get_trl_skills_dir())

    target_dir = resolve_target_path(target, scope)
    return _list_skills_in_dir(target_dir)


def _install_skill_to_dir(
    skill_name: str,
    target_dir: Path,
    source_dir: Path,
    force: bool = False,
) -> bool:
    """
    Install a skill to target directory.

    Args:
        skill_name (`str`): Name of skill to install.
        target_dir (`Path`): Target installation directory.
        source_dir (`Path`): Source directory containing skills.
        force (`bool`, defaults to `False`): Whether to overwrite if exists.

    Returns:
        `bool`: True if installed successfully.

    Raises:
        - `FileNotFoundError`: If skill doesn't exist in source_dir.
        - `FileExistsError`: If skill already installed and not force.
        - `PermissionError`: If no permission to write to target_dir.
        - `ValueError`: If source_dir entry exists but is not a directory.
        - `OSError`: If copying the skill fails.
    """
    source_skill = source_dir / skill_name

    # Check if source skill exists
    if not source_skill.exists():
        available = ", ".join(list_skills(target=source_dir))
        source_msg = f"source directory {source_dir}"
        if available:
            raise FileNotFoundError(f"Skill '{skill_name}' not found in {source_msg}. Available skills: {available}")
        raise FileNotFoundError(f"Skill '{skill_name}' not found in {source_msg}")

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


def install_skill(
    skill_name: str,
    target: str | Path,
    scope: str = "project",
    source: str | Path | None = None,
    force: bool = False,
) -> bool:
    """
    Install a skill.

    Args:
        skill_name (`str`): Name of skill to install.
        target (`str | Path`): Agent name (e.g., 'claude', 'codex') or directory path.
        scope (`str`, defaults to `"project"`):
            Scope for agent names: 'global' (user-level) or 'project' (current directory).
        source (`str | Path`, *optional*):
            Source directory containing skills. If `None`, defaults to TRL skills directory.
        force (`bool`, defaults to `False`): Whether to overwrite if skill already exists.

    Returns:
        `bool`: True if installed successfully.

    Raises:
        - `FileNotFoundError`: If skill doesn't exist in source.
        - `FileExistsError`: If skill already installed and not force.
        - `PermissionError`: If no permission to write to target.
        - `ValueError`:
           - If `scope` is invalid for a predefined agent target.
           - If `source` entry exists but is not a directory.
        - `OSError`: If copying the skill fails.

    Example:
        ```python
        from trl.skills import install_skill

        # Install to Claude's global skills directory
        install_skill("trl-training", target="claude", scope="global")

        # Install to custom directory
        install_skill("trl-training", target="/path/to/skills")

        # Overwrite existing installation
        install_skill("trl-training", target="claude", force=True)
        ```
    """
    target_dir = resolve_target_path(target, scope)
    source_dir = Path(source).expanduser().resolve() if source else _get_trl_skills_dir()
    return _install_skill_to_dir(skill_name, target_dir, source_dir, force)


def _uninstall_skill_from_dir(skill_name: str, target_dir: Path) -> bool:
    """
    Uninstall a skill from target directory.

    Args:
        skill_name (`str`): Name of skill to uninstall.
        target_dir (`Path`): Directory skill is installed in.

    Returns:
        `bool`: True if uninstalled successfully.

    Raises:
        - `FileNotFoundError`: If skill not installed.
        - `PermissionError`: If no permission to remove.
        - `OSError`: If removing the skill fails for another filesystem reason.
    """
    target_skill = target_dir / skill_name

    if not target_skill.exists():
        raise FileNotFoundError(f"Skill '{skill_name}' not installed at {target_dir}")

    # Remove symlink or directory
    try:
        shutil.rmtree(target_skill)
    except PermissionError as e:
        raise PermissionError(f"Cannot remove skill: {e}") from e
    except OSError as e:
        raise OSError(f"Failed to remove skill: {e}") from e

    return True


def uninstall_skill(skill_name: str, target: str | Path, scope: str = "project") -> bool:
    """
    Uninstall a skill.

    Args:
        skill_name (`str`): Name of skill to uninstall.
        target (`str | Path`): Agent name (e.g., 'claude', 'codex') or directory path.
        scope (`str`, defaults to `"project"`):
            Scope for agent names: 'global' (user-level) or 'project' (current directory).

    Returns:
        `bool`: True if uninstalled successfully.

    Raises:
        - `FileNotFoundError`: If skill not installed.
        - `PermissionError`: If no permission to remove.
        - `OSError`: If removing the skill fails for another filesystem reason.
        - `ValueError`: If `scope` is invalid for a predefined agent target.

    Example:
        ```python
        from trl.skills import uninstall_skill

        # Uninstall from Claude's global directory
        uninstall_skill("trl-training", target="claude", scope="global")

        # Uninstall from custom directory
        uninstall_skill("trl-training", target="/path/to/skills")
        ```
    """
    target_dir = resolve_target_path(target, scope)
    return _uninstall_skill_from_dir(skill_name, target_dir)
