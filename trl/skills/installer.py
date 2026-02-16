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
Skill installer for managing TRL skills in agent directories.

This module handles installation, uninstallation, and management of TRL skills, supporting both copy and symlink
installation methods.
"""

import os
import shutil
from pathlib import Path


class SkillInstaller:
    """
    Handles installation, uninstallation, and management of skills.

    This class provides methods to install TRL skills to agent-specific directories, supporting both copy (independent
    files) and symlink (auto-sync) methods.

    Args:
        source_skills_dir: Path to TRL skills directory (from get_skills_dir()).
    """

    def __init__(self, source_skills_dir: Path):
        self.source_skills_dir = source_skills_dir

    def install_skill(
        self,
        skill_name: str,
        target_dir: Path,
        force: bool = False,
        create_dirs: bool = True,
    ) -> bool:
        """
        Install a skill to target directory.

        Args:
            skill_name: Name of skill to install.
            target_dir: Directory to install to.
            force: Overwrite if exists.
            create_dirs: Create target directory if it doesn't exist.

        Returns:
            True if installed successfully.

        Raises:
            FileNotFoundError: If skill doesn't exist in TRL. FileExistsError: If skill already installed and not
            force. PermissionError: If no permission to write to target.
        """
        source_skill = self.source_skills_dir / skill_name

        # Check if source skill exists
        if not source_skill.exists():
            raise FileNotFoundError(
                f"Skill '{skill_name}' not found in TRL. Available skills: {', '.join(self._list_source_skills())}"
            )

        if not source_skill.is_dir():
            raise ValueError(f"Skill '{skill_name}' is not a directory")

        target_skill = target_dir / skill_name

        # Check if already exists
        if target_skill.exists() and not force:
            raise FileExistsError(
                f"Skill '{skill_name}' already installed at {target_skill}. Use --force to overwrite."
            )

        # Create target directory
        if create_dirs:
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

    def uninstall_skill(self, skill_name: str, target_dir: Path) -> bool:
        """
        Uninstall a skill from target directory.

        Args:
            skill_name: Name of skill to uninstall.
            target_dir: Directory skill is installed in.

        Returns:
            True if uninstalled successfully.

        Raises:
            FileNotFoundError: If skill not installed. PermissionError: If no permission to remove.
        """
        target_skill = target_dir / skill_name

        if not target_skill.exists():
            raise FileNotFoundError(f"Skill '{skill_name}' not installed at {target_dir}")

        # Remove symlink or directory
        try:
            if target_skill.is_symlink():
                target_skill.unlink()
            else:
                shutil.rmtree(target_skill)
        except OSError as e:
            raise PermissionError(f"Cannot remove skill: {e}") from e

        return True

    def list_installed_skills(self, target_dir: Path) -> list[dict]:
        """
        List skills installed in target directory.

        Args:
            target_dir: Directory to check

        Returns:
            List of dicts with skill info:
            - name: Skill name.
            - method: "copy" or "symlink".
            - path: Absolute path to installed skill.
            - source: Source path (for symlinks only).
        """
        if not target_dir.exists():
            return []

        installed = []

        try:
            for item in target_dir.iterdir():
                if not item.is_dir():
                    continue

                # Check if it's a valid skill (has SKILL.md)
                if not (item / "SKILL.md").exists():
                    continue

                is_symlink = item.is_symlink()

                info = {
                    "name": item.name,
                    "method": "symlink" if is_symlink else "copy",
                    "path": str(item.resolve()),
                }

                if is_symlink:
                    try:
                        source = os.readlink(item)
                    except OSError:
                        source = None
                    info["source"] = source

                installed.append(info)
        except PermissionError:
            # If we can't read the directory, return empty list
            return []

        return sorted(installed, key=lambda x: x["name"])

    def _list_source_skills(self) -> list[str]:
        """List available skills in source directory."""
        skills = []
        for item in self.source_skills_dir.iterdir():
            if item.is_dir() and (item / "SKILL.md").exists():
                skills.append(item.name)
        return sorted(skills)


__all__ = [
    "SkillInstaller",
]
