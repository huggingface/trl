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

from trl.skills import install_skill, list_agent_names, list_skills, resolve_target_path, uninstall_skill
from trl.skills.skills import _get_trl_skills_dir


class TestGetTrlSkillsDir:
    """Tests for _get_trl_skills_dir function."""

    def test_returns_path_object(self):
        """Test that returns a Path object."""
        skills_dir = _get_trl_skills_dir()
        assert isinstance(skills_dir, Path)

    def test_directory_exists(self):
        """Test that the returned directory exists."""
        skills_dir = _get_trl_skills_dir()
        assert skills_dir.exists(), f"Skills directory does not exist: {skills_dir}"

    def test_is_directory(self):
        """Test that the returned path is a directory."""
        skills_dir = _get_trl_skills_dir()
        assert skills_dir.is_dir(), f"Skills path is not a directory: {skills_dir}"

    def test_contains_skills_module(self):
        """Test that the path ends with 'skills' (the module name)."""
        skills_dir = _get_trl_skills_dir()
        assert skills_dir.name == "skills"


class TestListSkills:
    """Tests for list_skills function."""

    def test_returns_list(self):
        """Test that list_skills returns a list."""
        skills = list_skills()
        assert isinstance(skills, list)

    def test_contains_trl_training(self):
        """Test that list_skills includes the trl-training skill."""
        skills = list_skills()
        assert "trl-training" in skills

    def test_skills_are_sorted(self):
        """Test that skills are returned in sorted order."""
        skills = list_skills()
        assert skills == sorted(skills)

    def test_with_custom_directory(self, tmp_path):
        """Test list_skills with a custom directory."""
        # Create fake skills
        (tmp_path / "skill1").mkdir()
        (tmp_path / "skill1" / "SKILL.md").write_text("# Skill 1")
        (tmp_path / "skill2").mkdir()
        (tmp_path / "skill2" / "SKILL.md").write_text("# Skill 2")
        (tmp_path / "not-a-skill").mkdir()  # No SKILL.md

        skills = list_skills(tmp_path)
        assert skills == ["skill1", "skill2"]

    def test_empty_directory(self, tmp_path):
        """Test list_skills with an empty directory."""
        skills = list_skills(tmp_path)
        assert skills == []

    def test_nonexistent_directory(self, tmp_path):
        """Test list_skills with a non-existent directory."""
        nonexistent = tmp_path / "nonexistent"
        skills = list_skills(nonexistent)
        assert skills == []

    def test_ignores_files(self, tmp_path):
        """Test that list_skills ignores files, only returns directories."""
        (tmp_path / "skill1").mkdir()
        (tmp_path / "skill1" / "SKILL.md").write_text("# Skill 1")
        (tmp_path / "not-a-skill.txt").write_text("Not a skill")

        skills = list_skills(tmp_path)
        assert skills == ["skill1"]

    def test_requires_skill_md(self, tmp_path):
        """Test that directories without SKILL.md are ignored."""
        (tmp_path / "has-skill-md").mkdir()
        (tmp_path / "has-skill-md" / "SKILL.md").write_text("# Valid")
        (tmp_path / "no-skill-md").mkdir()
        (tmp_path / "no-skill-md" / "readme.md").write_text("# Invalid")

        skills = list_skills(tmp_path)
        assert skills == ["has-skill-md"]


class TestInstallSkill:
    """Tests for install_skill function."""

    def test_basic_installation(self, tmp_path):
        """Test basic skill installation."""
        target_dir = tmp_path / "target"

        result = install_skill("trl-training", target_dir)

        assert result is True
        assert (target_dir / "trl-training").exists()
        assert (target_dir / "trl-training" / "SKILL.md").exists()

    def test_creates_target_directory(self, tmp_path):
        """Test that install_skill creates the target directory if it doesn't exist."""
        target_dir = tmp_path / "nested" / "target"

        install_skill("trl-training", target_dir)

        assert target_dir.exists()
        assert (target_dir / "trl-training").exists()

    def test_skill_not_found(self, tmp_path):
        """Test that install_skill raises FileNotFoundError for non-existent skill."""
        target_dir = tmp_path / "target"

        with pytest.raises(FileNotFoundError, match="Skill 'nonexistent' not found"):
            install_skill("nonexistent", target_dir)

    def test_skill_already_exists_without_force(self, tmp_path):
        """Test that install_skill raises FileExistsError if skill exists and force=False."""
        target_dir = tmp_path / "target"

        # Install once
        install_skill("trl-training", target_dir)

        # Try to install again without force
        with pytest.raises(FileExistsError, match="already installed"):
            install_skill("trl-training", target_dir, force=False)

    def test_force_overwrites_existing(self, tmp_path):
        """Test that install_skill with force=True overwrites existing skill."""
        target_dir = tmp_path / "target"

        # Install once
        install_skill("trl-training", target_dir)

        # Modify the installed skill
        marker_file = target_dir / "trl-training" / "marker.txt"
        marker_file.write_text("This should be removed")

        # Install again with force
        result = install_skill("trl-training", target_dir, force=True)

        assert result is True
        assert (target_dir / "trl-training").exists()
        assert not marker_file.exists()  # Marker should be gone

    def test_force_overwrites_symlink(self, tmp_path):
        """Test that install_skill with force=True can overwrite a symlink."""
        target_dir = tmp_path / "target"
        target_dir.mkdir()

        # Create a symlink
        symlink = target_dir / "trl-training"
        symlink.symlink_to(_get_trl_skills_dir() / "trl-training")

        # Install with force should replace symlink with copy
        result = install_skill("trl-training", target_dir, force=True)

        assert result is True
        assert (target_dir / "trl-training").exists()
        assert not (target_dir / "trl-training").is_symlink()

    def test_skill_not_directory(self, tmp_path):
        """Test that install_skill raises ValueError if skill is not a directory."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        target_dir = tmp_path / "target"

        # Create a file instead of directory
        (source_dir / "fake-skill").write_text("not a directory")

        with pytest.raises(ValueError, match="is not a directory"):
            install_skill("fake-skill", target_dir, source=source_dir)

    def test_preserves_directory_structure(self, tmp_path):
        """Test that install_skill preserves the skill's directory structure."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"

        # Create a skill with subdirectories
        skill_dir = source_dir / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Test")
        (skill_dir / "subdir").mkdir()
        (skill_dir / "subdir" / "file.txt").write_text("content")

        install_skill("test-skill", target_dir, source=source_dir)

        assert (target_dir / "test-skill" / "SKILL.md").exists()
        assert (target_dir / "test-skill" / "subdir" / "file.txt").exists()
        assert (target_dir / "test-skill" / "subdir" / "file.txt").read_text() == "content"

    def test_install_to_same_directory_fails(self, tmp_path):
        """Test that installing to the same directory as source is handled correctly."""
        source_dir = tmp_path / "skills"
        source_dir.mkdir()

        # Create a skill
        skill_dir = source_dir / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test")

        # Try to install to same directory (should fail with exists error)
        with pytest.raises(FileExistsError):
            install_skill("test-skill", source_dir, source=source_dir, force=False)


class TestUninstallSkill:
    """Tests for uninstall_skill function."""

    def test_basic_uninstallation(self, tmp_path):
        """Test basic skill uninstallation."""
        target_dir = tmp_path / "target"

        # Install first
        install_skill("trl-training", target_dir)
        assert (target_dir / "trl-training").exists()

        # Uninstall
        result = uninstall_skill("trl-training", target_dir)

        assert result is True
        assert not (target_dir / "trl-training").exists()

    def test_skill_not_installed(self, tmp_path):
        """Test that uninstall_skill raises FileNotFoundError for non-existent skill."""
        target_dir = tmp_path / "target"
        target_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="not installed"):
            uninstall_skill("nonexistent", target_dir)

    def test_uninstall_from_nonexistent_directory(self, tmp_path):
        """Test uninstall_skill when target directory doesn't exist."""
        target_dir = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match="not installed"):
            uninstall_skill("trl-training", target_dir)

    def test_uninstall_removes_all_contents(self, tmp_path):
        """Test that uninstall removes the entire skill directory."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"

        # Create a skill with multiple files
        skill_dir = source_dir / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Test")
        (skill_dir / "file1.txt").write_text("content1")
        (skill_dir / "subdir").mkdir()
        (skill_dir / "subdir" / "file2.txt").write_text("content2")

        # Install and uninstall
        install_skill("test-skill", target_dir, source=source_dir)
        uninstall_skill("test-skill", target_dir)

        assert not (target_dir / "test-skill").exists()
        # Target directory itself should still exist
        assert target_dir.exists()

    def test_uninstall_doesnt_affect_other_skills(self, tmp_path):
        """Test that uninstalling one skill doesn't affect others."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"

        # Create two skills
        for skill_name in ["skill1", "skill2"]:
            skill_dir = source_dir / skill_name
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(f"# {skill_name}")

        # Install both
        install_skill("skill1", target_dir, source=source_dir)
        install_skill("skill2", target_dir, source=source_dir)

        # Uninstall one
        uninstall_skill("skill1", target_dir)

        # Check that only skill1 is removed
        assert not (target_dir / "skill1").exists()
        assert (target_dir / "skill2").exists()


class TestIntegration:
    """Integration tests for skills functions."""

    def test_full_workflow(self, tmp_path):
        """Test complete install -> list -> uninstall workflow."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"

        # Create skills
        for i in range(3):
            skill_dir = source_dir / f"skill{i}"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(f"# Skill {i}")

        # List available skills
        available = list_skills(target=source_dir)
        assert available == ["skill0", "skill1", "skill2"]

        # Install skills
        for skill in available:
            install_skill(skill, target_dir, source=source_dir)

        # List installed skills
        installed_dirs = [d.name for d in target_dir.iterdir() if d.is_dir()]
        assert sorted(installed_dirs) == ["skill0", "skill1", "skill2"]

        # Uninstall one skill
        uninstall_skill("skill1", target_dir)

        # Verify
        installed_dirs = [d.name for d in target_dir.iterdir() if d.is_dir()]
        assert sorted(installed_dirs) == ["skill0", "skill2"]

    def test_install_uninstall_cycle(self, tmp_path):
        """Test that we can install and uninstall the same skill multiple times."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"

        # Create skill
        skill_dir = source_dir / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Test")

        # Install -> Uninstall -> Install -> Uninstall
        for _ in range(2):
            install_skill("test-skill", target_dir, source=source_dir)
            assert (target_dir / "test-skill").exists()

            uninstall_skill("test-skill", target_dir)
            assert not (target_dir / "test-skill").exists()

    def test_force_reinstall_workflow(self, tmp_path):
        """Test the workflow of using force to update an installed skill."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"

        # Create initial skill version
        skill_dir = source_dir / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Version 1")

        # Install
        install_skill("test-skill", target_dir, source=source_dir)
        assert (target_dir / "test-skill" / "SKILL.md").read_text() == "# Version 1"

        # Update source skill
        (skill_dir / "SKILL.md").write_text("# Version 2")

        # Force reinstall
        install_skill("test-skill", target_dir, source=source_dir, force=True)
        assert (target_dir / "test-skill" / "SKILL.md").read_text() == "# Version 2"


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_skill_with_special_characters_in_name(self, tmp_path):
        """Test handling skills with special characters in names."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"

        # Create skill with hyphens and underscores (common in skill names)
        skill_name = "test-skill_v2"
        skill_dir = source_dir / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Test")

        # Should work fine
        install_skill(skill_name, target_dir, source=source_dir)
        assert (target_dir / skill_name).exists()

        uninstall_skill(skill_name, target_dir)
        assert not (target_dir / skill_name).exists()

    def test_empty_skill_directory(self, tmp_path):
        """Test installing a skill with only SKILL.md (no other files)."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"

        skill_dir = source_dir / "minimal-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Minimal")

        install_skill("minimal-skill", target_dir, source=source_dir)

        assert (target_dir / "minimal-skill" / "SKILL.md").exists()
        # Should only contain SKILL.md
        files = list((target_dir / "minimal-skill").iterdir())
        assert len(files) == 1
        assert files[0].name == "SKILL.md"

    def test_skill_with_hidden_files(self, tmp_path):
        """Test that hidden files are preserved during installation."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"

        skill_dir = source_dir / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Test")
        (skill_dir / ".hidden").write_text("hidden content")

        install_skill("test-skill", target_dir, source=source_dir)

        assert (target_dir / "test-skill" / ".hidden").exists()
        assert (target_dir / "test-skill" / ".hidden").read_text() == "hidden content"

    def test_list_skills_with_symlinks(self, tmp_path):
        """Test that list_skills handles symlinked skill directories."""
        source_dir = tmp_path / "source"
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        # Create a real skill
        skill_dir = source_dir / "real-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Real")

        # Create symlink to it
        (skills_dir / "linked-skill").symlink_to(skill_dir)

        # list_skills should include symlinked skills if they have SKILL.md
        skills = list_skills(target=skills_dir)
        assert "linked-skill" in skills


class TestListAgentNames:
    """Tests for list_agent_names function."""

    def test_returns_list(self):
        """Test that list_agent_names returns a list."""
        agents = list_agent_names()
        assert isinstance(agents, list)

    def test_contains_expected_agents(self):
        """Test that list includes expected agent names."""
        agents = list_agent_names()
        assert "claude" in agents
        assert "codex" in agents
        assert "opencode" in agents

    def test_agents_are_sorted(self):
        """Test that agent names are sorted."""
        agents = list_agent_names()
        assert agents == sorted(agents)


class TestResolveTargetPath:
    """Tests for resolve_target_path function."""

    def test_resolve_agent_name_project_scope(self):
        """Test resolving agent name with project scope."""
        path = resolve_target_path("claude", "project")
        assert path == Path("./.claude/skills").expanduser().resolve()

    def test_resolve_agent_name_global_scope(self):
        """Test resolving agent name with global scope."""
        path = resolve_target_path("claude", "global")
        assert path == Path("~/.claude/skills").expanduser().resolve()

    def test_resolve_custom_path_string(self):
        """Test resolving custom path as string."""
        path = resolve_target_path("/custom/path", "project")
        assert path == Path("/custom/path").resolve()

    def test_resolve_custom_path_object(self):
        """Test resolving Path object."""
        custom = Path("/custom/path")
        path = resolve_target_path(custom, "project")
        assert path == Path("/custom/path").resolve()

    def test_resolve_path_with_tilde(self):
        """Test that tilde expansion works."""
        path = resolve_target_path("~/my/skills", "project")
        assert path == Path("~/my/skills").expanduser().resolve()
        assert "~" not in str(path)

    def test_all_predefined_agents(self):
        """Test that all predefined agents can be resolved."""
        for agent in list_agent_names():
            for scope in ["project", "global"]:
                path = resolve_target_path(agent, scope)
                assert isinstance(path, Path)
                assert path.is_absolute()

    def test_invalid_scope_for_predefined_agent(self):
        """Test invalid scope raises ValueError for predefined agents."""
        with pytest.raises(ValueError, match="Invalid scope"):
            resolve_target_path("claude", "invalid")


class TestHighLevelAPI:
    """Tests for the new high-level API (target/scope instead of Path)."""

    def test_list_skills_with_target_string(self, tmp_path):
        """Test list_skills with target as string (custom path)."""
        # Create skills in target
        (tmp_path / "skill1").mkdir()
        (tmp_path / "skill1" / "SKILL.md").write_text("# Skill 1")

        skills = list_skills(target=str(tmp_path), scope="project")
        assert skills == ["skill1"]

    def test_list_skills_with_target_path(self, tmp_path):
        """Test list_skills with target as Path object."""
        (tmp_path / "skill1").mkdir()
        (tmp_path / "skill1" / "SKILL.md").write_text("# Skill 1")

        skills = list_skills(target=tmp_path, scope="project")
        assert skills == ["skill1"]

    def test_list_skills_without_target(self):
        """Test list_skills without target lists TRL's built-in skills."""
        skills = list_skills()
        assert isinstance(skills, list)
        assert "trl-training" in skills

    def test_install_skill_with_target_string(self, tmp_path):
        """Test install_skill with target as string."""
        result = install_skill("trl-training", target=str(tmp_path), scope="project")
        assert result is True
        assert (tmp_path / "trl-training").exists()

    def test_install_skill_with_target_path(self, tmp_path):
        """Test install_skill with target as Path object."""
        result = install_skill("trl-training", target=tmp_path, scope="project")
        assert result is True
        assert (tmp_path / "trl-training").exists()

    def test_install_skill_with_force(self, tmp_path):
        """Test install_skill with force parameter."""
        install_skill("trl-training", target=tmp_path)
        # Install again with force
        result = install_skill("trl-training", target=tmp_path, force=True)
        assert result is True

    def test_uninstall_skill_with_target_string(self, tmp_path):
        """Test uninstall_skill with target as string."""
        install_skill("trl-training", target=tmp_path)
        result = uninstall_skill("trl-training", target=str(tmp_path), scope="project")
        assert result is True
        assert not (tmp_path / "trl-training").exists()

    def test_uninstall_skill_with_target_path(self, tmp_path):
        """Test uninstall_skill with target as Path object."""
        install_skill("trl-training", target=tmp_path)
        result = uninstall_skill("trl-training", target=tmp_path, scope="project")
        assert result is True
        assert not (tmp_path / "trl-training").exists()

    def test_install_with_custom_source(self, tmp_path):
        """Test install_skill with custom source parameter."""
        source_dir = tmp_path / "source"
        target_dir = tmp_path / "target"

        # Create custom skill
        skill_dir = source_dir / "custom-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Custom")

        result = install_skill("custom-skill", target=target_dir, source=source_dir)
        assert result is True
        assert (target_dir / "custom-skill").exists()
