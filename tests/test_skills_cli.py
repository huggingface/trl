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

import argparse

import pytest

from trl.skills import install_skill
from trl.skills.cli import add_skills_subcommands, cmd_install, cmd_list, cmd_uninstall


class TestCLICommands:
    """Tests for CLI command handlers."""

    def test_cmd_list_without_target(self, capsys):
        """Test cmd_list without target (lists TRL skills)."""
        args = argparse.Namespace(target=None, scope="project")

        result = cmd_list(args)

        captured = capsys.readouterr()
        assert result == 0
        assert "TRL (available for installation)" in captured.out
        assert "trl-training" in captured.out
        assert "Use 'trl skills install" in captured.out

    def test_cmd_list_with_target(self, tmp_path, capsys):
        """Test cmd_list with target (lists installed skills)."""
        # Install a skill
        install_skill("trl-training", target=tmp_path)

        args = argparse.Namespace(target=str(tmp_path), scope="project")
        result = cmd_list(args)

        captured = capsys.readouterr()
        assert result == 0
        assert "trl-training" in captured.out
        assert str(tmp_path) in captured.out

    def test_cmd_list_empty_target(self, tmp_path, capsys):
        """Test cmd_list with empty target directory."""
        args = argparse.Namespace(target=str(tmp_path), scope="project")

        result = cmd_list(args)

        captured = capsys.readouterr()
        assert result == 0
        assert "No skills installed" in captured.out

    def test_cmd_install_single_skill(self, tmp_path, capsys):
        """Test cmd_install with single skill."""
        args = argparse.Namespace(skill="trl-training", all=False, target=str(tmp_path), scope="project", force=False)

        result = cmd_install(args)

        captured = capsys.readouterr()
        assert result == 0
        assert "✓" in captured.out
        assert "1/1 skills installed" in captured.out
        assert (tmp_path / "trl-training").exists()

    def test_cmd_install_all_skills(self, tmp_path, capsys):
        """Test cmd_install with --all flag."""
        args = argparse.Namespace(skill=None, all=True, target=str(tmp_path), scope="project", force=False)

        result = cmd_install(args)

        captured = capsys.readouterr()
        assert result == 0
        assert "✓" in captured.out
        assert "installed successfully" in captured.out
        assert (tmp_path / "trl-training").exists()

    def test_cmd_install_no_skill_or_all(self, capsys):
        """Test cmd_install without skill name or --all flag."""
        args = argparse.Namespace(skill=None, all=False, target="/tmp/test", scope="project", force=False)

        result = cmd_install(args)

        captured = capsys.readouterr()
        assert result == 1
        assert "Error: Either provide a skill name or use --all" in captured.out

    def test_cmd_install_both_skill_and_all(self, capsys):
        """Test cmd_install with both skill name and --all (error)."""
        args = argparse.Namespace(skill="trl-training", all=True, target="/tmp/test", scope="project", force=False)

        result = cmd_install(args)

        captured = capsys.readouterr()
        assert result == 1
        assert "Cannot specify both" in captured.out

    def test_cmd_install_nonexistent_skill(self, tmp_path, capsys):
        """Test cmd_install with non-existent skill."""
        args = argparse.Namespace(skill="nonexistent", all=False, target=str(tmp_path), scope="project", force=False)

        result = cmd_install(args)

        captured = capsys.readouterr()
        assert result == 1
        assert "✗" in captured.out
        assert "0/1 skills installed" in captured.out

    def test_cmd_install_already_exists(self, tmp_path, capsys):
        """Test cmd_install when skill already exists without force."""
        # Install once
        install_skill("trl-training", target=tmp_path)

        args = argparse.Namespace(skill="trl-training", all=False, target=str(tmp_path), scope="project", force=False)

        result = cmd_install(args)

        captured = capsys.readouterr()
        assert result == 1
        assert "✗" in captured.out
        assert "Use --force to overwrite" in captured.out

    def test_cmd_install_with_force(self, tmp_path, capsys):
        """Test cmd_install with --force to overwrite."""
        # Install once
        install_skill("trl-training", target=tmp_path)

        args = argparse.Namespace(skill="trl-training", all=False, target=str(tmp_path), scope="project", force=True)

        result = cmd_install(args)

        captured = capsys.readouterr()
        assert result == 0
        assert "✓" in captured.out
        assert "1/1 skills installed" in captured.out

    def test_cmd_uninstall_success(self, tmp_path, capsys):
        """Test cmd_uninstall with installed skill."""
        # Install first
        install_skill("trl-training", target=tmp_path)

        args = argparse.Namespace(skill="trl-training", target=str(tmp_path), scope="project")

        result = cmd_uninstall(args)

        captured = capsys.readouterr()
        assert result == 0
        assert "✓" in captured.out
        assert "has been removed" in captured.out
        assert not (tmp_path / "trl-training").exists()

    def test_cmd_uninstall_not_installed(self, tmp_path, capsys):
        """Test cmd_uninstall when skill is not installed."""
        args = argparse.Namespace(skill="nonexistent", target=str(tmp_path), scope="project")

        result = cmd_uninstall(args)

        captured = capsys.readouterr()
        assert result == 1
        assert "✗" in captured.out
        assert "Error:" in captured.out

    def test_cmd_install_creates_target_directory(self, tmp_path, capsys):
        """Test cmd_install creates target directory if it doesn't exist."""
        # Custom path that doesn't exist yet
        target_path = tmp_path / "new_directory"
        assert not target_path.exists()

        args = argparse.Namespace(
            skill="trl-training", all=False, target=str(target_path), scope="project", force=False
        )

        result = cmd_install(args)

        captured = capsys.readouterr()
        assert result == 0
        assert "✓" in captured.out
        assert target_path.exists()

    def test_cmd_uninstall_invalid_target(self, capsys):
        """Test cmd_uninstall with non-existent path."""
        args = argparse.Namespace(skill="trl-training", target="/nonexistent/invalid/path", scope="project")

        result = cmd_uninstall(args)

        captured = capsys.readouterr()
        assert result == 1
        assert "✗" in captured.out


class TestCLIArgumentParsing:
    """Tests for CLI argument parsing setup."""

    def test_add_skills_subcommands_creates_parsers(self):
        """Test that add_skills_subcommands creates the expected subparsers."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        add_skills_subcommands(subparsers)

        # Test that we can parse expected commands
        args = parser.parse_args(["list"])
        assert args.command == "list"
        assert hasattr(args, "func")

        args = parser.parse_args(["install", "trl-training", "--target", "claude"])
        assert args.command == "install"
        assert args.skill == "trl-training"
        assert args.target == "claude"

        args = parser.parse_args(["uninstall", "trl-training", "--target", "claude"])
        assert args.command == "uninstall"
        assert args.skill == "trl-training"

    def test_list_command_optional_target(self):
        """Test that list command has optional target."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_skills_subcommands(subparsers)

        # Should work without target
        args = parser.parse_args(["list"])
        assert args.target is None

        # Should work with target
        args = parser.parse_args(["list", "--target", "claude"])
        assert args.target == "claude"

    def test_install_command_requires_target(self):
        """Test that install command requires target."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_skills_subcommands(subparsers)

        # Should fail without target
        with pytest.raises(SystemExit):
            parser.parse_args(["install", "trl-training"])

    def test_scope_choices(self):
        """Test that scope parameter accepts valid choices."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_skills_subcommands(subparsers)

        # Valid scopes
        args = parser.parse_args(["install", "trl-training", "--target", "claude", "--scope", "project"])
        assert args.scope == "project"

        args = parser.parse_args(["install", "trl-training", "--target", "claude", "--scope", "global"])
        assert args.scope == "global"

        # Invalid scope should fail
        with pytest.raises(SystemExit):
            parser.parse_args(["install", "trl-training", "--target", "claude", "--scope", "invalid"])

    def test_install_all_flag(self):
        """Test install --all flag."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_skills_subcommands(subparsers)

        args = parser.parse_args(["install", "--all", "--target", "claude"])
        assert args.all is True
        assert args.skill is None

    def test_install_force_flag(self):
        """Test install --force flag."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_skills_subcommands(subparsers)

        args = parser.parse_args(["install", "trl-training", "--target", "claude", "--force"])
        assert args.force is True

    def test_default_scope_is_project(self):
        """Test that default scope is 'project'."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_skills_subcommands(subparsers)

        args = parser.parse_args(["install", "trl-training", "--target", "claude"])
        assert args.scope == "project"
