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
CLI commands for TRL skills installation and management.

This module provides command-line interface for installing TRL skills to various AI agent directories.
"""

import argparse
from pathlib import Path

from .agents import AGENT_REGISTRY, get_agent_target, list_agent_names
from .installer import InstallMethod, SkillInstaller
from .skills import get_skills_dir, list_skills


def add_skills_subcommands(subparsers: argparse._SubParsersAction) -> None:
    """
    Add skills subcommands to the parser.

    This creates nested subcommands under 'trl skills' for managing skill installations.

    Args:
        subparsers: Subparsers from 'trl skills' command
    """
    # Parent parser for common target options
    target_parser = argparse.ArgumentParser(add_help=False)
    target_parser.add_argument(
        "--target",
        required=True,
        help=f"Installation target: agent name ({', '.join(list_agent_names())}) or directory path",
    )
    target_parser.add_argument(
        "--scope",
        choices=["global", "project"],
        default="global",
        help="Scope for predefined agents: global (user-level like ~/.agent/skills/) or project (./agent/skills/)",
    )

    # trl skills install
    install_parser = subparsers.add_parser(
        "install",
        parents=[target_parser],
        help="Install skill to agent directory",
        description="Install a TRL skill to an AI agent's skills directory",
    )
    install_parser.add_argument("skill", nargs="?", help="Skill name to install (omit to use --all)")
    install_parser.add_argument("--all", action="store_true", help="Install all available TRL skills")
    install_parser.add_argument(
        "--method",
        choices=["copy", "symlink"],
        default="copy",
        help="Installation method: copy (independent files, default) or symlink (stay synced with TRL updates)",
    )
    install_parser.add_argument("--force", action="store_true", help="Overwrite if skill already exists")
    install_parser.set_defaults(func=cmd_install)

    # trl skills uninstall
    uninstall_parser = subparsers.add_parser(
        "uninstall",
        parents=[target_parser],
        help="Uninstall skill from agent directory",
        description="Remove a TRL skill from an AI agent's skills directory",
    )
    uninstall_parser.add_argument("skill", help="Skill name to uninstall")
    uninstall_parser.set_defaults(func=cmd_uninstall)

    # trl skills list-installed
    list_installed_parser = subparsers.add_parser(
        "list-installed",
        parents=[target_parser],
        help="List installed skills",
        description="Show skills installed in an agent's directory",
    )
    list_installed_parser.set_defaults(func=cmd_list_installed)


def resolve_target(args) -> tuple[Path, str]:
    """
    Resolve target directory from args.

    Args:
        args: Parsed arguments with 'target' and 'scope' attributes

    Returns:
        Tuple of (target_path, display_name)

    Raises:
        ValueError: If agent name is not recognized FileNotFoundError: If custom path doesn't exist
    """
    # Check if it's a predefined agent
    if args.target in AGENT_REGISTRY:
        agent = get_agent_target(args.target)
        try:
            path = agent.get_path(args.scope)
            display_name = f"{agent.display_name} ({path})"
            return path, display_name
        except ValueError as e:
            raise ValueError(str(e)) from e
    else:
        # Custom path
        path = Path(args.target).expanduser().resolve()
        return path, str(path)


def cmd_install(args):
    """Handle 'trl skills install' command."""
    # Check skill argument
    if not args.skill and not args.all:
        print("Error: Either provide a skill name or use --all to install all skills")
        print("Usage: trl skills install <skill> --target <target>")
        print("   or: trl skills install --all --target <target>")
        return 1

    if args.skill and args.all:
        print("Error: Cannot specify both a skill name and --all")
        return 1

    # Resolve target
    try:
        target_dir, display_name = resolve_target(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1

    # Get skills to install
    if args.all:
        skills_to_install = list_skills()
        if not skills_to_install:
            print("No skills available to install")
            return 1
        print(f"Installing {len(skills_to_install)} skills to {display_name}")
    else:
        skills_to_install = [args.skill]

    # Warn about symlinks
    if args.method == "symlink":
        print("\n⚠️  Warning: Symlinked skills will break if TRL is uninstalled or moved.")
        print("   Use --method copy if you want skills to survive TRL updates/uninstalls.\n")

    # Create installer
    installer = SkillInstaller(get_skills_dir())
    method = InstallMethod.SYMLINK if args.method == "symlink" else InstallMethod.COPY

    # Install each skill
    success_count = 0
    for skill_name in skills_to_install:
        try:
            print(f"Installing '{skill_name}'...", end=" ")
            installer.install_skill(
                skill_name=skill_name,
                target_dir=target_dir,
                method=method,
                force=args.force,
                create_dirs=True,
            )

            if method == InstallMethod.SYMLINK:
                print("✓ (symlink created)")
            else:
                print("✓ (file copied)")

            success_count += 1

        except FileNotFoundError as e:
            print("✗")
            print(f"  Error: {e}")
        except FileExistsError as e:
            print("✗")
            print(f"  Error: {e}")
            if not args.force:
                print("  Use --force to overwrite")
        except Exception as e:
            print("✗")
            print(f"  Error: {e}")

    # Summary
    print(f"\n{success_count}/{len(skills_to_install)} skills installed successfully")

    if success_count > 0:
        print(f"\nSkills are now available at: {target_dir}")
        print("You may need to restart your AI agent to use the new skills.")

    return 0 if success_count == len(skills_to_install) else 1


def cmd_uninstall(args):
    """Handle 'trl skills uninstall' command."""
    # Resolve target
    try:
        target_dir, display_name = resolve_target(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1

    if not target_dir.exists():
        print(f"Target directory doesn't exist: {target_dir}")
        return 1

    # Create installer
    installer = SkillInstaller(get_skills_dir())

    # Uninstall
    try:
        print(f"Uninstalling '{args.skill}' from {display_name}...", end=" ")
        installer.uninstall_skill(args.skill, target_dir)
        print("✓")
        print(f"\nSkill '{args.skill}' has been removed")
        return 0

    except FileNotFoundError as e:
        print("✗")
        print(f"Error: {e}")
        return 1
    except PermissionError as e:
        print("✗")
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print("✗")
        print(f"Error: {e}")
        return 1


def cmd_list_installed(args):
    """Handle 'trl skills list-installed' command."""
    # Resolve target
    try:
        target_dir, display_name = resolve_target(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1

    if not target_dir.exists():
        print(f"Target directory doesn't exist: {target_dir}")
        print("No skills installed")
        return 0

    # Create installer
    installer = SkillInstaller(get_skills_dir())

    # List installed
    installed = installer.list_installed_skills(target_dir)

    if not installed:
        print(f"No skills installed in {display_name}")
        return 0

    print(f"\nInstalled skills in {display_name}:\n")

    for skill in installed:
        method_str = skill["method"]
        if skill["method"] == "symlink":
            method_str = f"symlink → {skill['source']}"

        print(f"  {skill['name']}")
        print(f"    Method: {method_str}")
        print(f"    Path:   {skill['path']}")
        print()

    print(f"Total: {len(installed)} skill(s)")
    return 0


__all__ = [
    "add_skills_subcommands",
]
