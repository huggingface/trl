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

from .skills import install_skill, list_agent_names, list_skills, resolve_target_path, uninstall_skill


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
        choices=["project", "global"],
        default="project",
        help="Scope for predefined agents: project (./agent/skills/) or global (user-level like ~/.agent/skills/)",
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


def cmd_install(args):
    """Handle 'trl skills install' command."""
    # Validate arguments
    if not args.skill and not args.all:
        print("Error: Either provide a skill name or use --all to install all skills")
        print("Usage: trl skills install <skill> --target <target>")
        print("   or: trl skills install --all --target <target>")
        return 1

    if args.skill and args.all:
        print("Error: Cannot specify both a skill name and --all")
        return 1

    # Determine skills to install
    if args.all:
        skills_to_install = list_skills()
        if not skills_to_install:
            print("No skills available to install")
            return 1
        print(f"Installing {len(skills_to_install)} skills to {args.target}")
    else:
        skills_to_install = [args.skill]

    # Install each skill
    success_count = 0
    for skill_name in skills_to_install:
        try:
            print(f"Installing '{skill_name}'...", end=" ")
            install_skill(
                skill_name=skill_name,
                target=args.target,
                scope=args.scope,
                force=args.force,
            )
            print("✓")
            success_count += 1

        except FileNotFoundError as e:
            print("✗")
            print(f"  Error: {e}")
        except FileExistsError as e:
            print("✗")
            print(f"  Error: {e}")
            if not args.force:
                print("  Use --force to overwrite")
        except ValueError as e:
            print("✗")
            print(f"  Error: {e}")
        except Exception as e:
            print("✗")
            print(f"  Error: {e}")

    # Summary
    print(f"\n{success_count}/{len(skills_to_install)} skills installed successfully")

    if success_count > 0:
        target_path = resolve_target_path(args.target, args.scope)
        print(f"\nSkills are now available at: {target_path}")
        print("You may need to restart your AI agent to use the new skills.")

    return 0 if success_count == len(skills_to_install) else 1


def cmd_uninstall(args):
    """Handle 'trl skills uninstall' command."""
    try:
        print(f"Uninstalling '{args.skill}' from {args.target}...", end=" ")
        uninstall_skill(args.skill, target=args.target, scope=args.scope)
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
    except ValueError as e:
        print("✗")
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print("✗")
        print(f"Error: {e}")
        return 1


def cmd_list_installed(args):
    """Handle 'trl skills list-installed' command."""
    try:
        # List installed skills
        installed = list_skills(target=args.target, scope=args.scope)

        if not installed:
            print(f"No skills installed in {args.target}")
            return 0

        print(f"\nInstalled skills in {args.target}:\n")

        for skill in installed:
            print(f"  {skill}")

        print(f"\nTotal: {len(installed)} skill(s)")
        return 0

    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


__all__ = [
    "add_skills_subcommands",
]
