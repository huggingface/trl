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
Agent registry for TRL skills installation.

This module defines where different AI agents and coding tools look for skills, enabling easy installation of TRL
skills to the appropriate directories.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentTarget:
    """
    Definition of where an AI agent looks for skills.

    Attributes:
        name: Internal identifier (e.g., "claude").
        display_name: Human-readable name (e.g., "Claude Code").
        global_path: User-level skills directory path template.
        project_path: Project-level skills directory path template.
    """

    name: str
    display_name: str
    global_path: Path
    project_path: Path

    def get_path(self, scope: str = "project") -> Path:
        """
        Get the skills directory path for the specified scope.

        Args:
            scope: Either "project" (current directory) or "global" (user-level).

        Returns:
            Path to the skills directory.
        """
        path = self.project_path if scope == "project" else self.global_path
        return path.expanduser().resolve()


# Registry of known AI agents and coding tools
AGENT_REGISTRY = {
    "claude": AgentTarget(
        name="claude",
        display_name="Claude Code",
        global_path=Path("~/.claude/skills"),
        project_path=Path("./.claude/skills"),
    ),
    "codex": AgentTarget(
        name="codex",
        display_name="Codex",
        global_path=Path("~/.codex/skills"),
        project_path=Path("./.codex/skills"),
    ),
    "opencode": AgentTarget(
        name="opencode",
        display_name="OpenCode",
        global_path=Path("~/.config/opencode/skills"),
        project_path=Path(".opencode/skills"),
    ),
}


def get_agent_target(name: str) -> AgentTarget:
    """
    Get agent target definition by name.

    Args:
        name: Agent identifier (e.g., "claude")

    Returns:
        AgentTarget for the specified agent

    Raises:
        ValueError: If agent name is not recognized
    """
    if name not in AGENT_REGISTRY:
        available = ", ".join(sorted(AGENT_REGISTRY.keys()))
        raise ValueError(f"Unknown agent '{name}'. Available agents: {available}")
    return AGENT_REGISTRY[name]


def list_agent_names() -> list[str]:
    """
    List all registered agent names.

    Returns:
        Sorted list of agent names
    """
    return sorted(AGENT_REGISTRY.keys())


__all__ = [
    "AgentTarget",
    "AGENT_REGISTRY",
    "get_agent_target",
    "list_agent_names",
]
