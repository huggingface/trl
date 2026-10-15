"""Build each task's E2B template once, before training starts.

Harbor's E2B backend checks `alias_exists` and then builds, with no lock. The 8 rollouts of a GRPO
group start together, so on a task's first group they all build the same template at once and
cancel each other ("build is not in waiting state", "build was cancelled"). Templates persist on
E2B, so this costs one build per task, ever; existing ones are skipped.

    python prebuild_e2b_templates.py <tasks-dir> [--concurrency 8] [--dry-run]
"""

import argparse
import asyncio
import logging
import sys
import tempfile
from pathlib import Path

from harbor.environments.factory import EnvironmentFactory
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tasks_dir", type=Path, help="a folder of Harbor task dirs, or one containing `tasks/`")
    p.add_argument("--concurrency", type=int, default=8, help="distinct tasks built in parallel")
    p.add_argument("--dry-run", action="store_true", help="print template names without calling E2B")
    # The alias hashes only environment/, so a task.toml change (e.g. memory_mb) keeps the old, smaller build.
    p.add_argument("--force", action="store_true", help="rebuild templates that already exist")
    args = p.parse_args()

    root = args.tasks_dir / "tasks" if (args.tasks_dir / "tasks").is_dir() else args.tasks_dir
    task_dirs = sorted(d for d in root.iterdir() if (d / "task.toml").is_file())
    trial_dir = Path(tempfile.mkdtemp(prefix="prebuild-"))
    gate = asyncio.Semaphore(args.concurrency)
    failed = []

    async def build(task_dir: Path) -> None:
        task = Task(task_dir, disable_verification=True)
        # Same arguments Harbor's Trial uses, so the template alias matches what rollouts look up.
        env = EnvironmentFactory.create_environment(
            type=EnvironmentType.E2B,
            environment_dir=task.paths.environment_dir,
            environment_name=task.short_name,
            session_id="prebuild",
            trial_paths=TrialPaths(trial_dir),
            task_env_config=task.config.environment,
        )
        name = env._template_name
        if args.dry_run:
            print(f"{task_dir.name} -> {name} (cpus={env._effective_cpus}, memory_mb={env._effective_memory_mb})")
            return
        async with gate:
            try:
                if not args.force and await env._does_template_exist():
                    print(f"exists  {name}")
                    return
                print(f"build   {name}", flush=True)
                await env._create_template()
                print(f"built   {name}", flush=True)
            except Exception as exc:  # noqa: BLE001 - report every task, then fail once
                failed.append(name)
                print(f"FAILED  {name}: {exc}", flush=True)

    await asyncio.gather(*(build(d) for d in task_dirs))
    print(f"{len(task_dirs) - len(failed)}/{len(task_dirs)} templates ready")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
