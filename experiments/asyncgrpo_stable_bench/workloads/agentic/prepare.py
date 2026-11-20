"""TermGrade tasks -> Harbor task directories that run on HF Sandboxes.

    python prepare.py --tasks probe_tasks.txt --out /fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor/probe

Each TermGrade task is `FROM dsbase-venv312:local` + `RUN bash setup.sh || true` + tmux/asciinema + removal of the tests,
built locally and never pushed. HF Sandboxes only pull prebuilt images, so the shared part (the base image with tmux and
asciinema) is built once as a Docker Space (`base_image/`), and the per-task part runs at sandbox start: the task's
environment/ (setup.sh + prepare.sh) is uploaded to the workdir, and the task's healthcheck runs prepare.sh once, before
the agent starts. prepare.sh mirrors the rest of the task Dockerfile: `setup.sh` with errors ignored, then the tests and
the scripts are deleted, so the agent sees what it would see in the built image.

Changes from TermGrade, all forced by the sandbox backend:
  - network_mode no-network -> public (HF Sandboxes cannot disable internet, and setup.sh installs packages at start)
  - the per-task build step runs at sandbox start (it costs sandbox time on every episode)
Output: <out>/tasks/<task>/{task.toml, instruction.md, environment/{setup.sh,prepare.sh}, tests/, solution/}, and
<out>/tasks.jsonl (one row per task, with TermGrade's labels).
"""

import argparse
import csv
import json
import os
import shutil


PREPARE_SH = """#!/bin/bash
# Run once by the task healthcheck after the sandbox starts: the per-task half of TermGrade's Dockerfile
#   COPY setup.sh /tmp/ds_setup.sh; RUN bash /tmp/ds_setup.sh 2>/dev/null || true
#   RUN rm -f /tests/*.py /solution/*; rm -rf /tmp/ds_setup.sh /tmp/setup.sh /setup.sh; mkdir -p /solution
[ -f /var/tmp/.termgrade_ready ] && exit 0
cd /app
mv /app/setup.sh /tmp/ds_setup.sh
rm -f /app/prepare.sh
bash /tmp/ds_setup.sh > /dev/null 2>&1 || true
cd /app
rm -f /tests/*.py /solution/* 2>/dev/null; rm -rf /tmp/ds_setup.sh /tmp/setup.sh /setup.sh 2>/dev/null; mkdir -p /solution
touch /var/tmp/.termgrade_ready
exit 0
"""

TASK_TOML = """[task]
name = "termgrade/{task}"
description = "TermGrade {task} (ai-and/termgrade-environments)"

[metadata]
source = "ai-and/termgrade-environments"
category = "terminalops"
termgrade_split = "{split}"

[environment]
docker_image = "{image}"
workdir = "/app"

[environment.healthcheck]
command = "bash /app/prepare.sh"
timeout_sec = {setup_timeout}
retries = 1

[agent]
timeout_sec = {agent_timeout}

[verifier]
timeout_sec = 300.0
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/fsx/amine_dirhoussi/stable-bench-src/termgrade")
    p.add_argument("--tasks", required=True, help="file with one TermGrade task name per line")
    p.add_argument("--out", required=True)
    p.add_argument("--image", default="hf.co/spaces/aminediroHF/termgrade-base")
    p.add_argument("--setup-timeout", type=float, default=600.0)
    p.add_argument("--agent-timeout", type=float, default=600.0)
    args = p.parse_args()

    names = [line.strip() for line in open(args.tasks) if line.strip()]
    summary = {row["task"]: row for row in csv.DictReader(open(os.path.join(args.src, "task_summary.csv")))}
    out_tasks = os.path.join(args.out, "tasks")
    if os.path.isdir(out_tasks):
        shutil.rmtree(out_tasks)
    rows = []
    for name in names:
        src = os.path.join(args.src, "environments", "environments", name)
        dst = os.path.join(out_tasks, name)
        os.makedirs(os.path.join(dst, "environment"))
        shutil.copy(os.path.join(src, "instruction.md"), dst)
        shutil.copy(os.path.join(src, "environment", "setup.sh"), os.path.join(dst, "environment", "setup.sh"))
        with open(os.path.join(dst, "environment", "prepare.sh"), "w") as f:
            f.write(PREPARE_SH)
        shutil.copytree(os.path.join(src, "tests"), os.path.join(dst, "tests"))
        shutil.copytree(os.path.join(src, "solution"), os.path.join(dst, "solution"))
        meta = summary[name]
        with open(os.path.join(dst, "task.toml"), "w") as f:
            f.write(
                TASK_TOML.format(
                    task=name,
                    split=meta["split"],
                    image=args.image,
                    setup_timeout=args.setup_timeout,
                    agent_timeout=args.agent_timeout,
                )
            )
        rows.append(
            {
                k: meta[k]
                for k in ("task", "split", "checksum", "deepseek", "gemma", "gemma_think", "glm", "kimi", "qwen")
            }
        )
    with open(os.path.join(args.out, "tasks.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{len(rows)} tasks -> {out_tasks}")


if __name__ == "__main__":
    main()
