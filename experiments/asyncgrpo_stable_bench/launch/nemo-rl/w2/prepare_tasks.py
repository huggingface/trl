"""Harness glue for W2 on NeMo-RL's NeMo Gym `harbor_agent` route: the shared local-box TermGrade tasks, laid out the
way NeMo Gym's SingularityEnvironment reads them, plus the NeMo Gym input jsonl.

The shared task dirs (stable-bench-src/termgrade/harbor-local-sif/<split>/tasks) run their per-task setup through
`[environment.healthcheck] command = "bash /app/prepare.sh"` (Harbor 0.25). NeMo Gym pins Harbor `9dddd797`
(2026-01-26), which has no healthcheck, and its SingularityEnvironment instead runs `environment/files/setup.sh` from
a staging mount before the in-box server starts. So each copy gets `environment/files/` = the task's own setup.sh
and prepare.sh plus a 3-line setup.sh that puts them where prepare.sh expects them and runs it. Same image, same
setup, same tests, same timeouts; only the hook that runs the setup differs.

    python prepare_tasks.py <out_root> <rows per train file>
"""

import json
import os
import re
import shutil
import sys


SRC = "/fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor-local-sif"
DEPS = "/fsx/amine_dirhoussi/stable-bench-src/termgrade/nemogym-srvdeps"  # uvicorn + fastapi for cp312 (uv pip --target)
HOOK = """#!/bin/bash
# NeMo Gym SingularityEnvironment hook (environment/files/setup.sh) = the shared task's healthcheck `bash /app/prepare.sh`
mkdir -p /app
cp "$HARBOR_STAGING/task_setup.sh" /app/setup.sh
cp "$HARBOR_STAGING/prepare.sh" /app/prepare.sh
bash /app/prepare.sh
# NeMo Gym's in-box exec server needs uvicorn/fastapi, which the TermGrade image lacks. Gym's own write_min_setup_sh.py
# pip-installs them into the box's python3; here they stay out of it: a private copy plus a python3 wrapper at one of
# the paths Gym's bootstrap probes (/opt/conda/bin/python3), so the task's python3 and site-packages are unchanged.
mkdir -p /opt/hbdeps /opt/conda/bin
cp -r "$HARBOR_STAGING/srvdeps/." /opt/hbdeps/
printf '#!/bin/sh\nPYTHONPATH=/opt/hbdeps exec /usr/local/bin/python3 "$@"\n' > /opt/conda/bin/python3
chmod +x /opt/conda/bin/python3
"""


def convert(split, out_root):
    names = sorted(os.listdir(f"{SRC}/{split}/tasks"))
    for name in names:
        src, dst = f"{SRC}/{split}/tasks/{name}", f"{out_root}/{split}/tasks/{name}"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        toml = open(f"{dst}/task.toml").read()
        toml = re.sub(r"\n\[environment\.healthcheck\]\n(?:[^\[\n][^\n]*\n)*", "\n", toml)
        assert "healthcheck" not in toml, name
        open(f"{dst}/task.toml", "w").write(toml)
        files = f"{dst}/environment/files"
        os.makedirs(files)
        shutil.move(f"{dst}/environment/setup.sh", f"{files}/task_setup.sh")
        shutil.move(f"{dst}/environment/prepare.sh", f"{files}/prepare.sh")
        open(f"{files}/setup.sh", "w").write(HOOK)
        shutil.copytree(DEPS, f"{files}/srvdeps")
    return names


def rows(alias, names, n):
    # Sampling goes in responses_create_params: NeMo Gym's LLM sends these (not terminus-2's own 0.7 default) to vLLM.
    params = {"input": [], "temperature": 1.0, "top_p": 1.0, "max_output_tokens": 8192}
    return [
        {"instance_id": f"{alias}::{names[i % len(names)]}", "responses_create_params": params,
         "agent_ref": {"type": "responses_api_agents", "name": "harbor_agent"}}
        for i in range(n)
    ]


if __name__ == "__main__":
    out_root, n_train = sys.argv[1], int(sys.argv[2])
    for split in ("train", "eval"):
        names = convert(split, out_root)
        data = rows(split, names, n_train if split == "train" else len(names))
        with open(f"{out_root}/{split}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in data)
        print(split, len(names), "tasks,", len(data), "rows")
