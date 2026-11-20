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

"""Monitor a queued/running full-model Slurm experiment and retain private Trackio metrics."""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--space-id", default="edbeeching/trl-harbor-swarm-trackio")
    args = parser.parse_args()
    root = args.run_dir
    os.environ["TRACKIO_DIR"] = tempfile.mkdtemp(prefix="trl-full-trackio-")
    import trackio

    trackio.init(
        project="trl-harbor-swarm",
        name=f"full-16agents-monitor-{args.job}",
        space_id=args.space_id,
        private=True,
        auto_log_gpu=False,
        auto_log_cpu=False,
    )
    try:
        while True:
            state = subprocess.run(
                ["sacct", "-n", "-X", "-j", args.job, "--format=JobIDRaw,State,Elapsed", "-P"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            train = root / "experiment/train"
            teams = []
            path = train / "teams.jsonl"
            if path.exists():
                for line in path.read_text().splitlines():
                    try:
                        teams.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # A final line may still be being appended.
            scored = [team for team in teams if team["reward"] is not None]
            metrics = {
                "rollout/teams": len(teams),
                "rollout/scored": len(scored),
                "rollout/unscorable": len(teams) - len(scored),
                "rollout/generated_tokens": sum(team.get("generated_tokens", 0) for team in teams),
            }
            if scored:
                metrics["rollout/success_rate"] = sum(team["reward"] for team in scored) / len(scored)
                for difficulty in (10, 15):
                    selected = [team for team in scored if team["task"].startswith(f"agentescape-{difficulty}-")]
                    if selected:
                        metrics[f"rollout/difficulty_{difficulty}_success"] = sum(t["reward"] for t in selected) / len(
                            selected
                        )
            for key, value in read_json(train / "rollout-status.json", {}).items():
                metrics[f"rollout/{key}"] = value
            checkpoints = list((train / "checkpoint").glob("checkpoint-*/trainer_state.json"))
            metrics["train/optimizer_steps"] = max(
                (read_json(p, {}).get("global_step", 0) for p in checkpoints), default=0
            )
            smoke = read_json(root / "smoke/train/reload-check.json", {})
            stage = (
                "training"
                if (train / "manifest.json").exists()
                else "native_gates"
                if smoke
                else "gpu_smoke_or_queued"
            )
            rows = [line.split("|") for line in state.splitlines()]
            job_state = rows[0][1].split()[0] if rows else "UNKNOWN"
            terminal = job_state in {
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "PREEMPTED",
            }
            metrics["job/failed"] = int(terminal and job_state != "COMPLETED")
            metrics["job/running"] = int(job_state == "RUNNING")
            metrics["job/pending"] = int(job_state == "PENDING")
            validation_stage = stage
            if terminal:
                stage = job_state.lower()
            elif job_state == "PENDING":
                stage = "queued"
            record = {
                "timestamp": time.time(),
                "job": state,
                "stage": stage,
                "validation_stage": validation_stage,
                "metrics": metrics,
            }
            if terminal and job_state != "COMPLETED":
                for log in (root / "experiment/train.log", root / "smoke/train.log", root / "server-smoke.log"):
                    if log.exists():
                        errors = [
                            line
                            for line in log.read_text(errors="replace").splitlines()
                            if "[rank0]: RuntimeError:" in line or "[rank0]: Exception:" in line
                        ]
                        if errors:
                            record["failure_summary"] = errors[-1][:2000]
                            record["failure_log"] = str(log)
                            break
            manifest_path = root / "experiment.json"
            manifest = read_json(manifest_path, {})
            manifest.update(status=stage, job_state=job_state, last_monitor_update=record["timestamp"])
            if "failure_summary" in record:
                manifest["failure_summary"] = record["failure_summary"]
            manifest_temp = root / "experiment.tmp"
            manifest_temp.write_text(json.dumps(manifest, indent=2))
            manifest_temp.replace(manifest_path)
            temporary = root / "monitor-status.tmp"
            temporary.write_text(json.dumps(record, indent=2))
            temporary.replace(root / "monitor-status.json")
            with (root / "monitor.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            trackio.log(metrics)
            print(json.dumps(record), flush=True)
            if terminal:
                break
            time.sleep(30)
    finally:
        trackio.finish()


if __name__ == "__main__":
    main()
