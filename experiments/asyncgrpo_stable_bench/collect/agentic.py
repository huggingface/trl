"""W2 (agentic) helpers shared by every library's collector: episode records -> results/raw/<run_id>/episodes.jsonl in
the schema of collect/SCHEMA.md (W2), and per-step agentic fields for steps.jsonl.

Two sources of episode records:
  - `workloads/agentic/termgrade_env.py` writes `episodes.jsonl` in the run dir (TRL through OpenEnv; the probe).
  - any library that runs Harbor trials can instead point `episodes_from_trials` at its trials directory (or the
    `trials.tgz` archive): every Harbor trial writes `result.json` (phase times, reward, exception) and terminus-2
    writes `agent/trajectory.json` (one step per model call).
Standard library only.
"""

import io
import json
import os
import tarfile
from datetime import datetime


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else None


def model_seconds(steps):
    """Time inside model calls, estimated from terminus-2's trajectory. An agent step is stamped when its reply arrives;
    the interval since the previous step is the previous step's command execution plus this call. terminus-2 waits the
    `duration` it asked for after each command, so that wait (`command_wait_s`) is subtracted; the screen read and
    keystroke round trips are not, so this slightly overstates model time. None when the steps carry no waits."""
    if not steps or any("command_wait_s" not in s for s in steps):
        return None
    total, prev, prev_wait = 0.0, None, 0.0
    for step in steps:
        t = _ts(step.get("timestamp"))
        if t is None:
            continue
        if step.get("source") == "agent" and prev is not None:
            total += max(0.0, t - prev - prev_wait)
        prev, prev_wait = t, step.get("command_wait_s") or 0.0
    return total


def normalize(rec):
    """One recorded episode -> the SCHEMA.md W2 episode row."""
    phases = rec.get("phases") or {}
    wall = rec["end"] - rec["start"]
    model_s = model_seconds(rec.get("steps") or [])
    reward = rec.get("reward")
    budget = bool(rec.get("budget_stops")) or "Timeout" in str(rec.get("exception_type") or "")
    status = "failed" if reward is None else ("budget_scored" if budget else "scored")
    return {
        "start": rec["start"],
        "end": rec["end"],
        "wall_s": wall,
        "task": (rec.get("task") or "").split("/")[-1] or None,
        "reward": reward,
        "status": status,
        "error": rec.get("error"),
        "turns": rec.get("n_turns"),
        "completion_tokens": rec.get("completion_tokens"),
        "seq_tokens": rec.get("seq_tokens"),
        "prompt_tokens_sum": rec.get("prompt_tokens_sum"),
        "phases": phases,
        "model_s": model_s,
        "sandbox_s": (wall - model_s) if model_s is not None else None,
        "trial": rec.get("trial"),
    }


def read_episode_records(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def episodes_from_trials(path):
    """Episode records from a Harbor trials dir or trials.tgz (for libraries that run Harbor in-process)."""
    files = {}
    if path.endswith(".tgz"):
        with tarfile.open(path) as tar:
            for m in tar.getmembers():
                if m.isfile() and (m.name.endswith("/result.json") or m.name.endswith("/agent/trajectory.json")):
                    files[m.name.lstrip("./")] = tar.extractfile(m).read()
    else:
        for root, _, names in os.walk(path):
            for n in names:
                full = os.path.join(root, n)
                rel = os.path.relpath(full, path)
                if rel.endswith("/result.json") and rel.count("/") == 1 or rel.endswith("/agent/trajectory.json"):
                    files[rel] = open(full, "rb").read()
    recs = []
    for name, data in files.items():
        if not name.endswith("/result.json") or name.count("/") != 1:
            continue
        trial = name.split("/")[0]
        r = json.load(io.BytesIO(data))
        phases, start, end = {}, None, None
        for key, value in r.items():
            if isinstance(value, dict) and value.get("started_at") and value.get("finished_at"):
                phases[key] = _ts(value["finished_at"]) - _ts(value["started_at"])
        start, end = _ts(r.get("started_at")), _ts(r.get("finished_at"))
        steps = []
        traj = files.get(f"{trial}/agent/trajectory.json")
        try:
            traj and json.loads(traj)
        except json.JSONDecodeError:  # cut off by a killed job's archive: keep the episode, without its turns
            print(f"agentic: unreadable {trial}/agent/trajectory.json, turns left out")
            traj = None
        if traj:
            steps = [
                {
                    "source": s.get("source"),
                    "timestamp": s.get("timestamp"),
                    "command_wait_s": sum(
                        float((c.get("arguments") or {}).get("duration") or 0) for c in s.get("tool_calls") or []
                    ),
                }
                for s in json.load(io.BytesIO(traj)).get("steps", [])
            ]
        vr = r.get("verifier_result") or {}
        rewards = vr.get("rewards") or {}
        exc = r.get("exception_info") or {}
        recs.append(
            {
                "start": start,
                "end": end,
                "task": r.get("task_name"),
                "trial": trial,
                "reward": rewards.get("reward") if rewards else None,
                "error": exc.get("exception_message"),
                "exception_type": exc.get("exception_type"),
                "n_turns": sum(1 for s in steps if s["source"] == "agent"),
                "phases": phases,
                "steps": steps,
            }
        )
    return [r for r in recs if r["start"] and r["end"]]


def step_fields(episodes, t0, t1):
    """Agentic fields for the step window (t0, t1]: the episodes that finished in it."""
    window = [e for e in episodes if t0 < e["end"] <= t1]
    n = len(window)
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {
        "episodes_finished": n,
        "episode_wall_mean": mean([e["wall_s"] for e in window]),
        "turns_mean": mean([e["turns"] for e in window if e["turns"] is not None]),
        "tokens_per_episode_mean": mean([e["seq_tokens"] for e in window if e["seq_tokens"]]),
        "failed_episodes": sum(e["status"] == "failed" for e in window),
        "budget_episodes": sum(e["status"] == "budget_scored" for e in window),
        "episode_model_s_mean": mean([e["model_s"] for e in window if e["model_s"] is not None]),
        "episode_sandbox_s_mean": mean([e["sandbox_s"] for e in window if e["sandbox_s"] is not None]),
    }
