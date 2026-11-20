"""What every W2 client (TRL trainer, base-rate probe, eval) shares: the OpenEnv Harbor session factory with the
transport settings that long agent episodes need, the SHARED_W2 sampling policy, and a per-episode record.

Importable from the TRL rollout process (it is spawned and unpickles the factory), so this directory must be on
PYTHONPATH there too (launch/trl/w2.sbatch does it).
"""

import itertools
import json
import os
import threading
import time
from datetime import datetime

from harbor_env.harness import HarborSession, HarborSessionFactory
from openenv.harbor.client import HarborEnv


HARNESS = "terminus-2"
# Harbor environment type: HF Sandboxes, or `singularity` for the local-container variant (launch/trl/w2.sbatch SANDBOX=local).
SANDBOX = os.environ.get("W2_SANDBOX", "hf-sandbox")
# Full-vocabulary sampling at temperature 1, as in W1; repetition_penalty explicit so vLLM never fills it from the
# model's generation config. TRL builds the same dict from AsyncGRPOConfig (openenv_harness.py).
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}

_lock = threading.Lock()
_round_robin = itertools.count()


def trial_phases(trial_name):
    """Harbor's phase wall times (s) and terminus-2's per-step timestamps for one trial, read from the node-local trials
    dir (`OPENENV_HARBOR_TRIALS_DIR`). Empty when the trial dir isn't there."""
    root = os.environ.get("OPENENV_HARBOR_TRIALS_DIR", "")
    path = os.path.join(root, trial_name or "", "result.json")
    if not root or not trial_name or not os.path.exists(path):
        return {}, []
    result = json.load(open(path))
    phases = {}
    for key, value in result.items():
        if isinstance(value, dict) and value.get("started_at") and value.get("finished_at"):
            t0 = datetime.fromisoformat(value["started_at"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(value["finished_at"].replace("Z", "+00:00"))
            phases[key] = (t1 - t0).total_seconds()
    steps = []
    traj = os.path.join(root, trial_name, "agent", "trajectory.json")
    if os.path.exists(traj):
        for step in json.load(open(traj)).get("steps", []):
            metrics = step.get("metrics") or {}
            # terminus-2 stamps an agent step when the reply arrives, then runs its keystrokes and waits `duration`
            # seconds per command before the next call: that wait is sandbox time, not model time.
            wait = sum(float((c.get("arguments") or {}).get("duration") or 0) for c in step.get("tool_calls") or [])
            steps.append(
                {
                    "source": step.get("source"),
                    "timestamp": step.get("timestamp"),
                    "prompt_tokens": metrics.get("prompt_tokens"),
                    "completion_tokens": metrics.get("completion_tokens"),
                    "command_wait_s": wait,
                }
            )
    return phases, steps


class RecordedHarborSession(HarborSession):
    """`HarborSession` that appends one JSON line per finished episode to `$W2_EPISODES_FILE`: wall time, reward,
    error, turns, token counts, and Harbor's phase times (sandbox start, setup, agent, verifier)."""

    def wait_for_completion(self, timeout_s=None):
        start = time.time()
        code = super().wait_for_completion(timeout_s)
        path = os.environ.get("W2_EPISODES_FILE")
        if path:
            r = self.result
            agent = [t for t in r.turns if t.role == "agent"] if r else []
            phases, steps = trial_phases(r.trial_name if r else None)
            rec = {
                "task_index": self._task_index,
                "start": start,
                "end": time.time(),
                "exit_code": code,
                "ok": bool(r and r.ok),
                "reward": r.reward if r else None,
                "error": (r.error if r else "rollout call failed"),
                "exception_type": r.exception_type if r else None,
                "trial": r.trial_name if r else None,
                "task": r.task_name if r else None,
                "n_turns": r.n_turns if r else None,
                "n_roots": r.n_roots if r else None,
                "budget_stops": r.budget_stop_count if r else None,
                "server_wall_s": r.wall_s if r else None,
                "trainable_tokens": r.n_trainable_tokens if r else None,
                "completion_tokens": sum(len(t.completion_token_ids) for t in agent),
                "prompt_tokens_sum": sum(len(t.prompt_token_ids) for t in agent),
                "seq_tokens": len(agent[-1].prompt_token_ids) + len(agent[-1].completion_token_ids) if agent else 0,
                "finish_reasons": [t.finish_reason for t in agent],
                "findings": r.findings[:5] if r else None,
                "phases": phases,
                "steps": steps,
            }
            with _lock, open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return code


class TermGradeSessionFactory(HarborSessionFactory):
    """`HarborSessionFactory` whose per-rollout clients survive long, quiet episodes, and whose sessions record.

    The OpenEnv client pings its websocket every 20 s and drops it after 20 s without a pong. While the server boots
    dozens of sandboxes at once it can miss that window, and every in-flight rollout is lost
    (`keepalive ping timeout`). One rollout is one blocking `run_rollout` message, so the message timeout must also
    cover sandbox start + setup + agent + verifier.
    """

    def new_client(self) -> HarborEnv:
        # HARBOR_URLS (comma-separated): several Harbor servers on the same task set, one rollout per server in turn,
        # so no single server's event loop runs every agent (envs/trl/W2_STEP_TIME.md). Unset: the factory's server.
        urls = [u for u in os.environ.get("HARBOR_URLS", "").split(",") if u] or [self.server_url]
        return HarborEnv(
            base_url=urls[next(_round_robin) % len(urls)],
            max_message_size_mb=self._max_message_size_mb,
            message_timeout_s=3600.0,
            websocket_ping_interval_s=60.0,
            websocket_ping_timeout_s=600.0,
        )

    def create(self, task, seed=None, episode_id=None):
        session = super().create(task, seed=seed, episode_id=episode_id)
        session.__class__ = RecordedHarborSession
        return session
