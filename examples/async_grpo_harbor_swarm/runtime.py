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

"""Concurrent native Harbor/Pi trials with a private msgboard and exact OpenEnv captures."""

import asyncio
import json
import os
import shlex
import time
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from board import TeamBoard, tool_bundle
from openenv.core.harness.capture import CaptureServer
from openenv.core.harness.capture.sessions import Upstream
from openenv.harbor import seams
from openenv.harbor.contract import to_training_trace
from openenv.harbor.install_fixes import InterceptPi
from openenv.harbor.rollout import run_rollout


HERE = Path(__file__).parent
STOP_FILE = "/tmp/trl-team-stop"
STOP_EXTENSION = """import { existsSync } from "node:fs";
export default function (pi) {
  let timer;
  pi.on("agent_start", (_, ctx) => {
    timer = setInterval(() => { if (existsSync("/tmp/trl-team-stop")) ctx.abort(); }, 250);
  });
  pi.on("agent_end", () => clearInterval(timer));
  pi.on("session_shutdown", () => clearInterval(timer));
}
"""


@dataclass
class Settings:
    vllm_url: str
    capture_url: str
    output_dir: str
    model: str = "Qwen/Qwen3.5-4B"
    backend: str = "podman"
    agents: int = 2
    capture_port: int = 8300
    max_turns: int = 16
    agent_timeout: float = 300
    setup_timeout: float = 900
    max_output_tokens: int = 1024
    max_context: int = 16384
    temperature: float = 1.0


class SwarmPi(InterceptPi):
    """Pi owns the loop; a native extension aborts it when the team finishes."""

    def __init__(self, *args, **kwargs):
        self.board_env = {}
        super().__init__(*args, **kwargs)
        self.ready = None
        self.team_name = ""
        self.bundle = None
        self.max_context = 16384
        self.max_output_tokens = 1024

    @property
    def extra_env(self):
        return {**super().extra_env, **self.board_env}

    async def setup(self, environment):
        await super().setup(environment)
        await self.ensure_system_dependencies(environment, ("python3",))
        await environment.upload_file(self.bundle, "/tmp/swarm-tools.tgz")
        result = await self.exec_as_root(
            environment,
            "mkdir -p /opt/swarm-tools && python3 -m tarfile -e /tmp/swarm-tools.tgz /opt/swarm-tools && "
            "chmod -R a+rX /opt/swarm-tools && chmod a+x /opt/swarm-tools/board_tool.py && "
            "ln -sf /opt/swarm-tools/board_tool.py /usr/local/bin/flock-tool",
        )
        if result.return_code:
            raise RuntimeError("Installing the board tools failed")
        command = (
            "mkdir -p ~/.pi/agent/extensions; "
            f"printf %s {shlex.quote(STOP_EXTENSION)} > ~/.pi/agent/extensions/team-stop.ts; "
            "printf '%s' '{\"compaction\":{\"enabled\":false}}' > ~/.pi/agent/settings.json; "
            "flock-tool message_board.whoami"
        )
        result = await self.exec_as_agent(environment, command, env=self.extra_env)
        if result.return_code:
            raise RuntimeError("The agent cannot access its msgboard")
        update = (
            "import json,pathlib; p=pathlib.Path.home()/'.pi/agent/models.json'; "
            "d=json.loads(p.read_text()); "
            f"d['providers']['intercept']['models'][0].update(contextWindow={self.max_context}, "
            f"maxTokens={self.max_output_tokens}); p.write_text(json.dumps(d))"
        )
        result = await self.exec_as_agent(environment, "python3 -c " + shlex.quote(update))
        if result.return_code:
            raise RuntimeError("Cannot set Pi's context/output limits")

    async def run(self, instruction, environment, context):
        await self.ready()
        prompt = (HERE / "team.md").read_text()
        await super().run(f"You are {self.team_name}.\n{prompt}\n\n{instruction}", environment, context)


def pi_kwargs(base_url, session, model):
    return {
        "intercept_config": {"base_url": base_url, "api_key": session, "model": model},
        "version": "1.4.0",
        "reasoning_effort": "off",
    }


def validate_team(results, max_context, winner=None):
    """Native time-budget exhaustion is scorable; transport/setup/capture errors are not."""
    for result in results:
        if not result.ok and result.exception_type != "AgentTimeoutError":
            raise ValueError("A team member failed outside its native execution budget")
        if result.reward not in (0, 1):
            raise ValueError("A team member has no native binary grade")
    traces = [to_training_trace(result) for result in results]
    if any(not trace.turns for trace in traces):
        raise ValueError("Every agent needs a nonempty validated training trace")
    if any(
        len(turn.prompt_token_ids) + len(turn.completion_token_ids) > max_context
        for trace in traces
        for turn in trace.turns
    ):
        raise ValueError("Captured context exceeds the configured model limit")
    if winner is not None and results[winner].reward != 1:
        raise ValueError("Native verifier did not confirm the trusted success signal")
    return float(any(result.reward == 1 for result in results)), traces


class TeamRunner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = Path(settings.output_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.bundle = tool_bundle(self.root / "board-tools.tar.gz")
        self.capture = CaptureServer(
            llm_url=settings.vllm_url,
            model=settings.model,
            port=settings.capture_port,
            max_output_tokens=settings.max_output_tokens,
            max_model_calls=settings.max_turns,
            api_key=os.environ.get("VLLM_API_KEY"),
        )
        seams.SEAMS["trl-swarm-pi"] = seams.Seam(
            name="trl-swarm-pi",
            status="untested",
            dialect="openai_chat",
            import_path="runtime:SwarmPi",
            model_fmt="intercept/{model}",
            kwargs=pi_kwargs,
            step_limit=lambda n: {"max_turns": n},
        )

    async def start(self):
        await asyncio.to_thread(self.capture.start)

    async def close(self):
        await asyncio.to_thread(self.capture.stop)

    async def run(self, task_dir: Path, model: str, version: int = 0):
        cfg = self.settings
        team_id = uuid.uuid4().hex
        directory = self.root / "teams" / team_id
        directory.mkdir(parents=True, mode=0o700)
        names = [f"agent{i:03}" for i in range(cfg.agents)]
        board = TeamBoard(directory / "board", names)
        trials, ready = {}, set()
        barrier = asyncio.Event()
        active = asyncio.Event()
        scenario = tomllib.loads((task_dir / "task.toml").read_text())["metadata"]["scenario_sha256"]
        state = {
            "team_id": team_id,
            "task": task_dir.name,
            "model": model,
            "policy_version": version,
            "scenario_sha256": scenario,
            "reward": None,
            "stop_reason": "finished",
            "winner": None,
        }
        jobs = []
        monitor = None
        started = time.monotonic()
        try:
            await board.start()
            upstream = Upstream(llm_url=cfg.vllm_url, model=model, api_key=os.environ.get("VLLM_API_KEY"))
            client, level = await self.capture.app.state.upstreams.resolve(upstream)
            if level != "tokens":
                raise ValueError("The inference endpoint must supply exact tokens and sampling logprobs")

            async def ready_agent(name):
                ready.add(name)
                if len(ready) == cfg.agents:
                    state["execution_started"] = time.time()
                    barrier.set()
                    active.set()
                await asyncio.wait_for(barrier.wait(), cfg.setup_timeout)

            def created(name, trial):
                trials[name] = trial
                board.bind(trial, name)
                trial.config.agent.extra_allowed_hosts.append(urlsplit(cfg.capture_url).hostname)
                # OpenEnv's callback runs after Trial.create, before any environment starts.
                # Harbor 0.25 constructs the environment during create: rebuild it with the board mount.
                trial._init_agent_environment()
                trial.agent.team_name = name
                trial.agent.bundle = self.bundle
                trial.agent.max_context = cfg.max_context
                trial.agent.max_output_tokens = cfg.max_output_tokens
                trial.agent.ready = lambda: ready_agent(name)
                # Podman down --rmi local otherwise removes images shared by concurrent trials.
                if cfg.backend == "podman":
                    trial.config.environment.delete = False

            async def watch():
                await active.wait()
                command = "python -c " + shlex.quote(
                    "import json,urllib.request; "
                    "s=json.load(urllib.request.urlopen('http://127.0.0.1:8000/state',timeout=3)); "
                    "print('TRL_STATE='+json.dumps(s))"
                )
                while True:
                    for name, trial in list(trials.items()):
                        if jobs[names.index(name)].done():
                            continue
                        try:
                            result = await trial.agent_environment.service_exec(
                                command, service="puzzle", timeout_sec=5
                            )
                        except Exception:
                            if jobs[names.index(name)].done():
                                continue  # A completed trial can remove its room during the check.
                            raise
                        if result.return_code:
                            if jobs[names.index(name)].done():
                                continue
                            raise RuntimeError("Trusted puzzle state check failed")
                        payload = next(
                            line.removeprefix("TRL_STATE=")
                            for line in result.stdout.splitlines()
                            if line.startswith("TRL_STATE=")
                        )
                        evidence = json.loads(payload)
                        if evidence.get("won") is True and evidence.get("scenario_sha256") == scenario:
                            state.update(winner=name, stop_reason="first_success", success_at=time.time())
                            (directory / "success.json").write_text(json.dumps(evidence))

                            # Pi's extension calls ctx.abort(), allowing its native run to finish and grade.
                            async def stop_agent(n, t):
                                if jobs[names.index(n)].done():
                                    return
                                try:
                                    stopped = await t.agent.exec_as_root(
                                        t.agent_environment, f"touch {STOP_FILE}", timeout_sec=5
                                    )
                                    if stopped.return_code and not jobs[names.index(n)].done():
                                        raise RuntimeError("Cannot signal Pi to stop")
                                except Exception:
                                    if not jobs[names.index(n)].done():
                                        raise

                            await asyncio.gather(*(stop_agent(n, t) for n, t in trials.items()))
                            return
                    await asyncio.sleep(1)

            sampling = {
                "temperature": cfg.temperature,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": cfg.max_output_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            for name in names:
                jobs.append(
                    asyncio.create_task(
                        run_rollout(
                            task_dir=task_dir,
                            harness="trl-swarm-pi",
                            sandbox=cfg.backend,
                            registry=self.capture.registry,
                            intercept_url=cfg.capture_url,
                            model=model,
                            trials_dir=directory / name,
                            capture_level=level,
                            upstream=upstream,
                            inference=client,
                            purpose="train",
                            sampling=sampling,
                            agent_timeout_sec=cfg.agent_timeout,
                            agent_step_limit=cfg.max_turns,
                            on_trial_created=lambda trial, name=name: created(name, trial),
                        )
                    )
                )
            monitor = asyncio.create_task(watch())
            # Monitor failures and pre-barrier trial failures must not leave peers waiting indefinitely.
            pending = set(jobs)
            async with asyncio.timeout(cfg.setup_timeout + cfg.agent_timeout + 180):
                while pending:
                    if monitor.done() and not monitor.cancelled() and monitor.exception():
                        raise monitor.exception()
                    watched = pending | ({monitor} if not monitor.done() else set())
                    done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
                    if monitor in done and monitor.exception():
                        raise monitor.exception()
                    for job in done & pending:
                        if not barrier.is_set():
                            raise RuntimeError("A Harbor trial ended before all agents were ready")
                        pending.remove(job)
            results = [job.result() for job in jobs]
            for name, result in zip(names, results, strict=True):
                (directory / f"{name}.json").write_text(result.model_dump_json(indent=2))
            winner = names.index(state["winner"]) if state["winner"] else None
            state["reward"], traces = validate_team(results, cfg.max_context, winner)
            state["agent_rewards"] = [result.reward for result in results]
            state["capture_roots"] = [result.n_roots for result in results]
            state["generated_tokens"] = sum(len(turn.completion_token_ids) for trace in traces for turn in trace.turns)
            state["trained_tokens"] = sum(sum(turn.loss_mask) for trace in traces for turn in trace.turns)
            if state["winner"]:
                state["time_to_success"] = state["success_at"] - state["execution_started"]
            elif any(result.exception_type == "AgentTimeoutError" for result in results):
                state["stop_reason"] = "time_budget"
            elif any(result.budget_stop_count for result in results):
                state["stop_reason"] = "model_call_budget"
            return state, traces
        except Exception as exc:
            state.update(reward=None, stop_reason="unscorable", error=f"{type(exc).__name__}: {exc}")
            return state, []
        finally:
            if monitor:
                monitor.cancel()
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs, *([monitor] if monitor else []), return_exceptions=True)
            try:
                await board.close()
                messages = directory / "board" / "messages.jsonl"
                state["messages"] = len(messages.read_text().splitlines()) if messages.exists() else 0
            except Exception as exc:
                state.update(
                    reward=None, stop_reason="unscorable", error=f"Board cleanup failed: {type(exc).__name__}"
                )
            finally:
                state["wall_seconds"] = time.monotonic() - started
                (directory / "team.json").write_text(json.dumps(state, indent=2))
                with (self.root / "teams.jsonl").open("a") as output:
                    output.write(json.dumps(state) + "\n")
