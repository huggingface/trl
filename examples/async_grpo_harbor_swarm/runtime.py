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
import logging
import os
import shlex
import time
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from board import TeamBoard, tool_bundle
from harbor.trial.hooks import TrialEvent
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


async def control_command(command, *args, operation, **kwargs):
    """Retry a timed-out, idempotent Harbor control command once under container load."""
    for attempt in range(2):
        try:
            return await command(*args, timeout_sec=30, **kwargs)
        except RuntimeError as exc:
            # Harbor wraps subprocess timeouts in RuntimeError. Other errors must fail immediately.
            if not str(exc).startswith("Command timed out after "):
                raise RuntimeError(f"{operation}: {exc}") from exc
            if attempt == 1:
                raise RuntimeError(f"{operation}: timed out twice (30 seconds per attempt)") from exc
            logging.getLogger(__name__).warning("%s timed out; retrying once", operation)


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
    gateway_url: str | None = None
    admission_deadline: float = 0


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
        self.finished = False

    @property
    def extra_env(self):
        return {**super().extra_env, **self.board_env}

    async def install(self, environment):
        if not os.environ.get("SWARM_AGENT_OVERLAY"):
            return await super().install(environment)
        result = await self.exec_as_agent(
            environment,
            "mkdir -p ~/.nvm; "
            "printf '%s\\n' 'export NVM_DIR=/opt/nvm; . /opt/nvm/nvm.sh' > ~/.nvm/nvm.sh; "
            '. ~/.nvm/nvm.sh; test "$(pi --version)" = 1.4.0 && node --version',
        )
        if result.return_code:
            raise RuntimeError("Prebuilt Pi image is missing the pinned runtime")

    async def setup(self, environment):
        await super().setup(environment)
        await self.ensure_system_dependencies(environment, ("python3",))
        if not os.environ.get("SWARM_AGENT_OVERLAY"):
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
        try:
            await self.ready()
            prompt = (HERE / "team.md").read_text()
            await super().run(f"You are {self.team_name}.\n{prompt}\n\n{instruction}", environment, context)
        finally:
            self.finished = True


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
        self.active_teams = 0
        self.setup_teams = asyncio.Semaphore(2)
        self.grading_slots = asyncio.Semaphore(8)
        self.capture_loop = None
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

        @self.capture.app.middleware("http")
        async def remember_capture_loop(request, call_next):
            self.capture_loop = asyncio.get_running_loop()
            return await call_next(request)

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

    def write_status(self):
        path = self.root / "rollout-status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"active_teams": self.active_teams, "active_agents": self.active_teams * self.settings.agents})
        )
        temporary.replace(path)

    async def run(self, task_dir: Path, model: str, version: int = 0):
        cfg = self.settings
        self.active_teams += 1
        self.write_status()
        team_id = uuid.uuid4().hex
        directory = self.root / "teams" / team_id
        directory.mkdir(parents=True, mode=0o700)
        names = [f"agent{i:03}" for i in range(cfg.agents)]
        board = TeamBoard(directory / "board", names)
        trials, ready = {}, set()
        grading = set()
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
        agent_clients = []
        monitor = None
        setting_up = False
        started = time.monotonic()
        try:
            await self.setup_teams.acquire()
            setting_up = True
            await board.start()
            upstream = Upstream(llm_url=cfg.vllm_url, model=model, api_key=os.environ.get("VLLM_API_KEY"))
            client, level = await self.capture.app.state.upstreams.resolve(upstream)
            if level != "tokens":
                raise ValueError("The inference endpoint must supply exact tokens and sampling logprobs")

            async def ready_agent(name):
                nonlocal setting_up
                ready.add(name)
                if len(ready) == cfg.agents:
                    setting_up = False
                    self.setup_teams.release()
                    state["execution_started"] = time.time()
                    barrier.set()
                    active.set()
                await asyncio.wait_for(barrier.wait(), cfg.setup_timeout)

            def created(name, trial):
                trials[name] = trial

                async def admit_grading(event):
                    await self.grading_slots.acquire()
                    grading.add(name)

                trial.add_hook(TrialEvent.AGENT_END, admit_grading)
                board.bind(trial, name)
                trial.config.agent.extra_allowed_hosts.append(urlsplit(cfg.capture_url).hostname)
                # Configure the native backend before rebuilding with the board mount.
                if cfg.backend == "podman":
                    trial.config.environment.delete = False
                    trial.config.environment.import_path = "podman_runtime:SwarmPodmanEnvironment"
                    if overlay := os.environ.get("SWARM_AGENT_OVERLAY"):
                        trial.config.environment.extra_docker_compose = [overlay]
                trial._init_agent_environment()
                trial.agent.team_name = name
                trial.agent.bundle = self.bundle
                trial.agent.max_context = cfg.max_context
                trial.agent.max_output_tokens = cfg.max_output_tokens
                trial.agent.ready = lambda: ready_agent(name)

            async def watch():
                await active.wait()
                command = "python -c " + shlex.quote(
                    "import json,urllib.request; "
                    "s=json.load(urllib.request.urlopen('http://127.0.0.1:8000/state',timeout=3)); "
                    "print('TRL_STATE='+json.dumps(s))"
                )
                while True:
                    for name, trial in list(trials.items()):
                        if trial.agent.finished or jobs[names.index(name)].done():
                            continue
                        try:
                            result = await control_command(
                                trial.agent_environment.service_exec,
                                command,
                                service="puzzle",
                                operation=f"{team_id}/{name}: read trusted puzzle state",
                            )
                        except Exception:
                            if trial.agent.finished or jobs[names.index(name)].done():
                                continue  # A completed trial can remove its room during the check.
                            raise
                        if result.return_code:
                            if trial.agent.finished or jobs[names.index(name)].done():
                                continue
                            raise RuntimeError("Trusted puzzle state check failed")
                        _, marker, payload = (result.stdout or "").partition("TRL_STATE=")
                        if not marker:
                            raise RuntimeError(
                                f"Missing trusted state frame: stdout={result.stdout!r}, stderr={result.stderr!r}"
                            )
                        evidence, _ = json.JSONDecoder().raw_decode(payload)
                        if evidence.get("won") is True and evidence.get("scenario_sha256") == scenario:
                            state.update(winner=name, stop_reason="first_success", success_at=time.time())
                            (directory / "success.json").write_text(json.dumps(evidence))

                            # Pi's extension calls ctx.abort(), allowing its native run to finish and grade.
                            async def stop_agent(n, t):
                                if t.agent.finished or jobs[names.index(n)].done():
                                    return
                                try:
                                    stopped = await control_command(
                                        t.agent.exec_as_root,
                                        t.agent_environment,
                                        f"touch {STOP_FILE}",
                                        operation=f"{team_id}/{n}: signal Pi to stop",
                                    )
                                    if (
                                        stopped.return_code
                                        and not t.agent.finished
                                        and not jobs[names.index(n)].done()
                                    ):
                                        raise RuntimeError("Cannot signal Pi to stop")
                                except Exception:
                                    if not t.agent.finished and not jobs[names.index(n)].done():
                                        raise

                            await asyncio.gather(*(stop_agent(n, t) for n, t in trials.items()))
                            return
                    await asyncio.sleep(1)

            sampling = {
                "temperature": cfg.temperature,
                "top_p": 1.0,
                "top_k": -1,
            }

            async def run_agent(name):
                try:
                    return await run_trial_agent(name)
                finally:
                    if name in grading:
                        grading.remove(name)
                        self.grading_slots.release()

            async def run_trial_agent(name):
                agent_upstream, agent_client = upstream, client
                if cfg.gateway_url:
                    agent_upstream = Upstream(llm_url=f"{cfg.gateway_url}/agents/{team_id}-{name}/v1", model=model)
                    agent_client, agent_level = await self.capture.app.state.upstreams.resolve(agent_upstream)
                    agent_clients.append((agent_upstream, agent_client))
                    if agent_level != "tokens":
                        raise ValueError("Per-agent gateway must preserve exact token capture")
                return await run_rollout(
                    task_dir=task_dir,
                    harness="trl-swarm-pi",
                    sandbox=cfg.backend,
                    registry=self.capture.registry,
                    intercept_url=cfg.capture_url,
                    model=model,
                    trials_dir=directory / name,
                    capture_level=level,
                    upstream=agent_upstream,
                    inference=agent_client,
                    purpose="train",
                    sampling=sampling,
                    agent_timeout_sec=cfg.agent_timeout,
                    agent_step_limit=cfg.max_turns,
                    on_trial_created=lambda trial: created(name, trial),
                )

            for name in names:
                jobs.append(asyncio.create_task(run_agent(name)))
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
                        result = job.result()
                        name = names[jobs.index(job)]
                        (directory / f"{name}.json").write_text(result.model_dump_json(indent=2))
                        if not barrier.is_set():
                            raise RuntimeError(
                                f"{name} ended before all agents were ready: {result.exception_type}: {result.error}"
                            )
                        pending.remove(job)
            results = [job.result() for job in jobs]
            for name, result in zip(names, results, strict=True):
                (directory / f"{name}.json").write_text(result.model_dump_json(indent=2))
            winner = names.index(state["winner"]) if state["winner"] else None
            state["reward"], traces = validate_team(results, cfg.max_context, winner)
            if cfg.gateway_url:
                import httpx
                from policy_gateway import match_versions, read_records

                state["call_versions"] = {}
                async with httpx.AsyncClient(timeout=30) as http:
                    for name, result in zip(names, results, strict=True):
                        records = await read_records(
                            http,
                            f"{cfg.gateway_url}/records/{team_id}-{name}",
                            os.environ["SWARM_GATEWAY_TOKEN"],
                        )
                        state["call_versions"][name] = match_versions(result.turns, records)
                versions = [call["policy_version"] for calls in state["call_versions"].values() for call in calls]
                if not versions:
                    raise ValueError("Team has no versioned calls")
                state["policy_version"] = min(versions)
                state["newest_policy_version"] = max(versions)
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
            if cfg.gateway_url:
                import httpx

                async with httpx.AsyncClient(timeout=10) as http:
                    for name in names:
                        try:
                            await http.delete(
                                f"{cfg.gateway_url}/records/{team_id}-{name}",
                                headers={"x-swarm-control": os.environ["SWARM_GATEWAY_TOKEN"]},
                            )
                        except httpx.HTTPError:
                            pass
            if monitor:
                monitor.cancel()
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs, *([monitor] if monitor else []), return_exceptions=True)
            if setting_up:
                self.setup_teams.release()
            for agent_upstream, agent_client in agent_clients:
                self.capture.app.state.upstreams.forget(agent_upstream)
                if self.capture_loop and self.capture_loop.is_running():
                    await asyncio.wrap_future(
                        asyncio.run_coroutine_threadsafe(agent_client.aclose(), self.capture_loop)
                    )
            try:
                await board.close()
                messages = directory / "board" / "messages.jsonl"
                state["messages"] = len(messages.read_text().splitlines()) if messages.exists() else 0
            except Exception as exc:
                state.update(
                    reward=None, stop_reason="unscorable", error=f"Board cleanup failed: {type(exc).__name__}"
                )
            finally:
                self.active_teams -= 1
                self.write_status()
                state["wall_seconds"] = time.monotonic() - started
                (directory / "team.json").write_text(json.dumps(state, indent=2))
                with (self.root / "teams.jsonl").open("a") as output:
                    output.write(json.dumps(state) + "\n")
