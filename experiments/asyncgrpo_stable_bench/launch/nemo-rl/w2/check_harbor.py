"""CPU check of the W2 NeMo Gym harbor_agent route, run with the harbor_agent venv inside the NeMo-RL container.

For each task: one Harbor job per agent through NeMo Gym's own `run_harbor_job` and the job config its HarborAgent
builds (Gym SingularityEnvironment on the converted tasks, the w2-M.yaml timeouts):
- `oracle` (the task's solution) must score 1 and `nop` (does nothing) must score 0: the converted tasks' setup and
  verifier match the shared tasks (TRL's local-box validation gave 128/128 oracle passes, these tasks are not in the
  nop-pass list);
- `stub`: Terminus2NemoGym against a scripted OpenAI server that answers like NeMo-RL's vLLM HTTP server (token ids
  and logprobs in the message) with one terminus-2 reply that marks the task complete twice. It must reach the
  verifier (reward 0) and NeMo Gym's conversion must give output items carrying the stub's token ids.

    python check_harbor.py <tasks dir> <jobs dir> <task> [<task> ...]
"""

import asyncio
import json
import sys
import threading
from pathlib import Path

from aiohttp import web


sys.path.insert(0, "/opt/nemo-rl/3rdparty/Gym-workspace/Gym")
from responses_api_agents.harbor_agent.app import run_harbor_job  # noqa: E402
from responses_api_agents.harbor_agent.utils import HarborAgentUtils  # noqa: E402


REPLY = json.dumps({"analysis": "done", "plan": "nothing", "commands": [], "task_complete": True})
calls = []


async def chat(request):
    body = await request.json()
    calls.append(body)
    return web.json_response({
        "id": f"chatcmpl-{len(calls)}", "object": "chat.completion", "created": 0, "model": body.get("model"),
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": REPLY,
            "prompt_token_ids": [1, 2, 3], "generation_token_ids": [7, 8, 9], "generation_log_probs": [-0.1, -0.2, -0.3]}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
    })


def serve(port):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    loop = asyncio.new_event_loop()
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", port).start())
    loop.run_forever()


def job(tasks_dir, jobs_dir, task, agent, port):
    from harbor.models.job.config import JobConfig, LocalDatasetConfig, OrchestratorConfig
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, VerifierConfig

    if agent == "stub":
        agent_cfg = AgentConfig(
            import_path="responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym:Terminus2NemoGym",
            model_name="policy", override_timeout_sec=600.0,
            kwargs={"api_base": f"http://127.0.0.1:{port}/v1", "temperature": 1.0,
                    "responses_create_params": {"input": [], "temperature": 1.0, "top_p": 1.0, "max_output_tokens": 8192},
                    "max_turns": 30, "interleaved_thinking": False, "enable_summarize": False,
                    "proactive_summarization_threshold": 0, "collect_rollout_details": True,
                    "trajectory_config": {"raw_content": True},
                    "model_info": {"max_input_tokens": 65536, "max_output_tokens": 8192,
                                   "input_cost_per_token": 0.0, "output_cost_per_token": 0.0}})
    else:
        agent_cfg = AgentConfig(name=agent, override_timeout_sec=600.0)
    cfg = JobConfig(
        job_name=f"{agent}-{task}", jobs_dir=Path(jobs_dir), orchestrator=OrchestratorConfig(n_concurrent_trials=1, quiet=True),
        environment=EnvironmentConfig(
            import_path="responses_api_agents.harbor_agent.custom_envs.singularity.singularity:SingularityEnvironment",
            delete=False, kwargs={"workdir": "/app", "singularity_image_cache_dir": f"{jobs_dir}/sif-cache"}),
        verifier=VerifierConfig(override_timeout_sec=300.0), agents=[agent_cfg],
        datasets=[LocalDatasetConfig(path=Path(tasks_dir), task_names=[task])],
    )
    trial_dir = Path(asyncio.run(run_harbor_job(cfg.model_dump(mode="json"))))
    result = json.loads((trial_dir / "result.json").read_text())
    traj_path = trial_dir / "agent" / "trajectory.json"
    traj = json.loads(traj_path.read_text()) if traj_path.exists() else None
    reward = HarborAgentUtils.extract_reward(result.get("verifier_result"))
    out = HarborAgentUtils.trial_result_to_responses(result, traj)
    ids = [o.get("generation_token_ids") for o in out if isinstance(o, dict) and o.get("generation_token_ids")]
    exc = (result.get("exception_info") or {}).get("exception_type")
    return {"task": task, "agent": agent, "reward": reward, "exception": exc, "output_items": len(out),
            "items_with_token_ids": len(ids), "first_ids": ids[0] if ids else None, "trial": str(trial_dir)}


if __name__ == "__main__":
    tasks_dir, jobs_dir, tasks = sys.argv[1], sys.argv[2], sys.argv[3:]
    port = 18765
    threading.Thread(target=serve, args=(port,), daemon=True).start()
    rows = []
    for task in tasks:
        for agent in ("oracle", "nop", "stub"):
            try:
                row = job(tasks_dir, jobs_dir, task, agent, port)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                row = {"task": task, "agent": agent, "error": f"{type(e).__name__}: {e}"}
            print("CHECK", json.dumps(row), flush=True)
            rows.append(row)
    ok = all(
        r.get("reward") == {"oracle": 1.0, "nop": 0.0, "stub": 0.0}[r["agent"]]
        and (r["agent"] != "stub" or r.get("items_with_token_ids", 0) >= 1)
        for r in rows
    )
    print(f"stub model calls: {len(calls)}; first call keys: {sorted(calls[0]) if calls else None}")
    if calls:
        print("first call sampling:", {k: calls[0].get(k) for k in ("temperature", "top_p", "max_tokens", "max_completion_tokens")})
    print("CHECK_PASS" if ok else "CHECK_FAIL")
