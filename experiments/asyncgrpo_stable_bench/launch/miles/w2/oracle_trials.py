"""Run Harbor oracle trials on the singularity env with the miles agent function's EnvironmentConfig (sanity check)."""
import asyncio, json, os, sys, time
from pathlib import Path
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial

tasks_dir, trials_dir, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
names = sorted(p.name for p in tasks_dir.iterdir() if p.is_dir())[:n]

async def one(name):
    cfg = TrialConfig(task=TaskConfig(path=tasks_dir / name), agent=AgentConfig(name="oracle"),
                      environment=EnvironmentConfig(type=EnvironmentType("singularity"), delete=True,
                                                    kwargs=json.loads(os.environ.get("HARBOR_ENV_KWARGS", "{}"))),
                      trials_dir=trials_dir, timeout_multiplier=1.0)
    t0 = time.time()
    trial = await Trial.create(cfg)
    r = await trial.run()
    rew = r.verifier_result.rewards if r.verifier_result else None
    exc = r.exception_info.exception_type + ": " + r.exception_info.exception_message[:300] if r.exception_info else None
    print(f"TRIAL {name} reward={rew} exc={exc} wall={time.time()-t0:.1f}s", flush=True)

async def main():
    await asyncio.gather(*[one(x) for x in names])
asyncio.run(main())
