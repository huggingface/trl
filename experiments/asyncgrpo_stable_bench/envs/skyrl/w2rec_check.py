"""CPU check for SkyRL's recommended W2 arm (configs/skyrl/w2-M-recommended.overrides), no model needed.

1. Parses the overrides exactly like SkyRL's stock main_harbor_fully_async (HarborSkyRLConfig.from_cli_overrides, then
   default.yaml deep-merged with harbor_trial_config) and prints the merged trial config.
2. Runs scripted episodes through Harbor with that merged config, the way HarborGenerator builds each trial (deepcopy,
   task.path, TrialConfig.model_validate, Trial.create, trial.run), with the agent swapped for `oracle` (gold solution)
   and `nop` (does nothing), on train tasks from the local .sif task dir. Expected, as on the TRL harness: oracle 1.0,
   nop 0.0 except tasks in workloads/agentic/nop_pass_tasks.txt.
Usage: python w2rec_check.py <overrides> <tasks dir> <trials dir> <task> [<task> ...]
"""

import asyncio
import json
import sys
from copy import deepcopy

import yaml
from examples.train_integrations.harbor.entrypoints.main_harbor import HARBOR_DEFAULT_CONFIG, HarborSkyRLConfig, _deep_merge
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from omegaconf import OmegaConf

overrides, tasks_dir, trials_dir, tasks = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
args = [l.strip() for l in open(overrides) if l.strip() and not l.strip().startswith("#")]
args += [f"data.train_data=['{tasks_dir}']", "data.val_data=[]", f"harbor_trial_config.trials_dir={trials_dir}"]
cfg = HarborSkyRLConfig.from_cli_overrides(args)
defaults = yaml.safe_load(open(HARBOR_DEFAULT_CONFIG))
cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)
template = OmegaConf.to_container(OmegaConf.create(cfg.harbor_trial_config), resolve=True)
print("merged harbor_trial_config:\n" + yaml.safe_dump(template, sort_keys=False))
print("loss", cfg.trainer.algorithm.policy_loss_type, "geo mask", cfg.trainer.algorithm.off_policy_correction.sequence_mask_metric,
      "batch", cfg.trainer.train_batch_size, "x", cfg.generator.n_samples_per_prompt, "workers", cfg.trainer.fully_async.num_parallel_generation_workers,
      "overlong_filter", cfg.generator.apply_overlong_filtering, "max_seq_len", cfg.trainer.algorithm.max_seq_len,
      "remove_microbatch_padding", cfg.trainer.remove_microbatch_padding)
nop_pass = set(open("/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/workloads/agentic/nop_pass_tasks.txt").read().split())


async def one(task, agent):
    config = deepcopy(template)
    config["task"] = {"path": f"{tasks_dir}/{task}"}
    config["agent"] = {"name": agent, "override_timeout_sec": config["agent"]["override_timeout_sec"]}
    results = await (await Trial.create(TrialConfig.model_validate(config))).run()
    exc = results.exception_info.exception_type if results.exception_info else None
    reward = results.verifier_result.rewards["reward"] if results.verifier_result and results.verifier_result.rewards else None
    return {"task": task, "agent": agent, "reward": reward, "exception": exc}


async def main():
    rows = await asyncio.gather(*[one(t, a) for t in tasks for a in ("oracle", "nop")])
    ok = True
    for r in rows:
        want = 1.0 if r["agent"] == "oracle" or r["task"] in nop_pass else 0.0
        r["expected"] = want
        ok &= r["reward"] == want and r["exception"] is None
        print(json.dumps(r))
    print("PARITY", "OK" if ok else "MISMATCH")


asyncio.run(main())
