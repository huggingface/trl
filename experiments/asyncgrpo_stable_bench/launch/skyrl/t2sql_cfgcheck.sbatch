#!/bin/bash
#SBATCH -J skyrl-t2sql-cfg
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/skyrl/logs/%x-%j.out
B=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench
export HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1
mapfile -t ARGS < <(grep -v '^\s*#' $B/configs/skyrl/t2sql-35b.overrides | grep -v '^\s*$')
/fsx/amine_dirhoussi/envs/bench-skyrl/bin/python - "${ARGS[@]}" data.train_data="['/x.parquet']" data.val_data="[]" <<'PY'
import sys
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import validate_cfg
cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
validate_cfg(cfg)
a = cfg.trainer.algorithm
print("OK", a.policy_loss_type, a.loss_reduction, a.grpo_norm_by_std, a.off_policy_correction.tis_ratio_type, cfg.generator.inference_engine.engine_init_kwargs, cfg.generator.sampling_params)
PY
