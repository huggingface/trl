#!/bin/bash
# CPU check for W2 on NeMo-RL (hopper-cpu): (1) configs/nemo-rl/w2-M.yaml loads and validates with NeMo-RL's own config
# loader; (2) the NeMo Gym harbor_agent venv builds from its requirements.txt (as NeMo-RL builds it at spin-up);
# (3) check_harbor.py: oracle / nop / scripted-model episodes on converted tasks through Gym's own run_harbor_job.
#SBATCH -J nemo-w2-check
#SBATCH -p hopper-cpu
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/logs/%x-%j.out
set -uo pipefail
HERE=/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/nemo-rl/w2
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/nemo-rl-v0.8.0.sqsh
export RUN_DIR=/fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/w2-check-$SLURM_JOB_ID WORK=/scratch/$USER/nemo-w2-check-$SLURM_JOB_ID
mkdir -p $RUN_DIR $WORK
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_ON_NODE --mem=0 --container-image=$IMG --container-mounts=/fsx:/fsx,/scratch:/scratch \
  --container-workdir=/opt/nemo-rl --export=ALL,RUN_DIR=$RUN_DIR,WORK=$WORK bash -c '
set -x
export APPTAINER_IGNORE_FAKEROOT_COMMAND=1 APPTAINER_BIND=/etc/resolv.conf
export PATH=/opt/nemo_rl_venv/bin:$PATH HF_HUB_CACHE=/fsx/amine_dirhoussi/stable-bench-hub HF_HUB_OFFLINE=1
apptainer --version
python - <<EOF
from omegaconf import OmegaConf
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
register_omegaconf_resolvers()
c = load_config("/fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/configs/nemo-rl/w2-M.yaml")
c = MasterConfig(**OmegaConf.to_container(c, resolve=True))
import json
d = c.model_dump() if hasattr(c, "model_dump") else dict(c)
print("CFG", json.dumps({k: d[k] for k in ("grpo", "loss_fn")}, default=str)[:6000])
print("CFG policy", json.dumps({k: d["policy"][k] for k in ("model_name","train_global_batch_size","max_total_sequence_length","generation")}, default=str)[:4000])
print("CFG mcore", json.dumps({k: d["policy"]["megatron_cfg"][k] for k in ("tensor_model_parallel_size","sequence_parallel","optimizer","scheduler")}, default=str))
print("CFG env", json.dumps(d["env"], default=str)[:3000])
EOF
G=/opt/nemo-rl/3rdparty/Gym-workspace/Gym
V=$WORK/hbvenv
unset UV_PROJECT_ENVIRONMENT
cd $G/responses_api_agents/harbor_agent && uv venv -q --python 3.13 $V && VIRTUAL_ENV=$V UV_OVERRIDE='$HERE'/uv-override.txt uv pip install -q -r requirements.txt aiohttp 2>&1 | tail -5
$V/bin/python -c "import harbor, importlib.metadata as m; print(\"harbor\", m.version(\"harbor\"))"
cd $WORK && PYTHONPATH=$G timeout 4500 $V/bin/python '$HERE'/check_harbor.py /fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor-nemogym/train/tasks $WORK/jobs ${CHECK_TASKS:-task_00156_ce278a task_00193_a89fef}
for f in $(find $WORK/jobs -name trial.log | head -6); do echo "== $f"; tail -n 15 $f; done
for f in $(find $WORK/jobs -name test-stdout.txt | head -6); do echo "== $f"; tail -n 5 $f; done
cp -r $WORK/jobs $RUN_DIR/ 2>/dev/null
'
rm -rf $WORK
