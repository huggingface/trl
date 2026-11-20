#!/bin/bash
# Look inside the NeMo-RL container for the W2 (NeMo Gym harbor_agent) route: Gym checkout, its venv tooling, apptainer.
#SBATCH -J nemo-w2-probe
#SBATCH -p hopper-cpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0:40:00
#SBATCH -o /fsx/amine_dirhoussi/stable-bench-runs/nemo-rl/logs/%x-%j.out
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/nemo-rl-v0.8.0.sqsh
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_ON_NODE --mem=0 --container-image=$IMG --container-mounts=/fsx:/fsx,/scratch:/scratch --container-workdir=/opt/nemo-rl bash -c '
export PATH=/opt/nemo_rl_venv/bin:$PATH
G=/opt/nemo-rl/3rdparty/Gym-workspace/Gym
ls $G | head; ls $G/responses_api_agents | grep -i -E "harbor|terminus"; cat $G/responses_api_agents/harbor_agent/requirements.txt
git -C /opt/nemo-rl submodule status 2>&1 | head; cat $G/.git 2>/dev/null; ls /opt | head -30; ls /opt/gym* /opt/nemo_gym* 2>/dev/null | head
which uv apptainer singularity git; uv --version
env | grep -i -E "gym|uv_|nrl" | head -20
id; cat /proc/sys/user/max_user_namespaces; unshare -U true && echo userns_ok
source /fsx/amine_dirhoussi/stable-bench-src/container-tools/apptainer-env.sh; echo AP_LIBS=$AP_LIBS; which singularity apptainer
LD_LIBRARY_PATH=$AP_LIBS apptainer --version
LD_LIBRARY_PATH=$AP_LIBS APPTAINER_TMPDIR=/tmp/ap APPTAINER_CACHEDIR=/tmp/apc timeout 300 apptainer exec /fsx/amine_dirhoussi/stable-bench-src/container-tools/images/termgrade-base.sif bash -c "which python3; python3 -c \"import uvicorn, fastapi; print(\\\"uvicorn ok\\\")\"; ls /usr/local/bin | head; which tmux"
curl -sI https://pypi.org/simple/ | head -1; curl -sI https://github.com | head -1
'
