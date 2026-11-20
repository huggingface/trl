#!/bin/bash
#SBATCH --job-name=miles-w2-probe
#SBATCH --partition=hopper-cpu
#SBATCH --qos=normal
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/fsx/amine_dirhoussi/stable-bench-runs/miles/logs/%x-%j.out
# Inside the miles v0.1.3 container (Pyxis): resolve harbor@harbor-miles-v0.20.0 against the image env, install it,
# then check that the unprivileged Apptainer runs nested and that Harbor's singularity env runs oracle trials.
set -uo pipefail
IMG=/fsx/amine_dirhoussi/stable-bench-src/images/miles-v0.1.3.sqsh
W=/fsx/amine_dirhoussi/stable-bench-runs/miles/w2/probe-$SLURM_JOB_ID; mkdir -p $W
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
cp /admin/home/amine_dirhoussi/.local/bin/uv $W/uv
echo "host $(hostname) $(. /etc/os-release; echo $VERSION_ID) $(date -u)"
srun --ntasks=1 --mem=0 --cpus-per-task=$SLURM_CPUS_ON_NODE --container-image=$IMG --container-mounts=/fsx:/fsx,/scratch:/scratch --container-workdir=/root/miles \
  --export=ALL,W=$W bash -c '
set -x
. /etc/os-release; echo "container OS $VERSION_ID"; id; which python3; python3 --version
python3 -c "import sys; print(sys.prefix)"
ls -l /dev/fuse; cat /proc/sys/user/max_user_namespaces
python3 -m pip list 2>/dev/null | grep -i -E "^(pydantic|litellm|openai|httpx|fastapi|uvicorn|typer|rich|tenacity|jinja2|boto3|polars|supabase|orjson|websockets|packaging|filelock|platformdirs|requests|pyyaml) " | tee $W/image_pkgs.txt
export UV_CACHE_DIR=/fsx/amine_dirhoussi/stable-bench-runs/miles/w2/uv-cache
HSPEC="harbor @ git+https://github.com/harbor-framework/harbor@c699a7ec3bd9d8d07d779ff7f90c722a5d7a0c22"
$W/uv pip install --python $(which python3) --dry-run "$HSPEC" 2>&1 | tee $W/dryrun.txt
touch /root/w; ls -l /root/w && echo CONTAINER_WRITABLE
$W/uv pip install --python $(which python3) "$HSPEC" 2>&1 | tail -40 | tee $W/install.txt
python3 -c "import harbor, importlib.metadata as m; print(\"harbor\", m.version(\"harbor\"), harbor.__file__)"
python3 -c "import sglang, miles, litellm, pydantic, openai; print(\"imports ok\", pydantic.__version__, litellm.__version__ if hasattr(litellm,\"__version__\") else \"\", openai.__version__)"
source /fsx/amine_dirhoussi/stable-bench-src/container-tools/apptainer-env.sh
echo "AP=$AP"; apptainer --version
SIF=/fsx/amine_dirhoussi/stable-bench-src/container-tools/images/termgrade-base.sif
export LD_LIBRARY_PATH=${AP_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
time apptainer exec --fakeroot --containall --writable-tmpfs --no-mount home,tmp $SIF bash -c "id; cat /etc/os-release | head -2; python3 --version; getent hosts pypi.org"
echo "NESTED_EXEC_RC $?"
export HARBOR_ENV_KWARGS="{\"singularity_no_mount\":\"home,tmp\"}"
mkdir -p /scratch/$USER/w2probe-$SLURM_JOB_ID
time python3 /fsx/amine_dirhoussi/trl-wt-stable-bench/experiments/asyncgrpo_stable_bench/launch/miles/w2/oracle_trials.py \
   /fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor-local-sif/train/tasks $W/trials 4
echo ORACLE_RC $?
'
echo "END $(date -u)"
pkill -u $USER -f "w2probe-$SLURM_JOB_ID" 2>/dev/null; rm -rf /scratch/$USER/w2probe-$SLURM_JOB_ID
