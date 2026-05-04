#!/bin/bash
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:06:00
#SBATCH --qos=normal
#SBATCH --job-name=ep-ds-dbg3
#SBATCH --output=/fsx/amine_dirhoussi/trl/benchmark/logs/ep_ds_debug7.out
#SBATCH --error=/fsx/amine_dirhoussi/trl/benchmark/logs/ep_ds_debug7.err

source /fsx/amine_dirhoussi/trl/.venv/bin/activate
export PYTHONUNBUFFERED=1
torchrun --nproc_per_node=4 --master_port=29512 /fsx/amine_dirhoussi/trl/benchmark/test_ep_ds_loading_debug7.py
